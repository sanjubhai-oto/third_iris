#!/usr/bin/env python3
"""Behavior-clone the classical pursuit expert into a tiny, instant MLP policy.

Pipeline (Skydio CEILing recipe):
  1. roll out the classical EXPERT over thousands of randomized pursuit episodes (fast kinematic sim),
     logging (noisy_obs -> expert_action) pairs;
  2. train a small MLP (6 -> 64 -> 64 -> 3) to imitate;
  3. evaluate the POLICY vs the EXPERT on the SAME held-out episode configs (in-frame%, center_err,
     range_err) -- the policy should match the expert while running on noisy/dropout observations.

Runs on CPU (tiny net) so it does not contend with detector training on the GPU.

    python policy/train_imitation.py --episodes 4000 --epochs 40
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pursuit_env import PursuitEnv, rollout, VMAX, VZ_MAX, YAW_MAX

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs" / "policy"
ASCALE = np.array([VMAX, VZ_MAX, YAW_MAX], dtype=np.float32)   # action normalization


class MLP(nn.Module):
    def __init__(self, ni=7, nh=64, no=3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ni, nh), nn.ReLU(),
                                 nn.Linear(nh, nh), nn.ReLU(),
                                 nn.Linear(nh, no), nn.Tanh())   # tanh -> [-1,1], denorm by ASCALE

    def forward(self, x):
        return self.net(x)


def gen_dataset(n_eps, seed=0):
    env = PursuitEnv(seed=seed, dyn=True)            # FPV pitch + actuator lag + depth noise (AirSim-like)
    X, Y = [], []
    for _ in range(n_eps):
        obs = env.reset()
        done = False
        while not done:
            act = env.expert()
            X.append(obs.copy()); Y.append(act / ASCALE)        # normalized action label
            obs, _, done, _ = env.step(act)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def make_policy(model):
    model.eval()
    def fn(obs):
        with torch.no_grad():
            a = model(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).numpy()
        return a * ASCALE
    return fn


def evaluate(policy_fn, n=300, seed=12345):
    """Compare policy vs expert on identical episode configs."""
    env = PursuitEnv(seed=seed, dyn=True)
    rng = np.random.default_rng(seed)
    pol = {"in_frame": [], "center_err": [], "range_err": []}
    exp = {"in_frame": [], "center_err": [], "range_err": []}
    from pursuit_env import TRAJ
    for _ in range(n):
        cfg = {"gap": float(rng.uniform(8, 16)), "tspeed": float(rng.uniform(2, 9)),
               "traj": TRAJ[rng.integers(len(TRAJ))]}
        mp = rollout(env, policy_fn, dict(cfg))
        me = rollout(env, lambda o: env.expert(), dict(cfg))
        for k in pol:
            pol[k].append(mp[k]); exp[k].append(me[k])
    agg = lambda d: {k: float(np.mean(v)) for k, v in d.items()}
    return agg(pol), agg(exp)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--eval-eps", type=int, default=300)
    args = p.parse_args(argv)

    torch.manual_seed(0)
    print(f"[gen] rolling out expert over {args.episodes} episodes...")
    X, Y = gen_dataset(args.episodes)
    print(f"[gen] {len(X):,} (obs,act) pairs")
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)

    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.SmoothL1Loss()
    n = len(Xt)
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), Yt[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[bc] epoch {ep:3d}  loss {tot / n:.5f}")

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT / "bc_policy.pt")
    print(f"[save] {OUT / 'bc_policy.pt'}")

    print(f"[eval] policy vs expert on {args.eval_eps} identical episodes...")
    pol, exp = evaluate(make_policy(model), n=args.eval_eps)
    print(f"  {'metric':12s} {'EXPERT':>10s} {'POLICY(BC)':>12s}")
    for k in ("in_frame", "center_err", "range_err"):
        print(f"  {k:12s} {exp[k]:10.3f} {pol[k]:12.3f}")
    print("[done] BC policy ready for PyBullet RL fine-tune.")


if __name__ == "__main__":
    main()
