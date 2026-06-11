#!/usr/bin/env python3
"""PPO RL fine-tune the pursuit policy PAST the classical expert (hybrid: imitate -> RL).

Wraps the fast pursuit env (with first-order actuator dynamics = approx real-quad command lag, the
fidelity that matters for transfer) as a gymnasium env, warm-starts a PPO policy from the
behavior-cloned weights, and trains with 8 PARALLEL envs (the cheap realization of "multiple SITL").
Reward = keep-in-frame + tight centering + hold-standoff - control effort (already in env.step).

Eval compares EXPERT vs BC vs RL on identical episode configs, all WITH dynamics on (fair, realistic).

    python policy/rl_finetune.py --steps 800000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from pursuit_env import PursuitEnv, rollout, TRAJ, VMAX, VZ_MAX, YAW_MAX
from train_imitation import MLP, make_policy

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs" / "policy"
ASCALE = np.array([VMAX, VZ_MAX, YAW_MAX], dtype=np.float32)


class GymPursuit(gym.Env):
    """gymnasium wrapper: action in [-1,1]^3 (denormalized by ASCALE), obs 6-dim."""
    metadata = {}

    def __init__(self, seed=0):
        super().__init__()
        self.env = PursuitEnv(seed=seed, dyn=True)
        self.observation_space = spaces.Box(-np.inf, np.inf, (7,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.env.rng = np.random.default_rng(seed)
        return self.env.reset(), {}

    def step(self, action):
        obs, rew, done, info = self.env.step(np.asarray(action, np.float32) * ASCALE)
        return obs, float(rew), bool(done), False, info


def make_env(rank, seed=0):
    def _f():
        set_random_seed(seed + rank)
        return GymPursuit(seed=seed + rank)
    return _f


def warm_start(model, bc_path):
    """Copy BC hidden + action weights into the PPO policy mean network (best-effort)."""
    bc = MLP(); bc.load_state_dict(torch.load(bc_path, map_location="cpu"))
    try:
        pol = model.policy.mlp_extractor.policy_net      # [Linear,ReLU,Linear,ReLU]
        pol[0].load_state_dict(bc.net[0].state_dict())
        pol[2].load_state_dict(bc.net[2].state_dict())
        model.policy.action_net.load_state_dict(bc.net[4].state_dict())
        print("[warm] BC weights copied into PPO policy mean net")
    except Exception as e:
        print(f"[warm] skip ({e})")


def eval_three(bc_path, n=300, seed=4242):
    """EXPERT vs BC vs RL on identical configs, dynamics ON."""
    env = PursuitEnv(seed=seed, dyn=True)
    rng = np.random.default_rng(seed)
    bc = MLP(); bc.load_state_dict(torch.load(bc_path, map_location="cpu")); bc_fn = make_policy(bc)
    rl = PPO.load(OUT / "rl_policy.zip", device="cpu")
    rl_fn = lambda o: rl.predict(o, deterministic=True)[0] * ASCALE
    res = {"EXPERT": [], "BC": [], "RL": []}
    for _ in range(n):
        cfg = {"gap": float(rng.uniform(8, 16)), "tspeed": float(rng.uniform(2, 9)),
               "traj": TRAJ[rng.integers(len(TRAJ))]}
        res["EXPERT"].append(rollout(env, lambda o: env.expert(), dict(cfg)))
        res["BC"].append(rollout(env, bc_fn, dict(cfg)))
        res["RL"].append(rollout(env, rl_fn, dict(cfg)))
    print(f"  {'metric':12s} {'EXPERT':>9s} {'BC':>9s} {'RL':>9s}")
    for k in ("in_frame", "center_err", "range_err"):
        row = [np.mean([d[k] for d in res[m]]) for m in ("EXPERT", "BC", "RL")]
        print(f"  {k:12s} {row[0]:9.3f} {row[1]:9.3f} {row[2]:9.3f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=800000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--eval-eps", type=int, default=300)
    args = p.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    vec = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    model = PPO("MlpPolicy", vec, verbose=0, device="cpu", n_steps=1024, batch_size=2048,
                gae_lambda=0.95, gamma=0.995, ent_coef=0.002, learning_rate=3e-4,
                policy_kwargs=dict(activation_fn=nn.ReLU, net_arch=dict(pi=[64, 64], vf=[64, 64])))
    bc_path = OUT / "bc_policy.pt"
    if bc_path.exists():
        warm_start(model, bc_path)
    print(f"[ppo] training {args.steps:,} steps on {args.n_envs} envs (dynamics on)...")
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(OUT / "rl_policy.zip")
    print(f"[save] {OUT / 'rl_policy.zip'}")
    print(f"[eval] EXPERT vs BC vs RL on {args.eval_eps} identical episodes (dynamics on):")
    eval_three(bc_path, n=args.eval_eps)
    print("[done]")


if __name__ == "__main__":
    main()
