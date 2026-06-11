#!/usr/bin/env python3
"""Lightweight, instant runtime for the learned pursuit policy (deploy in AirSim / webui / hardware).

Loads either the behavior-cloned MLP (bc_policy.pt) or the RL-fine-tuned PPO (rl_policy.zip) and turns
the abstract detector+tracker observation into a body-frame command. Pure-CPU, sub-millisecond.

Observation (build with ``make_obs``): the SAME features the detector + image Kalman already produce,
so this drops straight onto the existing perception stack:

    obs = [ex, ey, vex, vey, range_err, closing]

    ex,ey     bbox center offset from image center, normalized to half-FOV (right+, down+)
    vex,vey   image-space target velocity (per second), from the Kalman tracker
    range_err (range - gap) / gap
    closing   d(range)/dt / VMAX

Returns action = [fwd, vz, yaw_rate] for moveByVelocityBodyFrameAsync(fwd, 0, vz, yaw=yaw_rate).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

VMAX, VZ_MAX, YAW_MAX = 8.0, 2.8, 60.0
ASCALE = np.array([VMAX, VZ_MAX, YAW_MAX], dtype=np.float32)
OUT = Path(__file__).resolve().parents[1] / "runs" / "policy"


def make_obs(ex, ey, vex, vey, rng, gap, fwd_prev=0.0):
    range_err = (rng - gap) / max(1e-3, gap)
    closing = 0.0 if not hasattr(make_obs, "_pr") else (rng - make_obs._pr) / 0.1 / VMAX
    make_obs._pr = rng
    return np.array([ex, ey, vex, vey, range_err, closing, fwd_prev / VMAX], dtype=np.float32)


class PursuitPolicy:
    """Unified loader. kind='rl' (default, best) or 'bc'."""

    def __init__(self, kind="rl"):
        self.kind = kind
        if kind == "rl":
            from stable_baselines3 import PPO
            self.model = PPO.load(OUT / "rl_policy.zip", device="cpu")
        else:
            import torch
            from train_imitation import MLP
            self.model = MLP()
            self.model.load_state_dict(torch.load(OUT / "bc_policy.pt", map_location="cpu"))
            self.model.eval()
            self._torch = torch

    def act(self, obs):
        """obs (6,) -> [fwd, vz, yaw_rate] body-frame command."""
        if self.kind == "rl":
            a = self.model.predict(np.asarray(obs, np.float32), deterministic=True)[0]
        else:
            with self._torch.no_grad():
                a = self.model(self._torch.from_numpy(np.asarray(obs, np.float32)).unsqueeze(0)
                               ).squeeze(0).numpy()
        return (a * ASCALE).astype(np.float32)


if __name__ == "__main__":
    # smoke: load whichever exists and print an action for a centered, too-far target
    import sys
    kind = sys.argv[1] if len(sys.argv) > 1 else "rl"
    pol = PursuitPolicy(kind)
    obs = make_obs(0.3, -0.1, 0.0, 0.0, rng=25.0, gap=12.0)
    print(f"{kind} action for ex=0.3 ey=-0.1 rng=25 gap=12 ->", pol.act(obs))
