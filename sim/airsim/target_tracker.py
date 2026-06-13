#!/usr/bin/env python3
"""World-frame target trajectory estimation + prediction, and a dual-path map.

Concept (standard in vision UAV tracking — Kalman/IMM target tracking, predict-through-occlusion):
  * Each frame, the target's WORLD position is reconstructed from the chaser's own pose (VIO / IMU /
    heading / baro) + the vision bearing & range -> a world N/E/D point.
  * A constant-ACCELERATION Kalman filter smooths it and, crucially, PREDICTS it when the detector
    misses -> the chaser keeps a target estimate (location + curve) even with no visual, so it can
    keep pointing/flying toward where the target will be and re-acquire.
  * Logging both the chaser path and the target path gives the trajectory MAP (curve) of both UAVs.

TargetCA: 9-state [pos(3), vel(3), acc(3)] constant-acceleration Kalman (follows curves, not just
straight lines like a constant-velocity filter). plot_trajectories(): top-down N-E map + altitude.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


class TargetCA:
    """3-D constant-acceleration Kalman for the target's world position. Update when seen; predict
    (coast) when missed. State x = [px,py,pz, vx,vy,vz, ax,ay,az]."""

    def __init__(self, q=3.0, r=0.6):
        self.x = None              # 9-vector
        self.P = None              # 9x9 covariance
        self.q = float(q)          # process (accel) noise
        self.r = float(r)          # measurement (position) noise

    @staticmethod
    def _F(dt):
        F = np.eye(9)
        for i in range(3):
            F[i, 3 + i] = dt
            F[i, 6 + i] = 0.5 * dt * dt
            F[3 + i, 6 + i] = dt
        return F

    def _Q(self, dt):
        # white-noise-jerk style process noise on the acceleration channel
        q = self.q
        Q = np.zeros((9, 9))
        for i in range(3):
            Q[i, i] = q * dt**4 / 4
            Q[3 + i, 3 + i] = q * dt**2
            Q[6 + i, 6 + i] = q
        return Q

    def _predict(self, dt):
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

    def update(self, z, dt):
        """z = measured world position (3,). Returns the position estimate (3,). Outlier-GATED: a
        measurement implausibly far from the prediction (bad range/bearing frame) is REJECTED and we
        coast on the model instead -> rejects the spikes that wreck a raw filter."""
        z = np.asarray(z, float)
        if self.x is None:
            self.x = np.zeros(9); self.x[:3] = z
            self.P = np.eye(9) * 10.0
            return self.x[:3].copy()
        self._predict(max(1e-3, dt))
        # gate: plausible per-frame move = speed*dt; reject jumps beyond that + a margin
        v = float(np.linalg.norm(self.x[3:6]))
        gate = max(6.0, 4.0 * v * max(1e-3, dt) + 4.0)
        if float(np.linalg.norm(z - self.x[:3])) > gate:
            return self.x[:3].copy()                      # outlier -> coast on prediction
        H = np.zeros((3, 9)); H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        R = np.eye(3) * self.r
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ H) @ self.P
        return self.x[:3].copy()

    def predict_only(self, dt):
        """No detection this frame -> coast. DECAY acceleration (then velocity over long gaps) so a
        constant-accel extrapolation can't blow up to hundreds of metres during a long dropout."""
        if self.x is None:
            return None
        self.x[6:9] *= 0.5          # kill accel extrapolation fast -> coast ~constant-velocity
        self.x[3:6] *= 0.97         # slowly bleed velocity too (a long-lost target shouldn't fly off)
        self._predict(max(1e-3, dt))
        return self.x[:3].copy()

    def pos(self):
        return None if self.x is None else self.x[:3].copy()

    def vel(self):
        return None if self.x is None else self.x[3:6].copy()


def plot_trajectories(rows, out_path, title="Target & chaser trajectories"):
    """rows = list of (t, ego_ned(3), tgt_truth_ned(3), tgt_est_ned(3), seen_bool).
    Saves a top-down N-E map + altitude-vs-time, both UAV curves, predicted-while-missed highlighted."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[traj] matplotlib unavailable ({e}); skipping plot")
        return None
    if not rows:
        return None
    t = np.array([r[0] for r in rows]); t = t - t.min()
    ego = np.array([r[1] for r in rows]); tru = np.array([r[2] for r in rows])
    est = np.array([r[3] for r in rows]); seen = np.array([r[4] for r in rows], bool)

    fig, (ax, axalt) = plt.subplots(1, 2, figsize=(14, 6))
    # top-down N-E map (East = x, North = y)
    ax.plot(ego[:, 1], ego[:, 0], "-", color="#0a84ff", lw=2, label="chaser path")
    ax.plot(tru[:, 1], tru[:, 0], "-", color="#2a9d8f", lw=2, label="target TRUE path")
    ax.plot(est[:, 1], est[:, 0], "--", color="#e63946", lw=1.5, label="target ESTIMATE")
    miss = ~seen
    if miss.any():
        ax.scatter(est[miss, 1], est[miss, 0], s=14, color="#ff9f1c", zorder=5,
                   label="predicted (no visual)")
    ax.scatter([ego[0, 1]], [ego[0, 0]], c="#0a84ff", marker="o", s=60)
    ax.scatter([tru[0, 1]], [tru[0, 0]], c="#2a9d8f", marker="o", s=60)
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)"); ax.set_title(title)
    ax.axis("equal"); ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9)
    # altitude vs time
    axalt.plot(t, -ego[:, 2], "-", color="#0a84ff", lw=2, label="chaser alt")
    axalt.plot(t, -tru[:, 2], "-", color="#2a9d8f", lw=2, label="target TRUE alt")
    axalt.plot(t, -est[:, 2], "--", color="#e63946", lw=1.5, label="target EST alt")
    axalt.set_xlabel("time (s)"); axalt.set_ylabel("altitude (m)"); axalt.set_title("Altitude")
    axalt.grid(alpha=0.3); axalt.legend(loc="best", fontsize=9)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)

    # accuracy: estimate vs truth, overall and during misses (the key number for predict-through-loss)
    err = np.linalg.norm(est - tru, axis=1)
    seen_err = err[seen].mean() if seen.any() else float("nan")
    miss_err = err[miss].mean() if miss.any() else float("nan")
    print(f"[traj] map -> {out_path}")
    print(f"[traj] target world-estimate error: overall {err.mean():.2f} m | "
          f"seen {seen_err:.2f} m | predicted-during-miss {miss_err:.2f} m ({int(miss.sum())} frames)")
    return out_path
