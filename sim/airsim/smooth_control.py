#!/usr/bin/env python3
"""Smoothing + prediction primitives for steady visual-servo tracking.

- ImageKalman: constant-velocity Kalman filter on the target's NORMALIZED image error (ex, ey in
  [-1,1]) -> smooths the noisy bbox AND predicts (lead) to counter detection latency, and can
  COAST on the prediction during brief detection gaps.
- helpers: deadband, rate_limit (slew-rate / jerk limiting), ema, pd.

Informed by UAV visual-servoing practice (Kalman target estimation + PD + rate limiting + deadband).
"""
from __future__ import annotations

import math
import numpy as np


class ImageKalman:
    def __init__(self, q=2.0, r=0.05):
        self.x = None              # [ex, ey, vex, vey]
        self.P = np.eye(4)
        self.q = q                 # process noise (target accel uncertainty)
        self.r = r                 # measurement noise (bbox jitter)
        self.alive = False
        self.miss = 0

    def reset(self):
        self.x = None
        self.alive = False
        self.miss = 0

    def _predict(self, dt):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        self.x = F @ self.x
        Q = np.zeros((4, 4))
        Q[0, 0] = Q[1, 1] = 0.25 * dt ** 4 * self.q
        Q[2, 2] = Q[3, 3] = dt ** 2 * self.q
        self.P = F @ self.P @ F.T + Q

    def update(self, ex, ey, dt):
        """Fuse a new measurement; returns smoothed (ex, ey)."""
        if self.x is None:
            self.x = np.array([ex, ey, 0.0, 0.0], float)
            self.P = np.eye(4)
            self.alive = True
            self.miss = 0
            return self.x[:2].copy()
        self._predict(max(dt, 1e-3))
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        R = np.eye(2) * self.r
        y = np.array([ex, ey], float) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.alive = True
        self.miss = 0
        return self.x[:2].copy()

    def coast(self, dt):
        """No measurement this frame: propagate the prediction. Returns (ex, ey) or None."""
        if self.x is None:
            return None
        self._predict(max(dt, 1e-3))
        self.miss += 1
        return self.x[:2].copy()

    def lead(self, t):
        """Predicted error t seconds ahead (anticipate the moving target)."""
        if self.x is None:
            return None
        return np.array([self.x[0] + self.x[2] * t, self.x[1] + self.x[3] * t])

    def speed(self):
        return 0.0 if self.x is None else float(math.hypot(self.x[2], self.x[3]))


def deadband(x, db):
    """Zero out small errors; keep continuity beyond the band."""
    if abs(x) < db:
        return 0.0
    return x - math.copysign(db, x)


def rate_limit(prev, target, max_delta):
    """Slew-rate / jerk limit: move prev toward target by at most max_delta."""
    d = max(-max_delta, min(max_delta, target - prev))
    return prev + d


def ema(prev, new, alpha):
    return new if prev is None else (alpha * new + (1 - alpha) * prev)


def pd(err, derr, kp, kd):
    return kp * err + kd * derr


if __name__ == "__main__":
    kf = ImageKalman()
    import random
    rnd = random.Random(0)
    print("step  meas_ex  smooth_ex  lead_ex")
    true = 0.0
    for i in range(8):
        true += 0.05
        meas = true + rnd.uniform(-0.04, 0.04)   # noisy bbox
        s = kf.update(meas, 0.0, 0.1)
        lead = kf.lead(0.2)
        print(f"{i:3d}  {meas:+.3f}   {s[0]:+.3f}    {lead[0]:+.3f}")
    print("coast:", kf.coast(0.1))
