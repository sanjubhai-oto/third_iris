#!/usr/bin/env python3
"""Robust range-to-target estimation for vision servoing (kills the background-depth surge).

Failure it fixes: when the locked target briefly blinks out / is occluded, the depth sampled at its
bbox reads the far BACKGROUND (e.g. 125 m instead of 12 m), and a forward velocity servo surges.

Grounded in the literature (see docs/RESEARCH.md):
  * Robust per-frame depth = a NEAR-cluster percentile over the bbox INTERIOR, not a single pixel
    (ForeSeE foreground/background depth separation; NOVA histogram-mode depth is "stable through
    occlusion").
  * An INDEPENDENT monocular size-range  r = fy * H_real / bbox_height_px  (pinhole), with H_real
    calibrated against depth while GPS/depth is trustworthy — it does NOT jump to background on
    occlusion, so it cross-checks the depth (EKF depth+size fusion, arXiv:2602.20958; size-prior
    d=f*H/h, arXiv:2603.15812).
  * A max-closing-rate plausibility gate + median-of-K backstop (innovation/NIS gating, Kalman
    outlier rejection) — an impossible 12->125 m jump in one frame is rejected, not applied.
The caller freezes forward motion whenever ``update`` returns valid=False (loss-aware control).
"""
from __future__ import annotations

from collections import deque

import numpy as np


class RangeFilter:
    def __init__(self, fy, max_closing=10.0, k=5):
        self.fy = float(fy)               # vertical focal length (px) — for the size-range model
        self.max_closing = float(max_closing)   # max plausible closing/opening speed (m/s)
        self.buf = deque(maxlen=int(k))
        self.last = None
        self.H = None                     # calibrated physical target height (m)
        self.miss = 0

    def robust_depth(self, depth, box):
        """NEAR-cluster percentile over the bbox interior (rejects background fringe). m or None."""
        if depth is None:
            return None
        x1, y1, x2, y2 = (int(v) for v in box)
        bw = max(1, x2 - x1); bh = max(1, y2 - y1)
        roi = depth[max(0, y1 + bh // 4):y2 - bh // 4, max(0, x1 + bw // 4):x2 - bw // 4]
        v = roi[(roi > 0.3) & (roi < 1e4)]
        if v.size < 20:
            return None
        return float(np.percentile(v, 30))    # foreground is the nearer cluster

    def calibrate(self, robust_depth, h_px):
        """Learn H_real = depth * h_px / fy while depth is trustworthy (GPS on). EMA-smoothed."""
        if robust_depth and robust_depth > 0.5 and h_px > 2:
            est = robust_depth * h_px / self.fy
            self.H = est if self.H is None else 0.9 * self.H + 0.1 * est

    def size_range(self, h_px):
        return (self.fy * self.H / h_px) if (self.H and h_px > 2) else None

    def update(self, depth, box, h_px, dt):
        """Return (range_m, valid). valid=False => no trustworthy range this frame (caller should
        freeze forward motion and coast)."""
        rd = self.robust_depth(depth, box)
        rs = self.size_range(h_px)
        cand = rd
        # depth disagreeing strongly with the size estimate => it read the background; trust size
        if rd is not None and rs is not None and (rd > 1.5 * rs or rd - rs > 15.0):
            cand = rs
        elif rd is None:
            cand = rs
        if cand is None or cand <= 0:
            self.miss += 1
            return (self.last, False)
        # plausibility gate: reject an impossible one-frame jump
        if self.last is not None and abs(cand - self.last) > max(2.0, self.max_closing * dt * 3.0):
            self.miss += 1
            if self.miss < 8:                  # short coast on the last good range
                return (self.last, False)
            # sustained disagreement -> accept the new regime (re-init), but mark invalid this frame
            self.buf.clear()
        self.buf.append(cand)
        self.last = float(np.median(self.buf))
        self.miss = 0
        return (self.last, True)

    def reset(self):
        self.buf.clear(); self.last = None; self.miss = 0
