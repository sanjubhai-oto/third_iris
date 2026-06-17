#!/usr/bin/env python3
"""Build a replay video from the chaser POV frames saved by chaser_strike.py (runs/videos/a2a_frames).
  python3 sim/gz/a2a/combine_a2a.py
"""
import glob, os, cv2
OUT = "/mnt/c/Users/admin/uav-vio-track/runs/videos"
fs = sorted(glob.glob(os.path.join(OUT, "a2a_frames", "c*.png")))
assert fs, "no a2a_frames"
h, w = cv2.imread(fs[0]).shape[:2]
vw = cv2.VideoWriter(os.path.join(OUT, "a2a_chase.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (w, h))
for f in fs:
    vw.write(cv2.imread(f))
vw.release()
print(f"A2A_VIDEO {len(fs)} frames -> {os.path.join(OUT, 'a2a_chase.mp4')}")
