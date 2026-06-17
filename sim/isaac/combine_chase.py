#!/usr/bin/env python3
"""Stitch chaser-POV (left) + spectator frame-both (right) into one side-by-side video for the browser.
  .venv\\Scripts\\python.exe sim\\isaac\\combine_chase.py
"""
import os, glob, cv2, numpy as np

OUT = r"C:\Users\admin\uav-vio-track\runs\videos"
CF, SF = os.path.join(OUT, "chase_frames"), os.path.join(OUT, "spec_frames")
cs = sorted(glob.glob(os.path.join(CF, "c*.png")))
ss = sorted(glob.glob(os.path.join(SF, "s*.png")))
n = min(len(cs), len(ss))
assert n > 0, "no frames"
h = 360
def load(p):
    im = cv2.imread(p); H, W = im.shape[:2]; return cv2.resize(im, (int(W * h / H), h))
first = np.hstack([load(cs[0]), load(ss[0])])
H, W = first.shape[:2]
vw = cv2.VideoWriter(os.path.join(OUT, "chase_combined.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (W, H))
for i in range(n):
    fr = np.hstack([load(cs[i]), load(ss[i])])
    cv2.putText(fr, "CHASER POV", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(fr, "SPECTATOR (both drones)", (W // 2 + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    vw.write(fr)
vw.release()
print(f"COMBINED {n} frames -> {os.path.join(OUT, 'chase_combined.mp4')}")
