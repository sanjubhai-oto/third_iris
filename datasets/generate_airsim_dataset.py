#!/usr/bin/env python3
"""Generate a YOLO drone detection+segmentation dataset from AirSim ground-truth segmentation.

Teleports the Target drone to many viewpoints in the Ego camera, captures RGB + segmentation,
and auto-extracts the Target's bounding box + polygon mask (class 0 = 'drone'). Includes some
negatives (target out of view). No manual labelling.

  python datasets/generate_airsim_dataset.py --num 700 --out datasets/airsim_drone

Output: datasets/airsim_drone/{images,labels}/{train,val} + data.yaml  (YOLO-seg format)
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import cv2
import cosysairsim as airsim

TARGET_ID = 25


def _decode(resp):
    g = lambda d, k: d[k] if k in d else d.get(k.encode())
    w, h, data = g(resp, "width"), g(resp, "height"), g(resp, "image_data_uint8")
    if not (w and h and data):
        return None
    return np.frombuffer(bytes(data), dtype=np.uint8).reshape(h, w, 3)


def grab_scene_seg(c):
    """Fetch Scene + Segmentation in ONE call so they come from the same rendered frame."""
    reqs = [airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False)]
    r = c.client.call("simGetImages", reqs, "Ego", False)
    if not r or len(r) < 2:
        return None, None
    return _decode(r[0]), _decode(r[1])


def target_mask(seg):
    """Target = the non-background region. Background = the most common color (sky/ground).
    Robust to AirSim's seg colormap differing between sessions (only Target has a distinct id)."""
    flat = seg.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    bg = colors[np.argmax(counts)]
    return (np.any(np.abs(seg.astype(int) - bg) > 8, axis=2)).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=700)
    ap.add_argument("--out", default="datasets/airsim_drone")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--neg-frac", type=float, default=0.12, help="fraction with target out of view")
    ap.add_argument("--min-area", type=int, default=20, help="min target pixels to label")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out)
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    c = airsim.MultirotorClient()
    c.confirmConnection()
    # Do NOT reset() or move Ego — both corrupt the seg buffer. Requires a freshly launched
    # AirSim with Ego at its spawn (0,0,0 facing +X).
    c.simSetSegmentationObjectID(".*", 0, True)
    ok = c.simSetSegmentationObjectID("Target.*", TARGET_ID, True)
    print(f"[info] set Target seg id: {ok}")

    kept = 0
    i = 0
    while kept < args.num:
        i += 1
        # IMPORTANT: never move Ego (simSetVehiclePose on Ego corrupts the seg buffer -> black).
        # Ego stays at spawn (0,0,0) facing +X; we teleport only the Target in front of it.
        negative = random.random() < args.neg_frac
        if negative:
            c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(500, 500, -50)), True, "Target")
        else:
            # Place inside the camera FOV cone (Ego at ground, +X forward), like the verified 12,0,-3.
            d = random.uniform(4, 25)
            tx = d
            ty = random.uniform(-0.40 * d, 0.40 * d)   # horizontal within ~+-22 deg
            tz = -random.uniform(1.0, 0.5 * d)          # above ground, within vertical FOV
            tyaw = random.uniform(-math.pi, math.pi)
            c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(tx, ty, tz),
                                            airsim.Quaternionr(0, 0, math.sin(tyaw/2), math.cos(tyaw/2))),
                                True, "Target")
        time.sleep(0.25)

        scene, seg = grab_scene_seg(c)
        if scene is None or seg is None:
            continue
        H, W = scene.shape[:2]
        m = target_mask(seg)

        lines = []
        if not negative and m.sum() >= args.min_area:
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = max(cnts, key=cv2.contourArea)
            eps = 0.004 * cv2.arcLength(cnt, True)
            poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
            if len(poly) >= 3:
                norm = " ".join(f"{x/W:.6f} {y/H:.6f}" for x, y in poly)
                lines.append(f"0 {norm}")
        elif not negative:
            continue  # target meant to be visible but too small/occluded -> skip

        split = "val" if random.random() < args.val_frac else "train"
        stem = f"airsim_{kept:05d}"
        cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), scene)
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        kept += 1
        if kept % 50 == 0:
            print(f"[gen] {kept}/{args.num} (last: {'neg' if not lines else 'drone'} )")

    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        f"nc: 1\nnames: ['drone']\n", encoding="utf-8")
    print(f"[done] {kept} images -> {out}")


if __name__ == "__main__":
    main()
