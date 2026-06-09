#!/usr/bin/env python3
"""Merge real (mjolnir + shahed) + synthetic (airsim) into one YOLO detection set.

Output classes: 0=drone, 1=bird. Real-world bird-vs-drone discrimination so the tracker never
locks onto a bird (the defense-grade failure mode). Per-dataset class remap:

    mjolnir : 0 Drone            -> 0 drone
    shahed  : 0 bird             -> 1 bird
              1 not  (negative)  -> DROP label rows (image kept as hard negative)
              2 shahed           -> 0 drone
    airsim  : 0 drone            -> 0 drone   (synthetic, optional)

Labels are remapped IN PLACE (datasets are regenerable). Builds datasets/uav_realmix.yaml with a
list of train/val image dirs (ultralytics accepts lists). Optional --shahed-cap subsamples the huge
shahed train split for a fast first run.
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RW = REPO / "datasets" / "realworld"

# per-dataset: {orig_class_id: new_class_id or None(drop)}
REMAP = {
    "mjolnir": {0: 0},
    "shahed": {0: 1, 1: None, 2: 0},
    "airsim": {0: 0},
}
NAMES = ["drone", "bird"]


def remap_labels(root: Path, mapping: dict, tag: str):
    n_files = n_rows = n_drop = 0
    for split in ("train", "valid", "val", "test"):
        ld = root / split / "labels"
        if not ld.exists():
            continue
        for lbl in ld.glob("*.txt"):
            out = []
            for line in lbl.read_text().splitlines():
                p = line.split()
                if not p:
                    continue
                c = int(float(p[0]))
                new = mapping.get(c, c)
                if new is None:
                    n_drop += 1
                    continue
                p[0] = str(new)
                out.append(" ".join(p))
                n_rows += 1
            lbl.write_text("\n".join(out) + ("\n" if out else ""))
            n_files += 1
    print(f"[remap:{tag}] {n_files} files, {n_rows} boxes kept, {n_drop} dropped")


def img_dir(root: Path, split: str) -> Path | None:
    for s in ([split] if split != "val" else ["valid", "val"]):
        d = root / s / "images"
        if d.exists() and any(d.iterdir()):
            return d
    return None


def cap_split(root: Path, n_keep: int, seed: int = 0):
    """Randomly keep only n_keep images (and their labels) in train/ — move the rest to a _held dir."""
    d = root / "train" / "images"
    if not d.exists():
        return
    imgs = sorted(d.iterdir())
    if len(imgs) <= n_keep:
        return
    random.Random(seed).shuffle(imgs)
    drop = imgs[n_keep:]
    held_i = root / "train" / "_held_images"; held_l = root / "train" / "_held_labels"
    held_i.mkdir(exist_ok=True); held_l.mkdir(exist_ok=True)
    lbldir = root / "train" / "labels"
    for im in drop:
        os.replace(im, held_i / im.name)
        lb = lbldir / (im.stem + ".txt")
        if lb.exists():
            os.replace(lb, held_l / lb.name)
    print(f"[cap] {root.name}: kept {n_keep}, moved {len(drop)} train imgs to _held")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shahed-cap", type=int, default=0, help="subsample shahed train to N (0=all)")
    p.add_argument("--no-airsim", action="store_true", help="exclude synthetic AirSim set")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    sources = []
    mj = RW / "mjolnir"; sh = RW / "shahed"; ai = REPO / "datasets" / "airsim_drone"
    if mj.exists():
        remap_labels(mj, REMAP["mjolnir"], "mjolnir"); sources.append(mj)
    if sh.exists():
        if args.shahed_cap:
            cap_split(sh, args.shahed_cap, args.seed)
        remap_labels(sh, REMAP["shahed"], "shahed"); sources.append(sh)
    if ai.exists() and not args.no_airsim:
        remap_labels(ai, REMAP["airsim"], "airsim"); sources.append(ai)
    if not sources:
        raise SystemExit("[err] no datasets found under datasets/realworld — run fetch first")

    trains, vals = [], []
    for s in sources:
        t = img_dir(s, "train"); v = img_dir(s, "val")
        if t:
            trains.append(t.as_posix())
        if v:
            vals.append(v.as_posix())
    if not vals:
        vals = trains
    out = REPO / "datasets" / "uav_realmix.yaml"
    body = ["# REAL (mjolnir+shahed) + SYNTHETIC (airsim) -- detection, 2 classes",
            "train:"] + [f"  - {t}" for t in trains] + ["val:"] + [f"  - {v}" for v in vals] + \
           [f"nc: {len(NAMES)}", f"names: {NAMES}"]
    out.write_text("\n".join(body) + "\n")
    print(f"[done] {len(trains)} train dir(s), {len(vals)} val dir(s) -> {out}")
    print(out.read_text())


if __name__ == "__main__":
    main()
