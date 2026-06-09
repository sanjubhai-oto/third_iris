#!/usr/bin/env python3
"""Fetch + normalize REAL air-to-air drone datasets for YOLO26 fine-tuning.

Closes the sim-to-real gap: our detector was fine-tuned only on synthetic AirSim drones, so it has
never seen a real drone against a real sky. This pulls real datasets and normalizes every label to a
single class ``0 = drone`` so they merge cleanly with the existing AirSim synthetic set.

Sources
-------
  * Roboflow  : the user's ``dronedetection-q7zbp/drone-tracking-daayo`` (needs a free API key).
  * Det-Fly   : 13,271 real drone-shot-from-drone images, CC BY 4.0 (manual link printed; hosted on
                Google Drive so it can't be reliably wget'd headless).

Usage
-----
    # Roboflow (auto-picks latest version, normalizes to single 'drone' class)
    python datasets/fetch_realworld.py --roboflow --api-key YOUR_KEY

    # then build the combined train/val yaml (real + synthetic)
    python datasets/fetch_realworld.py --combine
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RW = REPO / "datasets" / "realworld"
RF_DIR = RW / "roboflow_drone"
WS, PROJ = "dronedetection-q7zbp", "drone-tracking-daayo"


def _force_single_class(root: Path) -> int:
    """Rewrite every YOLO label under root so the class id is 0 (single 'drone' class).

    Handles both detection (5 cols) and segmentation (>=7 cols, polygon) label rows.
    Returns the number of label files touched.
    """
    n = 0
    for lbl in root.rglob("*.txt"):
        if lbl.name in ("classes.txt",):
            continue
        out = []
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            parts[0] = "0"          # collapse any class -> drone
            out.append(" ".join(parts))
        lbl.write_text("\n".join(out) + ("\n" if out else ""))
        n += 1
    return n


def fetch_roboflow(api_key: str, fmt: str = "yolov11") -> Path:
    """Download the Roboflow dataset (latest version) and collapse to single class."""
    from roboflow import Roboflow

    RF_DIR.parent.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WS).project(PROJ)
    versions = project.versions()
    if not versions:
        raise SystemExit("[err] no versions found on the Roboflow project")
    ver = versions[-1]                                  # latest
    vnum = int(str(ver.version).split("/")[-1])
    print(f"[roboflow] {WS}/{PROJ} v{vnum} -> {RF_DIR} (format={fmt})")
    if RF_DIR.exists():
        shutil.rmtree(RF_DIR)
    project.version(vnum).download(fmt, location=str(RF_DIR))
    touched = _force_single_class(RF_DIR)
    print(f"[roboflow] normalized {touched} label files to single class 'drone'")
    # rewrite the data.yaml to single class + absolute split paths ultralytics likes
    _write_yaml(RF_DIR, RF_DIR / "data.yaml")
    return RF_DIR


def _splits(root: Path):
    found = {}
    for s in ("train", "valid", "val", "test"):
        img = root / s / "images"
        if img.exists() and any(img.iterdir()):
            found[s] = img
    return found


def _write_yaml(root: Path, out: Path):
    s = _splits(root)
    train = s.get("train")
    val = s.get("valid") or s.get("val") or s.get("test") or train
    lines = [
        f"path: {root.as_posix()}",
        f"train: {train.relative_to(root).as_posix() if train else 'train/images'}",
        f"val: {val.relative_to(root).as_posix() if val else 'valid/images'}",
        "nc: 1",
        "names: ['drone']",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"[yaml] wrote {out}")


def combine():
    """Build a combined data.yaml mixing real (roboflow) + synthetic AirSim into one train/val set."""
    synth = REPO / "datasets" / "airsim_drone"
    sources = []
    if (RF_DIR / "data.yaml").exists():
        sources.append(RF_DIR)
    if synth.exists():
        sources.append(synth)
    if not sources:
        raise SystemExit("[err] nothing to combine — run --roboflow first")
    out = REPO / "datasets" / "uav_real.yaml"
    # ultralytics accepts a list of image dirs for train/val
    trains, vals = [], []
    for src in sources:
        s = _splits(src)
        if "train" in s:
            trains.append(s["train"].as_posix())
        for k in ("valid", "val", "test"):
            if k in s:
                vals.append(s[k].as_posix()); break
    if not vals:                       # fall back: use train as val (small sets)
        vals = trains
    body = ["# combined REAL (roboflow) + SYNTHETIC (airsim) — single class drone",
            "train:"] + [f"  - {t}" for t in trains] + ["val:"] + [f"  - {v}" for v in vals] + \
           ["nc: 1", "names: ['drone']"]
    out.write_text("\n".join(body) + "\n")
    print(f"[combine] {len(trains)} train dir(s), {len(vals)} val dir(s) -> {out}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roboflow", action="store_true", help="download the Roboflow drone dataset")
    p.add_argument("--api-key", default=None, help="Roboflow API key (free, from app.roboflow.com)")
    p.add_argument("--format", default="yolov11", help="roboflow export format")
    p.add_argument("--combine", action="store_true", help="build combined real+synthetic data.yaml")
    args = p.parse_args(argv)

    if args.roboflow:
        if not args.api_key:
            raise SystemExit("[err] --roboflow needs --api-key (free from app.roboflow.com > settings)")
        fetch_roboflow(args.api_key, args.format)
    if args.combine:
        combine()
    if not (args.roboflow or args.combine):
        print("Det-Fly (real air-to-air, CC BY 4.0): https://github.com/Jake-WU/Det-Fly")
        print("Run with --roboflow --api-key KEY  then  --combine")


if __name__ == "__main__":
    main()
