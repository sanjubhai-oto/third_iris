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

# Curated Roboflow Universe drone/FPV/UAV datasets -> merged into one 'drone' class so the detector
# sees MANY drone types, angles, and backgrounds (incl. FPV racing quads + indoor), not just one drone.
# (slug, workspace, project) — script tries each and skips any that fail to download.
UNIVERSE = [
    ("fpv-drone",     "object-detection-aw0vm", "fpv-drone-4posq"),   # FPV racing quads
    ("drone-fpv",     "salf",                   "drone-fpv"),         # more FPV
    ("drone-mixed",   "colleage-7thf7",         "drone-dataset-pw8lv"),  # ~1.9k mixed drones
    ("drone-yolov5",  "uav-detection",          "drone-yolov5-b4787"),   # UAV detection
    ("drone-track",   "dronedetection-q7zbp",   "drone-tracking-daayo"), # the original
    # breadth: multi-class sets — class-aware normalize keeps ONLY drone-like classes; bird/plane
    # frames become useful NEGATIVES (cut false positives). ~9.8k bird-vs-drone adds many angles/scales.
    ("bird-vs-drone", "drone-detection-project", "drone-vs-bird-detection"),
    ("bird-drone-2",  "antiuav-9-aniket",        "bird-and-drone"),
    ("drone-det-a",   "drone-detection-g4d3g",   "drone-detection-a1tsf"),
]

# class names that ARE the target (collapse to class 0); everything else (bird, airplane, helicopter,
# person, ...) is DROPPED so it is not mislabelled as a drone and its frames serve as negatives.
DRONE_WORDS = ("drone", "uav", "quad", "fpv", "multirotor", "multi-rotor", "vtol", "copter")


def _drone_class_ids(root: Path):
    """Return (set_of_drone_class_ids, n_classes) from data.yaml, or None if unknown -> assume single."""
    y = root / "data.yaml"
    if not y.exists():
        return None
    try:
        import yaml
        names = yaml.safe_load(y.read_text()).get("names")
    except Exception:
        return None
    if isinstance(names, dict):
        names = {int(k): v for k, v in names.items()}
    elif isinstance(names, list):
        names = {i: v for i, v in enumerate(names)}
    else:
        return None
    keep = {i for i, nm in names.items() if any(w in str(nm).lower() for w in DRONE_WORDS)}
    return keep, len(names)


def _force_single_class(root: Path) -> int:
    """Normalize labels to a single 'drone' class (id 0), CLASS-AWARE: for multi-class datasets keep
    only drone-like classes and DROP the rest (so birds/planes aren't mislabelled as drones; their
    frames become negatives). If the dataset's classes are unknown or none look drone-like, fall back
    to collapsing all (single-class assumption). Handles detection + segmentation rows."""
    info = _drone_class_ids(root)
    n = 0
    for lbl in root.rglob("*.txt"):
        if lbl.name in ("classes.txt",):
            continue
        out = []
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                cls = int(float(parts[0]))           # skip comments / malformed rows safely
            except ValueError:
                continue
            if info is None or not info[0] or cls in info[0]:   # keep drone-like (or all if unknown)
                parts[0] = "0"; out.append(" ".join(parts))
            # else: drop this row (bird/plane/etc.)
        lbl.write_text("\n".join(out) + ("\n" if out else ""))   # empty file => background negative
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


def fetch_universe(api_key: str, fmt: str = "yolov11"):
    """Download EVERY curated Universe dataset into datasets/realworld/<slug>, single 'drone' class."""
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    ok = []
    for slug, ws, proj in UNIVERSE:
        dst = RW / slug
        try:
            project = rf.workspace(ws).project(proj)
            versions = project.versions()
            if not versions:
                print(f"[skip] {ws}/{proj}: no versions"); continue
            vnum = int(str(versions[-1].version).split("/")[-1])
            if dst.exists():
                shutil.rmtree(dst)
            print(f"[get] {ws}/{proj} v{vnum} -> {dst}")
            project.version(vnum).download(fmt, location=str(dst))
            _force_single_class(dst)
            ok.append(slug)
        except Exception as e:
            print(f"[skip] {ws}/{proj}: {e}")
    print(f"[universe] downloaded {len(ok)}/{len(UNIVERSE)}: {ok}")
    return ok


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
    """Build a combined data.yaml from ALL datasets present: every datasets/realworld/* dataset (the
    Universe drone/FPV mix + mjolnir), the synthetic AirSim set, and any captured realdrone frames."""
    sources = []
    SKIP = {"shahed"}                              # flagged junk; excluded so it can't dominate/poison
    for d in sorted(RW.iterdir()) if RW.exists() else []:
        if d.is_dir() and d.name not in SKIP and _splits(d):
            sources.append(d)
    for extra in (REPO / "datasets" / "airsim_drone", REPO / "datasets" / "realdrone"):
        if extra.exists() and (_splits(extra) or (extra / "images").exists()):
            sources.append(extra)
    if not sources:
        raise SystemExit("[err] nothing to combine — run --universe --api-key KEY first")
    out = REPO / "datasets" / "uav_real.yaml"
    # ultralytics accepts a list of image dirs for train/val
    trains, vals = [], []
    for src in sources:
        s = _splits(src)
        if "train" in s:
            trains.append(s["train"].as_posix())
        elif (src / "images").exists():            # flat dataset (e.g. captured realdrone) -> train
            trains.append((src / "images").as_posix())
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
    p.add_argument("--roboflow", action="store_true", help="download the single original Roboflow dataset")
    p.add_argument("--universe", action="store_true",
                   help="download the CURATED multi-dataset drone/FPV/UAV mix (all angles/backgrounds)")
    p.add_argument("--api-key", default=None, help="Roboflow API key (free, from app.roboflow.com)")
    p.add_argument("--format", default="yolov11", help="roboflow export format")
    p.add_argument("--combine", action="store_true", help="build combined data.yaml from all datasets present")
    args = p.parse_args(argv)

    if args.universe:
        if not args.api_key:
            raise SystemExit("[err] --universe needs --api-key (free from app.roboflow.com > settings)")
        fetch_universe(args.api_key, args.format)
    if args.roboflow:
        if not args.api_key:
            raise SystemExit("[err] --roboflow needs --api-key (free from app.roboflow.com > settings)")
        fetch_roboflow(args.api_key, args.format)
    if args.combine:
        combine()
    if not (args.roboflow or args.universe or args.combine):
        print("Get a free key at app.roboflow.com > Settings > API Key, then:")
        print("  python datasets/fetch_realworld.py --universe --api-key KEY --combine")


if __name__ == "__main__":
    main()
