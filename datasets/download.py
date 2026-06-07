#!/usr/bin/env python3
"""Fetch (or print acquisition instructions for) air-to-air UAV datasets.

Several of these are gated behind registration or Google Drive and cannot be downloaded
unattended. This script downloads what it can and prints exact manual steps for the rest,
then expects the raw data under datasets/raw/<name>/. Convert to YOLO with to_yolo.py.

    python datasets/download.py --list
    python datasets/download.py --name det-fly
    python datasets/download.py --all
"""
from __future__ import annotations

import argparse
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"

# Each entry: annotation format hint + how to obtain. `url` is a best-effort direct link;
# `manual` lists steps when no unattended download exists.
DATASETS = {
    "det-fly": {
        "desc": "Air-to-air, ~13k images of a target multirotor from a flying camera. Boxes.",
        "ann": "voc",   # Pascal VOC xml
        "url": None,
        "manual": [
            "Repo: https://github.com/Jake-WU/Det-Fly",
            "Download the image+annotation archive from the Google Drive link in that README.",
            "Unzip so images + VOC xml land under datasets/raw/det-fly/.",
        ],
    },
    "ard100": {
        "desc": "100 air-to-air videos, extremely small targets (~0.01% of frame). Boxes per frame.",
        "ann": "mot",
        "url": None,
        "manual": [
            "Repo: https://github.com/Tianxiao-Lin/ARD-MAV  (ARD100 benchmark)",
            "Follow its README to download the videos + annotation txts.",
            "Place under datasets/raw/ard100/ (one subdir per sequence).",
        ],
    },
    "drone-vs-bird": {
        "desc": "Drone-vs-Bird Detection Challenge footage. Boxes; distinguishes drones from birds.",
        "ann": "mot",
        "url": None,
        "manual": [
            "Register at: https://wosdetc2023.wordpress.com/  (challenge dataset request).",
            "After approval, download videos + annotations to datasets/raw/drone-vs-bird/.",
        ],
    },
    "uavdb": {
        "desc": "UAV Detection & Segmentation; point-guided masks -> use for the seg head.",
        "ann": "coco",  # COCO json with segmentation
        "url": None,
        "manual": [
            "Paper/repo: search 'UAVDB Point-Guided Masks for UAV Detection and Segmentation'.",
            "Download images + COCO-style json (with 'segmentation') to datasets/raw/uavdb/.",
        ],
    },
}


def try_download(url: str, dest: Path) -> bool:
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get] {url} -> {dest}")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"[err] download failed: {e}")
        return False


def handle(name: str):
    info = DATASETS[name]
    target = RAW / name
    print(f"\n=== {name} ===\n{info['desc']}\nannotation format: {info['ann']}")
    if info["url"]:
        ok = try_download(info["url"], target / Path(info["url"]).name)
        if ok:
            print(f"[ok] saved under {target}. Now run: python datasets/to_yolo.py --name {name}")
            return
    print(f"[manual] No unattended download. Do this, then `to_yolo.py --name {name}`:")
    for step in info["manual"]:
        print(f"   - {step}")
    target.mkdir(parents=True, exist_ok=True)
    print(f"[dir] expected raw location: {target}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Fetch UAV datasets")
    p.add_argument("--list", action="store_true")
    p.add_argument("--name", choices=list(DATASETS))
    p.add_argument("--all", action="store_true")
    args = p.parse_args(argv)

    if args.list or (not args.name and not args.all):
        print("Available datasets:")
        for k, v in DATASETS.items():
            print(f"  {k:16s} {v['desc']}")
        return
    names = list(DATASETS) if args.all else [args.name]
    for n in names:
        handle(n)


if __name__ == "__main__":
    main()
