#!/usr/bin/env python3
"""Convert raw UAV datasets (VOC / COCO / MOT) into YOLO format + emit a data.yaml.

All UAVs are mapped to a single class `uav` (id 0) by default so datasets can be combined.
Reads from datasets/raw/<name>/, writes to datasets/yolo/<name>/{images,labels}/{train,val}.

    python datasets/to_yolo.py --name det-fly --format voc
    python datasets/to_yolo.py --name uavdb  --format coco --seg     # keep polygons for seg head
    python datasets/to_yolo.py --combine det-fly ard100 --out datasets/uav_det.yaml

YOLO label line (detect):  cls cx cy w h            (normalized)
YOLO label line (segment): cls x1 y1 x2 y2 ... xn yn (normalized polygon)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
YOLO = ROOT / "yolo"
CLASS_NAMES = ["uav"]


# ----------------------------------------------------------------------------- VOC
def convert_voc(name, val_split=0.2):
    import xml.etree.ElementTree as ET
    src = RAW / name
    xmls = sorted(src.rglob("*.xml"))
    if not xmls:
        raise SystemExit(f"no VOC xml under {src}")
    pairs = []
    for xml in xmls:
        tree = ET.parse(xml)
        root = tree.getroot()
        size = root.find("size")
        W, H = int(size.find("width").text), int(size.find("height").text)
        fn = root.find("filename").text
        img = _find_image(src, xml, fn)
        if img is None:
            continue
        lines = []
        for obj in root.findall("object"):
            b = obj.find("bndbox")
            x1, y1 = float(b.find("xmin").text), float(b.find("ymin").text)
            x2, y2 = float(b.find("xmax").text), float(b.find("ymax").text)
            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            w, h = (x2 - x1) / W, (y2 - y1) / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        pairs.append((img, lines))
    _write_split(name, pairs, val_split)


# ----------------------------------------------------------------------------- COCO
def convert_coco(name, seg=False, val_split=0.2):
    src = RAW / name
    jsons = sorted(src.rglob("*.json"))
    if not jsons:
        raise SystemExit(f"no COCO json under {src}")
    pairs = []
    for jf in jsons:
        coco = json.loads(jf.read_text(encoding="utf-8"))
        imgs = {im["id"]: im for im in coco["images"]}
        anns_by_img = {}
        for a in coco["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)
        for iid, im in imgs.items():
            img = _find_image(src, jf, im["file_name"])
            if img is None:
                continue
            W, H = im["width"], im["height"]
            lines = []
            for a in anns_by_img.get(iid, []):
                if seg and a.get("segmentation"):
                    poly = a["segmentation"][0]  # first polygon
                    norm = []
                    for i, v in enumerate(poly):
                        norm.append(v / W if i % 2 == 0 else v / H)
                    lines.append("0 " + " ".join(f"{v:.6f}" for v in norm))
                else:
                    x, y, w, h = a["bbox"]
                    cx, cy = (x + w / 2) / W, (y + h / 2) / H
                    lines.append(f"0 {cx:.6f} {cy:.6f} {w/W:.6f} {h/H:.6f}")
            pairs.append((img, lines))
    _write_split(name, pairs, val_split)


# ----------------------------------------------------------------------------- MOT
def convert_mot(name, val_split=0.2):
    """MOT layout: <seq>/img1/*.jpg + <seq>/gt/gt.txt (frame,id,x,y,w,h,...)."""
    import cv2
    src = RAW / name
    seqs = [d for d in src.iterdir() if (d / "img1").is_dir()]
    if not seqs:
        raise SystemExit(f"no MOT sequences (expect <seq>/img1/) under {src}")
    pairs = []
    for seq in seqs:
        gt = seq / "gt" / "gt.txt"
        per_frame = {}
        if gt.exists():
            for line in gt.read_text().splitlines():
                p = line.split(",")
                fr = int(float(p[0]))
                x, y, w, h = (float(p[i]) for i in range(2, 6))
                per_frame.setdefault(fr, []).append((x, y, w, h))
        for img in sorted((seq / "img1").glob("*.jpg")):
            fr = int(img.stem)
            im = cv2.imread(str(img))
            if im is None:
                continue
            H, W = im.shape[:2]
            lines = []
            for (x, y, w, h) in per_frame.get(fr, []):
                cx, cy = (x + w / 2) / W, (y + h / 2) / H
                lines.append(f"0 {cx:.6f} {cy:.6f} {w/W:.6f} {h/H:.6f}")
            pairs.append((img, lines))
    _write_split(name, pairs, val_split)


# ----------------------------------------------------------------------------- helpers
def _find_image(src, ann_path, filename):
    cand = ann_path.parent / filename
    if cand.exists():
        return cand
    hits = list(src.rglob(Path(filename).name))
    return hits[0] if hits else None


def _write_split(name, pairs, val_split):
    out = YOLO / name
    # Strided val pick so the split is representative even when raw data is ordered by sequence.
    stride = max(2, round(1 / val_split)) if val_split > 0 else 0
    n_val = sum(1 for i in range(len(pairs)) if stride and i % stride == 0)
    for idx, (img, lines) in enumerate(pairs):
        split = "val" if stride and idx % stride == 0 else "train"
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, img_dir / img.name)
        (lbl_dir / (img.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] {name}: {len(pairs)} images -> {out}  (val={n_val})")
    _write_yaml(out, out / "data.yaml")


def _write_yaml(dataset_dir, yaml_path):
    yaml_path.write_text(
        f"path: {dataset_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n",
        encoding="utf-8")
    print(f"[ok] wrote {yaml_path}")


def combine(names, out_yaml):
    """Emit one data.yaml whose train/val lists point at multiple converted datasets."""
    trains, vals = [], []
    for n in names:
        d = YOLO / n
        if not d.exists():
            raise SystemExit(f"{d} missing; convert {n} first")
        trains.append((d / "images" / "train").as_posix())
        vals.append((d / "images" / "val").as_posix())
    Path(out_yaml).write_text(
        f"train: {trains}\nval: {vals}\nnc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n",
        encoding="utf-8")
    print(f"[ok] combined {names} -> {out_yaml}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Convert UAV datasets to YOLO format")
    p.add_argument("--name")
    p.add_argument("--format", choices=["voc", "coco", "mot"])
    p.add_argument("--seg", action="store_true", help="(coco) keep polygons for segmentation")
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--combine", nargs="+", help="combine already-converted datasets")
    p.add_argument("--out", default=str(ROOT / "uav_det.yaml"))
    args = p.parse_args(argv)

    if args.combine:
        combine(args.combine, args.out)
        return
    if not (args.name and args.format):
        raise SystemExit("provide --name and --format, or --combine")
    {"voc": lambda: convert_voc(args.name, args.val_split),
     "coco": lambda: convert_coco(args.name, args.seg, args.val_split),
     "mot": lambda: convert_mot(args.name, args.val_split)}[args.format]()


if __name__ == "__main__":
    main()
