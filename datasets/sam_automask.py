#!/usr/bin/env python3
"""Turn YOLO box labels into segmentation polygons using SAM 2 (box-prompted).

Air-to-air datasets are boxes-only; the YOLO26 seg head needs masks. This prompts SAM 2 with
each YOLO box, takes the best mask, and rewrites labels as normalized polygons (YOLO-seg).

Install SAM 2 first (heavy):
    pip install "git+https://github.com/facebookresearch/sam2.git"
    # + download a checkpoint, e.g. sam2.1_hiera_small.pt

    python datasets/sam_automask.py --images datasets/yolo/det-fly/images/train \
        --labels datasets/yolo/det-fly/labels/train \
        --out    datasets/yolo/det-fly-seg/labels/train \
        --ckpt   checkpoints/sam2.1_hiera_small.pt --cfg sam2.1_hiera_s.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path


def yolo_box_to_xyxy(line, W, H):
    cls, cx, cy, w, h = (float(x) for x in line.split()[:5])
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    x2 = (cx + w / 2) * W
    y2 = (cy + h / 2) * H
    return int(cls), [x1, y1, x2, y2]


def mask_to_polygon(mask):
    import cv2
    import numpy as np
    m = (mask.astype("uint8") * 255)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    eps = 0.005 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    return approx if len(approx) >= 3 else None


def main(argv=None):
    import cv2
    import numpy as np
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    p = argparse.ArgumentParser(description="SAM2 box->mask for YOLO-seg labels")
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg", required=True, help="SAM2 model config name")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    sam = build_sam2(args.cfg, args.ckpt, device=args.device)
    predictor = SAM2ImagePredictor(sam)

    img_dir, lbl_dir, out_dir = Path(args.images), Path(args.labels), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = [q for ext in ("*.jpg", "*.png", "*.jpeg") for q in img_dir.glob(ext)]

    for img_path in imgs:
        lbl = lbl_dir / (img_path.stem + ".txt")
        if not lbl.exists():
            continue
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        H, W = image.shape[:2]
        predictor.set_image(image)

        out_lines = []
        for line in lbl.read_text().splitlines():
            if not line.strip():
                continue
            cls, box = yolo_box_to_xyxy(line, W, H)
            with torch.inference_mode():
                masks, scores, _ = predictor.predict(
                    box=np.array(box)[None, :], multimask_output=False)
            poly = mask_to_polygon(masks[0])
            if poly is None:
                continue
            norm = []
            for (x, y) in poly:
                norm += [x / W, y / H]
            out_lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in norm))
        (out_dir / (img_path.stem + ".txt")).write_text("\n".join(out_lines), encoding="utf-8")
        print(f"[ok] {img_path.name}: {len(out_lines)} masks")


if __name__ == "__main__":
    main()
