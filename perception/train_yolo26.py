#!/usr/bin/env python3
"""Fine-tune YOLO26 (detection or instance-segmentation) on UAV data.

Starts from a pretrained YOLO26 checkpoint and fine-tunes on the combined UAV dataset
produced by datasets/to_yolo.py. Defaults are tuned for small air-to-air targets on a
12 GB GPU (RTX 5070 Ti): large imgsz, mosaic on, AdamW auto.

Examples
--------
    # Detection
    python perception/train_yolo26.py --data datasets/uav_det.yaml --task detect --epochs 100

    # Instance segmentation (needs mask labels: SAM2 auto-masks or UAVDB)
    python perception/train_yolo26.py --data datasets/uav_seg.yaml --task segment --model yolo26s-seg.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def default_model(task: str) -> str:
    return "yolo26s-seg.pt" if task == "segment" else "yolo26s.pt"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fine-tune YOLO26 on UAV data")
    p.add_argument("--data", required=True, help="dataset yaml (see datasets/to_yolo.py output)")
    p.add_argument("--task", choices=["detect", "segment"], default="detect")
    p.add_argument("--model", default=None, help="pretrained weights (default yolo26s[-seg].pt)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1280, help="large helps tiny UAV targets")
    p.add_argument("--batch", type=int, default=-1, help="-1 = auto-fit to ~60%% VRAM")
    p.add_argument("--device", default=0)
    p.add_argument("--project", default=str(REPO_ROOT / "runs" / "train"))
    p.add_argument("--name", default="uav")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--patience", type=int, default=30, help="early-stop patience")
    p.add_argument("--workers", type=int, default=3, help="dataloader workers (low -> less RAM; avoids OOM)")
    return p.parse_args(argv)


def main(argv=None):
    from ultralytics import YOLO

    args = parse_args(argv)
    model_name = args.model or default_model(args.task)
    model = YOLO(model_name)
    print(f"[info] fine-tuning {model_name} on {args.data} (task={args.task})")

    model.train(
        data=args.data,
        task=args.task,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        patience=args.patience,
        workers=args.workers,     # fewer loader workers -> lower system RAM (prev run OOM-killed)
        cache=False,              # do NOT cache images in RAM
        # small-object friendly augmentation
        mosaic=1.0,
        close_mosaic=10,      # disable mosaic for last N epochs to sharpen small-box fit
        scale=0.5,
        cos_lr=True,
        plots=True,
    )
    # Quick val summary on the dataset's val split.
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    print(f"[done] best weights under {args.project}/{args.name}/weights/best.pt")
    print(f"[val] mAP50={getattr(metrics.box, 'map50', float('nan')):.4f} "
          f"mAP50-95={getattr(metrics.box, 'map', float('nan')):.4f}")


if __name__ == "__main__":
    main()
