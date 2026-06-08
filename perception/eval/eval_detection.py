#!/usr/bin/env python3
"""Detection / segmentation accuracy on a dataset's val (or test) split via Ultralytics val().

Reports mAP@50 and mAP@50-95 (box, and mask if task=segment).

    python perception/eval/eval_detection.py --data datasets/uav_det.yaml --model runs/train/uav/weights/best.pt
"""
from __future__ import annotations

import argparse


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Eval YOLO26 detection/segmentation mAP")
    p.add_argument("--data", required=True, help="dataset yaml")
    p.add_argument("--model", required=True, help="weights to evaluate")
    p.add_argument("--task", choices=["detect", "segment"], default="detect")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--device", default=0)
    return p.parse_args(argv)


def main(argv=None):
    from ultralytics import YOLO

    args = parse_args(argv)
    model = YOLO(args.model)
    m = model.val(data=args.data, task=args.task, imgsz=args.imgsz, split=args.split, device=args.device)

    print("\n=== Detection metrics ===")
    print(f"box  mAP50    : {m.box.map50:.4f}")
    print(f"box  mAP50-95 : {m.box.map:.4f}")
    if args.task == "segment" and getattr(m, "seg", None) is not None:
        print(f"mask mAP50    : {m.seg.map50:.4f}")
        print(f"mask mAP50-95 : {m.seg.map:.4f}")


if __name__ == "__main__":
    main()
