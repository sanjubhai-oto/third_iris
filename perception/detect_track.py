#!/usr/bin/env python3
"""YOLO26 detection / instance-segmentation + ByteTrack on video, images, webcam, or RTSP.

Air-to-air UAV tracking front-end for the uav-vio-track project. ByteTrack is provided
*inside* Ultralytics via `model.track(..., persist=True, tracker=...)` — no separate repo.

Examples
--------
    # Detection on a single image (smoke test)
    python perception/detect_track.py --source https://ultralytics.com/images/bus.jpg --task detect

    # Instance segmentation + tracking on a video, show window + save annotated mp4 + JSONL tracks
    python perception/detect_track.py --source drone.mp4 --task segment --show --save

    # Webcam, custom fine-tuned weights, only keep the 'uav' class
    python perception/detect_track.py --source 0 --model runs/uav/weights/best.pt --classes 0

Output
------
    runs/track/<name>/  annotated media (with --save)
    <out>/tracks.jsonl  one JSON object per detection: frame, track_id, cls, conf, xyxy, [mask_rle]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKER = REPO_ROOT / "perception" / "trackers" / "bytetrack_uav.yaml"


def default_model(task: str) -> str:
    """Pick a sensible pretrained YOLO26 checkpoint for the task."""
    return "yolo26n-seg.pt" if task == "segment" else "yolo26n.pt"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="YOLO26 + ByteTrack detect/segment/track")
    p.add_argument("--source", required=True,
                   help="image, video, dir, glob, webcam index (0), RTSP/HTTP url")
    p.add_argument("--task", choices=["detect", "segment"], default="detect")
    p.add_argument("--model", default=None,
                   help="weights path or name (default: yolo26n.pt / yolo26n-seg.pt). "
                        "Falls back to yolo11 if yolo26 weights are unavailable.")
    p.add_argument("--tracker", default=str(DEFAULT_TRACKER),
                   help="tracker yaml (default: tuned bytetrack_uav.yaml). Pass 'none' to disable tracking.")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="inference size; large helps the tiny air-to-air targets")
    p.add_argument("--conf", type=float, default=0.15, help="confidence threshold (low for small UAVs)")
    p.add_argument("--iou", type=float, default=0.6, help="NMS IoU (ignored: YOLO26 is end-to-end/NMS-free)")
    p.add_argument("--device", default=None, help="cuda device, e.g. 0, or 'cpu'")
    p.add_argument("--classes", type=int, nargs="*", default=None, help="filter to these class ids")
    p.add_argument("--show", action="store_true", help="display a live window")
    p.add_argument("--save", action="store_true", help="save annotated media")
    p.add_argument("--out", default=str(REPO_ROOT / "runs" / "track"), help="output project dir")
    p.add_argument("--name", default="exp", help="run name under --out")
    p.add_argument("--no-jsonl", action="store_true", help="do not dump per-frame track records")
    return p.parse_args(argv)


def load_model(model_name, task):
    from ultralytics import YOLO
    if model_name is None:
        model_name = default_model(task)
    try:
        return YOLO(model_name), model_name
    except Exception as e:  # weights not found / not yet published
        fallback = "yolo11n-seg.pt" if task == "segment" else "yolo11n.pt"
        print(f"[warn] could not load '{model_name}' ({e}); falling back to '{fallback}'.",
              file=sys.stderr)
        return YOLO(fallback), fallback


def mask_to_rle(mask) -> dict | None:
    """Compact COCO-style RLE so masks fit in JSONL without huge polygon dumps."""
    try:
        import numpy as np
        from pycocotools import mask as mask_utils
        m = np.asfortranarray(mask.astype("uint8"))
        rle = mask_utils.encode(m)
        rle["counts"] = rle["counts"].decode("ascii")
        return rle
    except Exception:
        return None


def main(argv=None):
    args = parse_args(argv)
    model, used = load_model(args.model, args.task)
    print(f"[info] model={used} task={args.task} imgsz={args.imgsz} conf={args.conf}")

    tracker = None if args.tracker.lower() == "none" else args.tracker
    out_dir = Path(args.out) / args.name
    jsonl_fh = None
    if not args.no_jsonl:
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_fh = open(out_dir / "tracks.jsonl", "w", encoding="utf-8")

    common = dict(
        source=args.source, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
        device=args.device, classes=args.classes, stream=True,
        show=args.show, save=args.save, project=args.out, name=args.name, exist_ok=True,
        verbose=False,
    )

    # Tracking needs the dedicated track() API (maintains Kalman state across frames via persist).
    if tracker:
        results = model.track(tracker=tracker, persist=True, **common)
    else:
        results = model.predict(**common)

    n_frames = n_dets = 0
    for frame_idx, r in enumerate(results):
        n_frames += 1
        boxes = getattr(r, "boxes", None)
        if boxes is None or boxes.id is None and tracker:
            # tracker active but nothing tracked this frame
            pass
        if boxes is None:
            continue
        ids = boxes.id
        masks = getattr(r, "masks", None)
        for i in range(len(boxes)):
            n_dets += 1
            rec = {
                "frame": frame_idx,
                "track_id": int(ids[i]) if ids is not None else None,
                "cls": int(boxes.cls[i]),
                "name": r.names.get(int(boxes.cls[i]), str(int(boxes.cls[i]))),
                "conf": round(float(boxes.conf[i]), 4),
                "xyxy": [round(float(v), 1) for v in boxes.xyxy[i].tolist()],
            }
            if masks is not None and args.task == "segment":
                rle = mask_to_rle(masks.data[i].cpu().numpy())
                if rle is not None:
                    rec["mask_rle"] = rle
            if jsonl_fh:
                jsonl_fh.write(json.dumps(rec) + "\n")

    if jsonl_fh:
        jsonl_fh.close()
        print(f"[info] wrote {out_dir / 'tracks.jsonl'}")
    print(f"[done] frames={n_frames} detections={n_dets}")


if __name__ == "__main__":
    main()
