#!/usr/bin/env python3
"""Run UAV detection + tracking on a real video file (e.g. Shahed footage) and write an annotated copy.
Reports per-frame detection rate, unique track IDs, mean confidence -> honest read on the detector.

Usage:
  python perception/detect_video.py "C:/path/to/shahed.mp4" [--weights runs/train/uav_diverse/weights/best.pt]
                                    [--conf 0.25] [--imgsz 960] [--max-sec 60]
"""
import argparse, os, sys, time
import numpy as np
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--weights", default="runs/train/uav_diverse/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--max-sec", type=float, default=0, help="limit processing seconds of video (0=all)")
    ap.add_argument("--out", default="runs/videos/detect_out.mp4")
    args = ap.parse_args()
    from ultralytics import YOLO

    if not os.path.exists(args.video):
        print(f"ERROR: video not found: {args.video}"); sys.exit(1)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}"); sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {args.video}  {W}x{H} @ {fps:.0f}fps  frames={total}")
    print(f"[model] {args.weights}  imgsz={args.imgsz} conf={args.conf}")

    model = YOLO(args.weights)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    maxf = int(args.max_sec * fps) if args.max_sec > 0 else total
    n = 0; det_frames = 0; ids = set(); confs = []
    t0 = time.time()
    while n < maxf:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        r = model.track(fr, persist=True, imgsz=args.imgsz, conf=args.conf,
                        tracker="bytetrack.yaml", verbose=False)[0]
        b = r.boxes
        if b is not None and len(b):
            det_frames += 1
            hasid = b.id is not None
            for i in range(len(b)):
                x1, y1, x2, y2 = b.xyxy[i].cpu().numpy().tolist()
                c = float(b.conf[i]); confs.append(c)
                tid = int(b.id[i]) if hasid else -1
                if tid >= 0: ids.add(tid)
                cv2.rectangle(fr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(fr, f"UAV{('#'+str(tid)) if tid>=0 else ''} {c:.2f}", (int(x1), int(y1)-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(fr, f"frame {n}  det {100*det_frames/max(1,n):.0f}%  ids {len(ids)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        vw.write(fr)
        if n % 50 == 0:
            print(f"  {n}/{maxf}  det-rate {100*det_frames/n:.0f}%  ids {len(ids)}", flush=True)
    cap.release(); vw.release()
    dt = time.time() - t0
    print("\n================ DETECTION ON VIDEO ================")
    print(f"frames={n}  detection-rate={100*det_frames/max(1,n):.1f}%  unique track IDs={len(ids)}  "
          f"mean conf={np.mean(confs):.2f}" if confs else f"frames={n}  detection-rate=0%  (NO detections)")
    print(f"annotated -> {args.out}   ({n/dt:.1f} fps proc)")

if __name__ == "__main__":
    main()
