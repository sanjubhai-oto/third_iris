#!/usr/bin/env python3
"""Stream an AirSim drone camera into YOLO26 + ByteTrack (air-to-air UAV tracking).

Uses the maintained Cosys-AirSim client (`cosysairsim`, drop-in for the old `airsim` package,
works on Python 3.12) to pull frames from the AirSim binary's camera over RPC, then runs the
same YOLO26 + ByteTrack pipeline as perception/detect_track.py.

Prereqs:
  - AirSim binary running (e.g. sim/airsim/Blocks/WindowsNoEditor/Blocks.exe)
  - settings.json at ~/Documents/AirSim/settings.json (this repo ships one with Ego + Target drones)
  - venv with: cosysairsim, ultralytics, torch (CUDA), opencv

Run:
  python sim/airsim/airsim_yolo_track.py --vehicle Ego --camera front_center --task detect --show

Note on detections: the stock yolo26n.pt is COCO-trained and will NOT recognize the AirSim
quadrotor as a UAV. Pass --model with a UAV-fine-tuned checkpoint (perception/train_yolo26.py)
to actually detect/track the Target drone. Without it, this still proves the live capture→infer
→track loop end to end.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKER = REPO_ROOT / "perception" / "trackers" / "bytetrack_uav.yaml"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="AirSim camera -> YOLO26 + ByteTrack")
    p.add_argument("--vehicle", default="Ego", help="vehicle name from settings.json")
    p.add_argument("--camera", default="front_center",
                   help="camera name/id (front_center, bottom_center, 0..4)")
    p.add_argument("--model", default="yolo26n.pt",
                   help="weights (use a UAV-fine-tuned model to detect the target drone)")
    p.add_argument("--task", choices=["detect", "segment"], default="detect")
    p.add_argument("--tracker", default=str(DEFAULT_TRACKER))
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--device", default=0)
    p.add_argument("--ip", default="127.0.0.1", help="AirSim RPC host")
    p.add_argument("--show", action="store_true")
    p.add_argument("--max-frames", type=int, default=0, help="0 = run until Ctrl-C")
    p.add_argument("--out", default=str(REPO_ROOT / "runs" / "airsim"))
    return p.parse_args(argv)


def _field(d, key):
    """ImageResponse comes back as a msgpack map; keys may be str or bytes."""
    if key in d:
        return d[key]
    bkey = key.encode()
    return d.get(bkey)


def get_frame(client, vehicle, camera):
    """Return an HxWx3 BGR uint8 image from the AirSim scene camera, or None.

    Calls the raw RPC with the 2-arg signature of the Microsoft AirSim v1.8.1 server.
    (The cosysairsim 3.x wrapper sends a 3rd 'external' arg that the older server rejects.)
    AirSim structs use MSGPACK_DEFINE_MAP, so a plain dict request matches the wire format.
    """
    import cosysairsim as airsim
    reqs = [airsim.ImageRequest(camera, airsim.ImageType.Scene, False, False)]
    # 3rd arg 'external'=False. Note: passing a vehicle name that does NOT exist in
    # settings.json makes the AirSim binary hard-crash here — make sure it matches.
    responses = client.client.call("simGetImages", reqs, vehicle, False)
    if not responses:
        return None
    r = responses[0]
    h, w = _field(r, "height"), _field(r, "width")
    data = _field(r, "image_data_uint8")
    if not h or not w or not data:
        return None
    img = np.frombuffer(bytes(data), dtype=np.uint8)
    if img.size != h * w * 3:
        return None
    return img.reshape(h, w, 3)  # AirSim returns BGR


def main(argv=None):
    import cv2
    import cosysairsim as airsim
    from ultralytics import YOLO

    args = parse_args(argv)

    print(f"[info] connecting to AirSim at {args.ip} ...")
    client = airsim.MultirotorClient(ip=args.ip)
    client.confirmConnection()
    print(f"[info] connected. vehicle={args.vehicle} camera={args.camera}")

    model = YOLO(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = open(out_dir / "tracks.jsonl", "w", encoding="utf-8")

    n = 0
    t0 = time.time()
    try:
        while True:
            frame = get_frame(client, args.vehicle, args.camera)
            if frame is None:
                time.sleep(0.05)
                continue
            results = model.track(
                frame, tracker=args.tracker, persist=True,
                imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
            r = results[0]

            boxes = r.boxes
            if boxes is not None and len(boxes):
                ids = boxes.id
                for i in range(len(boxes)):
                    jsonl.write(json.dumps({
                        "frame": n,
                        "track_id": int(ids[i]) if ids is not None else None,
                        "cls": int(boxes.cls[i]),
                        "name": r.names.get(int(boxes.cls[i]), str(int(boxes.cls[i]))),
                        "conf": round(float(boxes.conf[i]), 4),
                        "xyxy": [round(float(v), 1) for v in boxes.xyxy[i].tolist()],
                    }) + "\n")

            if args.show:
                cv2.imshow("AirSim + YOLO26/ByteTrack", r.plot())
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            n += 1
            if n % 30 == 0:
                fps = n / (time.time() - t0)
                ndet = 0 if boxes is None else len(boxes)
                print(f"[info] frame {n}  {fps:.1f} FPS  dets={ndet}")
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\n[info] stopped by user")
    finally:
        jsonl.close()
        if args.show:
            cv2.destroyAllWindows()
        print(f"[done] frames={n} tracks -> {out_dir / 'tracks.jsonl'}")


if __name__ == "__main__":
    main()
