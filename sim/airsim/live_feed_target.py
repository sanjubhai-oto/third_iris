#!/usr/bin/env python3
"""Live YOLO26+ByteTrack detection feed from the chaser (Ego) camera, while the Target circles.

Ego is PX4-backed (flown via QGC / virtual RC), so this script does NOT control Ego — it only
reads Ego's camera for the live detection feed. It controls the SimpleFlight Target to fly a
circle so there's something to watch/track.

  python sim/airsim/live_feed_target.py --alt 10 --radius 20
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import cv2
import cosysairsim as airsim
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[2]
TRACKER = str(REPO / "perception" / "trackers" / "bytetrack_uav.yaml")
TARGET_HOME = np.array([8.0, 0.0])  # from settings.json


def get_frame(client, vehicle="Ego", camera="front_center"):
    reqs = [airsim.ImageRequest(camera, airsim.ImageType.Scene, False, False)]
    r = client.client.call("simGetImages", reqs, vehicle, False)
    if not r:
        return None
    resp = r[0]
    g = lambda d, k: d[k] if k in d else d.get(k.encode())
    w, h, data = g(resp, "width"), g(resp, "height"), g(resp, "image_data_uint8")
    if not (w and h and data):
        return None
    img = np.frombuffer(bytes(data), dtype=np.uint8)
    return img.reshape(h, w, 3) if img.size == h * w * 3 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt", type=float, default=10.0)
    ap.add_argument("--radius", type=float, default=20.0)
    ap.add_argument("--model", default="yolo26n.pt")
    args = ap.parse_args()

    c = airsim.MultirotorClient()
    c.confirmConnection()

    # Control ONLY the Target (Ego is flown via QGC/virtual RC).
    c.enableApiControl(True, "Target")
    c.armDisarm(True, "Target")
    c.takeoffAsync(vehicle_name="Target").join()
    c.moveToZAsync(-args.alt, 3, vehicle_name="Target").join()
    p = c.simGetVehiclePose("Target").position
    cx, cy = TARGET_HOME[0] + p.x_val, TARGET_HOME[1] + p.y_val
    path = [airsim.Vector3r(float(cx + args.radius * math.cos(t)),
                            float(cy + args.radius * math.sin(t)),
                            float(-args.alt))
            for t in np.linspace(0, 8 * math.pi, 160)]
    c.moveOnPathAsync(path, 5, 99999, vehicle_name="Target")
    print("[info] Target circling. Live detection feed on Ego camera (press q to quit).")
    print("[info] Fly the chaser (Ego) in QGC: enable Virtual Joystick + Position mode.")

    model = YOLO(args.model)
    n = 0
    while True:
        f = get_frame(c, "Ego")
        if f is None:
            time.sleep(0.05)
            continue
        res = model.track(f, tracker=TRACKER, persist=True, imgsz=960, conf=0.2, verbose=False)
        ann = res[0].plot()
        nd = 0 if res[0].boxes is None else len(res[0].boxes)
        cv2.putText(ann, f"LIVE  Ego cam  yolo_dets={nd}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Ego LIVE feed - YOLO26 + ByteTrack", ann)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        n += 1
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
