#!/usr/bin/env python3
"""AirSim 2-drone visual tracking + following: Ego chases a moving Target.

- Target flies a circle.
- Ego runs YOLO26 + ByteTrack on its camera (the same perception algorithm) and follows the
  Target, keeping it in view at a fixed chase distance.

Detection note: the stock yolo26n.pt (COCO) has no "drone" class, so the follow control is
driven by the Target's simulator pose (a perfect-perception stand-in) while YOLO26+ByteTrack
runs live on the feed. Swap in a UAV-fine-tuned model + use its detection to make the follow
fully vision-locked (the control hook is marked below).

  python sim/airsim/follow_demo.py --duration 90 --alt 15 --radius 25 --chase 6
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

# Per-vehicle spawn offsets from settings.json (simGetVehiclePose is relative to each spawn).
HOMES = {"Ego": np.array([0.0, 0.0, 0.0]), "Target": np.array([8.0, 0.0, 0.0])}


def get_frame(client, vehicle, camera="front_center"):
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


def world_pos(client, name):
    p = client.simGetVehiclePose(name).position
    return HOMES[name] + np.array([p.x_val, p.y_val, p.z_val])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt", type=float, default=15.0)
    ap.add_argument("--radius", type=float, default=25.0)
    ap.add_argument("--chase", type=float, default=6.0, help="chase distance (m)")
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--duration", type=float, default=90.0)
    args = ap.parse_args()

    client = airsim.MultirotorClient()
    client.confirmConnection()
    for v in ("Ego", "Target"):
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
    print("[info] taking off both drones")
    client.takeoffAsync(vehicle_name="Target").join()
    client.takeoffAsync(vehicle_name="Ego").join()
    client.moveToZAsync(-args.alt, 3, vehicle_name="Target").join()
    client.moveToZAsync(-args.alt, 3, vehicle_name="Ego").join()

    # Target flies a circle (non-blocking).
    c = world_pos(client, "Target")
    path = [airsim.Vector3r(float(c[0] + args.radius * math.cos(t)),
                            float(c[1] + args.radius * math.sin(t)),
                            float(-args.alt))
            for t in np.linspace(0, 4 * math.pi, 80)]
    client.moveOnPathAsync(path, 6, int(args.duration) + 10, vehicle_name="Target")
    print("[info] Target circling; Ego following with YOLO26+ByteTrack")

    model = YOLO(args.model)
    t0 = time.time()
    n = 0
    while time.time() - t0 < args.duration:
        frame = get_frame(client, "Ego")
        if frame is None:
            time.sleep(0.05)
            continue
        res = model.track(frame, tracker=TRACKER, persist=True, imgsz=960, conf=0.2, verbose=False)

        ego = world_pos(client, "Ego")
        tgt = world_pos(client, "Target")
        d = tgt - ego
        horiz = float(np.linalg.norm(d[:2]))
        dirn = d[:2] / (horiz + 1e-6)
        # ---- FOLLOW CONTROL (swap `tgt` for YOLO-derived target pos once model detects drones) ----
        desired = np.array([tgt[0] - dirn[0] * args.chase,
                            tgt[1] - dirn[1] * args.chase, tgt[2]])
        err = desired - ego
        v = np.clip(err * 0.9, -7, 7)
        yaw = math.degrees(math.atan2(d[1], d[0]))
        client.moveByVelocityAsync(float(v[0]), float(v[1]), float(v[2]), 0.2,
                                   yaw_mode=airsim.YawMode(False, yaw), vehicle_name="Ego")

        ann = res[0].plot()
        ndet = 0 if res[0].boxes is None else len(res[0].boxes)
        cv2.putText(ann, f"dist={horiz:.1f}m alt={-ego[2]:.1f}m yolo_dets={ndet}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Ego cam: YOLO26+ByteTrack (following Target)", ann)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        n += 1
        if n % 30 == 0:
            print(f"[follow] t={time.time()-t0:.0f}s dist={horiz:.1f}m alt={-ego[2]:.1f}m dets={ndet}")

    client.hoverAsync(vehicle_name="Ego")
    cv2.destroyAllWindows()
    print(f"[done] frames={n}")


if __name__ == "__main__":
    main()
