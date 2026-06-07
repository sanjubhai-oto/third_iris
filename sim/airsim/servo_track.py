#!/usr/bin/env python3
"""Continuous visual-servoing tracker with dynamic trajectories + dual video recording.

Modes:
  --mode visual : FULLY VISUAL. Range is estimated from the target's bounding-box size (known
                  drone size + camera focal length); re-acquire is a visual yaw-search. NO target
                  ground truth is used at all.
  --mode fused  : vision for centering + target sim-pose for range/re-acquire (more robust).

- Target flies DYNAMIC, randomized trajectory patterns (type + params randomized each segment).
- Ego uses the fine-tuned YOLO26-SEG model (masks) + ByteTrack + image-based visual servoing to
  keep the target centered.
- Records two MP4s: runs/videos/unreal_feed.mp4 (clean photorealistic camera) and
  runs/videos/tracking_feed.mp4 (with masks/boxes/centering overlay).

  python sim/airsim/servo_track.py --mode visual --alt 20
"""
from __future__ import annotations

import argparse
import math
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import cv2
import cosysairsim as airsim
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[2]
TRACKER = str(REPO / "perception" / "trackers" / "bytetrack_uav.yaml")
import sys
sys.path.insert(0, str(REPO / "sim" / "airsim"))
from trajectories import TRAJECTORIES  # noqa: E402

EGO_HOME = np.array([0.0, 0.0, 0.0])
TARGET_HOME = np.array([8.0, 0.0, 0.0])
DRONE_SIZE_M = 0.7   # approx x500 visual width, for vision-only range estimation


def quat_yaw(q):
    x, y, z, w = q.x_val, q.y_val, q.z_val, q.w_val
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def get_frame(c, vehicle="Ego"):
    r = c.client.call("simGetImages",
                      [airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)],
                      vehicle, False)
    if not r:
        return None
    resp = r[0]
    g = lambda d, k: d[k] if k in d else d.get(k.encode())
    w, h, data = g(resp, "width"), g(resp, "height"), g(resp, "image_data_uint8")
    if not (w and h and data):
        return None
    img = np.frombuffer(bytes(data), dtype=np.uint8)
    return img.reshape(h, w, 3) if img.size == h * w * 3 else None


def world_pos(c, name, home):
    p = c.simGetVehiclePose(name).position
    return home + np.array([p.x_val, p.y_val, p.z_val])


def dynamic_trajectory(rng):
    """Pick a random trajectory type with randomized params -> (name, world-NED waypoints)."""
    alt = -rng.uniform(15, 25)
    cx, cy = rng.uniform(-15, 15), rng.uniform(-15, 15)
    name = rng.choice(list(TRAJECTORIES.keys()))
    f = TRAJECTORIES[name]
    if name == "circle":
        wps = f(center=(cx, cy, alt), radius=rng.uniform(15, 35), turns=rng.uniform(1.5, 3), points=180)
    elif name == "figure8":
        wps = f(center=(cx, cy, alt), size=rng.uniform(18, 35), turns=rng.uniform(1.5, 3), points=200)
    elif name == "spiral":
        wps = f(center=(cx, cy, alt), r0=rng.uniform(4, 8), r1=rng.uniform(25, 40),
                turns=rng.uniform(2, 4), points=220)
    elif name == "helix":
        wps = f(center=(cx, cy, alt), radius=rng.uniform(15, 28), turns=rng.uniform(2, 4),
                climb=rng.uniform(8, 18), points=220)
    elif name == "zigzag":
        wps = f(start=(cx, cy, alt), length=rng.uniform(50, 90), amplitude=rng.uniform(12, 28),
                segments=rng.randint(4, 8), points=220)
    elif name == "random":
        wps = f(center=(cx, cy, alt), extent=rng.uniform(25, 45),
                alt_range=(alt - 5, alt + 5), n=rng.randint(6, 10), points=220, seed=rng.randint(0, 9999))
    else:  # line
        ang = rng.uniform(0, 2 * math.pi)
        L = rng.uniform(40, 80)
        wps = f(start=(cx, cy, alt), end=(cx + L * math.cos(ang), cy + L * math.sin(ang), alt), points=140)
    return name, wps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(REPO / "runs/train/airsim_drone/weights/best.pt"))
    ap.add_argument("--mode", choices=["visual", "fused"], default="visual")
    ap.add_argument("--alt", type=float, default=20.0)
    ap.add_argument("--chase", type=float, default=10.0)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--per-traj", type=float, default=30.0)
    ap.add_argument("--seconds", type=float, default=300.0, help="total run time")
    args = ap.parse_args()

    rng = random.Random(7)
    model = YOLO(args.model)
    c = airsim.MultirotorClient()
    c.confirmConnection()
    for v in ("Ego", "Target"):
        c.enableApiControl(True, v)
        c.armDisarm(True, v)
    c.takeoffAsync(vehicle_name="Target").join()
    c.takeoffAsync(vehicle_name="Ego").join()
    c.moveToZAsync(-args.alt, 3, vehicle_name="Target").join()
    c.moveToZAsync(-args.alt, 3, vehicle_name="Ego").join()

    vid_dir = REPO / "runs" / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw_raw = vw_ann = None  # init on first frame

    focal_px = None
    trail = deque(maxlen=40)
    K_YAW, K_VZ, K_FWD = 90.0, 4.0, 0.6
    search_dir = 1.0
    t_start = time.time()
    traj_t0 = 0.0
    cur_traj = "-"
    print(f"[info] mode={args.mode}  recording -> {vid_dir}", flush=True)

    def new_traj():
        name, wps = dynamic_trajectory(rng)
        path = [airsim.Vector3r(float(n - TARGET_HOME[0]), float(e - TARGET_HOME[1]), float(d))
                for (n, e, d) in wps]
        c.moveOnPathAsync(path, rng.uniform(5, 8), int(args.per_traj) + 8, vehicle_name="Target")
        print(f"=== DYNAMIC TRAJECTORY: {name} ===", flush=True)
        return name

    cur_traj = new_traj(); traj_t0 = time.time()

    while time.time() - t_start < args.seconds:
        if time.time() - traj_t0 > args.per_traj:
            cur_traj = new_traj(); traj_t0 = time.time()

        frame = get_frame(c, "Ego")
        if frame is None:
            time.sleep(0.03); continue
        H, W = frame.shape[:2]
        if focal_px is None:
            focal_px = (W / 2.0) / math.tan(math.radians(90) / 2.0)  # hfov 90
        if vw_raw is None:
            vw_raw = cv2.VideoWriter(str(vid_dir / "unreal_feed.mp4"), fourcc, 12.0, (W, H))
            vw_ann = cv2.VideoWriter(str(vid_dir / "tracking_feed.mp4"), fourcc, 12.0, (W, H))
        raw = frame.copy()

        res = model.track(frame, tracker=TRACKER, persist=True, imgsz=960, conf=args.conf, verbose=False)[0]
        cxI, cyI = W / 2.0, H / 2.0
        ego = world_pos(c, "Ego", EGO_HOME)
        ego_yaw = quat_yaw(c.simGetVehiclePose("Ego").orientation)

        b = res.boxes
        have_vision = b is not None and len(b) > 0
        tcx = tcy = None
        if have_vision:
            i = int(np.argmax(b.conf.cpu().numpy()))
            x1, y1, x2, y2 = b.xyxy[i].tolist()
            tcx, tcy = (x1 + x2) / 2, (y1 + y2) / 2
            bbox_px = max(x2 - x1, y2 - y1)
            conf = float(b.conf[i])
            ex, ey = (tcx - cxI) / cxI, (tcy - cyI) / cyI

            if args.mode == "visual":
                rng_est = float(np.clip(DRONE_SIZE_M * focal_px / max(bbox_px, 1), 3, 70))  # VISION range
            else:
                tgt = world_pos(c, "Target", TARGET_HOME)
                rng_est = float(np.linalg.norm((tgt - ego)[:2]))                            # GT range

            yaw_rate = float(np.clip(K_YAW * ex, -120, 120))
            vz = float(np.clip(K_VZ * ey, -4, 4))
            vfwd = float(np.clip(K_FWD * (rng_est - args.chase), -7, 7))
            vx, vy = vfwd * math.cos(ego_yaw), vfwd * math.sin(ego_yaw)
            c.moveByVelocityAsync(vx, vy, vz, 0.15, yaw_mode=airsim.YawMode(True, yaw_rate), vehicle_name="Ego")
            trail.append((int(tcx), int(tcy)))
            source = f"VISION-SERVO (rng~{rng_est:.0f}m)"
            rng_show = rng_est
        else:
            conf = 0.0
            if args.mode == "visual":
                # FULLY-VISUAL re-acquire: spin in place to search (no ground truth)
                c.moveByVelocityAsync(0, 0, 0, 0.15, yaw_mode=airsim.YawMode(True, 50.0 * search_dir),
                                      vehicle_name="Ego")
                source = "VISUAL-SEARCH"
                rng_show = -1
            else:
                tgt = world_pos(c, "Target", TARGET_HOME)
                az = math.degrees(math.atan2(tgt[1] - ego[1], tgt[0] - ego[0]))
                rng_show = float(np.linalg.norm((tgt - ego)[:2]))
                vf = float(np.clip(K_FWD * (rng_show - args.chase), -6, 6))
                c.moveByVelocityAsync(vf * math.cos(math.radians(az)), vf * math.sin(math.radians(az)),
                                      float(np.clip(0.5 * (tgt[2] - ego[2]), -3, 3)), 0.15,
                                      yaw_mode=airsim.YawMode(False, az), vehicle_name="Ego")
                source = "LOCATION-REACQUIRE"

        # ---- overlays ----
        ann = res.plot()
        cv2.drawMarker(ann, (int(cxI), int(cyI)), (255, 255, 255), cv2.MARKER_CROSS, 30, 2)
        cv2.circle(ann, (int(cxI), int(cyI)), 60, (255, 255, 255), 1)
        if have_vision:
            cv2.circle(ann, (int(tcx), int(tcy)), 6, (0, 0, 255), -1)
            cv2.line(ann, (int(cxI), int(cyI)), (int(tcx), int(tcy)), (0, 0, 255), 2)
        for k in range(1, len(trail)):
            cv2.line(ann, trail[k - 1], trail[k], (0, 255, 255), 1)
        col = (0, 255, 0) if have_vision else (0, 165, 255)
        cv2.putText(ann, f"[{args.mode.upper()}] {cur_traj} | {source} conf={conf:.2f} alt={-ego[2]:.1f}m",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

        vw_raw.write(raw); vw_ann.write(ann)
        cv2.imshow("LIVE: visual-servo tracking (YOLO26-seg)", ann)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for vw in (vw_raw, vw_ann):
        if vw is not None:
            vw.release()
    cv2.destroyAllWindows()
    c.hoverAsync(vehicle_name="Ego")
    print(f"[done] videos saved in {vid_dir}", flush=True)


if __name__ == "__main__":
    main()
