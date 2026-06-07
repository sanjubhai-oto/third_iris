#!/usr/bin/env python3
"""Multi-trajectory, fused vision+location tracking & following in AirSim.

- Target drone flies through MULTIPLE trajectory types (circle, figure8, spiral, helix, zigzag,
  random, line) in sequence.
- Ego (chaser) detects the target with the fine-tuned YOLO26 model (VISION) + ByteTrack, reads the
  target's sim pose (LOCATION), FUSES both for guidance, and follows the target.
- Real-time monitoring: per-frame metrics to JSONL + event lines on stdout (trajectory switches,
  vision lost/reacquired) for live monitoring.

  python sim/airsim/multi_traj_track.py --model runs/train/airsim_drone/weights/best.pt --alt 20
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import cv2
import cosysairsim as airsim
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[2]
TRACKER = str(REPO / "perception" / "trackers" / "bytetrack_uav.yaml")

import sys
sys.path.insert(0, str(REPO / "sim" / "airsim"))
from trajectories import TRAJECTORIES          # noqa: E402
from fusion import bbox_to_bearing, fuse_guidance, follow_velocity, relative_location  # noqa: E402

EGO_HOME = np.array([0.0, 0.0, 0.0])
TARGET_HOME = np.array([8.0, 0.0, 0.0])


def quat_yaw(q):
    # cosysairsim Quaternionr has x_val,y_val,z_val,w_val
    x, y, z, w = q.x_val, q.y_val, q.z_val, q.w_val
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def get_frame(c, vehicle="Ego", camera="front_center"):
    req = [airsim.ImageRequest(camera, airsim.ImageType.Scene, False, False)]
    r = c.client.call("simGetImages", req, vehicle, False)
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


def best_detection(res, img_w, img_h):
    """Return the highest-confidence track as dict {cx,cy,conf,track_id} or None."""
    b = res.boxes
    if b is None or len(b) == 0:
        return None
    i = int(np.argmax(b.conf.cpu().numpy()))
    x1, y1, x2, y2 = b.xyxy[i].tolist()
    tid = int(b.id[i]) if b.id is not None else -1
    return {"cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "conf": float(b.conf[i]), "track_id": tid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(REPO / "runs/train/airsim_drone/weights/best.pt"))
    ap.add_argument("--alt", type=float, default=20.0)
    ap.add_argument("--chase", type=float, default=8.0)
    ap.add_argument("--per-traj", type=float, default=28.0, help="seconds per trajectory")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--order", default="circle,figure8,spiral,helix,zigzag,random,line")
    args = ap.parse_args()

    metrics = open(REPO / "runs" / "track_metrics.jsonl", "w", encoding="utf-8")
    model = YOLO(args.model)
    print(f"[info] model: {args.model}")

    c = airsim.MultirotorClient()
    c.confirmConnection()
    for v in ("Ego", "Target"):
        c.enableApiControl(True, v)
        c.armDisarm(True, v)
    print("[info] taking off both")
    c.takeoffAsync(vehicle_name="Target").join()
    c.takeoffAsync(vehicle_name="Ego").join()
    c.moveToZAsync(-args.alt, 3, vehicle_name="Target").join()
    c.moveToZAsync(-args.alt, 3, vehicle_name="Ego").join()

    trajs = [t.strip() for t in args.order.split(",") if t.strip() in TRAJECTORIES]
    print(f"[info] trajectories: {trajs}")

    vision_ok_prev = False
    for traj in trajs:
        # World-centered trajectory -> convert to Target-local (subtract Target home).
        wps = TRAJECTORIES[traj]() if traj != "line" else TRAJECTORIES[traj]()
        path = [airsim.Vector3r(float(n - TARGET_HOME[0]), float(e - TARGET_HOME[1]), float(d))
                for (n, e, d) in wps]
        c.moveOnPathAsync(path, 6, int(args.per_traj) + 8, vehicle_name="Target")
        print(f"\n=== TRAJECTORY: {traj}  ({len(wps)} wpts) ===", flush=True)

        t0 = time.time()
        frame_i = 0
        while time.time() - t0 < args.per_traj:
            frame = get_frame(c, "Ego")
            if frame is None:
                time.sleep(0.03)
                continue
            H, W = frame.shape[:2]
            res = model.track(frame, tracker=TRACKER, persist=True,
                              imgsz=960, conf=args.conf, verbose=False)[0]
            det = best_detection(res, W, H)

            ego = world_pos(c, "Ego", EGO_HOME)
            tgt = world_pos(c, "Target", TARGET_HOME)
            ego_yaw = quat_yaw(c.simGetVehiclePose("Ego").orientation)

            vision = None
            if det is not None and det["conf"] >= args.conf:
                yaw_err, _ = bbox_to_bearing(det["cx"], det["cy"], W, H)
                vision = {"cx": det["cx"], "cy": det["cy"], "conf": det["conf"],
                          "track_id": det["track_id"], "yaw_err": yaw_err}

            g = fuse_guidance(ego, ego_yaw, vision=vision, target_xyz=tuple(tgt), img_wh=(W, H))
            vx, vy, vz, yaw_deg = follow_velocity(ego, g["desired_yaw_rad"], g["range"],
                                                  chase_dist=args.chase, alt_match_d=tgt[2])
            c.moveByVelocityAsync(float(vx), float(vy), float(vz), 0.2,
                                  yaw_mode=airsim.YawMode(False, float(yaw_deg)), vehicle_name="Ego")

            rl = relative_location(tuple(ego), tuple(tgt))
            rec = {"t": round(time.time() - t0, 2), "traj": traj, "source": g["source"],
                   "vision": vision is not None, "conf": round(det["conf"], 3) if det else 0.0,
                   "track_id": det["track_id"] if det else -1,
                   "range": round(rl["range"], 1), "alt": round(-ego[2], 1),
                   "yaw_err_deg": round(math.degrees(g.get("yaw_err_rad", 0.0)), 1)}
            metrics.write(json.dumps(rec) + "\n")
            metrics.flush()

            # event lines for real-time monitoring
            vnow = vision is not None
            if vnow and not vision_ok_prev:
                print(f"[VISION] reacquired target id={det['track_id']} conf={det['conf']:.2f} "
                      f"range={rl['range']:.1f}m", flush=True)
            elif not vnow and vision_ok_prev:
                print(f"[VISION] LOST target (falling back to location) range={rl['range']:.1f}m", flush=True)
            vision_ok_prev = vnow

            ann = res.plot()
            cv2.putText(ann, f"{traj} | src={g['source']} conf={rec['conf']:.2f} "
                             f"rng={rec['range']}m alt={rec['alt']}m", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Multi-traj fused tracking (YOLO26+ByteTrack)", ann)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                metrics.close(); cv2.destroyAllWindows(); return
            frame_i += 1
            if frame_i % 30 == 0:
                print(f"  [{traj}] t={rec['t']}s src={g['source']} conf={rec['conf']:.2f} "
                      f"rng={rec['range']}m vision={vnow}", flush=True)

    metrics.close()
    cv2.destroyAllWindows()
    c.hoverAsync(vehicle_name="Ego")
    print("[done] all trajectories complete")


if __name__ == "__main__":
    main()
