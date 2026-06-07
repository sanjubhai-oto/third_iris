#!/usr/bin/env python3
"""GPS-acquire -> vision-lock -> SMOOTH auto-track, with depth ranging + shadow rejection.

Control re-engineered for steady, smooth tracking (see smooth_control.py):
  - Image-space KALMAN filter smooths the noisy bbox AND predicts (lead) to counter latency,
    and COASTS on the prediction through brief detection gaps.
  - Yaw is commanded as a RATE with slew-rate limiting + deadband (smooth slew, not instant snap).
  - Velocities are jerk-limited (rate-limited) -> smooth pitch/roll.
  - DEPTH gives range + rejects SHADOWS (flat-depth detections).

States: ACQUIRE (GPS slew until target in frame) -> LOCK -> TRACK (pure vision) -> (lost) ACQUIRE.
Chaser = PX4 (offboard via pymavlink, QGC connected). Records AVI. Logs runs/smooth_metrics.jsonl.

  python sim/airsim/acquire_track.py --alt 20 --seconds 240
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import cv2
import cosysairsim as airsim
from ultralytics import YOLO
from pymavlink import mavutil

REPO = Path(__file__).resolve().parents[2]
TRACKER = str(REPO / "perception" / "trackers" / "bytetrack_uav.yaml")
import sys
sys.path.insert(0, str(REPO / "sim" / "airsim"))
from trajectories import TRAJECTORIES                       # noqa: E402
from smooth_control import ImageKalman, deadband, rate_limit, ema  # noqa: E402

EGO_HOME = np.array([0.0, 0.0, 0.0])
TARGET_HOME = np.array([8.0, 0.0, 0.0])
MASK_VEL_YAW = 0b0000100111000111      # vel + yaw setpoint (ACQUIRE/climb)
MASK_VEL_YAWRATE = 0b0000010111000111  # vel + yaw RATE (smooth TRACK)
HFOV = 90.0

# smoothing / control gains
LEAD_T = 0.40; DB = 0.02          # predict ahead ~ system+yaw latency to cancel tracking lag
KI_YAW = 1.2; I_MAX = 0.35        # integral on bearing error -> removes steady-state lag
KP_YAW = 1.6; YR_MAX = 45.0; YR_SLEW = 130.0     # deg/s , deg/s per s
KP_VZ = 2.5; VZ_MAX = 2.5; VZ_SLEW = 5.0
KP_F = 0.5; VF_MAX = 5.0; VF_SLEW = 4.0


def quat_yaw(q):
    x, y, z, w = q.x_val, q.y_val, q.z_val, q.w_val
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def send_vel_yaw(m, vn, ve, vd, yaw_rad):
    m.mav.set_position_target_local_ned_send(0, m.target_system, m.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, MASK_VEL_YAW,
        0, 0, 0, float(vn), float(ve), float(vd), 0, 0, 0, float(yaw_rad), 0)


def send_vel_yawrate(m, vn, ve, vd, yaw_rate_rad):
    m.mav.set_position_target_local_ned_send(0, m.target_system, m.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, MASK_VEL_YAWRATE,
        0, 0, 0, float(vn), float(ve), float(vd), 0, 0, 0, 0, float(yaw_rate_rad))


def grab(ac):
    reqs = [airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True, False)]
    r = ac.client.call("simGetImages", reqs, "Ego", False)
    if not r or len(r) < 2:
        return None, None
    g = lambda d, k: d[k] if k in d else d.get(k.encode())
    s, dp = r[0], r[1]
    sw, sh = g(s, "width"), g(s, "height")
    scene = np.frombuffer(bytes(g(s, "image_data_uint8")), np.uint8).reshape(sh, sw, 3) if sw else None
    dw, dh, fd = g(dp, "width"), g(dp, "height"), g(dp, "image_data_float")
    depth = np.array(fd, np.float32).reshape(dh, dw) if (dw and fd) else None
    return scene, depth


def world_pos(ac, name, home):
    p = ac.simGetVehiclePose(name).position
    return home + np.array([p.x_val, p.y_val, p.z_val])


def is_real_object(depth, x1, y1, x2, y2, W, H):
    """Conservative shadow rejection. A SHADOW is coplanar with a finite surface AND has uniform
    depth; reject ONLY that clear case. When uncertain (sparse/noisy depth, or against sky) ACCEPT,
    so real small drones are never dropped. Returns (is_real, obj_depth, bg_depth)."""
    if depth is None:
        return True, None, None
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    inner = depth[max(0, y1 + bh // 5):y2 - bh // 5, max(0, x1 + bw // 5):x2 - bw // 5]
    iv = inner[(inner > 0.2) & (inner < 1e4)]
    if iv.size < 10:
        return True, None, None                       # can't judge -> accept
    obj_d, obj_std = float(np.median(iv)), float(np.std(iv))
    rx1, ry1, rx2, ry2 = max(0, x1 - bw), max(0, y1 - bh), min(W, x2 + bw), min(H, y2 + bh)
    ring = depth[ry1:ry2, rx1:rx2].copy()
    ring[max(0, y1 - ry1):y2 - ry1, max(0, x1 - rx1):x2 - rx1] = -1.0
    rv = ring[(ring > 0.2) & (ring < 1e4)]
    bg_d = float(np.median(rv)) if rv.size >= 10 else float("inf")
    if not math.isfinite(bg_d) or bg_d > 200:
        return True, obj_d, bg_d                       # against sky -> real
    coplanar = abs(bg_d - obj_d) < 0.8 and obj_std < 0.5   # flat & uniform == shadow on a surface
    return (not coplanar), obj_d, bg_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(REPO / "runs/train/airsim_drone/weights/best.pt"))
    ap.add_argument("--alt", type=float, default=20.0)
    ap.add_argument("--chase", type=float, default=18.0)   # larger standoff -> lower angular rate -> steadier
    ap.add_argument("--tgt-speed", type=float, default=3.5)  # target path speed (m/s)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=960)  # match training imgsz for best detection
    ap.add_argument("--per-traj", type=float, default=30.0)
    ap.add_argument("--seconds", type=float, default=240.0)
    args = ap.parse_args()

    rng = random.Random(11)
    model = YOLO(args.model)
    ac = airsim.MultirotorClient(); ac.confirmConnection()
    ac.enableApiControl(True, "Target"); ac.armDisarm(True, "Target")
    ac.takeoffAsync(vehicle_name="Target").join()
    ac.moveToZAsync(-args.alt, 3, vehicle_name="Target").join()

    def new_traj():
        name = rng.choice(list(TRAJECTORIES.keys()))
        alt = -rng.uniform(16, 24); cx, cy = rng.uniform(-12, 12), rng.uniform(-12, 12)
        wps = TRAJECTORIES[name]((cx, cy, alt), (cx + 50, cy, alt)) if name == "line" \
            else TRAJECTORIES[name]((cx, cy, alt))
        path = [airsim.Vector3r(float(n - TARGET_HOME[0]), float(e - TARGET_HOME[1]), float(d)) for n, e, d in wps]
        ac.moveOnPathAsync(path, args.tgt_speed, int(args.per_traj) + 8, vehicle_name="Target")
        print(f"=== DYNAMIC TRAJECTORY: {name} ===", flush=True)
        return name

    m = mavutil.mavlink_connection("udpin:0.0.0.0:14540"); m.wait_heartbeat()
    print(f"[info] PX4 sys={m.target_system}; arming + offboard", flush=True)
    for _ in range(20):
        send_vel_yaw(m, 0, 0, 0, 0); time.sleep(0.05)
    m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0, 1, 6, 0, 0, 0, 0, 0)
    m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
    tc = time.time()
    while time.time() - tc < 18:
        alt = -world_pos(ac, "Ego", EGO_HOME)[2]
        send_vel_yaw(m, 0, 0, -2.5 if alt < args.alt - 1 else 0.0, 0)
        if alt >= args.alt - 0.8:
            break
        time.sleep(0.05)
    print("[info] OFFBOARD. ACQUIRE -> LOCK -> TRACK (smooth)", flush=True)

    vid = REPO / "runs" / "videos"; vid.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG"); vw_raw = vw_ann = None
    sm = open(REPO / "runs" / "smooth_metrics.jsonl", "w", encoding="utf-8")
    kf = ImageKalman(q=2.5, r=0.05)
    prev_yr = prev_vz = prev_vf = 0.0; vf_s = None; i_yaw = 0.0
    cur = new_traj(); traj_t0 = time.time()
    state = "ACQUIRE"; lock_cnt = lost_cnt = shadows = 0
    trail = deque(maxlen=45); last_t = time.time(); t0 = time.time()
    try:
        while time.time() - t0 < args.seconds:
            now = time.time(); dt = min(0.3, max(0.02, now - last_t)); last_t = now
            if now - traj_t0 > args.per_traj:
                cur = new_traj(); traj_t0 = now
            scene, depth = grab(ac)
            if scene is None:
                send_vel_yaw(m, 0, 0, 0, 0); time.sleep(0.03); continue
            H, W = scene.shape[:2]; cxI, cyI = W / 2.0, H / 2.0
            if vw_raw is None:
                vw_raw = cv2.VideoWriter(str(vid / "unreal_feed.avi"), fourcc, 12.0, (W, H))
                vw_ann = cv2.VideoWriter(str(vid / "tracking_feed.avi"), fourcc, 12.0, (W, H))
            raw = scene.copy()
            res = model.track(scene, tracker=TRACKER, persist=True, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            ego = world_pos(ac, "Ego", EGO_HOME); ego_yaw = quat_yaw(ac.simGetVehiclePose("Ego").orientation)

            b = res.boxes; have = b is not None and len(b) > 0
            tcx = tcy = None; conf = 0.0; rng_m = None; shadow = False; sbox = None
            if have:
                i = int(np.argmax(b.conf.cpu().numpy())); x1, y1, x2, y2 = b.xyxy[i].tolist(); conf = float(b.conf[i])
                real, obj_d, bg_d = is_real_object(depth, x1, y1, x2, y2, W, H)
                if not real:
                    have = False; shadow = True; shadows += 1; sbox = (int(x1), int(y1), int(x2), int(y2))
                else:
                    tcx, tcy = (x1 + x2) / 2, (y1 + y2) / 2; rng_m = obj_d

            vn = ve = vd = 0.0; yaw_sp = ego_yaw; yr_cmd = 0.0; mode = "rate"
            if state == "ACQUIRE":
                tgt = world_pos(ac, "Target", TARGET_HOME)
                az = math.atan2(tgt[1] - ego[1], tgt[0] - ego[0]); yaw_sp = az
                vn, ve = 1.5 * math.cos(az), 1.5 * math.sin(az)
                vd = float(np.clip(0.4 * (tgt[2] - ego[2]), -2, 2))
                lock_cnt = lock_cnt + 1 if (have and conf >= args.conf) else 0
                if lock_cnt >= 4:
                    state = "TRACK"; lost_cnt = 0; kf.reset(); prev_yr = prev_vz = prev_vf = 0.0; i_yaw = 0.0
                    print("[STATE] LOCKED -> TRACK (smooth vision)", flush=True)
                send_vel_yaw(m, vn, ve, vd, yaw_sp); mode = "setpt"
            else:  # TRACK — GPS keeps target in frame; VISION centers precisely (KF-smoothed)
                tgt = world_pos(ac, "Target", TARGET_HOME)
                az_gps = math.atan2(tgt[1] - ego[1], tgt[0] - ego[0])
                rng_gps = float(math.hypot(tgt[0] - ego[0], tgt[1] - ego[1]))
                if have:
                    kf.update((tcx - cxI) / cxI, (tcy - cyI) / cyI, dt); lost_cnt = 0
                    exl, eyl = (float(v) for v in kf.lead(LEAD_T))
                    # VISION fine centering (KF-smoothed bearing) as a yaw setpoint -> PX4 damps it
                    bearing = math.atan(deadband(exl, DB) * math.tan(math.radians(HFOV / 2)))
                    i_yaw = float(np.clip(i_yaw + bearing * dt, -I_MAX, I_MAX))  # integral kills steady-state lag
                    yaw_sp = ego_yaw + bearing + KI_YAW * i_yaw
                    yr_cmd = math.degrees(bearing)
                    vz_t = float(np.clip(KP_VZ * deadband(eyl, DB), -VZ_MAX, VZ_MAX))
                    trail.append((int(tcx), int(tcy)))
                else:
                    lost_cnt += 1; kf.coast(dt)
                    yaw_sp = az_gps                     # GPS keeps the camera on the target
                    vz_t = float(np.clip(0.5 * (tgt[2] - ego[2]), -VZ_MAX, VZ_MAX))
                    yr_cmd = 0.0
                    if lost_cnt > 90:                    # truly gone for ~12s -> formal re-acquire
                        state = "ACQUIRE"; lock_cnt = 0; print("[STATE] LOST -> ACQUIRE (gps)", flush=True)
                vd = rate_limit(prev_vz, vz_t, VZ_SLEW * dt); prev_vz = vd
                r_use = rng_m if (have and rng_m is not None) else rng_gps
                vf_s = ema(vf_s, float(np.clip(KP_F * (r_use - args.chase), -VF_MAX, VF_MAX)), 0.3)
                vf = rate_limit(prev_vf, vf_s, VF_SLEW * dt); prev_vf = vf
                vn, ve = vf * math.cos(az_gps), vf * math.sin(az_gps)   # approach along bearing to target
                send_vel_yaw(m, vn, ve, vd, yaw_sp)

            sm.write(json.dumps({"t": round(now - t0, 2), "state": state, "vision": have,
                                 "conf": round(conf, 3), "yaw_rate_deg": round(yr_cmd, 2),
                                 "vfwd": round(float(math.hypot(vn, ve)), 2),
                                 "ex": round(float(kf.x[0]), 3) if kf.x is not None else 0.0,
                                 "range": round(rng_m, 1) if rng_m else -1, "shadows": shadows}) + "\n")
            sm.flush()

            # overlay
            ann = res.plot()
            cv2.drawMarker(ann, (int(cxI), int(cyI)), (255, 255, 255), cv2.MARKER_CROSS, 28, 2)
            if kf.x is not None and state == "TRACK":
                px, py = int(cxI + kf.x[0] * cxI), int(cyI + kf.x[1] * cyI)
                cv2.circle(ann, (px, py), 7, (0, 255, 0), 2)            # KF-smoothed target
                cv2.line(ann, (int(cxI), int(cyI)), (px, py), (0, 255, 0), 1)
            if have:
                cv2.line(ann, (int(cxI), int(cyI)), (int(tcx), int(tcy)), (0, 0, 255), 1)
            for k in range(1, len(trail)):
                cv2.line(ann, trail[k-1], trail[k], (0, 255, 255), 1)
            if shadow and sbox:
                cv2.rectangle(ann, sbox[:2], sbox[2:], (128, 128, 128), 2)
                cv2.putText(ann, "SHADOW rejected", (sbox[0], max(15, sbox[1]-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
            scol = (0, 255, 0) if state == "TRACK" else (0, 165, 255)
            rtxt = f"{rng_m:.1f}m" if rng_m else "--"
            cv2.putText(ann, f"{state} | {cur} | conf={conf:.2f} rng={rtxt} yawrate={yr_cmd:+.0f}d/s alt={-ego[2]:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, scol, 2)
            cv2.putText(ann, f"shadows_rejected={shadows}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 2)
            if depth is not None:
                dv = cv2.applyColorMap((np.clip(depth, 0, 40)/40*255).astype(np.uint8), cv2.COLORMAP_TURBO)
                dv = cv2.resize(dv, (W//4, H//4)); ann[10:10+H//4, W-10-W//4:W-10] = dv
            vw_raw.write(raw); vw_ann.write(ann)
            cv2.imshow("SMOOTH ACQUIRE->TRACK (KF + yaw-rate, vision+depth)", ann)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for vw in (vw_raw, vw_ann):
            if vw is not None:
                vw.release()
        sm.close(); cv2.destroyAllWindows()
        print("[done] videos (AVI) + runs/smooth_metrics.jsonl saved", flush=True)


if __name__ == "__main__":
    main()
