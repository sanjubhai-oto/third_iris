#!/usr/bin/env python3
"""Web UI for click-to-lock visual tracking (FPV, body-fixed camera).

- Browser shows the live Ego camera with YOLO detections.
- Operator CLICKS a confirmed detection to LOCK it -> ego flies TOWARD it, then holds the set GAP,
  keeping it centered by yawing the whole drone (FPV; no gimbal).
- Set the gap and clear the lock from the UI; live telemetry streamed.
- Target flies dynamic trajectories AND changes altitude so you can see the ego climb to follow.

Run:  python webui/app.py    then open http://localhost:5000
"""
from __future__ import annotations

import math
import random
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import cv2
import cosysairsim as airsim
from ultralytics import YOLO
from flask import Flask, Response, jsonify, request, render_template

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO / "sim" / "airsim"))
from trajectories import TRAJECTORIES                      # noqa: E402
from smooth_control import ImageKalman, deadband, ema       # noqa: E402
from guidance import Vec3KF, target_from_vision, standoff_command, vfov_from_hfov  # noqa: E402
from avoidance import apply_avoidance, clearance_and_escape    # noqa: E402
sys.path.insert(0, str(REPO / "vio"))
from vio_estimator import VIOEstimator                         # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import VideoReceiver, MavlinkReceiver, VIDEO_PROTOS, TELEM_PROTOS  # noqa: E402

TRACKER = str(REPO / "perception" / "trackers" / "botsort_uav.yaml")  # BoT-SORT+ReID+GMC (moving cam)
MODEL = str(REPO / "runs/train/airsim_drone/weights/best.pt")
EGO_HOME = np.array([0.0, 0.0, 0.0]); TARGET_HOME = np.array([8.0, 0.0, 0.0])
HFOV = 90.0; LEAD_T = 0.2; DB = 0.03; K_YR = 2.0; YR_MAX = 40.0  # proportional yaw-RATE, no integral

app = Flask(__name__)
LOCK = threading.Lock()
G = {"jpeg": None, "tel": {"state": "INIT"}, "gap": 12.0, "mode": "fused", "speed": 5.0,
     "click": None, "clear": False, "land": False, "running": True,
     # pluggable real-world receivers (default = AirSim sim)
     "video": {"proto": "airsim", "endpoint": ""},
     "telem_src": {"proto": "airsim", "endpoint": ""},
     "avoid": True,                # depth-based obstacle avoidance (needs AirSim depth)
     "jammed": False}              # GPS jammed -> navigate on VIO (camera+IMU) instead of GPS
# guidance modes: location | vision | fused | vision_after_arrival
# video protocols: airsim | rtsp | udp | http | device | file
# telemetry protocols: airsim | mavlink_udp | mavlink_serial
VID_RX = VideoReceiver()
MAV = {"rx": None, "key": None}      # active MavlinkReceiver + its (proto,endpoint) key


def telem_source_frame():
    """Return the active MAVLink receiver's snapshot, (re)connecting if the selection changed."""
    ts = G["telem_src"]; key = (ts["proto"], ts["endpoint"])
    if ts["proto"] == "airsim":
        if MAV["rx"] is not None:
            MAV["rx"].stop(); MAV["rx"] = None; MAV["key"] = None
        return None
    if key != MAV["key"]:
        if MAV["rx"] is not None:
            MAV["rx"].stop()
        MAV["rx"] = MavlinkReceiver(ts["proto"], ts["endpoint"]); MAV["rx"].start(); MAV["key"] = key
    return MAV["rx"].snapshot() if MAV["rx"] else None


def quat_yaw(q):
    x, y, z, w = q.x_val, q.y_val, q.z_val, q.w_val
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


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


def imu_of(ac):
    d = ac.getImuData(vehicle_name="Ego")
    a = np.array([d.linear_acceleration.x_val, d.linear_acceleration.y_val, d.linear_acceleration.z_val])
    w = np.array([d.angular_velocity.x_val, d.angular_velocity.y_val, d.angular_velocity.z_val])
    return a, w


def is_real(depth, x1, y1, x2, y2, W, H):
    if depth is None:
        return True, None
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    inner = depth[max(0, y1 + bh // 5):y2 - bh // 5, max(0, x1 + bw // 5):x2 - bw // 5]
    iv = inner[(inner > 0.2) & (inner < 1e4)]
    if iv.size < 10:
        return True, None
    obj_d, obj_std = float(np.median(iv)), float(np.std(iv))
    rx1, ry1, rx2, ry2 = max(0, x1 - bw), max(0, y1 - bh), min(W, x2 + bw), min(H, y2 + bh)
    ring = depth[ry1:ry2, rx1:rx2].copy()
    ring[max(0, y1 - ry1):y2 - ry1, max(0, x1 - rx1):x2 - rx1] = -1.0
    rv = ring[(ring > 0.2) & (ring < 1e4)]
    bg = float(np.median(rv)) if rv.size >= 10 else float("inf")
    if not math.isfinite(bg) or bg > 200:
        return True, obj_d
    return (not (abs(bg - obj_d) < 0.8 and obj_std < 0.5)), obj_d


def tracking_loop():
    rng = random.Random(7)
    model = YOLO(MODEL)
    ac = airsim.MultirotorClient(); ac.confirmConnection()
    ac.reset(); time.sleep(2.0)            # clean ground state (a prior run may leave drones airborne,
    #                                        which makes takeoffAsync().join() hang forever)
    for v in ("Ego", "Target"):
        ac.enableApiControl(True, v); ac.armDisarm(True, v)
    ac.takeoffAsync(vehicle_name="Target").join(); ac.takeoffAsync(vehicle_name="Ego").join()
    ac.moveToZAsync(-18, 3, vehicle_name="Target").join(); ac.moveToZAsync(-18, 3, vehicle_name="Ego").join()

    def new_traj():
        name = rng.choice(list(TRAJECTORIES.keys()))
        alt = -rng.uniform(12, 30)                       # wide altitude band -> ego must climb/descend
        cx, cy = rng.uniform(-12, 12), rng.uniform(-12, 12)
        wps = TRAJECTORIES[name]((cx, cy, alt), (cx + 50, cy, alt - rng.uniform(-8, 8))) if name == "line" \
            else TRAJECTORIES[name]((cx, cy, alt))
        path = [airsim.Vector3r(float(n - TARGET_HOME[0]), float(e - TARGET_HOME[1]), float(d)) for n, e, d in wps]
        ac.moveOnPathAsync(path, 3.5, 40, vehicle_name="Target")
        return name

    cur = new_traj(); traj_t0 = time.time()
    kf = ImageKalman(q=2.5, r=0.05)
    tkf = Vec3KF()                        # 3-D target position+velocity estimator (world NED)
    vio = None                            # VIO estimator (lazy-init once image size known)
    state = "DETECT"; locked_id = None; lost = 0; miss = 0; shadows = 0; i_yaw = 0.0
    prev_cmd = None; A_MAX = 2.4           # ego accel limit -> steady (low-tilt) camera platform
    prev_yaw = None; YAW_SLEW = 65.0       # yaw setpoint slew limit (deg/s) -> balanced smooth/keep-up
    vf_s = None; trail = deque(maxlen=60)  # ~5s at 12fps
    vis_hist = deque(maxlen=60); last = time.time(); fps = 0.0

    while G["running"]:
        now = time.time(); dt = min(0.3, max(0.02, now - last)); last = now
        fps = 0.9 * fps + 0.1 * (1.0 / dt)
        if now - traj_t0 > 40:
            cur = new_traj(); traj_t0 = now
        if G["land"]:
            ac.landAsync(vehicle_name="Ego"); ac.landAsync(vehicle_name="Target")
            G["land"] = False; state = "LANDED"; locked_id = None
        v = G["video"]
        if v["proto"] == "airsim":
            scene, depth = grab(ac)
        else:                                    # real-world video receiver (RTSP/UDP/HTTP/device/file)
            scene, depth = VID_RX.read(v["proto"], v["endpoint"])
        mav_snap = telem_source_frame()          # poll selected telemetry receiver (None for AirSim)
        if scene is None:
            time.sleep(0.03); continue
        H, W = scene.shape[:2]; cxI, cyI = W / 2.0, H / 2.0
        res = model.track(scene, tracker=TRACKER, persist=True, imgsz=960, conf=0.3, verbose=False)[0]
        ego = world_pos(ac, "Ego", EGO_HOME); ego_yaw = quat_yaw(ac.simGetVehiclePose("Ego").orientation)
        ego_q = ac.simGetVehiclePose("Ego").orientation
        tgt = world_pos(ac, "Target", TARGET_HOME)

        # ---- VIO: estimate ego pose from camera+IMU; anchor to GPS while available, free-run if jammed ----
        jammed = bool(G.get("jammed", False))
        vio_drift = None; nav_source = "GPS"
        if depth is not None and v["proto"] == "airsim":
            if vio is None:
                vio = VIOEstimator(hfov_deg=HFOV, width=W, height=H)
                vio.anchor(ego, [ego_q.x_val, ego_q.y_val, ego_q.z_val, ego_q.w_val])
            try:
                gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
                vout = vio.update(gray, depth, imu_of(ac), dt)
                if not jammed:
                    vio.anchor(ego, [ego_q.x_val, ego_q.y_val, ego_q.z_val, ego_q.w_val])  # GPS primes VIO
                else:
                    vio_drift = round(float(np.linalg.norm(vout["p"] - ego)), 2)            # vs truth (display)
            except Exception:
                vout = None
        else:
            vout = None
        if jammed and vout is not None:                      # navigate on VIO estimate
            ego_nav = vout["p"]; yaw_nav = vout["yaw"]; nav_source = "VIO"
        else:
            ego_nav = ego; yaw_nav = ego_yaw
        rng_gps = float(math.hypot(tgt[0] - ego_nav[0], tgt[1] - ego_nav[1]))
        az_gps = math.atan2(tgt[1] - ego_nav[1], tgt[0] - ego_nav[0])

        # collect valid (non-shadow) detections
        dets = []
        b = res.boxes
        if b is not None and len(b):
            for i in range(len(b)):
                x1, y1, x2, y2 = b.xyxy[i].tolist()
                _, od = is_real(depth, x1, y1, x2, y2, W, H)  # keep depth only (for range)
                real = True  # shadow filtering REMOVED -> detect & lock all UAVs
                tid = int(b.id[i]) if b.id is not None else -1
                d = {"box": (x1, y1, x2, y2), "cx": (x1+x2)/2, "cy": (y1+y2)/2,
                     "conf": float(b.conf[i]), "id": tid, "real": real, "depth": od}
                if not real:
                    shadows += 1
                dets.append(d)
        real_dets = [d for d in dets if d["real"]]
        vis_hist.append(1 if real_dets else 0)

        # ---- handle UI commands ----
        if G["clear"]:
            G["clear"] = False; locked_id = None; state = "DETECT"; kf.reset(); tkf.reset(); trail.clear()
        click = G["click"]
        if click is not None:
            G["click"] = None
            px, py = click[0] * W, click[1] * H
            inside = [d for d in real_dets if d["box"][0] <= px <= d["box"][2] and d["box"][1] <= py <= d["box"][3]]
            cand = inside or real_dets
            if cand:
                sel = min(cand, key=lambda d: (d["cx"]-px)**2 + (d["cy"]-py)**2)
                locked_id = sel["id"]; state = "TRACK"; lost = 0; miss = 0; prev_cmd = None; prev_yaw = None; kf.reset(); tkf.reset(); i_yaw = 0.0; trail.clear()

        gap = float(G["gap"])
        conf = 0.0; range_m = None; cur_det = None; clearance_m = None; avoiding = False
        vn = ve = vd = 0.0; yaw_sp = ego_yaw; yr_deg = 0.0
        source = "--"; reached = False

        if state == "TRACK":
            # find the locked target among current detections (by track id, else nearest real det)
            match = [d for d in real_dets if d["id"] == locked_id]
            # keep vision active even if ByteTrack reassigns the id: fall back to the most-centered detection
            cur_det = match[0] if match else (min(real_dets, key=lambda d: (d["cx"]-cxI)**2 + (d["cy"]-cyI)**2)
                                              if real_dets else None)
            mode = G["mode"]; sp = float(G["speed"])
            VFOV = vfov_from_hfov(HFOV, W, H)
            # ---- build a TARGET-POSITION measurement (world NED) from the chosen source ----
            vis_meas = None
            if cur_det is not None:
                conf = cur_det["conf"]; lost = 0
                ex, ey = (cur_det["cx"] - cxI) / cxI, (cur_det["cy"] - cyI) / cyI
                if cur_det["depth"] and cur_det["depth"] > 0.3:
                    vis_meas = target_from_vision(ego_nav, yaw_nav, ex, ey, cur_det["depth"], HFOV, VFOV)
                kf.update(ex, ey, dt)
                trail.append((int(cur_det["cx"]), int(cur_det["cy"])))
            else:
                lost += 1
                if lost > 150:
                    state = "DETECT"; locked_id = None; tkf.reset(); trail.clear()
            reached = rng_gps <= gap + 2.5
            # pick the measurement per guidance mode. When GPS is JAMMED there is no GPS target, so
            # force vision-only (target reconstructed from camera in the VIO frame).
            if jammed:
                meas, source = vis_meas, "vision(GPS-jammed)"
            elif mode == "location":
                meas, source = tgt, "gps"
            elif mode == "vision":
                meas, source = vis_meas, "vision"
            elif mode == "vision_after_arrival":
                meas, source = ((vis_meas, "vision(arrived)") if (reached and vis_meas is not None)
                                else (tgt, "gps(approach)"))
            else:  # fused = confidence-weighted blend of vision estimate + GPS
                if vis_meas is not None:
                    w = max(0.0, min(1.0, conf)); meas, source = w * vis_meas + (1 - w) * tgt, "fused"
                else:
                    meas, source = tgt, "fused(gps)"
            # feed the 3-D target Kalman; coast (predict) when no measurement this frame, but DECAY
            # the velocity feed-forward and slow down so a lost target never makes the ego fly off.
            if meas is not None:
                tp, tv = tkf.update(np.asarray(meas, float), dt); miss = 0
            else:
                miss += 1
                tp, tv = tkf.predict_only(dt)
                if tp is None:                                   # never seen yet -> bootstrap on GPS
                    tp, tv = tkf.update(tgt, dt)
                tv = tv * max(0.0, 1.0 - 0.2 * miss)
            eff_sp = sp * (1.0 if meas is not None else max(0.25, 1.0 - 0.15 * miss))
            clearance_m = None; avoiding = False
            if jammed and cur_det is not None and cur_det["depth"] and cur_det["depth"] > 0.3:
                # ===== GPS-DENIED: pure BODY-FRAME visual servo (drift-immune) =====
                # Uses ONLY the camera (depth range + image bearing), commanded in the body frame, so
                # it needs no world position/heading -> immune to VIO drift. VIO still runs for absolute
                # position awareness/telemetry, but the LOCK is held by vision alone.
                range_m = float(cur_det["depth"])
                exj = (cur_det["cx"] - cxI) / cxI; eyj = (cur_det["cy"] - cyI) / cyI
                fwd = float(np.clip(0.8 * (range_m - gap), -eff_sp, eff_sp))     # hold gap (fwd/back)
                vz_b = float(np.clip(2.2 * eyj, -2.5, 2.5))                      # center vertically (climb/descend)
                bearing = math.degrees(math.atan(exj * math.tan(math.radians(HFOV / 2))))
                yr = float(np.clip(K_YR * bearing, -YR_MAX, YR_MAX))             # center horizontally (yaw rate)
                if G.get("avoid", True) and depth is not None:                   # forward brake on obstacle
                    ahead, sev, _ = clearance_and_escape(depth)
                    if sev > 0.0:
                        fwd = (1.0 - sev) * fwd; avoiding = True
                        clearance_m = round(float(ahead), 1) if math.isfinite(ahead) else None
                ac.moveByVelocityBodyFrameAsync(fwd, 0.0, vz_b, 0.6,
                                                yaw_mode=airsim.YawMode(True, yr), vehicle_name="Ego")
                yr_deg = yr; prev_yaw = None; prev_cmd = None; source = "vision(GPS-jammed)"
            else:
                # ===== GPS AVAILABLE: world-frame standoff (pos-P + velocity feed-forward) =====
                vn, ve, vd, yaw_deg, range_m, yaw_err_deg = standoff_command(
                    ego_nav, yaw_nav, tp, tv, gap, eff_sp, lead_t=LEAD_T)
                # obstacle avoidance (depth = LiDAR-like): brake + steer around close obstacles
                if G.get("avoid", True) and v["proto"] == "airsim" and depth is not None:
                    (vn, ve, vd), clearance_m, avoiding = apply_avoidance(
                        [vn, ve, vd], depth, yaw_nav, max(eff_sp, 2.0))
                    clearance_m = round(float(clearance_m), 1) if math.isfinite(clearance_m) else None
                # acceleration-limit -> less tilt -> steady camera
                cmd = np.array([vn, ve, vd])
                if prev_cmd is None:
                    prev_cmd = cmd
                cmd = prev_cmd + np.clip(cmd - prev_cmd, -A_MAX * dt, A_MAX * dt)
                prev_cmd = cmd
                # absolute, slew-limited yaw setpoint (smooth)
                if prev_yaw is None:
                    prev_yaw = yaw_deg
                dyaw = math.degrees(math.atan2(math.sin(math.radians(yaw_deg - prev_yaw)),
                                               math.cos(math.radians(yaw_deg - prev_yaw))))
                prev_yaw = prev_yaw + max(-YAW_SLEW * dt, min(YAW_SLEW * dt, dyaw))
                ac.moveByVelocityAsync(float(cmd[0]), float(cmd[1]), float(cmd[2]), 0.6,
                                       yaw_mode=airsim.YawMode(False, float(prev_yaw)), vehicle_name="Ego")
                yr_deg = yaw_err_deg
        else:  # DETECT — GPS gently yaws (rate) the camera onto the target so operator can see & click
            yerr = math.atan2(math.sin(az_gps - ego_yaw), math.cos(az_gps - ego_yaw))
            yr_cmd = float(np.clip(K_YR * math.degrees(yerr), -YR_MAX, YR_MAX))
            ac.moveByVelocityAsync(0, 0, 0, 0.6,
                                   yaw_mode=airsim.YawMode(True, yr_cmd), vehicle_name="Ego")
            yr_deg = yr_cmd

        # ---- annotate ----
        ann = scene.copy()
        for d in dets:
            x1, y1, x2, y2 = (int(v) for v in d["box"])
            if not d["real"]:
                cv2.rectangle(ann, (x1, y1), (x2, y2), (128, 128, 128), 1)
                cv2.putText(ann, "shadow", (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128,128,128), 1)
            elif state == "TRACK" and cur_det is d:
                cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(ann, f"LOCKED id{d['id']} {d['conf']:.2f}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            else:
                cv2.rectangle(ann, (x1, y1), (x2, y2), (255, 200, 0), 1)
                cv2.putText(ann, f"id{d['id']} {d['conf']:.2f}", (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,200,0), 1)
        cv2.drawMarker(ann, (int(cxI), int(cyI)), (255, 255, 255), cv2.MARKER_CROSS, 26, 2)
        if state == "TRACK" and cur_det is not None:
            cv2.line(ann, (int(cxI), int(cyI)), (int(cur_det["cx"]), int(cur_det["cy"])), (0, 255, 0), 1)
        for k in range(1, len(trail)):                      # last-5s path only
            cv2.line(ann, trail[k-1], trail[k], (0, 255, 255), 1)
        if jammed:                                           # GPS-denied: navigating on VIO
            cv2.rectangle(ann, (0, 0), (W, H), (0, 165, 255), 5)
            cv2.putText(ann, f"GPS JAMMED - VIO NAV  drift~{vio_drift}m", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        if avoiding:                                         # obstacle-avoidance warning banner
            cv2.rectangle(ann, (0, 0), (W, H), (0, 0, 255), 6)
            cv2.putText(ann, f"AVOID OBSTACLE  clearance={clearance_m}m", (10, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if state == "DETECT":
            cv2.putText(ann, "DETECT - click a target box to LOCK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
        else:
            cv2.putText(ann, f"TRACK [{G['mode']}/{source}] gap={gap:.0f}m range={(range_m or 0):.1f}m", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
        if depth is not None:
            dv = cv2.applyColorMap((np.clip(depth, 0, 40)/40*255).astype(np.uint8), cv2.COLORMAP_TURBO)
            ann[10:10+H//4, W-10-W//4:W-10] = cv2.resize(dv, (W//4, H//4))
        ok, jpg = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 80])

        # centering error: normalized distance of the locked target from frame center (0 = perfect)
        center_err = None
        if state == "TRACK" and cur_det is not None:
            center_err = round(float(math.hypot((cur_det["cx"]-cxI)/cxI, (cur_det["cy"]-cyI)/cyI)), 3)
        with LOCK:
            if ok:
                G["jpeg"] = jpg.tobytes()
            G["tel"] = {"state": state, "locked": state == "TRACK", "target_id": locked_id if state=="TRACK" else None,
                        "conf": round(conf, 2), "range_m": round(range_m, 1) if range_m else None,
                        "gap": round(gap, 1), "alt_m": round(-ego[2], 1), "yaw_rate_deg": round(yr_deg, 1),
                        "center_err": center_err,
                        "ego_n": round(ego[0], 1), "ego_e": round(ego[1], 1), "ego_d": round(ego[2], 1),
                        "n_detections": len(real_dets), "vision_rate": round(100*sum(vis_hist)/max(1,len(vis_hist))),
                        "shadows": shadows, "fps": round(fps, 1),
                        "mode": G["mode"], "source": source, "reached": reached, "speed": round(float(G["speed"]), 1),
                        "video_src": G["video"]["proto"], "telem_src": G["telem_src"]["proto"],
                        "avoid": bool(G.get("avoid", True)), "clearance_m": clearance_m, "avoiding": avoiding,
                        "camera": "FPV (body-fixed)", "nav_source": nav_source, "jammed": jammed,
                        "vio_drift": vio_drift,
                        "mav": ("--" if mav_snap is None else
                                ("connected" if mav_snap.get("connected") else
                                 ("err: " + str(mav_snap.get("error"))[:30] if mav_snap.get("error") else "connecting…")))}


@app.after_request
def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            with LOCK:
                j = G["jpeg"]
            if j is None:
                time.sleep(0.05); continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n")
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/telemetry")
def telemetry():
    with LOCK:
        return jsonify(G["tel"])


@app.route("/select", methods=["POST"])
def select():
    d = request.get_json(force=True); G["click"] = (float(d["x"]), float(d["y"])); return ("", 204)


@app.route("/clear", methods=["POST"])
def clear():
    G["clear"] = True; return ("", 204)


@app.route("/set_gap", methods=["POST"])
def set_gap():
    G["gap"] = float(request.get_json(force=True)["gap"]); return ("", 204)


@app.route("/set_mode", methods=["POST"])
def set_mode():
    mo = str(request.get_json(force=True)["mode"])
    if mo in ("location", "vision", "fused", "vision_after_arrival"):
        G["mode"] = mo
    return ("", 204)


@app.route("/set_speed", methods=["POST"])
def set_speed():
    G["speed"] = max(0.5, min(12.0, float(request.get_json(force=True)["speed"]))); return ("", 204)


@app.route("/set_video_source", methods=["POST"])
def set_video_source():
    d = request.get_json(force=True); proto = str(d.get("proto", "airsim"))
    if proto in VIDEO_PROTOS:
        G["video"] = {"proto": proto, "endpoint": str(d.get("endpoint", ""))}
    return ("", 204)


@app.route("/set_telem_source", methods=["POST"])
def set_telem_source():
    d = request.get_json(force=True); proto = str(d.get("proto", "airsim"))
    if proto in TELEM_PROTOS:
        G["telem_src"] = {"proto": proto, "endpoint": str(d.get("endpoint", ""))}
    return ("", 204)


@app.route("/set_avoid", methods=["POST"])
def set_avoid():
    G["avoid"] = bool(request.get_json(force=True).get("avoid", True)); return ("", 204)


@app.route("/set_jam", methods=["POST"])
def set_jam():
    G["jammed"] = bool(request.get_json(force=True).get("jammed", False)); return ("", 204)


@app.route("/land", methods=["POST"])
def land():
    G["land"] = True; return ("", 204)


if __name__ == "__main__":
    threading.Thread(target=tracking_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
