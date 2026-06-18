#!/usr/bin/env python3
"""Air + GROUND tracking on PX4-SITL (Gazebo), driven from the SAME AirSim webui (templates/index.html):
live video (forward / down / DEPTH, switchable), click-to-select ANY detection (manual) or autolock,
3-D map, and follow/land/strike on a ground vehicle (UAVros suv).

Detection is OBJECT detection (YOLO), not colour:
  - AIR  : our UAV detector on /chaser/camera_front  (locks the flying target drone)
  - GROUND: COCO YOLO on /chaser/camera_down         (locks car/truck/person, e.g. the suv)
Guidance: MAVLink offboard body-velocity (deploy/mavlink_control.py). HIT/land on true range (gz pose).

Run (world up via launch_a2a.sh, runner + rover_mover flying):
  .venv\\Scripts\\python.exe webui\\app_a2a_sitl.py        # open http://localhost:5063
  (WSL alt: python3 webui/app_a2a_sitl.py)
"""
import os, sys, math, time, threading, json
os.environ.setdefault("GZ_IP", "127.0.0.1")
import numpy as np, cv2
from flask import Flask, render_template, Response, request, jsonify

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "deploy"))
sys.path.insert(0, os.path.join(REPO, "sim", "gz", "a2a"))
sys.path.insert(0, os.path.join(REPO, "sim", "airsim"))
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from gz.msgs10.pose_v_pb2 import Pose_V
from mavlink_control import MavBridge
from px4_util import arm_offboard
from range_filter import RangeFilter        # robust range: depth + bbox-size cross-check + gating
from smooth_control import ImageKalman, rate_limit   # smooth/predict bbox -> steady tracking
from ultralytics import YOLO

UAV_W = os.environ.get("UAV_MODEL", os.path.join(REPO, "runs/train/uav_diverse/weights/best.pt"))
COCO_W = os.environ.get("COCO_MODEL", "yolov8n.pt")          # auto-downloads; car/truck/person
GROUND_CLS = {0: "person", 2: "car", 5: "bus", 7: "truck"}
FW, FH = 640, 480                               # x500_mono_cam 1280x960 (4:3) downscaled for processing
CHASER = "x500_mono_cam_0"; AIR = "x500_mono_cam_1"        # chaser + air target (both standard x500_mono_cam)
FRONT_TOPIC = "/world/uav_a2a/model/x500_mono_cam_0/link/camera_link/sensor/camera/image"
DW, DH = 512, 384
PORT = int(os.environ.get("A2A_SITL_PORT", "5063"))
HFOV_F = 99.7                                   # x500_mono_cam front HFOV (deg) = 1.74 rad
VFOV_F = math.degrees(2 * math.atan(math.tan(math.radians(HFOV_F) / 2) * FH / FW))
FY_F = (FW / 2.0) / math.tan(math.radians(HFOV_F) / 2.0)   # focal in px (RGB coords) for size-range


def tgt_world_from_vision(ego, yaw, ex, ey, rng):
    """REAL-WORLD estimator: target world pos from CHASER GPS pose (ego,yaw) + camera bearing (ex,ey)
    + vision range. NO target coordinates used -> this is how it works on a real drone (chaser GPS only)."""
    thx = ex * math.tan(math.radians(HFOV_F) / 2.0); thy = ey * math.tan(math.radians(VFOV_F) / 2.0)
    fwd, right, down = rng, rng * thx, rng * thy
    fx, fy = math.cos(yaw), math.sin(yaw)        # world forward (chaser heading)
    rx, ry = math.sin(yaw), -math.cos(yaw)       # world right
    return [ego[0] + fwd * fx + right * rx, ego[1] + fwd * fy + right * ry, ego[2] - down]

app = Flask(__name__, template_folder=os.path.join(HERE, "templates"))

G = {"running": True, "autolock": True, "search": False, "manual_mode": False,
     "speed": 5.0, "gap": 6.0, "state": "DETECT", "locked": False, "jammed": False,
     "video_src": "front", "telem_src": "airsim", "avoid": False,
     "strike_mode": False, "strike_armed": False, "last_hit": "--",
     "action": "follow", "frame": None, "tel": {}, "sel": None, "clear": False,
     "target_type": "air",          # EXPLICIT selector: "air" (UAV) or "ground" (vehicle)
     "maxrange": 200.0}             # engage only targets within this range (m) -> chaser stays controlled

S = {"front": None, "down": None, "depth": None, "ego": None, "yaw": 0.0,
     "air": None, "ground": None, "stamp": 0.0}


# ---------- gz subscriptions (keep node refs) ----------
def gz_subs(world):
    def on_front(m):
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        bgr = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        S["front"] = cv2.resize(bgr, (FW, FH)) if (m.width != FW or m.height != FH) else bgr
        S["stamp"] = time.time()
    def on_down(m):
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        S["down"] = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    def on_depth(m):
        d = np.frombuffer(m.data, np.float32).reshape(m.height, m.width).copy()
        d[~np.isfinite(d)] = 0.0; S["depth"] = d
    def on_pose(m):
        for p in m.pose:
            if p.name == CHASER:
                S["ego"] = np.array([p.position.x, p.position.y, p.position.z]); q = p.orientation
                S["yaw"] = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
            elif p.name == AIR:
                S["air"] = np.array([p.position.x, p.position.y, p.position.z])
            elif p.name == "rover":
                S["ground"] = np.array([p.position.x, p.position.y, p.position.z])
    nodes = [Node(), Node(), Node(), Node()]
    nodes[0].subscribe(Image, FRONT_TOPIC, on_front)
    nodes[1].subscribe(Image, "/chaser/camera_down", on_down)
    nodes[2].subscribe(Image, "/chaser/depth_front", on_depth)
    nodes[3].subscribe(Pose_V, f"/world/{world}/pose/info", on_pose)
    return nodes


def yolo_boxes(model, img, conf, imgsz, classes=None):
    r = model.predict(img, imgsz=imgsz, conf=conf, classes=classes, verbose=False)[0]
    out = []
    b = r.boxes
    if b is not None:
        for i in range(len(b)):
            x1, y1, x2, y2 = b.xyxy[i].cpu().numpy().tolist()
            out.append({"box": [x1, y1, x2, y2], "conf": float(b.conf[i]),
                        "cls": int(b.cls[i]) if b.cls is not None else -1})
    return out


def is_real(depth, x1, y1, x2, y2):
    """Depth-based SHADOW rejection (ported from AirSim). A shadow is coplanar with a finite surface
    AND uniform in depth -> reject. Foreground drone (pops out) or against-sky -> accept. When depth is
    sparse/unknown -> accept (never drop a real small drone). depth is the forward depth image."""
    if depth is None:
        return True
    H, W = depth.shape
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    inner = depth[max(0, y1 + bh // 5):y2 - bh // 5, max(0, x1 + bw // 5):x2 - bw // 5]
    iv = inner[(inner > 0.2) & (inner < 1e4)]
    if iv.size < 10:
        return True
    obj_d, obj_std = float(np.median(iv)), float(np.std(iv))
    rx1, ry1, rx2, ry2 = max(0, x1 - bw), max(0, y1 - bh), min(W, x2 + bw), min(H, y2 + bh)
    ring = depth[ry1:ry2, rx1:rx2].copy()
    ring[max(0, y1 - ry1):y2 - ry1, max(0, x1 - rx1):x2 - rx1] = -1.0
    rv = ring[(ring > 0.2) & (ring < 1e4)]
    bg = float(np.median(rv)) if rv.size >= 10 else float("inf")
    if not math.isfinite(bg) or bg > 200:
        return True                                # against sky / no background -> real
    return not (abs(bg - obj_d) < 0.8 and obj_std < 0.5)   # coplanar + flat == shadow on a surface


def detect_sky_dots(bgr, top_frac=0.72, min_a=4, max_a=700, dev=10):
    """Any small, compact, distinct dot in the SKY region is a candidate UAV (real small drones at range
    are just dots). Local-contrast (img - blur) so it catches dark OR light specks vs smooth sky."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = g.shape; cut = int(H * top_frac)
    sky = g[:cut]
    d = cv2.absdiff(sky, cv2.GaussianBlur(sky, (0, 0), 7))
    m = cv2.morphologyEx((d > dev).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_a or a > max_a:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 2 or h < 2 or w > 5 * h or h > 5 * w or x <= 1 or y <= 1 or x + w >= W - 1:
            continue
        out.append({"box": [float(x), float(y), float(x + w), float(y + h)], "conf": 0.25, "cls": -1})
    return out[:8]


def depth_color(d):
    if d is None:
        return np.zeros((DH, DW, 3), np.uint8)
    v = np.clip(d, 0, 60) / 60.0
    return cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_JET)


def loop():
    from pymavlink import mavutil
    air_model = YOLO(UAV_W); coco = YOLO(COCO_W)
    br = MavBridge("udpin:0.0.0.0:14540")
    cmd = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw": 0.0, "mode": "takeoff", "alt": 9.0}

    HOLD = (0.0, 10.0, -9.0)                       # local NED hold ~ world (10,0,9): BELOW the ~15m target -> looks up vs sky
    def streamer():                               # the ONLY continuous br.m writer (proven to allow offboard)
        while True:
            if cmd["mode"] == "takeoff":          # POSITION setpoint takeoff (robust, like runner_mission)
                br.send_position_ned(0.0, 0.0, -cmd["alt"], 90.0)      # yaw 90 (NED) = face world +X corridor
            elif cmd["mode"] == "hold":           # POSITION hold -> contained + re-acquires (no dead-reckon)
                br.send_position_ned(HOLD[0], HOLD[1], HOLD[2], 90.0)
            else:
                br.send_body_velocity_xy(cmd["vx"], cmd["vy"], cmd["vz"], cmd["yaw"])
            time.sleep(0.05)
    threading.Thread(target=streamer, daemon=True).start()

    def arm_and_climb(timeout=60.0):
        cmd["mode"] = "takeoff"                    # stream position setpoint (0,0,-alt) -> climbs to alt
        arm_offboard(br.m, timeout=timeout, label="chaser")
        t0 = time.time()
        while time.time() - t0 < 14 and (S["ego"] is None or S["ego"][2] < cmd["alt"] - 0.6):
            time.sleep(0.1)
        print(f"[chaser] at alt {S['ego'][2]:.1f}m -> velocity tracking" if S["ego"] is not None else "climb done", flush=True)
        cmd["vx"] = cmd["vy"] = cmd["vz"] = cmd["yaw"] = 0.0; cmd["mode"] = "vel"

    time.sleep(1.0)
    arm_and_climb()

    lock = None; lock_miss = 0; rf = None      # {"kind":"air"|"ground", "box":..} + robust range filter
    ikf = ImageKalman(q=2.0, r=0.05)           # smooth + coast the bbox bearing -> steady servo
    prevc = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw": 0.0}; lost_t = 0.0; prev_video = "front"
    fno = 0; last = time.time(); last_armchk = time.time()
    air_dets = []; gnd_dets = []
    while G["running"]:
        if S["stamp"] == 0.0:
            time.sleep(0.02); continue
        fno += 1; now = time.time(); dt = max(0.02, now - last); last = now
        # re-arm-on-land watchdog: if PX4 dropped to disarmed (failsafe/land) outside a deliberate
        # LAND, bring it back up so the live demo keeps flying.
        if now - last_armchk > 2.5:
            last_armchk = now
            br.m.recv_match(type="HEARTBEAT", blocking=False)
            if not br.m.motors_armed() and G["state"] != "LAND":
                print("[chaser] disarmed -> re-arming + climbing", flush=True); arm_and_climb(timeout=20.0)
        front, down, depth = S["front"], S["down"], S["depth"]
        ego, yaw, air, gnd = S["ego"], S["yaw"], S["air"], S["ground"]

        # ---- detect (throttled + alternating; *_fresh = a REAL detection ran this frame) ----
        air_fresh = gnd_fresh = False
        if front is not None and fno % 4 == 0:
            air_dets = yolo_boxes(air_model, front, 0.12, 512)
            # reject the chaser's OWN prop-arms (big dark shapes at the frame EDGES) + ground blobs:
            # keep only central, not-too-large detections (the real distant drone is small + central).
            air_dets = [d for d in air_dets
                        if 0.16*FW < (d["box"][0]+d["box"][2])/2 < 0.84*FW
                        and (d["box"][1]+d["box"][3])/2 < 0.84*FH
                        and (d["box"][2]-d["box"][0])*(d["box"][3]-d["box"][1]) < 0.10*FW*FH]
            if depth is not None and air_dets:           # SHADOW rejection via forward depth
                dh, dw = depth.shape; sx = dw / FW; sy = dh / FH
                kept = []
                for d in air_dets:
                    b = d["box"]
                    if is_real(depth, b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy):
                        kept.append(d)
                    else:
                        G["shadows"] = G.get("shadows", 0) + 1
                air_dets = kept
            for dd in detect_sky_dots(front):        # "any sky dot = UAV candidate"; merge + dedup vs YOLO
                dc = ((dd["box"][0]+dd["box"][2])/2, (dd["box"][1]+dd["box"][3])/2)
                if not any(abs((d["box"][0]+d["box"][2])/2 - dc[0]) < 25 and
                           abs((d["box"][1]+d["box"][3])/2 - dc[1]) < 25 for d in air_dets):
                    air_dets.append(dd)
            air_fresh = len(air_dets) > 0
        if down is not None and fno % 4 == 2:
            gnd_dets = yolo_boxes(coco, down, 0.30, 416, classes=list(GROUND_CLS))
            gnd_fresh = len(gnd_dets) > 0

        # ---- TARGET SELECTOR: explicit UAV/ground choice from the webui (NOT tied to the video view) ----
        active_kind = G["target_type"]               # "air" or "ground"
        if active_kind != prev_video:                # selector changed -> drop a mismatched lock
            prev_video = active_kind
            if lock is not None and lock["kind"] != active_kind:
                lock = None; ikf.reset()
        # ---- selection (click / autolock / clear) ----
        if G["clear"]:
            lock = None; ikf.reset(); G["clear"] = False; G["strike_armed"] = False
        sel = G.pop("sel", None) if "sel" in G else None
        if sel is not None:
            pool = gnd_dets if active_kind == "ground" else air_dets
            iw, ih = (DW, DH) if active_kind == "ground" else (FW, FH)
            if pool:
                px, py = sel[0] * iw, sel[1] * ih
                d = min(pool, key=lambda d: (np.mean([d["box"][0], d["box"][2]]) - px) ** 2 +
                                            (np.mean([d["box"][1], d["box"][3]]) - py) ** 2)
                lock = {"kind": active_kind, "box": d["box"]}; ikf.reset()
        elif G["autolock"] and lock is None:
            pool = air_dets if active_kind == "air" else gnd_dets
            if active_kind == "air":                  # only lock targets within the engage envelope (maxrange)
                pool = [d for d in pool if (FY_F * 0.55 / max(2.0, d["box"][3]-d["box"][1])) <= G["maxrange"]]
            if pool:
                d = max(pool, key=lambda d: d["conf"]); lock = {"kind": active_kind, "box": d["box"]}; ikf.reset()

        # ---- refresh + SMOOTH: Kalman-update on a fresh detection, else COAST on the prediction ----
        rng = None; center_err = None; sxy = None
        if lock is not None:
            fresh = air_fresh if lock["kind"] == "air" else gnd_fresh
            pool = air_dets if lock["kind"] == "air" else gnd_dets
            iw, ih = (FW, FH) if lock["kind"] == "air" else (DW, DH)
            if fresh and pool:
                lb = lock["box"]; lc = ((lb[0]+lb[2])/2, (lb[1]+lb[3])/2)
                d = min(pool, key=lambda d: ((d["box"][0]+d["box"][2])/2-lc[0])**2 +
                                            ((d["box"][1]+d["box"][3])/2-lc[1])**2)
                lock["box"] = d["box"]; b = d["box"]
                exr = ((b[0]+b[2])/2 - iw/2)/(iw/2); eyr = ((b[1]+b[3])/2 - ih/2)/(ih/2)
                sxy = ikf.update(exr, eyr, dt)
            else:
                sxy = ikf.coast(dt)
                if sxy is None or ikf.miss > 30:       # ~3 s coasting with no detection -> truly lost
                    lock = None; ikf.reset()

        # ---- control (steer on the SMOOTHED/coasted bearing sxy; hover on brief loss -> no snap) ----
        G["locked"] = lock is not None
        if lock is None:
            lost_t += dt; G["tgt_est"] = None
            if rf is not None: rf.reset()
            if lost_t < 3.0:                          # brief loss -> HOVER in place (no snap to corridor)
                cmd["mode"] = "vel"; cmd["vx"] = cmd["vy"] = cmd["vz"] = cmd["yaw"] = 0.0
            else:                                     # long loss -> return to corridor (containment)
                cmd["mode"] = "hold"
            G["state"] = "SEARCH" if G["search"] else "DETECT"
        else:
            lost_t = 0.0; cmd["mode"] = "vel"
            ex, ey = float(sxy[0]), float(sxy[1]); center_err = float(math.hypot(ex, ey))
            b = lock["box"]; h_px = max(2.0, b[3] - b[1]); sp = float(G["speed"])
            if lock["kind"] == "air":
                G["state"] = "TRACK-AIR"; range_ok = False
                if rf is None:
                    rf = RangeFilter(fy=FY_F); rf.H = 0.55   # known target size (x500 ~0.55m): monocular range
                if depth is not None:                 # depth available -> depth + size cross-check
                    dep_rgb = cv2.resize(depth, (FW, FH))
                    rd = rf.robust_depth(dep_rgb, b)
                    if rd is not None: rf.calibrate(rd, h_px)
                    rng, range_ok = rf.update(dep_rgb, b, h_px, dt)
                else:                                 # no depth cam (x500_mono_cam) -> bbox-size range
                    rng, range_ok = rf.update(None, b, h_px, dt)
                cmd["yaw"] = float(np.clip(45 * ex, -40, 40))
                cmd["vz"] = float(np.clip(2.5 * ey, -2.0, 2.0))
                out_env = rng is not None and rng > G["maxrange"]    # beyond engage envelope -> don't chase
                if out_env:
                    cmd["vx"] = 0.0; range_ok = False; G["state"] = "OUT-OF-RANGE"
                elif range_ok and rng:
                    cmd["vx"] = float(np.clip(0.7 * (rng - G["gap"]), 0.0, sp))
                else:
                    cmd["vx"] = float(np.clip(sp * 0.5 * (1 - abs(ex)), 0.0, sp * 0.5))
                cmd["vy"] = 0.0
                if range_ok and rng and 1.0 < rng <= G["maxrange"] and ego is not None:   # VISION-ONLY estimate
                    est = tgt_world_from_vision(ego, yaw, ex, ey, rng)
                    pe = G.get("tgt_est"); G["tgt_est"] = est if not pe else [0.7*p+0.3*e for p, e in zip(pe, est)]
            else:  # ground (down cam) -- VISION-ONLY, no target coords
                cmd["yaw"] = 0.0
                cmd["vx"] = float(np.clip(-2.2 * ey, -sp, sp)); cmd["vy"] = float(np.clip(2.2 * ex, -sp, sp))
                centered = abs(ex) < 0.18 and abs(ey) < 0.18
                z = ego[2] if ego is not None else 8.0
                if G["strike_mode"] and G["strike_armed"]:
                    G["state"] = "STRIKE"; cmd["vz"] = 3.0 if centered else 1.0
                    if z < 1.6: G["last_hit"] = f"HIT car @ {z:.1f}m"; G["strike_armed"] = False
                elif G["action"] == "land":
                    G["state"] = "LAND"; cmd["vz"] = 1.2 if centered else 0.4
                    if z < 1.3:
                        G["last_hit"] = "LANDED on car"; G["action"] = "follow"; cmd["vx"]=cmd["vy"]=cmd["vz"]=0.0
                else:
                    G["state"] = "TRACK-GROUND"; cmd["vz"] = float(np.clip(-0.8*(9.0 - z), -1.5, 1.5))

        # ---- altitude governor: hold ~9 m so the chaser stays where it can see air+ground
        #      (LAND/STRIKE override to descend). Robust to takeoff overshoot. ----
        if ego is not None and G["state"] not in ("LAND", "STRIKE", "TRACK-AIR") and cmd["mode"] == "vel":
            cmd["vz"] = float(np.clip(-0.9 * (9.0 - ego[2]), -2.0, 2.0))   # hold ~9 m (below the ~15m air target)

        # ---- accel-limit velocity + slew-limit yaw-rate -> steady, compact platform (no shaking) ----
        if cmd["mode"] == "vel":
            cmd["vx"] = rate_limit(prevc["vx"], cmd["vx"], 3.0 * dt)
            cmd["vy"] = rate_limit(prevc["vy"], cmd["vy"], 3.0 * dt)
            cmd["vz"] = rate_limit(prevc["vz"], cmd["vz"], 2.5 * dt)
            cmd["yaw"] = rate_limit(prevc["yaw"], cmd["yaw"], 160.0 * dt)
        prevc = {k: cmd[k] for k in ("vx", "vy", "vz", "yaw")}

        # ---- annotate the active view ----
        src = G["video_src"]
        if src == "down":
            view = down.copy() if down is not None else np.zeros((DH, DW, 3), np.uint8)
            for d in gnd_dets:
                x1, y1, x2, y2 = [int(v) for v in d["box"]]
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 180, 255), 2)
                cv2.putText(view, GROUND_CLS.get(d["cls"], "obj"), (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)
        elif src == "depth":
            view = depth_color(depth); view = cv2.resize(view, (DW, DH))
        else:
            view = front.copy() if front is not None else np.zeros((FH, FW, 3), np.uint8)
            for d in air_dets:
                x1, y1, x2, y2 = [int(v) for v in d["box"]]
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if lock is not None and ((src == "down") == (lock["kind"] == "ground")):
            x1, y1, x2, y2 = [int(v) for v in lock["box"]]
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(view, f"LOCK {lock['kind']}", (x1, max(0, y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.rectangle(view, (0, 0), (view.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(view, f"PX4-SITL  {G['state']}  view:{src}  shadows_rej={G.get('shadows',0)}  {G['last_hit']}",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        ok, jpg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok: G["frame"] = jpg.tobytes()

        # ---- telemetry (index.html contract + 3D map) ----
        te = G.get("tgt_est")                     # VISION estimate (chaser GPS + camera) -> shown on map
        truth = air if (lock and lock["kind"] == "air") else (gnd if lock else None)
        vis_err = (round(float(np.linalg.norm(np.array(te) - truth)), 1)
                   if (te is not None and truth is not None) else None)   # honesty: vision vs truth
        targets = []
        for k, d in enumerate(air_dets):
            targets.append({"id": k, "conf": round(d["conf"], 2),
                            "cx": ((d["box"][0]+d["box"][2])/2)/FW, "cy": ((d["box"][1]+d["box"][3])/2)/FH})
        G["tel"] = {
            "state": G["state"], "locked": G["locked"], "nav_source": "VIO" if G["jammed"] else "GPS",
            "autolock": G["autolock"], "search": G["search"], "manual_mode": G["manual_mode"],
            "jammed": G["jammed"], "avoiding": G["avoid"], "fps": round(1.0/dt, 1),
            "range_m": round(rng, 1) if rng else None, "gap": G["gap"], "speed": G["speed"],
            "center_err": round(center_err, 2) if center_err is not None else None,
            "alt_m": round(float(ego[2]), 1) if ego is not None else None,
            "ego_n": float(ego[0]) if ego is not None else 0, "ego_e": float(ego[1]) if ego is not None else 0,
            "ego_d": -float(ego[2]) if ego is not None else 0,
            "tgt_n": float(te[0]) if te else 0, "tgt_e": float(te[1]) if te else 0,
            "tgt_d": -float(te[2]) if te else 0, "vis_err": vis_err, "nav": "VISION-ONLY (chaser GPS)",
            "n_detections": len(air_dets)+len(gnd_dets), "targets": targets[:12],
            "video_src": G["video_src"], "telem_src": G["telem_src"],
            "strike_mode": G["strike_mode"], "strike_armed": G["strike_armed"], "strike_msg": G["last_hit"],
            "lock_kind": lock["kind"] if lock else None, "n_air": len(air_dets), "n_ground": len(gnd_dets),
            "shadows": G.get("shadows", 0), "target_type": G["target_type"], "maxrange": G["maxrange"],
        }
        time.sleep(0.06)        # cap loop ~10 Hz -> leaves GIL for the Flask UI threads


# ---------- routes (index.html contract) ----------
@app.route("/")
def index(): return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            f = G["frame"]
            if f: yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n")
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/telemetry")
def telemetry(): return jsonify(G["tel"])

def _on(req):
    try: return bool(req.get_json(force=True).get("on", False))
    except Exception: return False

@app.route("/select", methods=["POST"])
def select():
    try:
        j = request.get_json(force=True); G["sel"] = (float(j["x"]), float(j["y"]))
    except Exception: pass
    return ("", 204)
@app.route("/strike_select", methods=["POST"])
def strike_select(): return select()
@app.route("/clear", methods=["POST"])
def clear(): G["clear"] = True; return ("", 204)
@app.route("/set_autolock", methods=["POST"])
def set_autolock(): G["autolock"] = _on(request); return ("", 204)
@app.route("/set_search", methods=["POST"])
def set_search(): G["search"] = _on(request); return ("", 204)
@app.route("/set_manual", methods=["POST"])
def set_manual(): G["manual_mode"] = _on(request); return ("", 204)
@app.route("/set_jam", methods=["POST"])
def set_jam(): G["jammed"] = _on(request); return ("", 204)
@app.route("/set_avoid", methods=["POST"])
def set_avoid(): G["avoid"] = _on(request); return ("", 204)
@app.route("/set_speed", methods=["POST"])
def set_speed():
    try: G["speed"] = float(request.get_json(force=True).get("speed", G["speed"]))
    except Exception: pass
    return ("", 204)
@app.route("/set_gap", methods=["POST"])
def set_gap():
    try: G["gap"] = float(request.get_json(force=True).get("gap", G["gap"]))
    except Exception: pass
    return ("", 204)
@app.route("/set_mode", methods=["POST"])
def set_mode(): return ("", 204)
@app.route("/set_target_type", methods=["POST"])
def set_target_type():
    try:
        t = request.get_json(force=True).get("type", "air")
        G["target_type"] = "ground" if t == "ground" else "air"
    except Exception: pass
    return jsonify(ok=True, target_type=G["target_type"])
@app.route("/set_maxrange", methods=["POST"])
def set_maxrange():
    try: G["maxrange"] = float(request.get_json(force=True).get("maxrange", G["maxrange"]))
    except Exception: pass
    return jsonify(ok=True, maxrange=G["maxrange"])
@app.route("/set_video_source", methods=["POST"])
def set_video_source():
    try:
        v = request.get_json(force=True).get("source", "front")
        G["video_src"] = v if v in ("front", "down", "depth") else "front"
    except Exception: pass
    return ("", 204)
@app.route("/set_telem_source", methods=["POST"])
def set_telem_source(): return ("", 204)
@app.route("/set_strike_mode", methods=["POST"])
def set_strike_mode(): G["strike_mode"] = _on(request); return ("", 204)
@app.route("/strike", methods=["POST"])
def strike(): G["strike_mode"] = True; G["strike_armed"] = True; return jsonify(ok=True)
@app.route("/abort_strike", methods=["POST"])
def abort_strike(): G["strike_armed"] = False; return jsonify(ok=True)
@app.route("/land", methods=["POST"])
def land(): G["action"] = "land"; return jsonify(ok=True)

if __name__ == "__main__":
    NODES = gz_subs(os.environ.get("A2A_WORLD", "uav_a2a"))   # keep refs alive (GC kills subs)
    threading.Thread(target=loop, daemon=True).start()
    print(f"[app_a2a_sitl] open http://localhost:{PORT}", flush=True)
    app.run(host="0.0.0.0", port=PORT, threaded=True)
