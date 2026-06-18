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
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from gz.msgs10.pose_v_pb2 import Pose_V
from mavlink_control import MavBridge
from px4_util import arm_offboard
from ultralytics import YOLO

UAV_W = os.environ.get("UAV_MODEL", os.path.join(REPO, "runs/train/uav_diverse/weights/best.pt"))
COCO_W = os.environ.get("COCO_MODEL", "yolov8n.pt")          # auto-downloads; car/truck/person
GROUND_CLS = {0: "person", 2: "car", 5: "bus", 7: "truck"}
FW, FH = 640, 360
DW, DH = 512, 384
PORT = int(os.environ.get("A2A_SITL_PORT", "5063"))
HFOV_F = 68.75                                  # forward cam HFOV (deg) = 1.20 rad
VFOV_F = math.degrees(2 * math.atan(math.tan(math.radians(HFOV_F) / 2) * FH / FW))


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
     "action": "follow", "frame": None, "tel": {}, "sel": None, "clear": False}

S = {"front": None, "down": None, "depth": None, "ego": None, "yaw": 0.0,
     "air": None, "ground": None, "stamp": 0.0}


# ---------- gz subscriptions (keep node refs) ----------
def gz_subs(world):
    def on_front(m):
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        S["front"] = cv2.cvtColor(a, cv2.COLOR_RGB2BGR); S["stamp"] = time.time()
    def on_down(m):
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        S["down"] = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    def on_depth(m):
        d = np.frombuffer(m.data, np.float32).reshape(m.height, m.width).copy()
        d[~np.isfinite(d)] = 0.0; S["depth"] = d
    def on_pose(m):
        for p in m.pose:
            if p.name == "jetray_chaser_0":
                S["ego"] = np.array([p.position.x, p.position.y, p.position.z]); q = p.orientation
                S["yaw"] = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
            elif p.name == "jetray_runner_1":
                S["air"] = np.array([p.position.x, p.position.y, p.position.z])
            elif p.name == "rover":
                S["ground"] = np.array([p.position.x, p.position.y, p.position.z])
    nodes = [Node(), Node(), Node(), Node()]
    nodes[0].subscribe(Image, "/chaser/camera_front", on_front)
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


def depth_color(d):
    if d is None:
        return np.zeros((DH, DW, 3), np.uint8)
    v = np.clip(d, 0, 60) / 60.0
    return cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_JET)


def loop():
    from pymavlink import mavutil
    air_model = YOLO(UAV_W); coco = YOLO(COCO_W)
    br = MavBridge("udpin:0.0.0.0:14540")
    cmd = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw": 0.0, "mode": "takeoff", "alt": 8.0}

    HOLD = (0.0, 10.0, -9.0)                       # local NED hold point ~ world (10,0,9): engagement area
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

    lock = None; lock_miss = 0      # {"kind":"air"|"ground", "box":..} in its source image
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

        # ---- detect (throttled + alternating so the Flask UI thread isn't GIL-starved) ----
        if front is not None and fno % 4 == 0:
            air_dets = yolo_boxes(air_model, front, 0.12, 512)
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
        if down is not None and fno % 4 == 2:
            gnd_dets = yolo_boxes(coco, down, 0.30, 416, classes=list(GROUND_CLS))

        # ---- selection (click / autolock / clear) ----
        if G["clear"]:
            lock = None; G["clear"] = False; G["strike_armed"] = False
        sel = G.pop("sel", None) if "sel" in G else None
        if sel is not None:
            sx, sy = sel
            src = G["video_src"]
            pool = gnd_dets if src == "down" else air_dets
            kind = "ground" if src == "down" else "air"
            iw, ih = (DW, DH) if src == "down" else (FW, FH)
            if pool:
                px, py = sx * iw, sy * ih
                d = min(pool, key=lambda d: (np.mean([d["box"][0], d["box"][2]]) - px) ** 2 +
                                            (np.mean([d["box"][1], d["box"][3]]) - py) ** 2)
                lock = {"kind": kind, "box": d["box"]}
        elif G["autolock"] and lock is None:
            if air_dets:
                d = max(air_dets, key=lambda d: d["conf"]); lock = {"kind": "air", "box": d["box"]}
            elif gnd_dets:
                d = max(gnd_dets, key=lambda d: d["conf"]); lock = {"kind": "ground", "box": d["box"]}

        # ---- refresh lock box from nearest current detection of same kind ----
        rng = None; center_err = None
        if lock is not None:
            pool = air_dets if lock["kind"] == "air" else gnd_dets
            if pool:
                lb = lock["box"]; lc = ((lb[0]+lb[2])/2, (lb[1]+lb[3])/2)
                d = min(pool, key=lambda d: ((d["box"][0]+d["box"][2])/2-lc[0])**2 +
                                            ((d["box"][1]+d["box"][3])/2-lc[1])**2)
                lock["box"] = d["box"]; lock_miss = 0
            else:
                lock_miss += 1
                if lock_miss > 25:                 # lost the locked target -> drop it (no dead-reckon away)
                    lock = None

        # ---- control ----
        G["locked"] = lock is not None
        if lock is None:
            cmd["mode"] = "hold"                   # position-hold at the engagement area + re-acquire
            cmd["vx"] = cmd["vy"] = cmd["vz"] = 0.0; cmd["yaw"] = 0.0
            G["state"] = "SEARCH" if G["search"] else "DETECT"; G["tgt_est"] = None
        elif lock["kind"] == "air":
            cmd["mode"] = "vel"; G["state"] = "TRACK-AIR"
            b = lock["box"]; cx = (b[0]+b[2])/2; cy = (b[1]+b[3])/2
            ex = (cx - FW/2)/(FW/2); ey = (cy - FH/2)/(FH/2); center_err = float(math.hypot(ex, ey))
            if depth is not None:                       # range from forward depth at bbox center
                du, dv = int(cx/FW*depth.shape[1]), int(cy/FH*depth.shape[0])
                roi = depth[max(0,dv-3):dv+3, max(0,du-3):du+3]; roi = roi[(roi > 0.3) & (roi < 250)]
                rng = float(np.median(roi)) if roi.size else None
            sp = float(G["speed"])
            cmd["yaw"] = float(np.clip(55*ex, -45, 45))
            cmd["vz"] = float(np.clip(3*ey, -2.5, 2.5))
            close = (rng - G["gap"]) if rng else (sp*(1-abs(ex)))
            cmd["vx"] = float(np.clip(0.8*close if rng else sp*(1-abs(ex)), 0.0, sp)); cmd["vy"] = 0.0
            if rng is not None and 1.0 < rng < 80.0 and ego is not None:   # VISION-ONLY target estimate
                est = tgt_world_from_vision(ego, yaw, ex, ey, rng)
                pe = G.get("tgt_est"); G["tgt_est"] = est if not pe else [0.6*p+0.4*e for p, e in zip(pe, est)]
        else:  # ground
            cmd["mode"] = "vel"
            b = lock["box"]; cx = (b[0]+b[2])/2; cy = (b[1]+b[3])/2
            ex = (cx - DW/2)/(DW/2); ey = (cy - DH/2)/(DH/2); center_err = float(math.hypot(ex, ey))
            cmd["yaw"] = 0.0; sp = float(G["speed"])
            cmd["vx"] = float(np.clip(-2.6*ey, -sp, sp)); cmd["vy"] = float(np.clip(2.6*ex, -sp, sp))
            centered = abs(ex) < 0.18 and abs(ey) < 0.18
            z = ego[2] if ego is not None else 8.0       # chaser GPS altitude (ego only)
            if G["strike_mode"] and G["strike_armed"]:
                G["state"] = "STRIKE"; cmd["vz"] = 3.0 if centered else 1.0
                if z < 1.6:                              # chaser near ground over the vision-locked car
                    G["last_hit"] = f"HIT car @ {z:.1f}m"; G["strike_armed"] = False
            elif G["action"] == "land":
                G["state"] = "LAND"; cmd["vz"] = 1.2 if centered else 0.4
                if z < 1.3:
                    G["last_hit"] = "LANDED on car"; G["action"] = "follow"
                    cmd["vx"] = cmd["vy"] = cmd["vz"] = 0.0
            else:
                G["state"] = "TRACK-GROUND"; cmd["vz"] = float(np.clip(-0.8*(8.0 - z), -1.5, 1.5))
            # NOTE: ground nav is VISION-ONLY (down-cam image servo above). No target GPS/coords used.

        # ---- altitude governor: hold ~9 m so the chaser stays where it can see air+ground
        #      (LAND/STRIKE override to descend). Robust to takeoff overshoot. ----
        if ego is not None and G["state"] not in ("LAND", "STRIKE"):
            cmd["vz"] = float(np.clip(-0.9 * (9.0 - ego[2]), -2.0, 2.0))

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
            "shadows": G.get("shadows", 0),
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
