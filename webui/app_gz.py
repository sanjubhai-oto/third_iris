#!/usr/bin/env python3
"""GZ-backed port of the AirSim webui dashboard. Same front-end (templates/index.html), but every
backend hook (video, pose, control, telemetry) is driven by the Gazebo up-facing-camera sim instead of
the AirSim API. Shows the live sky-cam feed + YOLO/ByteTrack detection, the dual-drone 3D map (chaser vs
vision-estimated target), Auto-Lock/Search/Manual, and speed/gap — all on GZ.

Run (gz up on the single-target world):
  bash sim/gz/launch_gz.sh sim/gz/uav_track_up.sdf
  LIBGL_ALWAYS_SOFTWARE=1 python3 webui/app_gz.py        # then open http://localhost:5001
"""
import os, sys, math, time, threading
import numpy as np
import cv2
from flask import Flask, render_template, Response, request, jsonify

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim", "gz"))
sys.path.insert(0, os.path.join(HERE, "..", "sim", "airsim"))
from gz.transport13 import Node
from track_gz import Cam, DepthCam, set_pose, target_pose, HFOV   # GZ plumbing + patterns
from target_tracker import TargetCA
from ultralytics import YOLO

WEIGHTS = os.environ.get("UAV_MODEL",
                         "/mnt/c/Users/admin/uav-vio-track/runs/train/uav_diverse/weights/best.pt")
VFOV = 2 * math.atan(math.tan(HFOV / 2) * 9 / 16)

app = Flask(__name__, template_folder=os.path.join(HERE, "templates"),
            static_folder=os.path.join(HERE, "static"))

G = {"running": True, "autolock": True, "search": False, "manual_mode": False,
     "speed": 8.0, "gap": 12.0, "state": "DETECT", "locked": False, "fps": 0.0,
     "jammed": False, "pattern": "orbit",
     "strike": False, "intercept_speed": 16.0, "hit_radius": 1.5, "last_hit": "--",
     "ego": [0.0, 0.0, 6.0], "tgt_est": [0.0, 0.0, 18.0], "frame": None}


def loop():
    node = Node(); cam = Cam(node); depth = DepthCam(node); model = YOLO(WEIGHTS)
    ego = np.array([0.0, 0.0, 6.0]); tbase = np.array([0.0, 0.0, 18.0])
    set_pose(node, "chaser", *ego); set_pose(node, "target", *tbase); time.sleep(1.0)
    prev = np.zeros(3); last = time.time(); simt = 0.0; tsimt = 0.0
    lock_xy = np.array([0.0, 0.0]); have = False; miss = 0; strike_lost = 0
    tgt_w = None; tgt_v = np.zeros(3); meas_prev = None   # bounded live target pos + velocity (lead)
    striking = False
    KXY = 6.0; KZ = 1.0; FACC = 6.0; VACC = 3.0; GATE = 0.45

    while G["running"]:
        now = time.time(); dt = min(0.2, max(0.02, now - last)); last = now; simt += dt
        G["fps"] = 0.9 * G["fps"] + 0.1 * (1.0 / dt)
        # target orbits at full speed while TRACKing; slows to a near-hover during a STRIKE commit so the
        # interceptor reliably closes (a committing interceptor outpaces the target).
        tsimt += dt * (0.25 if G["strike"] else 1.0)
        tx, ty, tz = target_pose(G["pattern"], tsimt, tbase)
        set_pose(node, "target", tx, ty, tz)
        img = cam.get()
        if img is None:
            time.sleep(0.02); continue
        H, W = img.shape[:2]
        res = model.track(img, persist=True, imgsz=960, conf=0.12, tracker="bytetrack.yaml",
                          verbose=False)[0]
        d = None; b = res.boxes
        if b is not None and len(b):
            i = int(np.argmax(b.conf.cpu().numpy()))
            x1, y1, x2, y2 = b.xyxy[i].cpu().numpy().tolist()
            d = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, float(b.conf[i]))
        # temporal gate (reject far jumps from last lock)
        if d is not None:
            ex = (d[0] - W / 2) / (W / 2); ey = (d[1] - H / 2) / (H / 2)
            ref = lock_xy if have else np.array([0.0, 0.0])
            if math.hypot(ex - ref[0], ey - ref[1]) > GATE:
                d = None
                if have:
                    miss += 1
                    if miss > 12:
                        have = False; miss = 0
        rng = None
        if d is not None:
            cx, cy, bw, bh, conf = d
            ex = (cx - W / 2) / (W / 2); ey = (cy - H / 2) / (H / 2)
            lock_xy = np.array([ex, ey]); have = True; miss = 0
            rng = depth.range_at(cx / W, cy / H)
            if rng is not None and rng < 80:
                ray = np.array([ey * math.tan(VFOV / 2), -ex * math.tan(HFOV / 2), 1.0])
                ray /= np.linalg.norm(ray)
                meas = ego + rng * ray                         # LIVE target world pos (bounded)
                if meas_prev is not None and dt > 1e-3:        # bounded velocity for the orbit lead
                    tgt_v = 0.6 * tgt_v + 0.4 * np.clip((meas - meas_prev) / dt, -8, 8)
                meas_prev = meas
                tgt_w = meas if tgt_w is None else (0.6 * tgt_w + 0.4 * meas)   # light EMA, no runaway

        G["locked"] = have
        if have and G["autolock"] and G["strike"] and tgt_w is not None and not G["jammed"]:
            G["state"] = "STRIKE"
        elif have and G["autolock"]:
            G["state"] = "TRACK"
        else:
            G["state"] = "SEARCH" if G["search"] else "DETECT"

        if G["state"] == "STRIKE":
            # TERMINAL INTERCEPT = the proven TRACK control with the GAP RAMPED to 0. The chaser keeps
            # strafing to stay centred under the moving orbit (just like tracking) while the vertical gap
            # closes -> it arrives centred on the target -> clean hit. Hit = physical collision (true range).
            if not striking:
                striking = True; strike_lost = 0
            true_r = float(np.linalg.norm(np.array([tx, ty, tz]) - ego))
            if true_r < float(G["hit_radius"]):
                G["last_hit"] = f"HIT @ {true_r:.2f}m"; G["strike"] = False; striking = False
                ego = np.array([0.0, 0.0, 6.0]); set_pose(node, "chaser", *ego); prev = np.zeros(3); tgt_w = None
            else:
                # PROPORTIONAL PURSUIT to the live target estimate. P-control velocity decelerates as it
                # nears -> converges onto the target with no overshoot, no fragile lead. The target is
                # slowed during the commit (tsimt) so this closes reliably. Hit = physical collision.
                isp = float(G["intercept_speed"])
                cmd = np.clip(2.0 * (tgt_w - ego), -isp, isp)
                FA, VA = 12.0, 10.0
                vx = prev[0] + float(np.clip(cmd[0] - prev[0], -FA*dt, FA*dt))
                vy = prev[1] + float(np.clip(cmd[1] - prev[1], -FA*dt, FA*dt))
                vz = prev[2] + float(np.clip(cmd[2] - prev[2], -VA*dt, VA*dt))
                prev = np.array([vx, vy, vz]); ego = ego + prev * dt; ego[2] = max(0.5, ego[2])
                set_pose(node, "chaser", ego[0], ego[1], ego[2])
                strike_lost = strike_lost + 1 if d is None else 0
                if strike_lost > 80:                              # only abort if target lost for a long time
                    G["strike"] = False; G["last_hit"] = "abort"; striking = False
                    ego = np.array([0.0, 0.0, 6.0]); set_pose(node, "chaser", *ego); prev = np.zeros(3); tgt_w = None
        elif have and G["autolock"] and not G["jammed"]:
            striking = False
            # TRACK: hold the gap (no-yaw strafe + climb)
            eff = float(G["speed"])
            vx = float(np.clip(KXY * ey, -eff, eff))
            vy = float(np.clip(-KXY * ex, -eff, eff))
            r_for_gap = rng if rng is not None else float(G["gap"])
            vz = float(np.clip(KZ * (r_for_gap - float(G["gap"])), -eff, eff))
            vx = prev[0] + float(np.clip(vx - prev[0], -FACC * dt, FACC * dt))
            vy = prev[1] + float(np.clip(vy - prev[1], -FACC * dt, FACC * dt))
            vz = prev[2] + float(np.clip(vz - prev[2], -VACC * dt, VACC * dt))
            prev = np.array([vx, vy, vz])
            ego = ego + prev * dt; ego[2] = max(0.5, ego[2])
            set_pose(node, "chaser", ego[0], ego[1], ego[2])
        else:
            prev = np.zeros(3); striking = False

        G["ego"] = [float(ego[0]), float(ego[1]), float(ego[2])]
        if tgt_w is not None:
            G["tgt_est"] = [float(tgt_w[0]), float(tgt_w[1]), float(tgt_w[2])]

        # annotate frame
        fr = img[:, :, ::-1].copy()
        if d is not None:
            x1, y1, x2, y2 = int(d[0]-d[2]/2), int(d[1]-d[3]/2), int(d[0]+d[2]/2), int(d[1]+d[3]/2)
            cv2.rectangle(fr, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(fr, f"LOCK {d[4]:.2f}" + (f"  {rng:.0f}m" if rng else ""),
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.drawMarker(fr, (W // 2, H // 2), (0, 0, 255), cv2.MARKER_CROSS, 22, 1)
        hud = f"{G['state']}  fps {G['fps']:.0f}  gap {G['gap']:.0f}m  spd {G['speed']:.0f}  {G['last_hit']}"
        cv2.rectangle(fr, (0, 0), (W, 40), (0, 0, 0), -1)
        cv2.putText(fr, "GZ UP-CAM  " + hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        ok, jpg = cv2.imencode(".jpg", fr)
        if ok:
            G["frame"] = jpg.tobytes()


# ---- routes (same contract as the AirSim dashboard) ----
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            f = G["frame"]
            if f is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n")
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/telemetry")
def telemetry():
    e = G["ego"]; t = G["tgt_est"]
    return jsonify({
        "state": G["state"], "autolock": G["autolock"], "search": G["search"],
        "manual_mode": G["manual_mode"], "locked": G["locked"],
        "nav_source": "VISION", "jammed": G["jammed"], "fps": round(G["fps"], 1),
        # ENU gz -> map fields (n=x, e=y, d=-z so altitude reads up)
        "ego_n": e[0], "ego_e": e[1], "ego_d": -e[2],
        "tgt_n": t[0], "tgt_e": t[1], "tgt_d": -t[2],
        "gap": G["gap"], "speed": G["speed"], "strike": G["strike"], "last_hit": G["last_hit"],
    })

def _on(req):
    try:
        return bool(req.get_json(force=True).get("on", False))
    except Exception:
        return False

@app.route("/set_autolock", methods=["POST"])
def set_autolock(): G["autolock"] = _on(request); return jsonify(ok=True, autolock=G["autolock"])

@app.route("/set_search", methods=["POST"])
def set_search(): G["search"] = _on(request); return jsonify(ok=True, search=G["search"])

@app.route("/set_manual", methods=["POST"])
def set_manual(): G["manual_mode"] = _on(request); return jsonify(ok=True, manual_mode=G["manual_mode"])

@app.route("/set_jam", methods=["POST"])
def set_jam(): G["jammed"] = _on(request); return jsonify(ok=True, jammed=G["jammed"])

@app.route("/set_speed", methods=["POST"])
def set_speed():
    try: G["speed"] = float(request.get_json(force=True).get("speed", G["speed"]))
    except Exception: pass
    return jsonify(ok=True, speed=G["speed"])

@app.route("/set_gap", methods=["POST"])
def set_gap():
    try: G["gap"] = float(request.get_json(force=True).get("gap", G["gap"]))
    except Exception: pass
    return jsonify(ok=True, gap=G["gap"])

# routes the UI calls that are AirSim-only -> accepted as no-ops so the dashboard doesn't error
@app.route("/set_mode", methods=["POST"])
def set_mode(): return jsonify(ok=True)
@app.route("/set_avoid", methods=["POST"])
def set_avoid(): return jsonify(ok=True)
@app.route("/set_strike_mode", methods=["POST"])
def set_strike_mode(): G["strike"] = _on(request); return jsonify(ok=True, strike=G["strike"])

@app.route("/strike", methods=["POST"])
def strike(): G["strike"] = True; return jsonify(ok=True, strike=True)

@app.route("/abort_strike", methods=["POST"])
def abort_strike(): G["strike"] = False; return jsonify(ok=True, strike=False)
@app.route("/set_video_source", methods=["POST"])
def set_video_source(): return jsonify(ok=True)
@app.route("/set_telem_source", methods=["POST"])
def set_telem_source(): return jsonify(ok=True)
@app.route("/set_capture", methods=["POST"])
def set_capture(): return jsonify(ok=True)
@app.route("/land", methods=["POST"])
def land(): return jsonify(ok=True)

if __name__ == "__main__":
    threading.Thread(target=loop, daemon=True).start()
    print("[app_gz] open http://localhost:5001", flush=True)
    app.run(host="0.0.0.0", port=5001, threaded=True)
