#!/usr/bin/env python3
"""Air + GROUND tracking demo (UAVros concept) on PX4-SITL. One chaser with a FORWARD camera (air UAV)
and a DOWNWARD camera (ground rover). Runs the ground behaviours one after another and records video:

  CLIMB -> AIR-TRACK (forward cam, follow the air runner)
        -> GROUND-FOLLOW (down cam, shadow the moving red rover)
        -> LAND (descend onto the rover, UAVros precision-landing)
        -> RESTRIKE climb -> STRIKE (dive onto the rover)

Vision: air = Hybrid YOLO+KLT (+dark-blob); ground = red color blob (UAVros-style). Guidance = MAVLink
offboard body-velocity (deploy/mavlink_control.py). Writes a COMBINED POV (forward | down, annotated) to
runs/videos/latest_chase.jpg + a frame sequence, and per-entity world poses to latest_telem_a2a.json so
the webui (app_a2a.py) shows the live feed + 3-D map of chaser/air/ground.

Webui override: reads runs/videos/a2a_ctrl.json {"mode":"auto|air|ground","action":"follow|land|strike"}
each loop; default auto runs the sequence.

  python3 sim/gz/a2a/air_ground_demo.py
"""
import argparse, json, math, os, sys, threading, time
os.environ.setdefault("GZ_IP", "127.0.0.1")
import numpy as np, cv2
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from gz.msgs10.pose_v_pb2 import Pose_V
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "deploy"))
from detector import HybridDetector, detect_ground_color, DEFAULT_WEIGHTS
from px4_util import arm_offboard
from mavlink_control import MavBridge

OUT = "/mnt/c/Users/admin/uav-vio-track/runs/videos"
CF = os.path.join(OUT, "a2a_frames")
CTRL = os.path.join(OUT, "a2a_ctrl.json")


def safe_replace(tmp, dst, tries=6):
    """Windows/9P: os.replace fails if the webui has the target open for reading. Retry, then skip."""
    for _ in range(tries):
        try:
            os.replace(tmp, dst); return True
        except PermissionError:
            time.sleep(0.01)
    try: os.remove(tmp)
    except Exception: pass
    return False
FW, FH = 640, 360       # forward cam
DW, DH = 512, 384       # down cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=14540)
    ap.add_argument("--world", default="uav_a2a")
    ap.add_argument("--chaser", default="jetray_chaser_0")
    ap.add_argument("--air", default="jetray_runner_1")
    ap.add_argument("--rover", default="rover")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    args = ap.parse_args()
    os.makedirs(CF, exist_ok=True)
    for f in os.listdir(CF):
        try: os.remove(os.path.join(CF, f))
        except Exception: pass

    # ---- gz subscriptions (keep node refs) ----
    S = {"fwd": None, "down": None, "ego": None, "air": None, "rover": None, "st": 0.0}
    def on_fwd(m):
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        S["fwd"] = cv2.cvtColor(a, cv2.COLOR_RGB2BGR); S["st"] = time.time()
    def on_down(m):
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        S["down"] = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    def on_pose(m):
        for p in m.pose:
            if p.name == args.chaser:
                S["ego"] = np.array([p.position.x, p.position.y, p.position.z])
                q = p.orientation
                S["ego_yaw"] = math.atan2(2 * (q.w * q.z + q.x * q.y),
                                          1 - 2 * (q.y * q.y + q.z * q.z))
            elif p.name == args.air: S["air"] = np.array([p.position.x, p.position.y, p.position.z])
            elif p.name == args.rover: S["rover"] = np.array([p.position.x, p.position.y, p.position.z])
    n1 = Node(); n1.subscribe(Image, "/chaser/camera_front", on_fwd)
    n2 = Node(); n2.subscribe(Image, "/chaser/camera_down", on_down)
    n3 = Node(); n3.subscribe(Pose_V, f"/world/{args.world}/pose/info", on_pose)

    det_air = HybridDetector(args.weights, every=5, conf=0.12, imgsz=640)

    # ---- offboard ----
    br = MavBridge(f"udpin:0.0.0.0:{args.port}")
    cmd = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw": 0.0}
    def streamer():
        while True:
            br.send_body_velocity_xy(cmd["vx"], cmd["vy"], cmd["vz"], cmd["yaw"]); time.sleep(0.05)
    threading.Thread(target=streamer, daemon=True).start()
    time.sleep(1.0)
    arm_offboard(br.m, timeout=60.0, label="chaser")

    def read_ctrl():
        try:
            with open(CTRL) as f: return json.load(f)
        except Exception:
            return {"mode": "auto", "action": "follow"}

    def telem(phase, target):
        ego, air, rov = S["ego"], S["air"], S["rover"]
        drones = []
        if ego is not None: drones.append({"name": args.chaser, "role": "chaser", "color": "#3aa0ff",
                                           "x": float(ego[0]), "y": float(ego[1]), "z": float(ego[2])})
        if air is not None: drones.append({"name": args.air, "role": "air-target", "color": "#39e6a3",
                                           "x": float(air[0]), "y": float(air[1]), "z": float(air[2])})
        if rov is not None: drones.append({"name": args.rover, "role": "ground-rover", "color": "#ff4d4d",
                                           "x": float(rov[0]), "y": float(rov[1]), "z": float(rov[2])})
        d = {"t": int(time.time() * 10) % 100000, "drones": drones, "phase": phase, "target": target}
        dst = os.path.join(OUT, "latest_telem_a2a.json")
        with open(dst + ".tmp", "w") as f: json.dump(d, f)
        safe_replace(dst + ".tmp", dst)

    def write_pov(fwd, down, da, dg, phase, target, info):
        L = cv2.resize(fwd, (int(FW * 360 / FH), 360)) if fwd is not None else np.zeros((360, 640, 3), np.uint8)
        R = cv2.resize(down, (int(DW * 360 / DH), 360)) if down is not None else np.zeros((360, 480, 3), np.uint8)
        if fwd is not None and da:
            x1, y1 = int(da["cx"] - da["w"] / 2), int(da["cy"] - da["h"] / 2)
            x2, y2 = int(da["cx"] + da["w"] / 2), int(da["cy"] + da["h"] / 2)
            sx, sy = L.shape[1] / FW, L.shape[0] / FH
            c = (0, 255, 0) if target == "air" else (120, 120, 120)
            cv2.rectangle(L, (int(x1*sx), int(y1*sy)), (int(x2*sx), int(y2*sy)), c, 2)
            cv2.putText(L, "AIR UAV", (int(x1*sx), int(y1*sy)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        if down is not None and dg:
            x1, y1 = int(dg["cx"] - dg["w"] / 2), int(dg["cy"] - dg["h"] / 2)
            x2, y2 = int(dg["cx"] + dg["w"] / 2), int(dg["cy"] + dg["h"] / 2)
            sx, sy = R.shape[1] / DW, R.shape[0] / DH
            c = (0, 0, 255) if target == "ground" else (120, 120, 120)
            cv2.rectangle(R, (int(x1*sx), int(y1*sy)), (int(x2*sx), int(y2*sy)), c, 2)
            cv2.putText(R, "ROVER", (int(x1*sx), int(y1*sy)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        cv2.putText(L, "FORWARD (air)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(R, "DOWN (ground)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        fr = np.hstack([L, R])
        cv2.rectangle(fr, (0, 0), (fr.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(fr, f"PX4-SITL  PHASE: {phase}  TARGET: {target}  {info}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tmp = os.path.join(OUT, "latest_chase.tmp.jpg")
        cv2.imwrite(tmp, fr); safe_replace(tmp, os.path.join(OUT, "latest_chase.jpg"))
        return fr

    # ---- climb to working altitude ----
    print("[demo] climbing to 8 m", flush=True)
    cmd["vz"] = -2.0; t0 = time.time()
    while time.time() - t0 < 7.0 and (S["ego"] is None or S["ego"][2] < 8.0):
        time.sleep(0.1)
    cmd["vz"] = 0.0

    # ---- phase schedule (auto). ctrl.json can override target/action. ----
    AIR_S, GFOLLOW_S = 22.0, 18.0
    KYAW, VMAX = 55.0, 5.0
    KG = 2.6                                  # ground down-cam servo gain (m/s per unit image error)
    phase = "AIR-TRACK"; t_phase = time.time(); fno = 0
    relaunch_done = False
    print("[demo] AIR-TRACK", flush=True)
    while True:
        if S["st"] == 0.0:
            time.sleep(0.02); continue
        fno += 1
        fwd, down = S["fwd"], S["down"]
        ego, air, rov = S["ego"], S["air"], S["rover"]
        ctrl = read_ctrl(); mode = ctrl.get("mode", "auto"); action = ctrl.get("action", "follow")
        now = time.time(); info = ""

        # ---------- decide phase ----------
        if mode == "air":
            phase = "AIR-TRACK"
        elif mode == "ground":
            phase = {"follow": "GROUND-FOLLOW", "land": "LAND", "strike": "STRIKE"}.get(action, "GROUND-FOLLOW")
        else:  # auto sequence
            if phase == "AIR-TRACK" and now - t_phase > AIR_S:
                phase = "GROUND-FOLLOW"; t_phase = now; print("[demo] GROUND-FOLLOW", flush=True)
            elif phase == "GROUND-FOLLOW" and now - t_phase > GFOLLOW_S:
                phase = "LAND"; t_phase = now; print("[demo] LAND on rover", flush=True)

        da = det_air(fwd) if fwd is not None else None
        dg = detect_ground_color(down) if down is not None else None
        target = "air" if phase == "AIR-TRACK" else "ground"

        # ---------- control ----------
        if phase == "AIR-TRACK":
            # forward-cam visual servo (yaw to bearing, vz to image-y, fwd toward target)
            if da is not None:
                ex = (da["cx"] - FW / 2) / (FW / 2); ey = (da["cy"] - FH / 2) / (FH / 2)
                cmd["yaw"] = float(np.clip(KYAW * ex, -45, 45))
                cmd["vz"] = float(np.clip(3.0 * ey, -2.5, 2.5))
                cmd["vx"] = float(np.clip(VMAX * (1.0 - abs(ex)), 1.0, VMAX)); cmd["vy"] = 0.0
                info = f"air lock"
            else:
                cmd["vx"] = 1.0; cmd["vy"] = 0.0; cmd["vz"] = 0.0; cmd["yaw"] = 0.0
                info = "air search"
        else:
            # GROUND phases use the DOWN camera. Center the rover with body vx/vy; hold heading.
            cmd["yaw"] = 0.0
            if dg is None and rov is not None and ego is not None:
                # ACQUIRE: rover not yet under the down cam -> GPS-level approach toward its XY
                # (UAVros: GPS gets you close, vision does the terminal centering/landing).
                yw = S.get("ego_yaw", 0.0)
                dx, dy = float(rov[0] - ego[0]), float(rov[1] - ego[1])
                bx = dx * math.cos(yw) + dy * math.sin(yw)        # world->body
                by = -dx * math.sin(yw) + dy * math.cos(yw)
                cmd["vx"] = float(np.clip(0.7 * bx, -VMAX, VMAX))
                cmd["vy"] = float(np.clip(0.7 * by, -VMAX, VMAX))
                z = ego[2]; cmd["vz"] = float(np.clip(-0.8 * (8.0 - z), -1.5, 1.5))
                info = f"approach rover  d=({dx:.1f},{dy:.1f})"
            elif dg is not None:
                ex = (dg["cx"] - DW / 2) / (DW / 2); ey = (dg["cy"] - DH / 2) / (DH / 2)
                # down cam (looking -Z): image-up(-ey) -> body +X forward; image-right(+ex) -> body +Y right
                cmd["vx"] = float(np.clip(-KG * ey, -VMAX, VMAX))
                cmd["vy"] = float(np.clip(KG * ex, -VMAX, VMAX))
                centered = abs(ex) < 0.18 and abs(ey) < 0.18
                z = ego[2] if ego is not None else 8.0
                if phase == "GROUND-FOLLOW":
                    cmd["vz"] = float(np.clip(0.8 * (8.0 - z) * -1.0, -1.5, 1.5))  # hold ~8 m
                    info = f"shadow rover  err=({ex:.2f},{ey:.2f})"
                elif phase == "LAND":
                    cmd["vz"] = 1.2 if centered else 0.4          # descend (down+) faster when centered
                    info = f"descending z={z:.1f} centered={centered}"
                    if z < 1.1:
                        cmd["vx"] = cmd["vy"] = cmd["vz"] = 0.0
                        print(f"[demo] LANDED on rover (z={z:.2f})", flush=True)
                        phase = "RESTRIKE"; t_phase = now
                elif phase == "STRIKE":
                    cmd["vz"] = 3.0 if centered else 1.0           # dive
                    info = f"STRIKE diving z={z:.1f}"
                    rz = rov[2] if rov is not None else 0.3
                    if (z - rz) < 1.0:
                        cmd["vx"] = cmd["vy"] = cmd["vz"] = 0.0
                        print(f"[demo] HIT rover (z={z:.2f})", flush=True); phase = "DONE"; t_phase = now
            else:
                cmd["vx"] = cmd["vy"] = 0.0; cmd["vz"] = 0.0; info = "ground search"

        # RESTRIKE: climb back up, then auto -> STRIKE
        if phase == "RESTRIKE":
            cmd["vx"] = cmd["vy"] = 0.0; cmd["yaw"] = 0.0
            z = ego[2] if ego is not None else 0
            cmd["vz"] = -2.5
            info = f"climb for strike z={z:.1f}"
            if z > 7.0:
                phase = "STRIKE"; t_phase = now; print("[demo] STRIKE run", flush=True)

        fr = write_pov(fwd, down, da, dg, phase, target, info)
        telem(phase, target)
        if fr is not None:                              # frame sequence for the recording
            try: cv2.imwrite(os.path.join(CF, f"c{fno:05d}.png"), fr)
            except Exception: pass

        if phase == "DONE" and now - t_phase > 3.0:
            print("[demo] sequence complete", flush=True); break
        time.sleep(0.02)

    print("[demo] done", flush=True)


if __name__ == "__main__":
    main()
