#!/usr/bin/env python3
"""CHASER vision-guided strike (PX4 SITL offboard). Real onboard pipeline:
  forward camera (/chaser/camera_front) -> Hybrid YOLO+KLT detector -> image-based visual servoing
  -> body-frame velocity setpoints (yaw toward target, climb/descend to center it, close range) sent to
  PX4 via deploy/mavlink_control.py MavBridge on udp 14540. Steering is 100% VISION; on brief loss it
  coasts the last command, else hovers. HIT is declared on TRUE range (gz /pose/info) < hit_radius.

Writes runs/videos/latest_chase.jpg (annotated POV) + latest_telem.json (world ego/tgt) so the existing
webui (webui/app_isaac.py, :5060) shows the live POV + dual-drone 3D map.

  # world up (launch_a2a.sh) + runner flying (runner_mission.py), then:
  python3 sim/gz/a2a/chaser_strike.py --secs 90
"""
import argparse, json, math, os, sys, threading, time
os.environ.setdefault("GZ_IP", "127.0.0.1")
import numpy as np, cv2
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from gz.msgs10.pose_v_pb2 import Pose_V
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "deploy"))
from detector import HybridDetector, DEFAULT_WEIGHTS
from px4_util import arm_offboard
from mavlink_control import MavBridge

OUT = "/mnt/c/Users/admin/uav-vio-track/runs/videos"
CF = os.path.join(OUT, "a2a_frames")
W, H = 640, 360


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=14540)
    ap.add_argument("--world", default="uav_a2a")
    ap.add_argument("--chaser", default="jetray_chaser_0")
    ap.add_argument("--runner", default="jetray_runner_1")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--secs", type=float, default=90.0)
    ap.add_argument("--vmax", type=float, default=4.5)
    ap.add_argument("--hit", type=float, default=1.5)
    ap.add_argument("--climb-alt", type=float, default=8.0)   # ~match runner altitude for the tail-chase
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(CF, exist_ok=True)
    for f in os.listdir(CF):
        try: os.remove(os.path.join(CF, f))
        except Exception: pass

    # ---- gz subscriptions (keep node refs; anonymous Node() gets GC'd and the sub dies) ----
    S = {"bgr": None, "stamp": 0.0, "ego": None, "tgt": None}
    def on_img(m):
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        S["bgr"] = cv2.cvtColor(a, cv2.COLOR_RGB2BGR); S["stamp"] = time.time()
    def on_pose(m):
        for p in m.pose:
            if p.name == args.chaser: S["ego"] = np.array([p.position.x, p.position.y, p.position.z])
            elif p.name == args.runner: S["tgt"] = np.array([p.position.x, p.position.y, p.position.z])
    ncam = Node(); ncam.subscribe(Image, "/chaser/camera_front", on_img)
    npose = Node(); npose.subscribe(Pose_V, f"/world/{args.world}/pose/info", on_pose)

    det = HybridDetector(args.weights, every=5, conf=0.12, imgsz=896)

    # ---- offboard via MavBridge: stream the latest body-velocity at 20 Hz ----
    br = MavBridge(f"udpin:0.0.0.0:{args.port}")
    cmd = {"fwd": 0.0, "vz": 0.0, "yaw": 0.0}
    def streamer():
        while True:
            br.send_body_velocity(cmd["fwd"], cmd["vz"], cmd["yaw"]); time.sleep(0.05)
    threading.Thread(target=streamer, daemon=True).start()
    time.sleep(1.0)
    arm_offboard(br.m, timeout=60.0, label="chaser")

    # climb to acquisition altitude (body vz<0 = up)
    print(f"[chaser] climbing to ~{args.climb_alt} m", flush=True)
    cmd["vz"] = -2.0
    t0 = time.time()
    while time.time() - t0 < 8.0 and (S["ego"] is None or S["ego"][2] < args.climb_alt):
        time.sleep(0.1)
    cmd["vz"] = 0.0

    # ---- vision servo / strike loop ----
    KYAW, KVZ = 55.0, 3.0
    n = locks = lost = vis_steer = 0
    rmin = 1e9; hit = False; last_stamp = -1.0
    t0 = time.time()
    print("[chaser] engaging (vision servo)", flush=True)
    while time.time() - t0 < args.secs:
        if S["stamp"] == last_stamp:
            time.sleep(0.005); continue
        last_stamp = S["stamp"]; fr = S["bgr"].copy(); n += 1
        out = det(fr)
        ego, tgt = S["ego"], S["tgt"]
        rng = float(np.linalg.norm(tgt - ego)) if (ego is not None and tgt is not None) else None
        if rng is not None: rmin = min(rmin, rng)

        if out:
            locks += 1; vis_steer += 1; lost = 0
            ex = (out["cx"] - W / 2) / (W / 2); ey = (out["cy"] - H / 2) / (H / 2)
            cmd["yaw"] = float(np.clip(KYAW * ex, -45, 45))      # yaw toward target (CW +)
            cmd["vz"] = float(np.clip(KVZ * ey, -3.0, 3.0))      # center vertically (img-down=+ -> descend)
            cmd["fwd"] = float(np.clip(args.vmax * (1.0 - abs(ex)), 1.5, args.vmax))  # close; slow if off-axis
        else:
            lost += 1
            if lost <= 8:                                        # brief coast: hold heading, stop closing
                cmd["fwd"] = 0.0; cmd["vz"] = 0.0; cmd["yaw"] *= 0.5
            elif lost <= 160:                                    # reacquire: hold heading, creep forward
                cmd["fwd"] = 2.0; cmd["vz"] = 0.0; cmd["yaw"] = 0.0
            else:                                                # long loss -> stop running away, slow yaw scan
                cmd["fwd"] = 0.0; cmd["vz"] = 0.0; cmd["yaw"] = 15.0

        if rng is not None and rng < args.hit:
            hit = True; cmd["fwd"] = cmd["vz"] = cmd["yaw"] = 0.0
            print(f"[chaser] HIT at range {rng:.2f} m (t={time.time()-t0:.1f}s)", flush=True)

        # annotated POV + telemetry for the webui
        if out:
            x1 = int(out["cx"] - out["w"] / 2); y1 = int(out["cy"] - out["h"] / 2)
            x2 = int(out["cx"] + out["w"] / 2); y2 = int(out["cy"] + out["h"] / 2)
            c = (0, 255, 0) if out["method"] == "YOLO" else (0, 200, 255)
            cv2.rectangle(fr, (x1, y1), (x2, y2), c, 2)
            cv2.putText(fr, f'TARGET [{out["method"]}]', (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        tag = "HIT" if hit else ("LOCK" if out else "SEARCH")
        for k, s in enumerate([f"CHASER POV  PX4-SITL  vision-servo  [{tag}]",
                               f"range {rng:5.1f} m" if rng is not None else "range --",
                               f"lock {100*locks/max(1,n):.0f}%  fwd {cmd['fwd']:.1f} yaw {cmd['yaw']:.0f}"]):
            cv2.putText(fr, s, (8, 18 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        tmp = os.path.join(OUT, "latest_chase.tmp.jpg")
        cv2.imwrite(tmp, fr); os.replace(tmp, os.path.join(OUT, "latest_chase.jpg"))
        cv2.imwrite(os.path.join(CF, f"c{n:04d}.png"), fr)       # POV sequence for the replay video
        if ego is not None and tgt is not None:
            telem = {"ego": [float(x) for x in ego], "tgt": [float(x) for x in tgt],
                     "range": round(rng, 2), "tgo": 0.0, "step": n,
                     "locked": bool(out), "hit": hit}
            tj = os.path.join(OUT, "latest_telem.json")
            with open(tj + ".tmp", "w") as f: json.dump(telem, f)
            os.replace(tj + ".tmp", tj)

        if hit:
            time.sleep(2.0); break

    dt = time.time() - t0
    print(f"[chaser] DONE frames={n} lock={100*locks/max(1,n):.0f}% "
          f"vision-steered={100*vis_steer/max(1,n):.0f}% min_range={rmin:.2f}m hit={hit} cam={n/dt:.1f}Hz",
          flush=True)


if __name__ == "__main__":
    main()
