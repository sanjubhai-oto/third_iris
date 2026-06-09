#!/usr/bin/env python3
"""Scenario test + recorder for the standoff air-to-air guidance.

Reproduces the real-drone use case and proves the controller is centered, doesn't spin, and
matches the target's speed:

  * Ego launches and holds at ~5 m altitude; Target is placed ~30 m ahead, facing the Ego.
  * Phase APPROACH: GPS-aided standoff guidance closes from ~30 m to the set gap.
  * Phase VISION-ONLY: the controller uses **only the camera** (bbox bearing + depth range) to
    estimate the target's 3-D position — no GPS in the loop — while the Target flies a sequence of
    patterns that include lateral crossing and altitude changes. This phase is recorded to an AVI.
  * Per-frame metrics (centering error, range vs gap, altitude-tracking error, lost frames) are
    logged and summarized per-pattern and overall at the end (GPS truth used for *scoring only*).

Run (AirSim Blocks running, fine-tuned weights present):
    python sim/airsim/scenario_eval.py --gap 12 --pattern-secs 16 --record
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
import sys
sys.path.insert(0, str(REPO / "sim" / "airsim"))
from guidance import (Vec3KF, target_from_vision, target_from_vision_cam, standoff_command,  # noqa: E402
                      vfov_from_hfov, look_at_angles, camera_rel_quat, euler_R, R_to_quat)
from smooth_control import ImageKalman          # noqa: E402  image-space predictor (coast through dropouts)

MODEL = str(REPO / "runs/train/airsim_drone/weights/best.pt")
TRACKER = str(REPO / "perception" / "trackers" / "botsort_uav.yaml")  # BoT-SORT+ReID+GMC (matches webui)
EGO_HOME = np.array([0.0, 0.0, 0.0])
TARGET_HOME = np.array([8.0, 0.0, 0.0])
HFOV = 90.0
START_RANGE = 30.0      # initial Ego<->Target separation (m)
START_ALT = 5.0         # Ego launch altitude (m)


# ---------------------------------------------------------------- AirSim helpers
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


def quat_yaw(q):
    x, y, z, w = q.x_val, q.y_val, q.z_val, q.w_val
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_rpy(q):
    """Return (roll, pitch, yaw) in degrees from an AirSim quaternion."""
    x, y, z, w = q.x_val, q.y_val, q.z_val, q.w_val
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(sp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


class RandomWalker:
    """Smooth, bounded random-walk velocity for the Target — 'very random' horizontal motion plus
    continual altitude changes, but acceleration-limited so it stays physically plausible (and so the
    chaser has a fair chance to follow). The safe-box leash in the loop keeps it from escaping."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.v = np.zeros(3)
        self.target_v = np.zeros(3)
        self.t_next = 0.0

    def step(self, t, dt):
        if t >= self.t_next:
            ang = self.rng.uniform(0, 2 * math.pi)
            spd = self.rng.uniform(0.6, 2.6)                 # horizontal speed
            vz = self.rng.uniform(-1.6, 1.6)                 # climb/descend
            self.target_v = np.array([spd * math.cos(ang), spd * math.sin(ang), vz])
            self.t_next = t + self.rng.uniform(1.2, 3.0)
        # accel-limited approach to the new random heading -> smooth, not teleporting
        dv = np.clip(self.target_v - self.v, -2.0 * dt, 2.0 * dt)
        self.v = self.v + dv
        return float(self.v[0]), float(self.v[1]), float(self.v[2])


def world_pos(ac, name, home):
    p = ac.simGetVehiclePose(name).position
    return home + np.array([p.x_val, p.y_val, p.z_val])


def depth_at(depth, cx, cy, W, H):
    if depth is None:
        return None
    k = 4
    x1, y1 = max(0, int(cx) - k), max(0, int(cy) - k)
    patch = depth[y1:int(cy) + k, x1:int(cx) + k]
    v = patch[(patch > 0.3) & (patch < 1e4)]
    return float(np.median(v)) if v.size else None


# ---------------------------------------------------------------- target patterns
def pattern_velocity(name, t):
    """World-NED velocity (vn, ve, vd) for the Target at local pattern time t (s).

    Zero-mean oscillations so the target stays in its area while the chaser must continuously
    match its motion. Each pattern stresses a different failure mode.
    """
    if name == "orbit":            # lateral crossing — the classic 'spin' trap
        w = 0.40; R = 5.0
        return (-R * w * math.sin(w * t), R * w * math.cos(w * t), 0.0)
    if name == "figure8_alt":      # crossing + altitude coupling
        w = 0.35
        return (1.8 * math.cos(w * t), 1.8 * math.sin(2 * w * t), -0.9 * math.sin(0.5 * w * t))
    if name == "climb_descend":    # mostly vertical — does the ego climb/descend dynamically?
        return (0.5 * math.sin(0.22 * t), 0.0, 1.1 * math.sin(0.25 * t))
    if name == "zigzag":           # lateral weave — speed-matching stress (realistic, not a step)
        return (0.0, 1.7 * math.sin(0.4 * t), -0.6 * math.sin(0.3 * t))
    if name == "recede_approach":  # range in/out — bidirectional gap hold
        return (1.6 * math.sin(0.20 * t), 0.0, 0.0)
    return (0.0, 0.0, 0.0)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=12.0)
    ap.add_argument("--speed", type=float, default=8.0, help="Ego max speed (m/s)")
    ap.add_argument("--pattern-secs", type=float, default=16.0)
    ap.add_argument("--patterns", default="orbit,figure8_alt,climb_descend,zigzag,recede_approach")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--gimbal", action="store_true",
                    help="software-stabilized gimbal: point the camera at the target each frame so it "
                         "stays centered regardless of airframe tilt (FPV camera decoupled)")
    ap.add_argument("--gimbal-frame", choices=["world", "relative"], default="world",
                    help="how simSetCameraPose interprets the orientation (world vs vehicle-relative)")
    ap.add_argument("--body-servo", action="store_true",
                    help="in vision-only phases, use pure body-frame visual servo instead of "
                         "world-frame standoff reconstruction")
    ap.add_argument("--out", default=str(REPO / "runs" / "videos"))
    ap.add_argument("--imgsz", type=int, default=960, help="YOLO inference size for eval")
    ap.add_argument("--conf", type=float, default=0.30, help="YOLO confidence threshold for eval")
    ap.add_argument("--op-alt", type=float, default=25.0,
                    help="operating altitude (m) for the tracking test — Blocks has structures below "
                         "~20m, so we launch at 5m then climb here to fly in clear air")
    args = ap.parse_args()
    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]

    OP_ALT = float(args.op_alt)
    # safe operating box for the Target so it can't wander into a Blocks structure / the ground
    SAFE_CENTER = np.array([START_RANGE, 0.0, -OP_ALT])
    BOX_R = 9.0                          # horizontal leash radius (m) around SAFE_CENTER
    ALT_MIN, ALT_MAX = 18.0, 34.0        # altitude band (m above ground) — stays above Blocks

    model = YOLO(MODEL)
    ac = airsim.MultirotorClient(); ac.confirmConnection()
    # reset to a clean ground state (a prior session may have left the drones airborne, which makes
    # takeoffAsync().join() hang forever waiting for a takeoff that never "completes")
    print("[setup] resetting sim to ground", flush=True)
    ac.reset(); time.sleep(2.0)
    for v in ("Ego", "Target"):
        ac.enableApiControl(True, v); ac.armDisarm(True, v)
    print("[setup] taking off both vehicles (launch alt ~5m)", flush=True)
    ft = ac.takeoffAsync(vehicle_name="Target"); fe = ac.takeoffAsync(vehicle_name="Ego")
    ft.join(); fe.join()
    # honour 'launch from 5m' then climb to the clear operating altitude before the test
    ac.moveToZAsync(-START_ALT, 2.5, vehicle_name="Ego").join()
    print(f"[setup] Ego launched at {START_ALT:.0f}m; climbing both to {OP_ALT:.0f}m (clear air)", flush=True)

    # place Ego at (0,0,-OP_ALT) facing +N; Target START_RANGE ahead (+N) facing the Ego
    ego_w = np.array([0.0, 0.0, -OP_ALT])
    tgt_w = SAFE_CENTER.copy()
    el = ego_w - EGO_HOME; tl = tgt_w - TARGET_HOME
    ac.moveToPositionAsync(float(el[0]), float(el[1]), float(el[2]), 4,
                           yaw_mode=airsim.YawMode(False, 0.0), vehicle_name="Ego")
    ac.moveToPositionAsync(float(tl[0]), float(tl[1]), float(tl[2]), 4,
                           yaw_mode=airsim.YawMode(False, 180.0), vehicle_name="Target").join()
    time.sleep(2.0)
    # collision-info baseline (to detect NEW collisions during the run)
    coll_count = {"Ego": 0, "Target": 0}
    coll_stamp = {v: ac.simGetCollisionInfo(vehicle_name=v).time_stamp for v in ("Ego", "Target")}
    print(f"[setup] Ego @{OP_ALT:.0f}m, Target @{START_RANGE}m facing Ego (safe box r={BOX_R}m, "
          f"alt {ALT_MIN:.0f}-{ALT_MAX:.0f}m)", flush=True)

    tkf = Vec3KF(q=6.0, r=0.6)
    writer = None
    if args.record:
        Path(args.out).mkdir(parents=True, exist_ok=True)
    log = []          # (phase, pattern, t, center_err, range, ego_alt, tgt_alt, alt_err, lost)
    traj = []         # (t, pattern, ego_n,e,d, roll, pitch, yaw, tgt_n,e,d) — 3-D path + attitude
    last = time.time()
    locked_box = None
    # fixed camera mount offset (vehicle frame); gimbal only changes ORIENTATION each frame
    cam_pos = airsim.Vector3r(0.50, 0.0, 0.10)
    if args.gimbal:
        print("[gimbal] software-stabilized camera pointing ENABLED", flush=True)

    def detect(scene):
        res = model.track(scene, tracker=TRACKER, persist=True, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        out = []
        b = res.boxes
        if b is not None and len(b):
            for i in range(len(b)):
                x1, y1, x2, y2 = b.xyxy[i].tolist()
                out.append({"box": (x1, y1, x2, y2), "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
                            "conf": float(b.conf[i])})
        return out

    def run_phase(phase, pattern, secs, vision_only):
        nonlocal last, writer, locked_box
        t0 = time.time(); pat_t0 = time.time(); lost_n = 0
        walker = RandomWalker(seed=1)
        cam_yaw = cam_pitch = None               # commanded gimbal world orientation (rad)
        prev_cmd = None                          # for ego acceleration limiting (steady camera)
        A_MAX = 2.6                              # max ego accel (m/s^2) -> caps tilt ~atan(2.6/9.8)=15deg
        prev_yaw = None                          # for yaw setpoint slew limiting (smooth, low-rate yaw)
        YAW_SLEW = 80.0                          # max yaw setpoint change (deg/s) — smooth yet keeps up
        imk = ImageKalman(q=2.0, r=0.05)         # image-space bearing predictor (coast through dropouts)
        last_range = float(args.gap)             # hold last good range during a dropout
        COAST_MAX = 6                            # frames to keep panning on prediction before declaring lost
        while time.time() - t0 < secs:
            now = time.time(); dt = min(0.3, max(0.02, now - last)); last = now
            scene, depth = grab(ac)
            if scene is None:
                time.sleep(0.02); continue
            H, W = scene.shape[:2]; cxI, cyI = W / 2.0, H / 2.0
            VFOV = vfov_from_hfov(HFOV, W, H)
            ego_q = ac.simGetVehiclePose("Ego").orientation
            ego = world_pos(ac, "Ego", EGO_HOME); ego_yaw = quat_yaw(ego_q)
            tgt = world_pos(ac, "Target", TARGET_HOME)

            # drive the target along the pattern, but KEEP IT INSIDE A SAFE BOX so it can never fly
            # into a Blocks structure or the ground: blend in a restoring velocity past the leash
            # radius / altitude band, and a hard shove-back if it actually collides.
            if pattern is not None:
                if pattern == "random":
                    vN, vE, vD = walker.step(now - pat_t0, dt)
                else:
                    vN, vE, vD = pattern_velocity(pattern, now - pat_t0)
                off = tgt - SAFE_CENTER
                hoff = off[:2]; hr = float(np.hypot(hoff[0], hoff[1]))
                rN = rE = rD = 0.0
                if hr > BOX_R:                                   # horizontal leash
                    rN = -1.6 * (hr - BOX_R) * hoff[0] / hr
                    rE = -1.6 * (hr - BOX_R) * hoff[1] / hr
                alt = -tgt[2]
                if alt < ALT_MIN:                                # too low -> climb (up = -vz)
                    rD = -1.6 * (ALT_MIN - alt)
                elif alt > ALT_MAX:                              # too high -> descend
                    rD = 1.6 * (alt - ALT_MAX)
                ci = ac.simGetCollisionInfo(vehicle_name="Target")
                if ci.has_collided and ci.time_stamp != coll_stamp["Target"]:
                    coll_stamp["Target"] = ci.time_stamp; coll_count["Target"] += 1
                    print(f"[guard] Target collision #{coll_count['Target']} with "
                          f"{ci.object_name!r} -> shoving back to safe center", flush=True)
                if ci.has_collided and hr > 1e-3:               # strong push toward center
                    rN += -4.0 * hoff[0] / hr; rE += -4.0 * hoff[1] / hr
                ac.moveByVelocityAsync(float(vN + rN), float(vE + rE), float(vD + rD), 0.3,
                                       vehicle_name="Target")

            dets = detect(scene)
            # pick the locked detection: nearest to the PREDICTED image position (image-Kalman), else
            # last box, else centre. With a single target there are no distractors, so always accept the
            # nearest detection (a hard gate would reject the real target when the prediction drifts).
            cur = None
            if dets:
                if imk.alive:
                    pe = imk.lead(0.0)               # current predicted (ex, ey)
                    px, py = cxI + pe[0] * cxI, cyI + pe[1] * cyI
                elif locked_box is not None:
                    px, py = locked_box
                else:
                    px, py = cxI, cyI
                cur = min(dets, key=lambda d: (d["cx"] - px) ** 2 + (d["cy"] - py) ** 2)
                locked_box = (cur["cx"], cur["cy"])

            # ---- target-position measurement ----
            meas = None; ex = ey = dval = None
            if cur is not None:
                ex, ey = (cur["cx"] - cxI) / cxI, (cur["cy"] - cyI) / cyI
                dval = depth_at(depth, cur["cx"], cur["cy"], W, H)
                if vision_only:
                    if dval and dval > 0.3:
                        # with the gimbal the camera world orientation != airframe yaw -> reconstruct
                        # using the orientation we commanded the gimbal to last frame.
                        if args.gimbal and cam_yaw is not None:
                            meas = target_from_vision_cam(ego, cam_yaw, cam_pitch, ex, ey, dval, HFOV, VFOV)
                        else:
                            meas = target_from_vision(ego, ego_yaw, ex, ey, dval, HFOV, VFOV)
                else:
                    meas = tgt                     # approach phase uses GPS truth
            if meas is not None:
                tp, tv = tkf.update(np.asarray(meas, float), dt); lost_n = 0
            else:
                lost_n += 1
                tp, tv = tkf.predict_only(dt)
                if tp is None:
                    tp, tv = tkf.update(tgt, dt)
                tv = tv * max(0.0, 1.0 - 0.2 * lost_n)            # decay FF: don't chase a ghost

            # ---- GIMBAL: point the camera straight at the (lead) target, decoupled from airframe ----
            if args.gimbal and tp is not None:
                cam_yaw, cam_pitch = look_at_angles(ego, np.asarray(tp, float) + np.asarray(tv, float) * 0.3)
                if args.gimbal_frame == "world":     # simSetCameraPose orientation is WORLD
                    qx, qy, qz, qw = R_to_quat(euler_R(0.0, cam_pitch, cam_yaw))
                else:                                 # orientation is RELATIVE to the vehicle body
                    qx, qy, qz, qw = camera_rel_quat((ego_q.x_val, ego_q.y_val, ego_q.z_val, ego_q.w_val),
                                                     cam_yaw, cam_pitch)
                try:
                    ac.simSetCameraPose("front_center",
                                        airsim.Pose(cam_pos, airsim.Quaternionr(qx, qy, qz, qw)),
                                        vehicle_name="Ego")
                except Exception:
                    pass
            # slow down and stop coasting hard when the target is not currently seen
            eff_speed = args.speed * (1.0 if meas is not None else max(0.25, 1.0 - 0.15 * lost_n))
            if args.body_servo and vision_only:
                # ===== ROBUST BODY-FRAME VISUAL SERVO with COAST-THROUGH-DROPOUT =====
                # Detected -> update the image-Kalman and servo on its smoothed+lead estimate. Detector
                # MISS -> keep panning on the prediction (constant-velocity) for up to COAST_MAX frames
                # so a fast lateral target stays in frame and re-acquires, instead of leaving the view.
                if cur is not None:
                    imk.update(ex, ey, dt)
                    ex_s, ey_s = float(ex), float(ey)                      # RAW measurement -> tight centering
                    if dval and dval > 0.3:
                        last_range = 0.5 * float(dval) + 0.5 * last_range   # smooth + hold the range
                    coasting = False
                else:
                    pe = imk.coast(dt)                                      # no detection -> predict
                    coasting = pe is not None
                    if coasting:
                        ex_s = float(pe[0] + imk.x[2] * 0.10)               # extrapolate slightly ahead
                        ey_s = float(pe[1] + imk.x[3] * 0.10)               # to keep panning with the target
                tracking = imk.alive and imk.miss <= COAST_MAX
                if tracking:
                    rng = last_range
                    # forward holds the gap; eased off while coasting so a stale range can't surge us
                    fwd = float(np.clip(0.8 * (rng - args.gap), -eff_speed, eff_speed)) * (0.4 if coasting else 1.0)
                    vz_b = float(np.clip(2.2 * ey_s, -2.8, 2.8))
                    bearing = math.degrees(math.atan(ex_s * math.tan(math.radians(HFOV / 2))))
                    yaw_err = bearing
                    yr = float(np.clip(2.2 * bearing, -60, 60))
                    ac.moveByVelocityBodyFrameAsync(fwd, 0.0, vz_b, 0.4,
                                                    yaw_mode=airsim.YawMode(True, yr), vehicle_name="Ego")
                else:
                    # truly lost -> stop translating, slow scan toward the last-seen side to re-find it
                    rng = last_range; yaw_err = 0.0
                    scan = 25.0 * (1.0 if (imk.x is not None and imk.x[0] >= 0) else -1.0)
                    ac.moveByVelocityBodyFrameAsync(0.0, 0.0, 0.0, 0.4,
                                                    yaw_mode=airsim.YawMode(True, scan), vehicle_name="Ego")
                    if imk.miss > COAST_MAX * 3:
                        imk.reset()
                prev_cmd = None; prev_yaw = None
            else:
                vn, ve, vd, yaw_deg, rng, yaw_err = standoff_command(
                    ego, ego_yaw, tp, tv, args.gap, eff_speed)
                # EGO STEADINESS: acceleration-limit the velocity command so the multirotor doesn't
                # pitch/roll hard (a body-fixed camera swings with tilt). Caps tilt to ~atan(A_MAX/g).
                cmd = np.array([vn, ve, vd])
                if prev_cmd is None:
                    prev_cmd = cmd
                cmd = prev_cmd + np.clip(cmd - prev_cmd, -A_MAX * dt, A_MAX * dt)
                prev_cmd = cmd
                vn, ve, vd = float(cmd[0]), float(cmd[1]), float(cmd[2])
                # slew-limit the yaw setpoint -> smooth, low-rate yaw (no snapping on random heading jumps)
                if prev_yaw is None:
                    prev_yaw = yaw_deg
                dyaw = math.degrees(math.atan2(math.sin(math.radians(yaw_deg - prev_yaw)),
                                               math.cos(math.radians(yaw_deg - prev_yaw))))
                yaw_cmd = prev_yaw + max(-YAW_SLEW * dt, min(YAW_SLEW * dt, dyaw))
                prev_yaw = yaw_cmd
                ac.moveByVelocityAsync(vn, ve, vd, 0.4,
                                       yaw_mode=airsim.YawMode(False, float(yaw_cmd)), vehicle_name="Ego")

            # ---- metrics (GPS truth for scoring only) + ego attitude/trajectory ----
            roll, pitch, eyaw = quat_rpy(ego_q)
            cerr = float(math.hypot((cur["cx"] - cxI) / cxI, (cur["cy"] - cyI) / cyI)) if cur else None
            true_rng = float(math.hypot(tgt[0] - ego[0], tgt[1] - ego[1]))
            alt_err = float((-tgt[2]) - (-ego[2]))
            log.append((phase, pattern or "-", now - t0, cerr, true_rng,
                        -ego[2], -tgt[2], alt_err, cur is None))
            traj.append((now - t0, pattern or "-", ego[0], ego[1], ego[2], roll, pitch, eyaw,
                         tgt[0], tgt[1], tgt[2]))

            # ---- annotate + record ----
            ann = scene.copy()
            for d in dets:
                x1, y1, x2, y2 = (int(v) for v in d["box"])
                col = (0, 255, 0) if d is cur else (255, 200, 0)
                cv2.rectangle(ann, (x1, y1), (x2, y2), col, 2 if d is cur else 1)
            cv2.drawMarker(ann, (int(cxI), int(cyI)), (255, 255, 255), cv2.MARKER_CROSS, 28, 2)
            if cur is not None:
                cv2.line(ann, (int(cxI), int(cyI)), (int(cur["cx"]), int(cur["cy"])), (0, 255, 0), 1)
            tag = "VISION-ONLY" if vision_only else "APPROACH(gps)"
            cv2.putText(ann, f"{tag} [{pattern or '-'}]", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if vision_only else (0, 200, 255), 2)
            cv2.putText(ann, f"range={rng:4.1f}m gap={args.gap:.0f}m  center_err={cerr if cerr is None else round(cerr,3)}"
                             f"  yaw_err={yaw_err:+5.1f}  alt_err={alt_err:+4.1f}m",
                        (10, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if args.record:
                if writer is None:
                    fn = str(Path(args.out) / "vision_only_track.avi")
                    writer = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (W, H))
                    print(f"[record] writing {fn}")
                if vision_only:
                    writer.write(ann)
        return

    try:
        # PHASE 1 — GPS-aided approach from 30 m to the gap (no recording, target hovers)
        print("[phase] APPROACH 30m -> gap", flush=True)
        run_phase("approach", None, 14.0, vision_only=False)
        # PHASE 2 — pure vision-only tracking through the pattern sequence (recorded)
        for p in patterns:
            print(f"[phase] VISION-ONLY pattern={p}", flush=True)
            run_phase("vision", p, args.pattern_secs, vision_only=True)
    finally:
        if writer is not None:
            writer.release(); print("[record] released video cleanly", flush=True)
        for v in ("Ego", "Target"):
            try:
                ac.hoverAsync(vehicle_name=v)
            except Exception:
                pass

    # ---------------------------------------------------------------- summary
    arr = log
    def stats(rows):
        ce = np.array([r[3] for r in rows if r[3] is not None])
        ae = np.array([r[7] for r in rows])
        rg = np.array([r[4] for r in rows])
        lost = sum(1 for r in rows if r[8])
        return (len(rows), lost,
                (float(ce.mean()), float(np.percentile(ce, 95)), float(ce.max())) if ce.size else (None, None, None),
                (float(np.sqrt(np.mean(ae ** 2)))),
                (float(rg.mean()), float(rg.min()), float(rg.max())))

    print("\n================ SCENARIO RESULTS ================")
    print(f"{'pattern':<16}{'frames':>7}{'lost':>6}{'cerr_mean':>11}{'cerr_p95':>10}{'cerr_max':>10}"
          f"{'altRMS':>9}{'range[min/mean/max]':>22}")
    vis_rows = [r for r in arr if r[0] == "vision"]
    for p in patterns:
        rows = [r for r in vis_rows if r[1] == p]
        if not rows:
            continue
        n, lost, ce, altrms, rg = stats(rows)
        cem = "  n/a " if ce[0] is None else f"{ce[0]:.3f}"
        cep = "  n/a " if ce[1] is None else f"{ce[1]:.3f}"
        cex = "  n/a " if ce[2] is None else f"{ce[2]:.3f}"
        print(f"{p:<16}{n:>7}{lost:>6}{cem:>11}{cep:>10}{cex:>10}{altrms:>9.2f}"
              f"   {rg[1]:.1f}/{rg[0]:.1f}/{rg[2]:.1f}")
    if vis_rows:
        n, lost, ce, altrms, rg = stats(vis_rows)
        print("-" * 81)
        print(f"{'ALL VISION':<16}{n:>7}{lost:>6}{ce[0]:>11.3f}{ce[1]:>10.3f}{ce[2]:>10.3f}{altrms:>9.2f}"
              f"   {rg[1]:.1f}/{rg[0]:.1f}/{rg[2]:.1f}")
        inframe = 100.0 * (1 - lost / max(1, n))
        print(f"\nin-frame: {inframe:.1f}%   center_err mean={ce[0]:.3f} (0=perfect, 1=frame edge)   "
              f"alt-track RMS={altrms:.2f}m   gap={args.gap:.0f}m")
        verdict = "PASS" if (ce[0] < 0.18 and inframe > 95 and altrms < 3.0) else "NEEDS TUNING"
        print(f"collisions: Ego={coll_count['Ego']} Target={coll_count['Target']}")
        print(f"VERDICT: {verdict}")

    # ---- ego 3-D trajectory + attitude analysis (how steady is the camera platform?) ----
    if traj:
        T = np.array([[r[2], r[3], r[4], r[5], r[6], r[7]] for r in traj])  # n,e,d,roll,pitch,yaw
        roll, pitch = T[:, 3], T[:, 4]
        yaw = np.unwrap(np.radians(T[:, 5]))
        yaw_rate = np.degrees(np.diff(yaw)) / 0.21                       # approx per-frame dt
        def rms(a): return float(np.sqrt(np.mean(a ** 2)))
        print("\n---------------- EGO PLATFORM STEADINESS ----------------")
        print(f"position span  N:[{T[:,0].min():.1f},{T[:,0].max():.1f}] "
              f"E:[{T[:,1].min():.1f},{T[:,1].max():.1f}] alt:[{-T[:,2].max():.1f},{-T[:,2].min():.1f}] m")
        print(f"roll  (deg): rms={rms(roll):.2f}  max|{np.abs(roll).max():.1f}|")
        print(f"pitch (deg): rms={rms(pitch):.2f}  max|{np.abs(pitch).max():.1f}|")
        print(f"yaw-rate(deg/s): rms={rms(yaw_rate):.1f}  max|{np.abs(yaw_rate).max():.1f}|")
        try:
            cfile = Path(args.out) / "ego_trajectory.csv"
            Path(args.out).mkdir(parents=True, exist_ok=True)
            with open(cfile, "w") as fh:
                fh.write("t,pattern,ego_n,ego_e,ego_d,roll_deg,pitch_deg,yaw_deg,tgt_n,tgt_e,tgt_d\n")
                for r in traj:
                    fh.write(",".join(str(round(x, 3)) if isinstance(x, float) else str(x) for x in r) + "\n")
            print(f"[traj] wrote {cfile} ({len(traj)} samples)")
        except Exception as e:
            print(f"[traj] write failed: {e}")
        # persist a metrics file so results survive any stdout buffering
        try:
            mfile = Path(args.out) / "scenario_metrics.txt"
            Path(args.out).mkdir(parents=True, exist_ok=True)
            with open(mfile, "w") as fh:
                fh.write(f"gap={args.gap} op_alt={OP_ALT} speed={args.speed}\n")
                fh.write(f"in_frame_pct={inframe:.1f}\ncenter_err_mean={ce[0]:.3f}\n")
                fh.write(f"center_err_p95={ce[1]:.3f}\ncenter_err_max={ce[2]:.3f}\n")
                fh.write(f"alt_rms_m={altrms:.2f}\nlost_frames={lost}/{n}\n")
                fh.write(f"collisions_ego={coll_count['Ego']} collisions_target={coll_count['Target']}\n")
                for p in patterns:
                    rows = [r for r in vis_rows if r[1] == p]
                    if not rows:
                        continue
                    pn, pl, pce, palt, prg = stats(rows)
                    cm = "n/a" if pce[0] is None else f"{pce[0]:.3f}"
                    cp = "n/a" if pce[1] is None else f"{pce[1]:.3f}"
                    fh.write(f"  {p}: cerr_mean={cm} cerr_p95={cp} "
                             f"alt_rms={palt:.2f} lost={pl}/{pn} range={prg[1]:.1f}/{prg[0]:.1f}/{prg[2]:.1f}\n")
                fh.write(f"VERDICT={verdict}\n")
            print(f"[metrics] wrote {mfile}", flush=True)
        except Exception as e:
            print(f"[metrics] write failed: {e}", flush=True)
    print("==================================================", flush=True)


if __name__ == "__main__":
    main()
