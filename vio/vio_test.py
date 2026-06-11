#!/usr/bin/env python3
"""Standalone VIO drift test: prime from GPS, then JAM GPS and free-run, measure drift vs truth.

Flies the Ego on a known path. Phase PRIME: anchor VIO to ground truth each frame (GPS available).
Phase JAM: stop anchoring -> VIO dead-reckons from camera+IMU only; we log estimated vs true position
and report drift growth. This validates the estimator before wiring it into the live tracker.

Run (AirSim Blocks running):  python -u vio/vio_test.py
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import cv2
import cosysairsim as airsim

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO / "vio"))
from vio_estimator import VIOEstimator      # noqa: E402

EGO_HOME = np.array([0.0, 0.0, 0.0])


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


def truth(ac):
    k = ac.simGetGroundTruthKinematics(vehicle_name="Ego")
    p = np.array([k.position.x_val, k.position.y_val, k.position.z_val]) + EGO_HOME
    o = k.orientation
    return p, np.array([o.x_val, o.y_val, o.z_val, o.w_val])


def imu_of(ac):
    d = ac.getImuData(vehicle_name="Ego")
    a = np.array([d.linear_acceleration.x_val, d.linear_acceleration.y_val, d.linear_acceleration.z_val])
    w = np.array([d.angular_velocity.x_val, d.angular_velocity.y_val, d.angular_velocity.z_val])
    return a, w


def baro_of(ac):
    return float(ac.getBarometerData(vehicle_name="Ego").altitude)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["umeyama", "open3d"], default="umeyama")
    args = ap.parse_args()
    ac = airsim.MultirotorClient(); ac.confirmConnection()
    print("[setup] reset + takeoff", flush=True)
    ac.reset(); time.sleep(2.0)
    ac.enableApiControl(True, "Ego"); ac.armDisarm(True, "Ego")
    ac.takeoffAsync(vehicle_name="Ego").join()
    ac.moveToZAsync(-14, 2.5, vehicle_name="Ego").join()
    time.sleep(1.0)

    scene, depth = grab(ac)
    H, W = scene.shape[:2]
    est = VIOEstimator(hfov_deg=90.0, width=W, height=H, backend=args.backend)
    print(f"[vio] backend = {est.backend}", flush=True)
    p0, q0 = truth(ac); est.anchor(p0, q0)
    last = time.time()
    log = []                                   # (t, phase, drift, true_dist, vo_ok)
    dist = 0.0; prev_true = p0.copy()

    # a known path: gentle box + slow climb, FIXED heading (minimise yaw to isolate translation VO).
    # Velocity targets are low-passed + accel-limited so the airframe banks gently (smooth attitude
    # also gives cleaner optical flow).
    legs = [(1.5, 0.0, -0.25), (0.0, 1.5, 0.0), (-1.5, 0.0, 0.25), (0.0, -1.5, 0.0)]  # vN,vE,vD m/s
    A_MAX = 0.8
    vcmd = np.zeros(3)

    t0 = time.time()
    PRIME_S, JAM_S = 8.0, 22.0
    print(f"[run] PRIME {PRIME_S:.0f}s (GPS on) then JAM {JAM_S:.0f}s (VIO only)", flush=True)
    while time.time() - t0 < PRIME_S + JAM_S:
        now = time.time(); dt = now - last; last = now
        el = now - t0
        leg = np.array(legs[int(el // 4) % len(legs)], float)        # 4s legs, gentler turns
        v_lp = 0.25 * leg + 0.75 * vcmd
        vcmd = vcmd + np.clip(v_lp - vcmd, -A_MAX * dt, A_MAX * dt)
        ac.moveByVelocityAsync(float(vcmd[0]), float(vcmd[1]), float(vcmd[2]), 0.3,
                               yaw_mode=airsim.YawMode(False, 0), vehicle_name="Ego")

        scene, depth = grab(ac)
        if scene is None:
            time.sleep(0.02); continue
        gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        out = est.update(gray, depth, imu_of(ac), dt, baro_alt=baro_of(ac))
        tp, tq = truth(ac)
        dist += float(np.linalg.norm(tp - prev_true)); prev_true = tp

        gps_on = el < PRIME_S
        if gps_on:
            est.anchor(tp, tq)                  # GPS available -> keep VIO primed (zero drift)
            est.calibrate_baro(baro_of(ac))     # pin baro reference while GPS is on
            phase = "PRIME"
        else:
            phase = "JAM"
        drift = float(np.linalg.norm(out["p"] - tp))
        log.append((el, phase, drift, dist, out["vo_ok"]))
        if len(log) % 6 == 0:
            print(f"  t={el:5.1f}s {phase}  drift={drift:5.2f}m  travelled={dist:5.1f}m  "
                  f"vo_ok={out['vo_ok']} coast={out['coast']}", flush=True)

    ac.hoverAsync(vehicle_name="Ego")
    jam = [r for r in log if r[1] == "JAM"]
    if jam:
        d = np.array([r[2] for r in jam])
        jam_dist = jam[-1][3] - jam[0][3]
        vo_rate = 100.0 * np.mean([1 if r[4] else 0 for r in jam])
        print("\n================ VIO DRIFT (GPS jammed) ================", flush=True)
        print(f"jam duration  : {jam[-1][0]-jam[0][0]:.1f}s   distance flown during jam: {jam_dist:.1f}m")
        print(f"drift mean    : {d.mean():.2f} m")
        print(f"drift final   : {d[-1]:.2f} m")
        print(f"drift max     : {d.max():.2f} m")
        print(f"drift as %dist: {100.0*d[-1]/max(0.1,jam_dist):.1f}% of distance flown")
        print(f"VO success    : {vo_rate:.0f}% of jam frames")
        verdict = "GOOD" if d[-1] < 3.0 else ("USABLE" if d[-1] < 8.0 else "HIGH DRIFT")
        print(f"VERDICT: {verdict}")
        print("========================================================", flush=True)


if __name__ == "__main__":
    main()
