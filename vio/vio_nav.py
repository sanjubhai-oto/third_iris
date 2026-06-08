#!/usr/bin/env python3
"""GPS-denied waypoint NAVIGATION on VIO alone — build + test.

Proves the chaser can fly to coordinates with NO target in view and NO GPS, using only the camera+IMU
(+magnetometer) VIO pose. Phase PRIME (GPS on): anchor VIO to truth and calibrate the compass. Phase
NAV (GPS jammed): fly a sequence of world-NED waypoints, steering purely on the VIO position estimate;
at each waypoint we log the TRUE position error vs ground truth — that is the real GPS-denied
navigation accuracy.

Run (AirSim Blocks running):  python -u vio/vio_nav.py [--no-mag]
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
OP_ALT = 18.0


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


def mag_of(ac):
    m = ac.getMagnetometerData(vehicle_name="Ego").magnetic_field_body
    return np.array([m.x_val, m.y_val, m.z_val])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-mag", action="store_true", help="disable magnetometer yaw-aid (to compare)")
    args = ap.parse_args()
    use_mag = not args.no_mag

    ac = airsim.MultirotorClient(); ac.confirmConnection()
    print("[setup] reset + takeoff + climb", flush=True)
    ac.reset(); time.sleep(2.0)
    ac.enableApiControl(True, "Ego"); ac.armDisarm(True, "Ego")
    ac.takeoffAsync(vehicle_name="Ego").join()
    ac.moveToZAsync(-OP_ALT, 2.5, vehicle_name="Ego").join(); time.sleep(1.0)

    scene, depth = grab(ac); H, W = scene.shape[:2]
    est = VIOEstimator(hfov_deg=90.0, width=W, height=H)
    p0, q0 = truth(ac)
    last = time.time()

    # PRIME: GPS available -> anchor + calibrate compass
    print(f"[prime] 7s GPS-on, mag_aid={use_mag}", flush=True)
    t0 = time.time()
    while time.time() - t0 < 7.0:
        now = time.time(); dt = now - last; last = now
        scene, depth = grab(ac)
        if scene is None:
            time.sleep(0.02); continue
        gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        est.update(gray, depth, imu_of(ac), dt, mag=mag_of(ac) if use_mag else None)
        tp, tq = truth(ac); est.anchor(tp, tq)
        if use_mag:
            est.calibrate_mag(mag_of(ac))
        ac.moveByVelocityAsync(0, 0, 0, 0.2, vehicle_name="Ego")

    start = est.p.copy()
    # square + an altitude change, world-NED, relative to where we are now (GPS LAST-KNOWN)
    wps = [start + np.array([14, 0, 0]),
           start + np.array([14, 14, 0]),
           start + np.array([0, 14, -4]),
           start + np.array([0, 0, 0])]
    names = ["A (N14)", "B (NE)", "C (E14,+4m up)", "D (home)"]
    print("[NAV] GPS JAMMED — navigating on VIO only", flush=True)

    results = []
    KP = 0.5; VMAX = 4.0; ARRIVE = 1.8; A_MAX = 2.0   # gentler gain + accel limit -> smooth, no pitch shake
    vcmd = np.zeros(3)
    for wp, nm in zip(wps, names):
        wt0 = time.time()
        while True:
            now = time.time(); dt = now - last; last = now
            scene, depth = grab(ac)
            if scene is None:
                time.sleep(0.02); continue
            gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
            est.update(gray, depth, imu_of(ac), dt, mag=mag_of(ac) if use_mag else None)
            err_vec = wp - est.p
            v_raw = np.clip(KP * err_vec, -VMAX, VMAX)
            v_lp = 0.4 * v_raw + 0.6 * vcmd                              # low-pass
            vcmd = vcmd + np.clip(v_lp - vcmd, -A_MAX * dt, A_MAX * dt)  # accel limit -> smooth pitch
            v = vcmd
            yaw_deg = math.degrees(math.atan2(err_vec[1], err_vec[0])) if np.hypot(err_vec[0], err_vec[1]) > 1.0 else None
            ymode = airsim.YawMode(False, yaw_deg) if yaw_deg is not None else airsim.YawMode(False, 0)
            ac.moveByVelocityAsync(float(v[0]), float(v[1]), float(v[2]), 0.5,
                                   yaw_mode=ymode, vehicle_name="Ego")
            vio_dist = float(np.linalg.norm(wp - est.p))      # what the drone THINKS
            if vio_dist < ARRIVE or time.time() - wt0 > 18.0:
                tp, _ = truth(ac)
                true_err = float(np.linalg.norm(tp - wp))     # the REAL accuracy
                vio_drift = float(np.linalg.norm(est.p - tp))
                timeout = time.time() - wt0 > 18.0
                results.append((nm, true_err, vio_drift, timeout))
                print(f"  reached {nm}: VIO thinks d={vio_dist:.2f}m | TRUE error={true_err:.2f}m | "
                      f"VIO drift={vio_drift:.2f}m{' (TIMEOUT)' if timeout else ''}", flush=True)
                break
    ac.hoverAsync(vehicle_name="Ego")

    errs = np.array([r[1] for r in results])
    print("\n========= GPS-DENIED WAYPOINT NAV (VIO only) =========", flush=True)
    print(f"mag_aid={use_mag}  waypoints={len(results)}")
    print(f"true arrival error  mean={errs.mean():.2f}m  max={errs.max():.2f}m")
    print(f"return-to-home error: {results[-1][1]:.2f}m")
    verdict = "GOOD" if errs.mean() < 3.0 else ("USABLE" if errs.mean() < 7.0 else "HIGH DRIFT")
    print(f"VERDICT: {verdict}")
    print("======================================================", flush=True)


if __name__ == "__main__":
    main()
