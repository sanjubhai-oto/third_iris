#!/usr/bin/env python3
"""Test our GPS-denied VIO on the jetray flown by PX4 SITL in Gazebo.

Flies the jetray (MAVLink offboard: takeoff + a square), feeds the DOWN RGB-D camera + IMU into our
VIOEstimator (the one developed on AirSim), anchors ONCE to ground truth at start, then free-runs
(GPS-denied) and reports drift vs the gz ground-truth pose.

Frames: gz world is ENU + body FLU; our VIO is NED + body FRD. Convert:
  pos ENU(x=E,y=N,z=U) -> NED(N,E,D) = (y, x, -z);  body FLU -> FRD = (x,-y,-z) for accel & gyro.

Run in WSL (PX4 SITL jetray already flying):
  LIBGL_ALWAYS_SOFTWARE=1 python3 sim/gz/vio_sitl.py --world <world>
"""
import argparse, math, threading, time, sys, os
import numpy as np
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vio"))
from vio_estimator import VIOEstimator
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from gz.msgs10.imu_pb2 import IMU
from gz.msgs10.pose_v_pb2 import Pose_V
from pymavlink import mavutil

def enu2ned(p):   return np.array([p[1], p[0], -p[2]])
def flu2frd(v):   return np.array([v[0], -v[1], -v[2]])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="uav_colosseum")
    ap.add_argument("--secs", type=float, default=60.0)
    ap.add_argument("--model", default="jetray_0")
    args = ap.parse_args()

    node = Node()
    S = {"gray": None, "depth": None, "imu": None, "truth": None, "tquat": None}
    def on_rgb(m):
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        S["gray"] = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    def on_depth(m):
        S["depth"] = np.frombuffer(m.data, np.float32).reshape(m.height, m.width)
    def on_imu(m):
        a = m.linear_acceleration; g = m.angular_velocity
        S["imu"] = (flu2frd([a.x, a.y, a.z]), flu2frd([g.x, g.y, g.z]))
    def on_pose(m):
        for p in m.pose:
            if p.name == args.model:
                S["truth"] = enu2ned([p.position.x, p.position.y, p.position.z])
                q = p.orientation; S["tquat"] = [q.x, q.y, q.z, q.w]
                return
    node.subscribe(Image, "/jetray/cam_vio", on_rgb)
    node.subscribe(Image, "/jetray/cam_vio_depth", on_depth)
    node.subscribe(IMU, f"/world/{args.world}/model/{args.model}/link/base_link/sensor/imu_sensor/imu", on_imu)
    node.subscribe(Pose_V, f"/world/{args.world}/pose/info", on_pose)

    # ---- flight: arm, offboard, takeoff, fly a square (so the camera sees translation) ----
    def fly():
        m = mavutil.mavlink_connection("udpin:0.0.0.0:14540")
        m.wait_heartbeat(); print("[fly] heartbeat", flush=True)
        def sp(n, e, d):
            m.mav.set_position_target_local_ned_send(
                0, m.target_system, m.target_component, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111111000, n, e, d, 0,0,0, 0,0,0, 0,0)
        for _ in range(20): sp(0,0,-5); time.sleep(0.05)
        m.mav.command_long_send(m.target_system, m.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0, 1, 6, 0,0,0,0,0)   # OFFBOARD
        m.arducopter_arm() if hasattr(m,"arducopter_arm") else None
        m.mav.command_long_send(m.target_system, m.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1,0,0,0,0,0,0)
        print("[fly] armed + offboard, climbing", flush=True)
        sq = [(0,0,-5)]*60 + [(8,0,-5)]*60 + [(8,8,-6)]*60 + [(0,8,-6)]*60 + [(0,0,-5)]*60
        for (n,e,d) in sq:
            sp(n,e,d); time.sleep(0.1)
        print("[fly] path done", flush=True)
    threading.Thread(target=fly, daemon=True).start()

    # ---- VIO: wait for sensors + motion, anchor once, free-run, measure drift ----
    while any(S[k] is None for k in ("gray","depth","imu","truth")):
        time.sleep(0.1)
    H, W = S["gray"].shape
    vio = VIOEstimator(hfov_deg=85.9, width=W, height=H, max_depth=60.0, vo_depth_max=40.0)
    # wait until it climbs a bit (so VO has parallax) then anchor
    t0 = time.time()
    while -S["truth"][2] < 2.0 and time.time()-t0 < 30: time.sleep(0.2)
    vio.anchor(S["truth"], S["tquat"]); anchor_truth = S["truth"].copy()
    print(f"[vio] anchored at {S['truth'].round(2)} (alt {-S['truth'][2]:.1f}m); free-running GPS-denied", flush=True)

    last = time.time(); errs = []; path_truth = 0.0; prev_truth = S["truth"].copy()
    t0 = time.time()
    while time.time() - t0 < args.secs:
        now = time.time(); dt = min(0.2, max(0.02, now-last)); last = now
        g, d, imu, truth = S["gray"], S["depth"], S["imu"], S["truth"]
        if g is None or d is None: time.sleep(0.02); continue
        vio.update(g, d, imu, dt)                        # GPS-denied: no re-anchor
        pe = np.asarray(vio.p, float)                    # VIO world position estimate
        err = float(np.linalg.norm(pe - truth))
        errs.append(err)
        path_truth += float(np.linalg.norm(truth - prev_truth)); prev_truth = truth.copy()
        time.sleep(0.02)

    errs = np.array(errs)
    print("\n================ VIO-on-SITL DRIFT ================")
    print(f"frames={len(errs)} path_len={path_truth:.1f}m  drift mean={errs.mean():.2f}m "
          f"final={errs[-1]:.2f}m  drift%%={100*errs[-1]/max(1,path_truth):.1f}%")

if __name__ == "__main__":
    main()
