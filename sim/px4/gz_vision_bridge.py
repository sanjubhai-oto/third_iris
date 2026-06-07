#!/usr/bin/env python3
"""Gazebo ground-truth -> PX4 VISION_POSITION_ESTIMATE (a VIO stand-in for SITL).

Runs INSIDE WSL2. Subscribes to the Gazebo model pose (ENU), converts to NED, and streams
VISION_POSITION_ESTIMATE to PX4 at ~30 Hz so EKF2 can navigate GPS-denied. This emulates a
perfect VIO so the offboard + external-vision pipeline can be validated before swapping in a
real estimator (OpenVINS) on the AirSim/sim camera.

  python3 sim/px4/gz_vision_bridge.py --model x500_0 --rate 30

Frames: Gazebo world is ENU; PX4 vision expects NED. n=E_y, e=E_x, d=-E_z; yaw_ned=pi/2-yaw_enu.
"""
from __future__ import annotations

import argparse
import math
import time

from pymavlink import mavutil
from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V


def quat_to_euler(x, y, z, w):
    """Return (roll, pitch, yaw) from a quaternion."""
    # roll
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    # pitch
    sinp = 2 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    # yaw
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


class Bridge:
    def __init__(self, model, mav):
        self.model = model
        self.mav = mav
        self.latest = None  # (n,e,d,roll,pitch,yaw)

    def on_pose(self, msg: Pose_V):
        # PX4 SITL is lockstep: timestamps MUST be in sim time, which is the gz header stamp.
        usec = msg.header.stamp.sec * 1_000_000 + msg.header.stamp.nsec // 1000
        for p in msg.pose:
            if p.name == self.model:
                ex, ey, ez = p.position.x, p.position.y, p.position.z
                r, pit, yaw_e = quat_to_euler(p.orientation.x, p.orientation.y,
                                              p.orientation.z, p.orientation.w)
                # ENU -> NED
                n, e, d = ey, ex, -ez
                roll_n, pitch_n = r, -pit
                yaw_n = math.atan2(math.sin(math.pi / 2 - yaw_e), math.cos(math.pi / 2 - yaw_e))
                self.latest = (usec, n, e, d, roll_n, pitch_n, yaw_n)
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="x500_0")
    ap.add_argument("--topic", default="/world/default/pose/info")
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--mav", default="udpout:127.0.0.1:14580",
                    help="PX4 onboard MAVLink receive endpoint")
    args = ap.parse_args()

    mav = mavutil.mavlink_connection(args.mav, source_system=1, source_component=197)
    bridge = Bridge(args.model, mav)

    node = Node()
    if not node.subscribe(Pose_V, args.topic, bridge.on_pose):
        raise SystemExit(f"failed to subscribe to {args.topic}")
    print(f"[ok] subscribed to {args.topic}, streaming VISION_POSITION_ESTIMATE -> {args.mav}")

    t0 = time.time()
    period = 1.0 / args.rate
    n = 0
    while True:
        if bridge.latest is not None:
            usec, x, y, z, roll, pitch, yaw = bridge.latest
            mav.mav.vision_position_estimate_send(usec, x, y, z, roll, pitch, yaw)
            n += 1
            if n % 60 == 0:
                print(f"[tx] {n} msgs  pos N={x:.2f} E={y:.2f} D={z:.2f} yaw={math.degrees(yaw):.0f}")
        time.sleep(period)


if __name__ == "__main__":
    main()
