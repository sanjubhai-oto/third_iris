#!/usr/bin/env python3
"""Drive the GROUND ROVER (UGV) kinematically along a slow ground path via gz set_pose. The chaser
detects it from its downward camera (color) and follows/lands/strikes (UAVros concept). Smooth motion
by interpolating between waypoints; stays on the ground (z=0.3).

  python3 sim/gz/a2a/rover_mover.py [--speed 1.5]
"""
import argparse, math, os, time
os.environ.setdefault("GZ_IP", "127.0.0.1")
from gz.transport13 import Node
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.boolean_pb2 import Boolean

# ground waypoints (world x,y), a slow loop the chaser can shadow from above
WPS = [(10, -6), (24, -6), (28, 4), (16, 10), (4, 6), (6, -6), (10, -6)]
Z = 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="uav_a2a")
    ap.add_argument("--model", default="rover")
    ap.add_argument("--speed", type=float, default=1.5, help="m/s ground speed")
    args = ap.parse_args()

    node = Node()
    def setp(x, y, yaw):
        req = Pose(); req.name = args.model
        req.position.x = float(x); req.position.y = float(y); req.position.z = Z
        req.orientation.z = math.sin(yaw / 2.0); req.orientation.w = math.cos(yaw / 2.0)
        node.request(f"/world/{args.world}/set_pose", req, Pose, Boolean, 300)

    print(f"[rover] driving {args.model} at {args.speed} m/s", flush=True)
    dt = 0.05
    while True:
        for i in range(len(WPS) - 1):
            (x0, y0), (x1, y1) = WPS[i], WPS[i + 1]
            seg = math.hypot(x1 - x0, y1 - y0)
            yaw = math.atan2(y1 - y0, x1 - x0)
            steps = max(1, int(seg / (args.speed * dt)))
            for s in range(steps):
                a = s / steps
                setp(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a, yaw)
                time.sleep(dt)


if __name__ == "__main__":
    main()
