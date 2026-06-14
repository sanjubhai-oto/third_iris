#!/usr/bin/env python3
"""Continuously drive the 3 target drones (gentle patterns) so the GUI scene is alive. Runs as any user."""
import time, math
from gz.transport13 import Node
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.boolean_pb2 import Boolean

WORLD = "uav_track_team"
node = Node()
def sp(n, x, y, z):
    r = Pose(); r.name = n; r.position.x = x; r.position.y = y; r.position.z = z; r.orientation.w = 1.0
    node.request(f"/world/{WORLD}/set_pose", r, Pose, Boolean, 300)

sp("chaser", 0, 0, 6)
t0 = time.time()
while True:
    t = time.time() - t0
    sp("target_red",   3 + 2.0*math.cos(0.4*t),  2 + 2.0*math.sin(0.4*t),  11)
    sp("target_green", -3 + 1.8*math.sin(0.5*t), 2 + 1.8*math.cos(0.3*t),  11)
    sp("target_blue",  2 + 1.5*math.sin(0.7*t),  -3 + 2.0*math.sin(0.35*t), 11)
    time.sleep(0.05)
