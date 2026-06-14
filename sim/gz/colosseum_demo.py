#!/usr/bin/env python3
"""3D-view engagement animation for the colosseum world: target orbits above the arena; the chaser
tracks it at a standoff gap, then periodically commits a climb-and-strike and resets. Pose-driven
(ground truth) purely for the live 3D Gazebo view — the vision-guided version (YOLO up-cam) runs in the
webui. Runnable by any user (gz-transport + numpy only)."""
import time, math
import numpy as np
from gz.transport13 import Node
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.boolean_pb2 import Boolean

WORLD = "uav_colosseum"
node = Node()
def sp(n, p, yaw=0.0):
    r = Pose(); r.name = n
    r.position.x, r.position.y, r.position.z = float(p[0]), float(p[1]), float(p[2])
    r.orientation.z = math.sin(yaw/2); r.orientation.w = math.cos(yaw/2)
    node.request(f"/world/{WORLD}/set_pose", r, Pose, Boolean, 300)

ego = np.array([0.0, 0.0, 6.0]); t0 = time.time(); last = time.time()
sp("chaser", ego); sp("target", [7, 0, 17])
while True:
    now = time.time(); dt = min(0.1, now - last); last = now
    t = now - t0
    ang = 0.30 * t
    tgt = np.array([7*math.cos(ang), 7*math.sin(ang), 17 + 2*math.sin(0.15*t)])
    sp("target", tgt, yaw=ang + math.pi/2)
    cyc = t % 16.0
    if cyc < 10.0:            # TRACK at 6 m standoff below
        setp = np.array([tgt[0], tgt[1], tgt[2] - 6.0]); spd = 6.0
    else:                     # STRIKE: climb onto the target
        setp = tgt.copy(); spd = 9.0
    step = setp - ego; n = float(np.linalg.norm(step))
    if n > 1e-3:
        ego = ego + step / n * min(n, spd * dt)
    ego[2] = max(0.5, ego[2])
    sp("chaser", ego)
    time.sleep(0.02)
