#!/usr/bin/env python3
"""Smoke-test the gz set_pose service + image subscribe from Python (gz.transport13)."""
import time, sys
import numpy as np
from gz.transport13 import Node
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.image_pb2 import Image

WORLD = "uav_track_up"
node = Node()

# 1) move the target to a known spot via set_pose
req = Pose()
req.name = "target"
req.position.x = 2.0; req.position.y = 0.0; req.position.z = 18.0
req.orientation.w = 1.0
res, ok = node.request(f"/world/{WORLD}/set_pose", req, Pose, Boolean, 1000)
print("set_pose target -> result:", res, "data:", ok.data if res else None)

# 2) confirm a camera frame still flows
got = {}
def cb(msg):
    got.setdefault("img", np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3))
node.subscribe(Image, "/jetray/camera_up", cb)
t0 = time.time()
while "img" not in got and time.time() - t0 < 10:
    time.sleep(0.1)
print("frame:", None if "img" not in got else got["img"].shape, "mean",
      None if "img" not in got else round(float(got["img"].mean()), 1))
