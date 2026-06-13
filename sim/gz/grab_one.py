#!/usr/bin/env python3
"""Grab ONE frame from a gz camera topic -> save PNG + print stats. De-risks WSL camera rendering.
Usage (in WSL): python3 grab_one.py [topic_substr]   (default substr: camera_up)
"""
import sys, time
import numpy as np

substr = sys.argv[1] if len(sys.argv) > 1 else "camera_up"

from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

node = Node()
# discover the camera topic
topics = node.topic_list()
cam = next((t for t in topics if substr in t), None)
print("topics:", [t for t in topics if "camera" in t or "image" in t])
if cam is None:
    print(f"NO topic matching {substr!r}; all topics:", topics)
    sys.exit(1)
print("subscribing:", cam)

got = {}
def cb(msg):
    if "img" in got:
        return
    h, w = msg.height, msg.width
    buf = np.frombuffer(msg.data, np.uint8)
    try:
        img = buf.reshape(h, w, 3)
    except Exception:
        img = buf.reshape(h, w, -1)
    got["img"] = img
    got["fmt"] = msg.pixel_format_type

node.subscribe(Image, cam, cb)

t0 = time.time()
while "img" not in got and time.time() - t0 < 20:
    time.sleep(0.1)

if "img" not in got:
    print("TIMEOUT: no image received in 20s")
    sys.exit(2)

img = got["img"]
print(f"GOT frame {img.shape} fmt={got['fmt']} mean={img.mean():.1f} "
      f"min={img.min()} max={img.max()} std={img.std():.1f}")
try:
    import cv2
    cv2.imwrite("/mnt/c/Users/admin/uav-vio-track/runs/videos/gz_up_grab.png", img[:, :, ::-1])
    print("saved runs/videos/gz_up_grab.png")
except Exception as e:
    np.save("/mnt/c/Users/admin/uav-vio-track/runs/videos/gz_up_grab.npy", img)
    print("cv2 unavailable, saved .npy:", e)
