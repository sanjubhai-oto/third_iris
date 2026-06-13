#!/usr/bin/env python3
"""Capture FRESH AirSim frames for the brochure: FPV lock shots at varied geometry + depth view."""
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "docs"))
from gen_brochure_assets import hud_box, frame_chrome, OUT

import cosysairsim as airsim
from ultralytics import YOLO

model = YOLO(str(REPO / "runs/train/airsim_drone/weights/best.pt"))
c = airsim.MultirotorClient()
c.confirmConnection()
for v in ("Ego", "Target"):
    c.enableApiControl(True, v)
    c.armDisarm(True, v)
c.takeoffAsync(vehicle_name="Ego")
c.takeoffAsync(vehicle_name="Target").join()
time.sleep(1)

EGO_ALT = -28.0                       # open sky, above every Blocks obstacle


def set_pose(vehicle, x, y, z, yaw):
    half = yaw / 2.0
    c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x, y, z),
                                    airsim.Quaternionr(0.0, 0.0, math.sin(half), math.cos(half))),
                        True, vehicle)

shots = [
    ("airsim_track_0", 10.0, 0.0, -1.5),     # close, slightly above ego
    ("airsim_track_1", 16.0, 18.0, 1.0),     # mid, off-axis right, below
    ("airsim_track_2", 24.0, -12.0, -3.0),   # far, left, higher
    ("airsim_hero", 7.0, 6.0, -2.0),         # hero close-up
]


def grab(vehicle="Ego"):
    reqs = [airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True, False)]
    raw = c.client.call("simGetImages", reqs, vehicle, False)
    rs = [airsim.ImageResponse.from_msgpack(r) for r in raw]
    rgb = np.frombuffer(rs[0].image_data_uint8, np.uint8).reshape(rs[0].height, rs[0].width, 3)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    dep = airsim.list_to_2d_float_array(rs[1].image_data_float, rs[1].width, rs[1].height)
    return rgb, dep


for name, ahead, side, dz in shots:
    yaw = math.atan2(side, ahead)
    set_pose("Ego", 0, 0, EGO_ALT, yaw)                  # hover in open sky, camera on bearing
    set_pose("Target", ahead, side, EGO_ALT + dz, yaw + math.pi)
    c.hoverAsync(vehicle_name="Ego")
    time.sleep(0.8)
    rgb, dep = grab()
    r = model.predict(rgb, imgsz=960, conf=0.25, verbose=False)[0]
    if len(r.boxes):
        b = max(r.boxes, key=lambda bb: float(bb.conf))
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        rng = float(np.nanmedian(dep[max(0, y1):y2, max(0, x1):x2])) if y2 > y1 and x2 > x1 else ahead
        if not np.isfinite(rng) or rng <= 0.5:
            rng = math.hypot(ahead, side)
        hud_box(rgb, x1, y1, x2, y2, "drone", float(b.conf), rng)
        det = "LOCK"
    else:
        det = "NO-DET"
    frame_chrome(rgb, "THIRD IRIS // AIRSIM LIVE", "SIM FEED // VISION ONLY")
    cv2.imwrite(str(OUT / f"{name}.jpg"), rgb, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[cap] {name} {det} range={ahead:.0f}/{side:.0f}")

# depth view for avoidance page: aim at the block stacks from low altitude for layered ranges
set_pose("Ego", -38.0, 4.0, -9.0, 0.0)
time.sleep(0.8)
rgb, dep = grab()
d = np.clip(dep, 0, 40.0) / 40.0
dm = (255 * (1.0 - d)).astype(np.uint8)
col = cv2.applyColorMap(dm, cv2.COLORMAP_OCEAN)
col = cv2.convertScaleAbs(col, alpha=1.15, beta=4)
h, w = col.shape[:2]
for frac, lab in ((0.25, "10M"), (0.5, "20M"), (0.75, "30M")):
    cv2.putText(col, lab, (int(w * frac), h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 220, 255), 1, cv2.LINE_AA)
frame_chrome(col, "THIRD IRIS // DEPTH FIELD", "OBSTACLE SENSE // 90X60 FOV")
cv2.imwrite(str(OUT / "airsim_depth.jpg"), col, [cv2.IMWRITE_JPEG_QUALITY, 92])
print("[cap] airsim_depth")

c.armDisarm(False, "Ego"); c.armDisarm(False, "Target")
print("[done]")
