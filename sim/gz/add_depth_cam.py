#!/usr/bin/env python3
"""Idempotently add an UP-facing DEPTH camera to the jetray gz model (true range to overhead target).
Run inside WSL: python3 add_depth_cam.py
"""
import re, sys
MODEL = "/root/PX4-Autopilot/Tools/simulation/gz/models/jetray/model.sdf"

DEPTH = """      <!-- SKY depth camera: true range to an overhead target (pitch=-pi/2 = up) -->
      <sensor name="camera_up_depth" type="depth_camera">
        <pose>0 0 0.46 0 -1.5707963 0</pose>
        <topic>jetray/camera_up_depth</topic>
        <camera>
          <horizontal_fov>2.094</horizontal_fov>
          <image>
            <width>320</width>
            <height>180</height>
          </image>
          <clip>
            <near>0.1</near>
            <far>200</far>
          </clip>
        </camera>
        <always_on>1</always_on>
        <update_rate>15</update_rate>
      </sensor>
"""

src = open(MODEL).read()
if "camera_up_depth" in src:
    print("camera_up_depth already present; nothing to do"); sys.exit(0)
m = re.search(r'(<sensor name="camera_up".*?</sensor>\n)', src, re.S)
if not m:
    print("ERROR: camera_up sensor not found (run add_up_cam.py first)"); sys.exit(1)
out = src[:m.end()] + DEPTH + src[m.end():]
open(MODEL, "w").write(out)
print("inserted camera_up_depth after camera_up")
