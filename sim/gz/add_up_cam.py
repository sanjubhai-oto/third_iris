#!/usr/bin/env python3
"""Idempotently add an UP-facing camera sensor to the jetray gz model (keeps camera_front).
SDF camera looks along +X; Gazebo is Z-UP, so pitch -pi/2 points the optical axis straight up.
Run inside WSL: python3 add_up_cam.py
"""
import re, sys

MODEL = "/root/PX4-Autopilot/Tools/simulation/gz/models/jetray/model.sdf"

UP_CAM = """      <!-- SKY camera: mounted on body top, facing straight UP (+z). pitch=-pi/2 -->
      <sensor name="camera_up" type="camera">
        <gz_frame_id>base_link</gz_frame_id>
        <pose>0 0 0.46 0 -1.5707963 0</pose>
        <topic>jetray/camera_up</topic>
        <camera>
          <horizontal_fov>2.094</horizontal_fov>
          <image>
            <width>960</width>
            <height>540</height>
          </image>
          <clip>
            <near>0.05</near>
            <far>2000</far>
          </clip>
        </camera>
        <always_on>1</always_on>
        <update_rate>15</update_rate>
        <visualize>true</visualize>
      </sensor>
"""

src = open(MODEL).read()
if "camera_up" in src:
    print("camera_up already present; nothing to do")
    sys.exit(0)

# insert right after the camera_front sensor's closing </sensor>
m = re.search(r'(<sensor name="camera_front".*?</sensor>\n)', src, re.S)
if not m:
    print("ERROR: could not find camera_front sensor block")
    sys.exit(1)
out = src[:m.end()] + UP_CAM + src[m.end():]
open(MODEL, "w").write(out)
print("inserted camera_up sensor after camera_front")
