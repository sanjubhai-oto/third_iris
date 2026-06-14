#!/usr/bin/env python3
"""Configure the jetray gz model with a DOWN-facing RGB-D camera for VIO (ground = reliable features +
depth). Removes the up/sky cameras (keeps 2 sensors so software-GL renders). Run in WSL."""
import re, sys
MODEL = "/root/PX4-Autopilot/Tools/simulation/gz/models/jetray/model.sdf"
src = open(MODEL).read()

# drop any existing camera sensors we added (camera_up / camera_up_depth / camera_front / camera_vio*)
src = re.sub(r'\s*<!--[^>]*camera[^>]*-->', '', src)
src = re.sub(r'\s*<sensor name="camera_[^"]*".*?</sensor>\n', '\n', src, flags=re.S)

VIO = """      <!-- DOWN-facing VIO camera (RGB) -->
      <sensor name="camera_vio" type="camera">
        <pose>0 0 -0.05 0 1.5707963 0</pose>
        <topic>jetray/cam_vio</topic>
        <camera><horizontal_fov>1.5</horizontal_fov>
          <image><width>640</width><height>480</height></image>
          <clip><near>0.1</near><far>200</far></clip></camera>
        <always_on>1</always_on><update_rate>20</update_rate>
      </sensor>
      <!-- DOWN-facing VIO depth -->
      <sensor name="camera_vio_depth" type="depth_camera">
        <pose>0 0 -0.05 0 1.5707963 0</pose>
        <topic>jetray/cam_vio_depth</topic>
        <camera><horizontal_fov>1.5</horizontal_fov>
          <image><width>640</width><height>480</height></image>
          <clip><near>0.1</near><far>200</far></clip></camera>
        <always_on>1</always_on><update_rate>20</update_rate>
      </sensor>
"""
if "camera_vio" in src:
    print("camera_vio already present"); sys.exit(0)
m = re.search(r'(<sensor name="imu_sensor".*?</sensor>\n)', src, re.S)
if not m:
    print("ERROR: imu_sensor not found"); sys.exit(1)
src = src[:m.end()] + VIO + src[m.end():]
open(MODEL, "w").write(src)
print("set DOWN-facing camera_vio (RGB+depth); cameras now:",
      re.findall(r'sensor name="(camera_[^"]*)"', src))
