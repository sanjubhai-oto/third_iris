#!/usr/bin/env python3
"""Generate two PX4 gz jetray VARIANTS (additive; base jetray untouched) so the two SITL vehicles get
distinct, non-colliding camera topics:

  jetray_chaser  -> ONE forward-facing RGB camera on topic /chaser/camera_front (640x360@12). Tail-chase
                    air-to-air vision input. (Down VIO cams stripped to save WSL software-GL render budget.)
  jetray_runner  -> NO cameras (pure flight target; saves render budget).

Both reuse the base jetray meshes via `model://jetray/...` (resolve through GZ_SIM_RESOURCE_PATH), so no
mesh copying. Idempotent: safe to re-run. Writes into $PX4_GZ_MODELS (PX4 Tools/simulation/gz/models).

  python3 sim/gz/a2a/make_models.py            # auto-detect PX4 dir
  python3 sim/gz/a2a/make_models.py --px4 /root/PX4-Autopilot
"""
import argparse, os, re, sys

def find_px4(arg):
    for d in ([arg] if arg else []) + ["/root/PX4-Autopilot", "/opt/PX4-Autopilot",
                                       os.path.expanduser("~/PX4-Autopilot")]:
        if d and os.path.isdir(os.path.join(d, "Tools/simulation/gz/models/jetray")):
            return d
    sys.exit("ERROR: PX4-Autopilot with a jetray model not found (pass --px4 <dir>)")

CONFIG = """<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>{desc}</description>
</model>
"""

FWD_CAM = """      <!-- FORWARD tracking camera (air-to-air). Mounted WELL forward+up (0.35, 0, 0.15) to clear the
           body/props (no self-occlusion), tilted UP ~10deg (pitch -0.17) to compensate for the nose-down
           pitch of forward flight so a same-altitude target stays framed. -->
      <sensor name="camera_front" type="camera">
        <gz_frame_id>base_link</gz_frame_id>
        <pose>0.35 0 0.15 0 -0.17 0</pose>
        <topic>chaser/camera_front</topic>
        <camera>
          <horizontal_fov>1.20</horizontal_fov>
          <image><width>640</width><height>360</height></image>
          <clip><near>0.05</near><far>2000</far></clip>
        </camera>
        <always_on>1</always_on>
        <update_rate>12</update_rate>
      </sensor>
"""

DOWN_CAM = """      <!-- DOWNWARD camera (UAVros-style) for tracking a GROUND vehicle / precision landing. -->
      <sensor name="camera_down" type="camera">
        <gz_frame_id>base_link</gz_frame_id>
        <pose>0 0 -0.05 0 1.5707963 0</pose>
        <topic>chaser/camera_down</topic>
        <camera>
          <horizontal_fov>1.50</horizontal_fov>
          <image><width>512</width><height>384</height></image>
          <clip><near>0.05</near><far>500</far></clip>
        </camera>
        <always_on>1</always_on>
        <update_rate>12</update_rate>
      </sensor>
      <!-- FORWARD DEPTH camera (range + the webui depth view, like AirSim DepthPlanar). -->
      <sensor name="depth_front" type="depth_camera">
        <gz_frame_id>base_link</gz_frame_id>
        <pose>0.35 0 0.15 0 -0.17 0</pose>
        <topic>chaser/depth_front</topic>
        <camera>
          <horizontal_fov>1.20</horizontal_fov>
          <image><width>320</width><height>180</height></image>
          <clip><near>0.1</near><far>300</far></clip>
        </camera>
        <always_on>1</always_on>
        <update_rate>10</update_rate>
      </sensor>
"""

# the contiguous down-VIO camera region in the base jetray model.sdf (RGB + depth, with its comment)
VIO_BLOCK = re.compile(r'[ \t]*<!-- DOWN-facing VIO camera.*?camera_vio_depth.*?</sensor>\n', re.S)

def build(base_sdf, new_name, cam_block):
    sdf = re.sub(r"<model name=(['\"])jetray\1", f"<model name='{new_name}'", base_sdf, count=1)
    sdf, n = VIO_BLOCK.subn(cam_block, sdf)
    if n != 1:
        sys.exit(f"ERROR: expected exactly 1 VIO camera block to replace, found {n}")
    return sdf

def write_model(models_dir, name, sdf, desc):
    d = os.path.join(models_dir, name); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "model.sdf"), "w") as f: f.write(sdf)
    with open(os.path.join(d, "model.config"), "w") as f: f.write(CONFIG.format(name=name, desc=desc))
    print(f"  wrote {d}/model.sdf ({len(sdf)} bytes)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--px4", default=os.environ.get("PX4_DIR"))
    args = ap.parse_args()
    px4 = find_px4(args.px4)
    models = os.path.join(px4, "Tools/simulation/gz/models")
    base = open(os.path.join(models, "jetray", "model.sdf")).read()
    print(f"PX4_DIR={px4}\nmodels={models}")
    write_model(models, "jetray_chaser", build(base, "jetray_chaser", FWD_CAM + DOWN_CAM),
                "jetray chaser: forward cam /chaser/camera_front (air) + down cam /chaser/camera_down (ground)")
    write_model(models, "jetray_runner", build(base, "jetray_runner", ""),
                "jetray runner: no cameras (flight target)")
    print("OK: jetray_chaser + jetray_runner generated.")

if __name__ == "__main__":
    main()
