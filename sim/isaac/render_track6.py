#!/usr/bin/env python3
"""Isaac Sim 6 (headless, fast): realistic field + sky, a TARGET UAV (iris quad) flying an orbit overhead,
a chaser camera looking up. Renders the camera each frame -> PNG sequence + truth CSV. Our YOLO/VIO run
separately on the frames (decoupled, no torch conflict).
Run:  C:\\isaacsim6\\python.bat sim\\isaac\\render_track6.py
"""
import os, csv, math
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 1280, "height": 720})

import numpy as np
from pxr import UsdLux, UsdGeom, Gf, Sdf
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
from pxr import Usd, UsdPhysics
import isaacsim.core.utils.numpy.rotations as rot_utils
import imageio.v2 as imageio

IRIS = r"C:\Users\admin\uav-vio-track\sim\pegasus\extensions\pegasus.simulator\pegasus\simulator\assets\Robots\Iris\iris.usd"
FRAMES = r"C:\Users\admin\uav-vio-track\runs\videos\isaac_frames"
os.makedirs(FRAMES, exist_ok=True)
for f in os.listdir(FRAMES):
    try: os.remove(os.path.join(FRAMES, f))
    except Exception: pass

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

GroundPlane("/World/Field", size=2000.0, color=np.array([0.30, 0.42, 0.22]))
sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
sun.CreateIntensityAttr(2500.0); sun.CreateAngleAttr(0.53)
UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 35.0))
# realistic cloudy sky from NVIDIA's available assets (matches real air-to-air backdrop); blue fallback
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Sky"))
dome.CreateIntensityAttr(1100.0)
try:
    from isaacsim.storage.native import get_assets_root_path
except Exception:
    from omni.isaac.nucleus import get_assets_root_path
_root = get_assets_root_path(); _sky = False
if _root:
    for tex in ["/NVIDIA/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr",
                "/NVIDIA/Assets/Skies/Clear/qwantani_4k.hdr",
                "/NVIDIA/Assets/Skies/Cloudy/champagne_castle_1_4k.hdr"]:
        try:
            dome.CreateTextureFileAttr().Set(_root + tex)
            dome.CreateTextureFormatAttr().Set("latlong"); _sky = True; print("SKY_HDRI:", tex); break
        except Exception as e:
            print("sky skip", repr(e))
if not _sky:
    dome.CreateColorAttr(Gf.Vec3f(0.45, 0.62, 0.92)); print("SKY: blue fallback")

# target UAV (iris quad), scaled up, pose driven each step (orbits overhead -> in the up-camera FOV)
add_reference_to_stage(IRIS, "/World/Target")
target_prim = stage.GetPrimAtPath("/World/Target")
# strip physics so the target is a pure VISUAL prim -> gravity won't pull it out of frame and our
# per-frame xformOp translate is respected (rigid bodies ignore xformOp edits after the first step).
for p in Usd.PrimRange(target_prim):
    if p.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Set(False)
    if p.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)
tgt = UsdGeom.Xformable(target_prim)
tgt.ClearXformOpOrder()
t_op = tgt.AddTranslateOp(); s_op = tgt.AddScaleOp(); s_op.Set(Gf.Vec3d(4.0, 4.0, 4.0))
DEBUG_MARK = os.environ.get("ISAAC_MARK") == "1"
if DEBUG_MARK:
    from isaacsim.core.api.objects import DynamicCuboid
    DynamicCuboid("/World/Mark", position=np.array([6.0, 0.0, 10.0]),
                  scale=np.array([1.5, 1.5, 1.5]), color=np.array([1.0, 0.0, 0.0]))

# chaser camera looks ~49deg UP toward +X. Isaac Camera identity = +X forward, +Z up; pitch up = -Y rot.
cam = Camera(prim_path="/World/chaser_cam", position=np.array([0.0, 0.0, 3.0]),
             frequency=30, resolution=(1280, 720),
             orientation=rot_utils.euler_angles_to_quats(np.array([0, -49, 0]), degrees=True))
world.reset(); cam.initialize()

rows = []
N = int(os.environ.get("ISAAC_N", "150"))
for i in range(N):
    t = i / 30.0
    # target flies up-forward of the chaser (in the camera FOV) at ~10m -> clear drone vs sky
    tx, ty, tz = 7.0 + 2.0*math.cos(0.4*t), 2.0*math.sin(0.4*t), 11.0 + 1.2*math.sin(0.25*t)
    t_op.Set(Gf.Vec3d(tx, ty, tz))
    world.step(render=True)
    rgb = cam.get_rgba()
    if rgb is None or rgb.size == 0:
        continue
    imageio.imwrite(os.path.join(FRAMES, f"f{i:04d}.png"), rgb[:, :, :3].astype(np.uint8))
    rows.append((i, tx, ty, tz))
    if i % 30 == 0:
        print(f"frame {i}/{N}  target=({tx:.1f},{ty:.1f},{tz:.1f})", flush=True)

with open(os.path.join(FRAMES, "truth.csv"), "w", newline="") as fcsv:
    csv.writer(fcsv).writerows([("frame", "tx", "ty", "tz")] + rows)
print(f"ISAAC_RENDER_DONE: {len(rows)} frames -> {FRAMES}")
app.close()
