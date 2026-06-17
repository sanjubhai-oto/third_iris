#!/usr/bin/env python3
"""Find the Isaac Camera orientation that frames a fixed marker at (6,0,10) from cam (0,0,3).
Sweeps euler candidates, captures each, reports red-marker pixel %. Run with C:\\isaacsim6\\python.bat."""
import os
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 1280, "height": 720})
import numpy as np
from pxr import UsdLux, Sdf, Gf
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane, DynamicCuboid
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
GroundPlane("/World/Field", size=2000.0, color=np.array([0.30, 0.42, 0.22]))
sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun")); sun.CreateIntensityAttr(2500.0)
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Sky"))
dome.CreateIntensityAttr(1100.0); dome.CreateColorAttr(Gf.Vec3f(0.45, 0.62, 0.92))
DynamicCuboid("/World/Mark", position=np.array([6.0, 0.0, 10.0]), scale=np.array([2.0, 2.0, 2.0]),
              color=np.array([1.0, 0.0, 0.0]))
cam = Camera(prim_path="/World/cam", position=np.array([0.0, 0.0, 3.0]), frequency=30,
             resolution=(1280, 720))
world.reset(); cam.initialize()
for _ in range(5):
    world.step(render=True)

cands = [(0,0,0),(0,49,0),(0,-49,0),(0,90,0),(0,-90,0),(49,0,0),(-49,0,0),
         (0,0,49),(0,49,90),(0,-49,90),(90,0,-90),(-49,0,-90),(0,131,0),(0,-131,0)]
best = None
for e in cands:
    cam.set_world_pose(np.array([0.0,0.0,3.0]),
                       rot_utils.euler_angles_to_quats(np.array(e), degrees=True))
    for _ in range(3):
        world.step(render=True)
    rgb = cam.get_rgba()
    if rgb is None or rgb.size == 0:
        print(e, "no frame"); continue
    im = rgb[:, :, :3].astype(np.int16)
    red = ((im[:,:,0] > 120) & (im[:,:,1] < 80) & (im[:,:,2] < 80))
    pct = 100*red.mean()
    cx = cy = -1
    if red.any():
        ys, xs = np.where(red); cx, cy = int(xs.mean()), int(ys.mean())
    print(f"euler={e}  red%={pct:.3f}  center=({cx},{cy})", flush=True)
    if best is None or pct > best[1]:
        best = (e, pct, cx, cy)
print("BEST:", best)
app.close()
