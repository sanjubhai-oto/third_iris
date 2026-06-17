#!/usr/bin/env python3
"""Static probe of Isaac Camera convention: 5 colored cuboids on the axes around cam(0,0,3),
identity orientation, ONE render. Reports which color is visible + its pixel centroid -> tells us
the camera forward/up/right axes. No pose changes (avoids the headless USD-ref crash)."""
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

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
GroundPlane("/World/Field", size=2000.0, color=np.array([0.30, 0.42, 0.22]))
sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun")); sun.CreateIntensityAttr(2500.0)
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Sky"))
dome.CreateIntensityAttr(1100.0); dome.CreateColorAttr(Gf.Vec3f(0.45, 0.62, 0.92))

C = (0.0, 0.0, 3.0)
marks = {  # name: (pos, rgb)
  "+X_red":     ((C[0]+10, C[1],    C[2]),   (1,0,0)),
  "-X_yellow":  ((C[0]-10, C[1],    C[2]),   (1,1,0)),
  "+Y_magenta": ((C[0],    C[1]+10, C[2]),   (1,0,1)),
  "-Y_cyan":    ((C[0],    C[1]-10, C[2]),   (0,1,1)),
  "+Z_orange":  ((C[0],    C[1],    C[2]+10), (1,0.5,0)),
}
for i,(nm,(p,c)) in enumerate(marks.items()):
    DynamicCuboid(f"/World/M{i}", position=np.array(p, float),
                  scale=np.array([2.5,2.5,2.5]), color=np.array(c, float))

cam = Camera(prim_path="/World/cam", position=np.array(C), frequency=30, resolution=(1280,720))
world.reset(); cam.initialize()
for _ in range(8):
    world.step(render=True)
rgb = cam.get_rgba()[:, :, :3].astype(np.int16)
import imageio.v2 as imageio
imageio.imwrite(r"C:\Users\admin\uav-vio-track\runs\videos\axis_probe.png", rgb.astype(np.uint8))

def find(mask):
    if not mask.any(): return None
    ys, xs = np.where(mask); return (int(xs.mean()), int(ys.mean()), int(mask.sum()))
R,G,B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
tests = {
 "+X_red":     (R>120)&(G<70)&(B<70),
 "-X_yellow":  (R>120)&(G>120)&(B<80),
 "+Y_magenta": (R>120)&(G<80)&(B>120),
 "-Y_cyan":    (R<80)&(G>120)&(B>120),
 "+Z_orange":  (R>120)&(G>60)&(G<160)&(B<70),
}
print("IMG center = (640,360)")
for nm,m in tests.items():
    print(nm, "->", find(m))
app.close()
