#!/usr/bin/env python3
"""Isaac Sim 6 air-to-air CHASER vs TARGET, vision-in-the-loop (headless).

A TARGET drone weaves/recedes through the sky. A CHASER drone carries a camera; each step we render
the chaser's view, detect the target IN-IMAGE (pure-numpy dark-blob vs bright sky -> true onboard
vision, no external NN process), and steer the chaser by image-based visual servoing (point so the
target stays centered) while closing range to a standoff (range estimated from apparent size).

Outputs per step (atomic) for the live browser view:
  runs/videos/latest_chase.jpg  (chaser POV + lock box + HUD)
  runs/videos/latest_spec.jpg   (fixed spectator cam: both drones in the sky)
and PNG sequences in runs/videos/chase_frames/ , spec_frames/ for video building.

Run:  C:\\isaacsim6\\python.bat sim\\isaac\\chase_sim6.py        (full)
      set ISAAC_STEPS=15 & ...                                   (smoke test)
"""
import os, csv, math, json
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp
GUI = os.environ.get("ISAAC_GUI") == "1"          # ISAAC_GUI=1 -> live Isaac Sim window
app = SimulationApp({"headless": not GUI, "width": 1280, "height": 720})

import numpy as np
from pxr import UsdLux, UsdGeom, Gf, Sdf, Usd, UsdPhysics
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
import imageio.v2 as imageio
try:
    from PIL import Image, ImageDraw
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False

IRIS = r"C:\Users\admin\uav-vio-track\sim\pegasus\extensions\pegasus.simulator\pegasus\simulator\assets\Robots\Iris\iris.usd"
OUT = r"C:\Users\admin\uav-vio-track\runs\videos"
CF = os.path.join(OUT, "chase_frames"); SF = os.path.join(OUT, "spec_frames")
for d in (CF, SF):
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        try: os.remove(os.path.join(d, f))
        except Exception: pass

W, H = 1280, 720
CX, CY = W / 2, H / 2
STEPS = int(os.environ.get("ISAAC_STEPS", "260"))
DT = 0.05

# ----- scene -----
world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
GroundPlane("/World/Field", size=4000.0, color=np.array([0.30, 0.42, 0.22]))
sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
sun.CreateIntensityAttr(2500.0); sun.CreateAngleAttr(0.53)
UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 35.0))
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Sky")); dome.CreateIntensityAttr(1100.0)
try:
    from isaacsim.storage.native import get_assets_root_path
except Exception:
    from omni.isaac.nucleus import get_assets_root_path
_root = get_assets_root_path(); _sky = False
if _root:
    try:
        dome.CreateTextureFileAttr().Set(_root + "/NVIDIA/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr")
        dome.CreateTextureFormatAttr().Set("latlong"); _sky = True; print("SKY_HDRI ok")
    except Exception as e:
        print("sky skip", repr(e))
if not _sky:
    dome.CreateColorAttr(Gf.Vec3f(0.45, 0.62, 0.92)); print("SKY blue fallback")

def strip_physics(prim):
    for p in Usd.PrimRange(prim):          # pure visual -> obeys xformOp every frame, no gravity
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Set(False)
        if p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)

# orientation via AddRotateXYZOp (AddOrientOp/quat was silently ignored on these prims);
# rot (0, -pitch, yaw): identity cam looks +X, pitch up = -Y rot, yaw = +Z rot.
# --- target: standalone drone, translate+rotate each frame ---
add_reference_to_stage(IRIS, "/World/Target"); strip_physics(stage.GetPrimAtPath("/World/Target"))
txf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Target")); txf.ClearXformOpOrder()
tgt_t = txf.AddTranslateOp(); tgt_r = txf.AddRotateXYZOp(); txf.AddScaleOp().Set(Gf.Vec3d(2.6, 2.6, 2.6))

# --- chaser RIG: a parent Xform holds the body + the camera eye; move/orient the PARENT, the
#     camera inherits it via the hierarchy (robust; no fighting the Camera wrapper's own transform) ---
UsdGeom.Xform.Define(stage, "/World/Chaser")
cxf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Chaser"))
ch_t = cxf.AddTranslateOp(); ch_r = cxf.AddRotateXYZOp()
add_reference_to_stage(IRIS, "/World/Chaser/Body"); strip_physics(stage.GetPrimAtPath("/World/Chaser/Body"))
bxf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Chaser/Body")); bxf.ClearXformOpOrder()
bxf.AddScaleOp().Set(Gf.Vec3d(1.8, 1.8, 1.8))
cam = Camera(prim_path="/World/Chaser/Eye", position=np.array([0.0, 0.0, 0.0]),
             frequency=30, resolution=(W, H))           # local identity -> looks along parent +X (default ~47deg FOV)

# follow-cam spectator (updated each frame): behind+above the chaser, framing chaser(near)+target(far)
UsdGeom.Xform.Define(stage, "/World/Spec")
sxf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Spec"))
spec_t = sxf.AddTranslateOp(); spec_r = sxf.AddRotateXYZOp()
spec = Camera(prim_path="/World/Spec/Eye", position=np.array([0.0, 0.0, 0.0]),
              frequency=30, resolution=(W, H))

world.reset(); cam.initialize(); spec.initialize()

# ----- helpers -----
def look_dir(yaw, pitch):
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    return np.array([cp * cy, cp * sy, sp])           # +X fwd, yaw about Z, pitch up

# --- NEW GEOMETRY: target cruises straight & level (constant alt), point A -> B, heading +Y (its
#     "front" faces +Y). Chaser starts ~20 m ABOVE and ahead, dives tangentially, strikes the FRONT.
TGT_ALT = 18.0
TGT_V = np.array([0.0, 5.0, 0.0])      # 5 m/s along +Y, constant altitude
TGT_P0 = np.array([0.0, -25.0, TGT_ALT])

def target_pos(t):
    return TGT_P0 + TGT_V * t           # straight line, same altitude the whole flight

def detect(rgb, roi=None):
    """numpy dark-blob detector: compact dark object vs bright sky. -> (u,v,x0,y0,x1,y1).
    roi=(cu,cv,rad): search ONLY a window around (cu,cv) -> local visual TRACK (holds lock, ignores
    the HDRI band). roi=None: global ACQUIRE. Pure vision, no ground truth."""
    g = rgb[:, :, :3].astype(np.float32).mean(2)
    if roi is not None:
        cu, cv, rad = roi
        x0r = max(0, int(cu - rad)); x1r = min(W, int(cu + rad))
        y0r = max(0, int(cv - rad)); y1r = min(H, int(cv + rad))
        gm = np.full_like(g, 255.0); gm[y0r:y1r, x0r:x1r] = g[y0r:y1r, x0r:x1r]; g = gm
    gmin = float(g.min())
    if np.median(g[g < 255.0]) - gmin < 25.0 if roi is not None else np.median(g) - gmin < 25.0:
        return None                                   # no clearly-dark object -> just sky
    mask = g < gmin + 70.0
    mask[mask.sum(1) > 0.5 * W, :] = False            # drop full-width dark rows (bands)
    mask[:, mask.sum(0) > 0.5 * H] = False            # drop full-height dark cols
    ys, xs = np.where(mask)
    if len(xs) < 12:
        return None
    u = float(np.median(xs)); v = float(np.median(ys))
    x0, x1 = np.percentile(xs, 2), np.percentile(xs, 98)
    y0, y1 = np.percentile(ys, 2), np.percentile(ys, 98)
    if (x1 - x0) > 0.5 * W or (y1 - y0) > 0.5 * H:    # still spans frame -> not a drone
        return None
    return (u, v, float(x0), float(y0), float(x1), float(y1))

def grab(c):
    r = c.get_rgba()
    if r is None or r.size == 0:
        return None
    return r[:, :, :3].astype(np.uint8)

def save_atomic(arr_or_img, path, draw=None):
    tmp = path + ".tmp.jpg"
    if HAVE_PIL and isinstance(arr_or_img, Image.Image):
        arr_or_img.save(tmp, quality=82)
    else:
        imageio.imwrite(tmp, arr_or_img)
    os.replace(tmp, path)

# warmup
for _ in range(6):
    world.step(render=True)

VC = 10.0          # chaser strike speed (m/s), > target so it overtakes/strikes
HIT_R = 2.0        # impact radius (m)
# fixed side-on offset direction for the spectator frame-both cam (from +X, slightly above)
SPEC_DIR = np.array([1.0, -0.25, 0.45]); SPEC_DIR = SPEC_DIR / np.linalg.norm(SPEC_DIR)

def intercept_dir(C, T, vt, Vc):
    """Proportional-nav / predicted-intercept-point lead: solve for time-to-go on a collision course
    so the chaser arrives where the target WILL be -> tangential closure that strikes the FRONT.
    Returns (unit_dir, tgo). Truth-based guidance (down-looking-at-ground makes pure vision unreliable
    here -- vision still runs as a monitor; honest: this strike is PN-guided, not vision-steered)."""
    rel = T - C
    a = float(vt.dot(vt) - Vc * Vc); b = float(2.0 * rel.dot(vt)); c = float(rel.dot(rel))
    tgo = None
    if abs(a) < 1e-6:
        if abs(b) > 1e-6: tgo = -c / b
    else:
        disc = b * b - 4 * a * c
        if disc >= 0.0:
            sq = math.sqrt(disc)
            for r in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)):
                if r > 1e-3 and (tgo is None or r < tgo): tgo = r
    pip = T + vt * tgo if (tgo and tgo > 0) else T
    d = pip - C; n = float(np.linalg.norm(d))
    return (d / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])), (tgo or 0.0)

def engage(save=True):
    """Top-attack strike: target cruises straight & level; chaser starts ~20 m ABOVE + ahead and dives
    on a PN lead course to hit the target's FRONT. Returns per-step log rows."""
    T0 = target_pos(0.0)
    C = np.array([10.0, 25.0, TGT_ALT + 20.0])         # 20 m above target alt, ahead (+Y) & offset (+X)
    d0 = T0 - C
    yaw = math.degrees(math.atan2(d0[1], d0[0]))
    pitch = math.degrees(math.asin(d0[2] / np.linalg.norm(d0)))
    roi_c, roi_r = None, 90.0
    rows = []; rmin = 1e9; hit_i = -1
    print(f"START strike: steps={STEPS} C0={C.tolist()} alt+20 above target  GUI={GUI}", flush=True)
    for i in range(STEPS):
        t = i * DT
        T = target_pos(t)
        # --- guidance: lead-intercept, move chaser; gimbal camera onto the target (down-looking dive) ---
        dirv, tgo = intercept_dir(C, T, TGT_V, VC)
        C = C + dirv * VC * DT
        d = T - C; nd = float(np.linalg.norm(d))
        yaw = math.degrees(math.atan2(d[1], d[0]))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, d[2] / nd))))

        # poses: target heading +Y (front faces +Y); chaser rig body+eye inherit
        tgt_t.Set(Gf.Vec3d(float(T[0]), float(T[1]), float(T[2]))); tgt_r.Set(Gf.Vec3f(0.0, 0.0, 90.0))
        ch_t.Set(Gf.Vec3d(float(C[0]), float(C[1]), float(C[2])))
        ch_r.Set(Gf.Vec3f(0.0, float(-pitch), float(yaw)))
        # spectator: LOCKED on both -> aim at midpoint, stand back by the pair's separation so BOTH stay
        # framed the whole dive (fixed side-on direction; distance auto-scales with separation).
        mid = 0.5 * (C + T); dist = max(22.0, 0.75 * nd + 16.0)
        _sp = mid + SPEC_DIR * dist
        _dd = mid - _sp; _n = float(np.linalg.norm(_dd))
        _ys = math.degrees(math.atan2(_dd[1], _dd[0])); _ps = math.degrees(math.asin(_dd[2] / _n))
        spec_t.Set(Gf.Vec3d(float(_sp[0]), float(_sp[1]), float(_sp[2])))
        spec_r.Set(Gf.Vec3f(0.0, float(-_ps), float(_ys)))
        world.step(render=True)

        img = grab(cam)
        if img is None:
            det = None
        elif roi_c is not None:
            det = detect(img, roi=(roi_c[0], roi_c[1], roi_r)) or detect(img)
        else:
            det = detect(img)
        if det is not None:
            roi_c = (det[0], det[1]); roi_r = max(70.0, 2.2 * (det[5] - det[3]))
        rng = nd; rmin = min(rmin, rng)

        if save and img is not None:
            if HAVE_PIL:
                pim = Image.fromarray(img); dr = ImageDraw.Draw(pim)
                if det is not None:
                    dr.rectangle([det[2], det[3], det[4], det[5]], outline=(255, 200, 0), width=3)
                    dr.text((det[2], max(0, det[3] - 12)), "TARGET UAV", fill=(255, 200, 0))
                vtag = "VISION LOCK" if det is not None else "no-vis (ground bg)"
                for k, s in enumerate([f"CHASER POV  TOP-ATTACK STRIKE (PN lead guidance)",
                                       f"range {rng:5.1f} m   tgo {tgo:4.1f}s   step {i}/{STEPS}",
                                       f"detector: {vtag}"]):
                    dr.text((12, 10 + 16 * k), s, fill=(0, 255, 255))
                save_atomic(pim, os.path.join(OUT, "latest_chase.jpg"))
                imageio.imwrite(os.path.join(CF, f"c{i:04d}.png"), np.asarray(pim))
            else:
                save_atomic(img, os.path.join(OUT, "latest_chase.jpg"))
                imageio.imwrite(os.path.join(CF, f"c{i:04d}.png"), img)
            simg = grab(spec)
            if simg is not None:
                save_atomic(simg, os.path.join(OUT, "latest_spec.jpg"))
                imageio.imwrite(os.path.join(SF, f"s{i:04d}.png"), simg)

        rows.append((i, round(t, 2), *[round(float(x), 2) for x in C], round(yaw, 1), round(pitch, 1),
                     round(rng, 2), 1 if det is not None else 0))
        if i % 20 == 0:
            print(f"i={i} range={rng:5.1f} tgo={tgo:4.1f} pitch={pitch:6.1f} det={det is not None}", flush=True)
        if rng < HIT_R:
            hit_i = i; print(f"HIT at i={i} range={rng:.2f}m (front strike)", flush=True); break
    locks = sum(r[-1] for r in rows)
    print(f"STRIKE_DONE steps={len(rows)} lock={100*locks/max(1,len(rows)):.0f}% "
          f"miss/min_range={rmin:.2f}m hit={hit_i>=0}", flush=True)
    return rows

if GUI:
    try:
        import omni.kit.viewport.utility as vpu
        vp = vpu.get_active_viewport()
        try: vp.camera_path = "/World/Spec/Eye"
        except Exception: vp.set_active_camera("/World/Spec/Eye")
        print("VIEWPORT -> frame-both spectator (locked on chaser+target)", flush=True)
    except Exception as e:
        print("viewport set skip (default cam):", repr(e), flush=True)
    print("LIVE_GUI up — watch the Isaac Sim window (strike loops).", flush=True)
    while app.is_running():
        engage(save=False)
else:
    rows = engage(save=True)
    with open(os.path.join(OUT, "chase_log.csv"), "w", newline="") as f:
        csv.writer(f).writerows([("i", "t", "cx", "cy", "cz", "yaw", "pitch", "range", "locked")] + rows)
    app.close()
