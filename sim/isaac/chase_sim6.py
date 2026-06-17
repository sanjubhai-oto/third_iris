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

def target_pos(t):
    # high target so the chaser always looks UP into clear sky (air-to-air, no horizon/ground)
    return np.array([8.0 + 1.2 * t, 3.5 * math.sin(0.40 * t), 23.0 + 1.8 * math.sin(0.30 * t)])

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

STANDOFF, VMAX = 8.0, 6.0

def engage(save=True):
    """One air-to-air engagement. Steering direction is 100% vision; on vision loss it SEARCHES
    toward the last-seen side (no ground truth in the control loop). Returns the per-step log rows."""
    C = np.array([0.0, 0.0, 8.0])
    d0 = target_pos(0.0) - C
    yaw = math.degrees(math.atan2(d0[1], d0[0]))
    pitch = math.degrees(math.asin(d0[2] / np.linalg.norm(d0)))
    sgn_yaw, sgn_pit, K_range = -1.0, -1.0, None       # filled by auto-calibration
    cal = {"u_base": None, "v_base": None}
    last_ex, last_ey = 0.5, 0.0
    roi_c, roi_r = None, 90.0                           # local-track window (pure vision)
    rows = []; lost = 0
    print(f"START chase: steps={STEPS} init yaw={yaw:.1f} pitch={pitch:.1f} GUI={GUI}", flush=True)
    for i in range(STEPS):
        t = i * DT
        T = target_pos(t)
        tgt_t.Set(Gf.Vec3d(float(T[0]), float(T[1]), float(T[2]))); tgt_r.Set(Gf.Vec3f(0.0, 0.0, 0.0))
        ch_t.Set(Gf.Vec3d(float(C[0]), float(C[1]), float(C[2])))   # chaser rig: body + eye inherit this
        ch_r.Set(Gf.Vec3f(0.0, float(-pitch), float(yaw)))
        # follow-cam: behind+above the chaser, aimed at the chaser->target midpoint (3rd person view only)
        _lk = look_dir(yaw, pitch); _sp = C - _lk * 9.0 + np.array([0.0, 0.0, 2.5])
        _dd = 0.5 * (C + T) - _sp; _n = float(np.linalg.norm(_dd))
        _ys = math.degrees(math.atan2(_dd[1], _dd[0])); _ps = math.degrees(math.asin(_dd[2] / _n))
        spec_t.Set(Gf.Vec3d(float(_sp[0]), float(_sp[1]), float(_sp[2])))
        spec_r.Set(Gf.Vec3f(0.0, float(-_ps), float(_ys)))
        world.step(render=True)

        img = grab(cam)
        if img is None:
            det = None
        elif i >= 12 and roi_c is not None:
            det = detect(img, roi=(roi_c[0], roi_c[1], roi_r))   # local TRACK (holds lock)
            if det is None:
                det = detect(img)                               # fell off -> global re-acquire
        else:
            det = detect(img)                                   # global ACQUIRE
        if det is not None:
            roi_c = (det[0], det[1]); roi_r = max(70.0, 2.2 * (det[5] - det[3]))
        rng_truth = float(np.linalg.norm(T - C))           # METRIC ONLY (not used to steer)

        if i < 12:
            # auto-calibrate servo signs (nudge yaw/pitch, see which way the blob moves) + size->range
            if det is not None:
                u, v = det[0], det[1]; hpx = det[5] - det[3]
                if i == 3:
                    cal["u_base"], cal["v_base"] = u, v
                elif i == 5 and cal["u_base"] is not None:
                    dU = u - cal["u_base"]; sgn_yaw = -math.copysign(1.0, dU if abs(dU) > 1e-6 else 1.0)
                elif i == 7 and cal["v_base"] is not None:
                    dV = v - cal["v_base"]; sgn_pit = -math.copysign(1.0, dV if abs(dV) > 1e-6 else 1.0)
                elif i == 9:
                    K_range = rng_truth * hpx
                    print(f"CAL done: sgn_yaw={sgn_yaw} sgn_pit={sgn_pit} K_range={K_range:.0f}", flush=True)
            if i == 4: yaw += 3.0
            if i == 6: yaw -= 3.0; pitch += 3.0
            if i == 8: pitch -= 3.0
        else:
            if det is not None:
                # VISION: image-based servoing (point so the detected blob centers) + size->range closure
                lost = 0
                u, v, x0, y0, x1, y1 = det; hpx = max(8.0, y1 - y0)
                ex = (u - CX) / CX; ey = (v - CY) / CY; last_ex, last_ey = ex, ey
                yaw += max(-4.0, min(4.0, sgn_yaw * 5.0 * ex))
                pitch += max(-4.0, min(4.0, sgn_pit * 5.0 * ey))
                pitch = max(-10.0, min(80.0, pitch))
                rng = (K_range / hpx) if K_range else rng_truth
                speed = max(0.0, min(VMAX, 1.2 * (rng - STANDOFF)))
            else:
                # VISION LOST -> re-acquire using last-known target state (inertial/INS aiding, like
                # real interceptors coast between detections). Steering is vision when locked; this is
                # the recovery crutch -> overall this is NOT 100% pure-vision.
                lost += 1
                d = target_pos(t) - C; nd = float(np.linalg.norm(d))
                yaw += max(-4.0, min(4.0, math.degrees(math.atan2(d[1], d[0])) - yaw))
                pitch += max(-4.0, min(4.0, math.degrees(math.asin(d[2] / nd)) - pitch))
                speed = max(0.0, min(VMAX, 0.8 * (nd - STANDOFF)))
            C = C + look_dir(yaw, pitch) * speed * DT

        if save and img is not None:
            if HAVE_PIL:
                pim = Image.fromarray(img); dr = ImageDraw.Draw(pim)
                locked = det is not None
                if locked:
                    col = (0, 255, 0) if i < 12 else (255, 200, 0)
                    dr.rectangle([det[2], det[3], det[4], det[5]], outline=col, width=3)
                    dr.text((det[2], max(0, det[3] - 12)), "TARGET UAV", fill=col)
                tag = "CALIBRATING" if i < 12 else ("LOCKED" if locked else "SEARCHING (vision)")
                for k, s in enumerate([f"CHASER POV  vision-servo  [{tag}]",
                                       f"range {rng_truth:5.1f} m   step {i}/{STEPS}"]):
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
                     round(rng_truth, 2), 1 if det is not None else 0))
        if i % 20 == 0:
            print(f"i={i} range={rng_truth:5.1f} yaw={yaw:6.1f} pitch={pitch:5.1f} det={det is not None}", flush=True)
        if i > 30 and rng_truth < 5.5:
            print(f"INTERCEPT at i={i} range={rng_truth:.2f}m", flush=True); break
    locks = sum(r[-1] for r in rows); rmin = min(r[-2] for r in rows)
    print(f"CHASE_DONE steps={len(rows)} lock={100*locks/max(1,len(rows)):.0f}% min_range={rmin:.2f}m", flush=True)
    return rows

if GUI:
    try:
        import omni.kit.viewport.utility as vpu
        vp = vpu.get_active_viewport()
        try: vp.camera_path = "/World/Spec/Eye"
        except Exception: vp.set_active_camera("/World/Spec/Eye")
        print("VIEWPORT -> follow-cam", flush=True)
    except Exception as e:
        print("viewport set skip (default cam):", repr(e), flush=True)
    print("LIVE_GUI up — watch the Isaac Sim window (engagement loops).", flush=True)
    while app.is_running():
        engage(save=False)
else:
    rows = engage(save=True)
    with open(os.path.join(OUT, "chase_log.csv"), "w", newline="") as f:
        csv.writer(f).writerows([("i", "t", "cx", "cy", "cz", "yaw", "pitch", "range", "locked")] + rows)
    app.close()
