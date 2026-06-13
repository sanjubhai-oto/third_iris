#!/usr/bin/env python3
"""UP-FACING-camera air-to-air VIO tracking in Gazebo (the real sky-cam AirSim couldn't render).

Chaser carries a sky-facing camera (jetray/camera_up). A target drone maneuvers overhead. The
chaser tracks it PURELY from the up-camera with the no-yaw control law (hold heading; STRAFE in the
horizontal plane to keep the target centred; CLIMB to hold the vertical gap) — yawing a fixed up-cam
only rotates the image, so we never yaw (this is what killed the yaw-shake in AirSim).

Kinematic: target is driven on a pattern via /world/<w>/set_pose; the chaser pose is integrated from
the vision-derived velocity command and also set via set_pose (isolates perception+guidance, like the
AirSim scenario_eval test). Detector: classical dark-blob-on-sky (default, robust for the clean-sky
case, no deps) or YOLO (--yolo, universal detector) if available.

Run in WSL (software GL):
  bash sim/gz/launch_gz.sh
  LIBGL_ALWAYS_SOFTWARE=1 python3 sim/gz/track_gz.py --pattern orbit_climb --secs 40
"""
import argparse, math, time, threading
import numpy as np

from gz.transport13 import Node
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.image_pb2 import Image

WORLD = "uav_track_up"
CAM_TOPIC = "/jetray/camera_up"
DEPTH_TOPIC = "/jetray/camera_up_depth"
HFOV = 2.094                      # rad (matches model camera_up)

# ---------------- latest-frame subscriber ----------------
class Cam:
    def __init__(self, node):
        self.lock = threading.Lock(); self.img = None; self.stamp = 0.0
        node.subscribe(Image, CAM_TOPIC, self._cb)
    def _cb(self, msg):
        a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        with self.lock:
            self.img = a; self.stamp = time.time()
    def get(self):
        with self.lock:
            return None if self.img is None else self.img.copy()

class DepthCam:
    """Up-facing depth camera -> true range to the overhead target (FLOAT32 depth image)."""
    def __init__(self, node):
        self.lock = threading.Lock(); self.d = None
        node.subscribe(Image, DEPTH_TOPIC, self._cb)
    def _cb(self, msg):
        try:
            a = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
        except Exception:
            return
        with self.lock:
            self.d = a
    def range_at(self, u_frac, v_frac):
        """Sample depth at fractional image coords (0..1). Median of a small patch, ignoring inf/nan."""
        with self.lock:
            if self.d is None:
                return None
            d = self.d
        h, w = d.shape
        cu, cv = int(np.clip(u_frac * w, 0, w - 1)), int(np.clip(v_frac * h, 0, h - 1))
        patch = d[max(0, cv-3):cv+4, max(0, cu-3):cu+4].ravel()
        patch = patch[np.isfinite(patch) & (patch > 0.1) & (patch < 190)]
        return float(np.median(patch)) if patch.size else None

# ---------------- detectors ----------------
def detect_blob(img):
    """Dark drone on sky -> CLOUD-ROBUST: a drone is among the DARKEST, COMPACT regions. Clouds are
    BRIGHT (never in the darkest percentile) and LARGE (filtered by area/compactness). Returns
    (cx,cy,w,h,conf) or None. Robust where a global mean-k*std threshold collapses on cloudy frames."""
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thr = float(np.percentile(g, 0.4))          # darkest ~0.4% of pixels (the drone), cloud-immune
    mask = (g <= thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None; best_area = 0.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 6 or area > 9000:             # noise / huge dark band
            continue
        x, y, w, h = cv2.boundingRect(c)
        ar = w / max(h, 1)
        if ar < 0.25 or ar > 4.0:               # reject long thin (horizon/edges); drone ~ blob
            continue
        if area > best_area:
            best_area = area; best = (x + w / 2.0, y + h / 2.0, float(w), float(h),
                                      min(1.0, area / 2000.0))
    return best

class YoloDet:
    def __init__(self, weights, imgsz=960, conf=0.12):
        from ultralytics import YOLO
        self.m = YOLO(weights); self.imgsz = imgsz; self.conf = conf
    def __call__(self, img):
        r = self.m.predict(img, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        b = r.boxes
        if b is None or not len(b):
            return None
        i = int(np.argmax(b.conf.cpu().numpy()))
        x1, y1, x2, y2 = b.xyxy[i].cpu().numpy().tolist()
        return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, float(b.conf[i]))

# ---------------- target patterns (gz Z-up world) ----------------
def target_pose(pattern, t, base):
    bx, by, bz = base
    if pattern == "orbit":
        R = 6.0; w = 0.35
        return (bx + R * math.cos(w * t), by + R * math.sin(w * t), bz)
    if pattern == "orbit_climb":
        R = 6.0; w = 0.35
        return (bx + R * math.cos(w * t), by + R * math.sin(w * t), bz + 0.6 * t)
    if pattern == "skyward":
        return (bx + 1.5 * math.sin(0.3 * t), by + 1.5 * math.cos(0.3 * t), bz + 0.8 * t)
    if pattern == "zigzag":
        return (bx + 5.0 * math.sin(0.5 * t), by + 3.0 * math.sin(0.9 * t), bz)
    return (bx, by, bz)

# ---------------- set_pose helper ----------------
def set_pose(node, name, x, y, z, yaw=0.0):
    req = Pose(); req.name = name
    req.position.x = float(x); req.position.y = float(y); req.position.z = float(z)
    req.orientation.z = math.sin(yaw / 2); req.orientation.w = math.cos(yaw / 2)
    node.request(f"/world/{WORLD}/set_pose", req, Pose, Boolean, 300)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="orbit_climb",
                    choices=["orbit", "orbit_climb", "skyward", "zigzag"])
    ap.add_argument("--secs", type=float, default=40.0)
    ap.add_argument("--gap", type=float, default=12.0, help="vertical gap to hold (m)")
    ap.add_argument("--speed", type=float, default=8.0)
    ap.add_argument("--commit", action="store_true",
                    help="track-then-commit terminal intercept: hold the gap, then drive gap->0 onto the "
                         "PREDICTED intercept point (PIP/PN-style), the Ukraine/Israel terminal-kill pattern")
    ap.add_argument("--commit-after", type=float, default=14.0, help="seconds to track before committing")
    ap.add_argument("--hit-radius", type=float, default=1.5, help="declare intercept within this range (m)")
    ap.add_argument("--intercept-speed", type=float, default=16.0, help="full closing speed in commit phase (m/s)")
    ap.add_argument("--yolo", default="", help="path to YOLO weights (else classical blob detector)")
    ap.add_argument("--imgsz", type=int, default=960, help="YOLO inference size (960 = better small-target recall)")
    ap.add_argument("--conf", type=float, default=0.12, help="YOLO confidence threshold")
    ap.add_argument("--tgt0", type=float, nargs=3, default=[0, 0, 18.0])
    ap.add_argument("--ego0", type=float, nargs=3, default=None,
                    help="chaser start (default = directly under target at the gap)")
    ap.add_argument("--out", default="/mnt/c/Users/admin/uav-vio-track/runs/videos")
    args = ap.parse_args()

    import cv2
    node = Node()
    cam = Cam(node)
    depth = DepthCam(node)
    det = YoloDet(args.yolo, args.imgsz, args.conf) if args.yolo else detect_blob
    print(f"[track] detector = {'YOLO '+args.yolo if args.yolo else 'classical dark-blob'}")

    tbase = np.array(args.tgt0, float)
    # chaser starts directly under the target at the gap (so it tracks from the gap, not a long close-in)
    ego0 = args.ego0 if args.ego0 is not None else [tbase[0], tbase[1], max(0.5, tbase[2] - args.gap)]
    ego = np.array(ego0, float)             # chaser world pos (x,y,z), yaw fixed 0
    set_pose(node, "chaser", *ego)
    set_pose(node, "target", *tbase)
    time.sleep(1.0)
    # wait for first frame
    t0w = time.time()
    while cam.get() is None and time.time() - t0w < 10:
        time.sleep(0.1)
    if cam.get() is None:
        print("[track] ERROR no camera frames"); return

    KXY = 6.0; KZ = 1.0
    FWD_ACC = 6.0; VZ_ACC = 3.0
    prev_vx = prev_vy = prev_vz = 0.0
    Ksize = None                            # range = Ksize / bbox_h (calibrated on 1st detect)
    # TEMPORAL GATE: target starts centred (chaser launched under it). Reject any detection that jumps
    # too far from the last lock -> kills false locks (cloud edges / horizon) that otherwise drag the
    # chaser off (mirrors the image-Kalman gate in the AirSim servo). lock_xy in normalized image coords.
    lock_xy = np.array([0.0, 0.0]); have_lock = False
    GATE = 0.45; miss = 0
    VFOV = 2 * math.atan(math.tan(HFOV / 2) * 9 / 16)   # 960x540 -> vertical FOV
    Pt = None; Vt = np.zeros(3); phase = "TRACK"        # target world pos estimate + velocity (for PIP)
    hit = False; hit_t = None; min_range = 1e9
    last = time.time(); simt = 0.0
    log = []                                 # (t, seen, ex, ey, range, ego(3), tgt_truth(3))
    vw = None
    H = W = None
    frames = 0; seen_n = 0

    # drive the WHOLE sim on accumulated step-time (not wall-clock) so slow CPU YOLO stays consistent
    while simt < args.secs:
        now = time.time(); dt = min(0.2, max(0.02, now - last)); last = now
        simt += dt; tt = simt
        # drive target on its pattern (truth)
        tx, ty, tz = target_pose(args.pattern, tt, tbase)
        set_pose(node, "target", tx, ty, tz)
        tgt = np.array([tx, ty, tz])

        img = cam.get()
        if img is None:
            continue
        H, W = img.shape[:2]
        d = det(img)
        frames += 1
        # temporal gate: only accept a detection consistent with the last lock (near centre to acquire)
        if d is not None:
            _ex = (d[0] - W / 2) / (W / 2); _ey = (d[1] - H / 2) / (H / 2)
            ref = lock_xy if have_lock else np.array([0.0, 0.0])
            if math.hypot(_ex - ref[0], _ey - ref[1]) > GATE:
                d = None
                if have_lock:
                    miss += 1                        # jump rejected -> coast on the gate, stay locked
                    if miss > 12:                    # long miss -> drop lock, allow re-acquire anywhere
                        have_lock = False; miss = 0
        if d is not None:
            seen_n += 1
            cx, cy, bw, bh, conf = d
            ex = (cx - W / 2) / (W / 2); ey = (cy - H / 2) / (H / 2)
            lock_xy = np.array([ex, ey]); have_lock = True; miss = 0
            # RANGE: prefer the up depth camera (true range); fall back to size-proxy if depth missing
            rng = depth.range_at(cx / W, cy / H)
            if rng is None:
                if Ksize is None:
                    Ksize = max(1.0, (tbase[2] - ego[2])) * max(bh, 1.0)
                rng = Ksize / max(bh, 1.0)
            # NO-YAW strafe + climb (chaser yaw fixed at 0):
            #  image-down (ey>0) -> target toward world +X ; image-right (ex>0) -> world -Y
            eff = args.speed
            vx = float(np.clip(KXY * ey, -eff, eff))
            vy = float(np.clip(-KXY * ex, -eff, eff))
            vz = float(np.clip(KZ * (rng - args.gap), -eff, eff))   # climb to hold gap (z up +)
            # reconstruct the TARGET WORLD POSITION from the up-cam ray (chaser yaw 0): the ray points
            # up (+Z) tilted by the image offset. Used for the PIP terminal intercept.
            ray = np.array([ey * math.tan(VFOV / 2), -ex * math.tan(HFOV / 2), 1.0])
            ray = ray / np.linalg.norm(ray)
            P_new = ego + rng * ray
            if Pt is None:
                Pt = P_new
            else:
                Vt = 0.7 * Vt + 0.3 * ((P_new - Pt) / dt)          # EMA target velocity (world)
                Pt = 0.5 * Pt + 0.5 * P_new                        # EMA target position
        else:
            ex = ey = 0.0; rng = float('nan')
            vx = vy = vz = 0.0                                       # lost -> hold (no spin)

        # ---- TRACK-THEN-COMMIT terminal intercept (PIP / PN-style) ----
        # Phase 1 holds the gap (above). Phase 2 (commit): drive onto the PREDICTED intercept point
        # P_t + V_t*t_go at full closing speed -> gap collapses to 0 = kinetic intercept (ram/net), the
        # pattern Ukraine FPV interceptors + Israeli fire-control use. Range from the world estimate so it
        # works even on the frames the detector misses (coast through loss).
        if args.commit and Pt is not None:
            cur_range = float(np.linalg.norm(Pt - ego))
            min_range = min(min_range, cur_range)
            if simt >= args.commit_after:
                phase = "COMMIT"
                t_go = cur_range / max(1.0, args.intercept_speed)
                pip = Pt + Vt * t_go                                # predicted intercept point (lead)
                dirv = pip - ego; n = np.linalg.norm(dirv)
                if n > 1e-3:
                    cmd = dirv / n * args.intercept_speed
                    vx, vy, vz = float(cmd[0]), float(cmd[1]), float(cmd[2])
                if cur_range < args.hit_radius and not hit:
                    hit = True; hit_t = simt
                    print(f"[INTERCEPT] HIT at t={simt:.1f}s range={cur_range:.2f}m "
                          f"(target world {Pt.round(1)}, chaser {ego.round(1)})", flush=True)

        # accel-limit -> steady platform (looser in COMMIT so the terminal dash can accelerate)
        fa = 12.0 if phase == "COMMIT" else FWD_ACC
        va = 10.0 if phase == "COMMIT" else VZ_ACC
        vx = prev_vx + float(np.clip(vx - prev_vx, -fa * dt, fa * dt))
        vy = prev_vy + float(np.clip(vy - prev_vy, -fa * dt, fa * dt))
        vz = prev_vz + float(np.clip(vz - prev_vz, -va * dt, va * dt))
        prev_vx, prev_vy, prev_vz = vx, vy, vz
        ego = ego + np.array([vx, vy, vz]) * dt
        ego[2] = max(0.5, ego[2])
        set_pose(node, "chaser", ego[0], ego[1], ego[2])
        log.append((tt, d is not None, ex, ey, rng, ego.copy(), tgt.copy()))
        if hit:
            break

        # annotated video
        fr = img[:, :, ::-1].copy()
        if d is not None:
            cv2.circle(fr, (int(d[0]), int(d[1])), 8, (0, 255, 0), 2)
            cv2.rectangle(fr, (int(d[0]-d[2]/2), int(d[1]-d[3]/2)),
                          (int(d[0]+d[2]/2), int(d[1]+d[3]/2)), (0, 255, 0), 2)
        cv2.drawMarker(fr, (W//2, H//2), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
        cv2.putText(fr, f"t={tt:4.1f} {'LOCK' if d else 'LOST'} gapV={(tgt[2]-ego[2]):4.1f}m",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if vw is None:
            import os; os.makedirs(args.out, exist_ok=True)
            vw = cv2.VideoWriter(f"{args.out}/gz_up_track.mp4",
                                 cv2.VideoWriter_fourcc(*"mp4v"), 12, (W, H))
        vw.write(fr)

    if vw is not None:
        vw.release()
    # ---- metrics ----
    inframe = 100.0 * seen_n / max(1, frames)
    seen = [r for r in log if r[1]]
    cerr = np.mean([math.hypot(r[2], r[3]) for r in seen]) if seen else float('nan')
    vgap = np.array([(r[6][2] - r[5][2]) for r in log])
    vgap_err = np.sqrt(np.mean((vgap - args.gap) ** 2)) if len(vgap) else float('nan')
    hsep = np.array([math.hypot(r[6][0]-r[5][0], r[6][1]-r[5][1]) for r in log])
    print("\n================ GZ UP-CAM TRACK ================")
    print(f"pattern={args.pattern} frames={frames} in-frame={inframe:.1f}% "
          f"center_err={cerr:.3f} vgap_RMS={vgap_err:.2f}m horiz_sep[mean/max]={hsep.mean():.1f}/{hsep.max():.1f}m")
    print(f"video -> {args.out}/gz_up_track.mp4")

    # trajectory plot
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ego_a = np.array([r[5] for r in log]); tgt_a = np.array([r[6] for r in log])
        t_a = np.array([r[0] for r in log])
        fig, (ax, az) = plt.subplots(1, 2, figsize=(14, 6))
        ax.plot(tgt_a[:,0], tgt_a[:,1], '-', color="#2a9d8f", lw=2, label="target")
        ax.plot(ego_a[:,0], ego_a[:,1], '-', color="#0a84ff", lw=2, label="chaser")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.axis("equal"); ax.grid(alpha=.3)
        ax.legend(); ax.set_title(f"Top-down (up-cam track, {args.pattern})")
        az.plot(t_a, tgt_a[:,2], '-', color="#2a9d8f", lw=2, label="target alt")
        az.plot(t_a, ego_a[:,2], '-', color="#0a84ff", lw=2, label="chaser alt")
        az.set_xlabel("t (s)"); az.set_ylabel("Z (m)"); az.grid(alpha=.3); az.legend()
        az.set_title("Altitude (gap held)")
        fig.tight_layout(); fig.savefig(f"{args.out}/gz_up_track_map.png", dpi=110)
        print(f"map -> {args.out}/gz_up_track_map.png")
    except Exception as e:
        print("plot skipped:", e)

if __name__ == "__main__":
    main()
