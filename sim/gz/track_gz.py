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
    """Dark drone on bright sky -> threshold + largest contour. Returns (cx,cy,w,h,conf) or None."""
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    m, s = float(g.mean()), float(g.std())
    thr = max(0, m - 2.2 * s)
    mask = (g < thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 12:                       # too small / noise
        return None
    x, y, w, h = cv2.boundingRect(c)
    return (x + w / 2.0, y + h / 2.0, float(w), float(h), min(1.0, area / 4000.0))

class YoloDet:
    def __init__(self, weights):
        from ultralytics import YOLO
        self.m = YOLO(weights)
    def __call__(self, img):
        r = self.m.predict(img, imgsz=640, conf=0.20, verbose=False)[0]
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
    ap.add_argument("--yolo", default="", help="path to YOLO weights (else classical blob detector)")
    ap.add_argument("--ego0", type=float, nargs=3, default=[0, 0, 3.0])
    ap.add_argument("--tgt0", type=float, nargs=3, default=[0, 0, 18.0])
    ap.add_argument("--out", default="/mnt/c/Users/admin/uav-vio-track/runs/videos")
    args = ap.parse_args()

    import cv2
    node = Node()
    cam = Cam(node)
    depth = DepthCam(node)
    det = YoloDet(args.yolo) if args.yolo else detect_blob
    print(f"[track] detector = {'YOLO '+args.yolo if args.yolo else 'classical dark-blob'}")

    ego = np.array(args.ego0, float)        # chaser world pos (x,y,z), yaw fixed 0
    tbase = np.array(args.tgt0, float)
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
        if d is not None:
            seen_n += 1
            cx, cy, bw, bh, conf = d
            ex = (cx - W / 2) / (W / 2); ey = (cy - H / 2) / (H / 2)
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
        else:
            ex = ey = 0.0; rng = float('nan')
            vx = vy = vz = 0.0                                       # lost -> hold (no spin)

        # accel-limit -> steady platform
        vx = prev_vx + float(np.clip(vx - prev_vx, -FWD_ACC * dt, FWD_ACC * dt))
        vy = prev_vy + float(np.clip(vy - prev_vy, -FWD_ACC * dt, FWD_ACC * dt))
        vz = prev_vz + float(np.clip(vz - prev_vz, -VZ_ACC * dt, VZ_ACC * dt))
        prev_vx, prev_vy, prev_vz = vx, vy, vz
        ego = ego + np.array([vx, vy, vz]) * dt
        ego[2] = max(0.5, ego[2])
        set_pose(node, "chaser", ego[0], ego[1], ego[2])
        log.append((tt, d is not None, ex, ey, rng, ego.copy(), tgt.copy()))

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
