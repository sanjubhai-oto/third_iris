#!/usr/bin/env python3
"""Auto-label an x500 air-target dataset in Gazebo (no manual labels) to fine-tune YOLO for the
low-contrast gz x500 quad. Uses the capture_cam.sdf rig: a STATIC camera (no drift/self-arm) + a
kinematic x500 'target' teleported around it. Background subtraction isolates ONLY the target -> bbox.

The target is placed at varying range + lateral + VERTICAL offsets so the background varies
(sky / horizon / ground) without moving the camera.

Run:  bash: gz sim -s -r sim/gz/a2a/capture_cam.sdf   (software GL), then:
      python3 datasets/capture_x500_dataset.py --num 800
"""
import argparse, math, os, random, time
os.environ.setdefault("GZ_IP", "127.0.0.1")
import numpy as np, cv2
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.boolean_pb2 import Boolean

REPO = "/mnt/c/Users/admin/uav-vio-track"
OUT = os.path.join(REPO, "datasets", "x500_air")
WORLD = "capworld"; TARGET = "target"
CAM = "/capcam/image"
CAMZ = 8.0; HFOV = 1.74
S = {"img": None, "stamp": 0.0}


def setpose(node, name, x, y, z, yaw=0.0):
    req = Pose(); req.name = name
    req.position.x = float(x); req.position.y = float(y); req.position.z = float(z)
    req.orientation.z = math.sin(yaw/2); req.orientation.w = math.cos(yaw/2)
    node.request(f"/world/{WORLD}/set_pose", req, Pose, Boolean, 300)


def grab():
    t0 = time.time(); s0 = S["stamp"]
    while S["stamp"] == s0 and time.time() - t0 < 2.0:
        time.sleep(0.02)
    return S["img"]


def bbox_from_diff(bg, fg, thr=16, min_area=25):
    d = cv2.absdiff(cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY), cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY))
    m = (d > thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.dilate(m, np.ones((3, 3), np.uint8), 1)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    x, y, w, h = cv2.boundingRect(c)
    H, W = fg.shape[:2]
    if w > 0.6 * W or h > 0.6 * H:
        return None
    return (x, y, w, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=800)
    ap.add_argument("--val", type=float, default=0.15)
    args = ap.parse_args()
    for s in ("train", "val"):
        os.makedirs(os.path.join(OUT, "images", s), exist_ok=True)
        os.makedirs(os.path.join(OUT, "labels", s), exist_ok=True)
    node = Node(); node.subscribe(Image, CAM, lambda m: S.update(
        img=cv2.cvtColor(np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3), cv2.COLOR_RGB2BGR),
        stamp=time.time()))
    rng = random.Random(7)
    print("[cap] waiting for camera ...", flush=True)
    while grab() is None:
        time.sleep(0.2)
    H, W = S["img"].shape[:2]
    vfov = 2*math.atan(math.tan(HFOV/2) * H / W)
    setpose(node, TARGET, 600, 600, 40); time.sleep(0.5)
    bg = grab().copy()
    n = saved = 0
    while n < args.num:
        if n % 120 == 0 and n > 0:                       # refresh background (cam is static; cheap safety)
            setpose(node, TARGET, 600, 600, 40); time.sleep(0.3); bg = grab().copy()
        d = rng.uniform(5, 38)
        la = rng.uniform(-0.42, 0.42) * HFOV
        va = rng.uniform(-0.42, 0.42) * vfov
        tx = d * math.cos(la); ty = d * math.sin(la); tz = max(0.4, CAMZ + d * math.tan(va))
        setpose(node, TARGET, tx, ty, tz, rng.uniform(0, 6.28)); time.sleep(0.16)
        fg = grab()
        if fg is None:
            continue
        n += 1
        bb = bbox_from_diff(bg, fg)
        split = "val" if rng.random() < args.val else "train"
        stem = f"x{n:05d}"
        cv2.imwrite(os.path.join(OUT, "images", split, stem + ".jpg"), fg)
        lbl = os.path.join(OUT, "labels", split, stem + ".txt")
        if bb is not None:
            x, y, w, h = bb
            with open(lbl, "w") as f:
                f.write(f"0 {(x+w/2)/W:.6f} {(y+h/2)/H:.6f} {w/W:.6f} {h/H:.6f}\n")
            saved += 1
        else:
            open(lbl, "w").close()
        if n % 50 == 0:
            print(f"[cap] {n} frames, {saved} labeled", flush=True)
    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write(f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n  0: uav\n")
    print(f"[cap] DONE: {n} frames, {saved} labeled (+{n-saved} neg) -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
