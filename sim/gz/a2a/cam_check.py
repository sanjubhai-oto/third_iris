#!/usr/bin/env python3
"""Stage-3 offline detection check: subscribe the chaser forward camera (/chaser/camera_front), run the
Hybrid YOLO+KLT detector, annotate, and report lock% + achieved camera Hz. Optionally HOLD the chaser at
a fixed vantage via gz set_pose (kinematic) so the flying runner is in its forward FOV for the check.

  # in another shell the runner must be flying (runner_mission.py)
  python3 sim/gz/a2a/cam_check.py --secs 40 --hold "0,0,6,0"
"""
import argparse, math, os, sys, threading, time
os.environ.setdefault("GZ_IP", "127.0.0.1")          # match the gz server's transport scope (else no frames)
import numpy as np, cv2
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
sys.path.insert(0, os.path.dirname(__file__))
from detector import HybridDetector, DEFAULT_WEIGHTS

OUT = "/mnt/c/Users/admin/uav-vio-track/runs/videos"


def hold_chaser(world, model, x, y, z, yaw):
    """Continuously set_pose the chaser to a fixed vantage (kinematic) for the detection check."""
    from gz.msgs10.pose_pb2 import Pose
    from gz.msgs10.boolean_pb2 import Boolean
    node = Node()
    while True:
        req = Pose(); req.name = model
        req.position.x = x; req.position.y = y; req.position.z = z
        req.orientation.z = math.sin(yaw / 2.0); req.orientation.w = math.cos(yaw / 2.0)
        node.request(f"/world/{world}/set_pose", req, Pose, Boolean, 300)
        time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/chaser/camera_front")
    ap.add_argument("--world", default="uav_a2a")
    ap.add_argument("--chaser", default="jetray_chaser_0")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--secs", type=float, default=40.0)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--hold", default="", help='hold chaser at "x,y,z,yaw" (kinematic) for the check')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if args.hold:
        x, y, z, yaw = (float(v) for v in args.hold.split(","))
        threading.Thread(target=hold_chaser, args=(args.world, args.chaser, x, y, z, yaw),
                         daemon=True).start()
        print(f"[cam_check] holding chaser at {args.hold}")

    S = {"bgr": None, "stamp": 0.0}
    def on_img(m):
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)   # RGB
        S["bgr"] = cv2.cvtColor(a, cv2.COLOR_RGB2BGR); S["stamp"] = time.time()
    cam_node = Node(); cam_node.subscribe(Image, args.topic, on_img)   # keep ref (GC kills subscription)
    print(f"[cam_check] subscribed {args.topic}; waiting for frames ...")
    t0 = time.time()
    while S["bgr"] is None and time.time() - t0 < 15:
        time.sleep(0.1)
    if S["bgr"] is None:
        print("[cam_check] ERROR: no camera frames"); return

    det = HybridDetector(args.weights, every=args.every, conf=args.conf, imgsz=args.imgsz)
    n = locks = 0; last_stamp = -1.0; t0 = time.time()
    methods = {}
    while time.time() - t0 < args.secs:
        if S["stamp"] == last_stamp:
            time.sleep(0.005); continue
        last_stamp = S["stamp"]; fr = S["bgr"].copy(); n += 1
        out = det(fr)
        if out:
            locks += 1; methods[out["method"]] = methods.get(out["method"], 0) + 1
            x1 = int(out["cx"] - out["w"] / 2); y1 = int(out["cy"] - out["h"] / 2)
            x2 = int(out["cx"] + out["w"] / 2); y2 = int(out["cy"] + out["h"] / 2)
            c = (0, 255, 0) if out["method"] == "YOLO" else (0, 200, 255)
            cv2.rectangle(fr, (x1, y1), (x2, y2), c, 2)
            cv2.putText(fr, f'TARGET [{out["method"]}] {out["conf"]:.2f}', (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        hz = n / max(1e-3, time.time() - t0)
        cv2.putText(fr, f"cam_check  lock {100*locks/max(1,n):.0f}%  {hz:.1f} Hz  f{n}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tmp = os.path.join(OUT, "latest_chase.tmp.jpg")
        cv2.imwrite(tmp, fr); os.replace(tmp, os.path.join(OUT, "latest_chase.jpg"))
    dt = time.time() - t0
    print(f"[cam_check] frames={n} lock={100*locks/max(1,n):.0f}% cam={n/dt:.1f}Hz methods={methods}")


if __name__ == "__main__":
    main()
