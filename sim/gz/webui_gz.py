#!/usr/bin/env python3
"""Live browser detection feed for the GZ up-camera (a webui you open in Windows: http://localhost:8088).
Subscribes to /jetray/camera_up, drives the 3 target drones on gentle patterns, runs YOLO+ByteTrack, and
MJPEG-streams the annotated frames. WSL2 forwards localhost to Windows, so just open the URL.

Run in WSL (gz already up on the team world):
  LIBGL_ALWAYS_SOFTWARE=1 python3 sim/gz/webui_gz.py --yolo <weights>
"""
import argparse, math, time, threading
import numpy as np
from flask import Flask, Response
from gz.transport13 import Node
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.image_pb2 import Image

WORLD = "uav_track_team"
app = Flask(__name__)
state = {"frame": None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--port", type=int, default=8088)
    args = ap.parse_args()
    import cv2
    from ultralytics import YOLO

    node = Node()
    def sp(n, x, y, z):
        r = Pose(); r.name = n; r.position.x = x; r.position.y = y; r.position.z = z; r.orientation.w = 1.0
        node.request(f"/world/{WORLD}/set_pose", r, Pose, Boolean, 300)
    last = {"img": None}
    def cb(m):
        last["img"] = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
    node.subscribe(Image, "/jetray/camera_up", cb)
    model = YOLO(args.yolo)
    bases = {"target_red": (3, 2, 11), "target_green": (-3, 2, 11), "target_blue": (2, -3, 11)}
    sp("chaser", 0, 0, 6)
    COLORS = {}

    def worker():
        t0 = time.time()
        while True:
            t = time.time() - t0
            sp("target_red",   3 + 2.0*math.cos(0.4*t),  2 + 2.0*math.sin(0.4*t),  11)
            sp("target_green", -3 + 1.8*math.sin(0.5*t), 2 + 1.8*math.cos(0.3*t),  11)
            sp("target_blue",  2 + 1.5*math.sin(0.7*t),  -3 + 2.0*math.sin(0.35*t), 11)
            img = last["img"]
            if img is None:
                time.sleep(0.05); continue
            res = model.track(img, persist=True, imgsz=args.imgsz, conf=args.conf,
                              tracker="bytetrack.yaml", verbose=False)[0]
            fr = img[:, :, ::-1].copy(); H, W = fr.shape[:2]; n = 0
            b = res.boxes
            if b is not None and b.id is not None:
                for i in range(len(b)):
                    x1, y1, x2, y2 = b.xyxy[i].cpu().numpy().tolist(); tid = int(b.id[i])
                    c = COLORS.setdefault(tid, tuple(int(v) for v in np.random.randint(60, 255, 3)))
                    cv2.rectangle(fr, (int(x1), int(y1)), (int(x2), int(y2)), c, 3)
                    cv2.putText(fr, f"DRONE {tid}", (int(x1), int(y1)-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2); n += 1
            cv2.rectangle(fr, (0, 0), (W, 44), (0, 0, 0), -1)
            cv2.putText(fr, f"GZ UP-CAM  YOLO+ByteTrack  -  tracking {n} drones   t={t:5.1f}s",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            ok, jpg = cv2.imencode(".jpg", fr)
            if ok:
                state["frame"] = jpg.tobytes()
            time.sleep(0.01)
    threading.Thread(target=worker, daemon=True).start()

    @app.route("/")
    def index():
        return ('<html><body style="margin:0;background:#111;text-align:center">'
                '<h2 style="color:#eee;font-family:sans-serif">GZ Sim — UAV detection feed</h2>'
                '<img src="/stream" style="max-width:98vw"></body></html>')

    @app.route("/stream")
    def stream():
        def gen():
            while True:
                f = state["frame"]
                if f is not None:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n")
                time.sleep(0.05)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    print(f"[webui_gz] open http://localhost:{args.port}  (Ctrl-C to stop)", flush=True)
    app.run(host="0.0.0.0", port=args.port, threaded=True)

if __name__ == "__main__":
    main()
