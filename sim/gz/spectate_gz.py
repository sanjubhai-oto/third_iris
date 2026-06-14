#!/usr/bin/env python3
"""Live 3D view of the colosseum sim in the BROWSER (WSLg won't paint the native gz window). Streams the
spectator camera (/spectator) as MJPEG and drives the engagement (target orbits, chaser tracks then
climb-strikes, resets). Open http://<wsl-ip>:5002 . Root-runnable (flask + gz-transport + numpy)."""
import time, math, threading
import numpy as np
from flask import Flask, Response
from gz.transport13 import Node
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.image_pb2 import Image

WORLD = "uav_colosseum"
app = Flask(__name__)
S = {"frame": None, "phase": "TRACK"}

def run():
    import cv2
    node = Node()
    def sp(n, p, yaw=0.0):
        r = Pose(); r.name = n
        r.position.x, r.position.y, r.position.z = float(p[0]), float(p[1]), float(p[2])
        r.orientation.z = math.sin(yaw/2); r.orientation.w = math.cos(yaw/2)
        node.request(f"/world/{WORLD}/set_pose", r, Pose, Boolean, 300)
    last_img = {}
    node.subscribe(Image, "/spectator", lambda m: last_img.__setitem__(
        "a", np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)))

    ego = np.array([0.0, 0.0, 6.0]); t0 = time.time(); last = time.time()
    sp("chaser", ego); sp("target", [7, 0, 17])
    while True:
        now = time.time(); dt = min(0.1, now - last); last = now; t = now - t0
        ang = 0.30 * t
        tgt = np.array([7*math.cos(ang), 7*math.sin(ang), 17 + 2*math.sin(0.15*t)])
        sp("target", tgt, yaw=ang + math.pi/2)
        cyc = t % 16.0
        if cyc < 10.0:
            setp = np.array([tgt[0], tgt[1], tgt[2]-6.0]); spd = 6.0; S["phase"] = "TRACK (gap 6m)"
        else:
            setp = tgt.copy(); spd = 9.0; S["phase"] = "STRIKE"
        step = setp - ego; n = float(np.linalg.norm(step))
        if n > 1e-3:
            ego = ego + step/n * min(n, spd*dt)
        ego[2] = max(0.5, ego[2]); sp("chaser", ego)

        img = last_img.get("a")
        if img is not None:
            fr = img[:, :, ::-1].copy(); H, W = fr.shape[:2]
            true_r = float(np.linalg.norm(tgt - ego))
            cv2.rectangle(fr, (0, 0), (W, 40), (0, 0, 0), -1)
            cv2.putText(fr, f"GZ COLOSSEUM  {S['phase']}  chaser-target {true_r:4.1f}m",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            ok, jpg = cv2.imencode(".jpg", fr)
            if ok:
                S["frame"] = jpg.tobytes()
        time.sleep(0.02)

@app.route("/")
def idx():
    return ('<html><body style="margin:0;background:#111;text-align:center">'
            '<h2 style="color:#eee;font-family:sans-serif">GZ Colosseum — live 3D view</h2>'
            '<img src="/stream" style="max-width:99vw"></body></html>')

@app.route("/stream")
def stream():
    def gen():
        while True:
            f = S["frame"]
            if f is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n")
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    threading.Thread(target=run, daemon=True).start()
    print("[spectate_gz] open http://localhost:5002", flush=True)
    app.run(host="0.0.0.0", port=5002, threaded=True)
