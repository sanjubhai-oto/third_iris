#!/usr/bin/env python3
"""Standalone LIVE detection+tracking UI for a real camera — NO AirSim, NO sim dependency.

Streams the camera with YOLO boxes + ByteTrack IDs to the browser. Built because the full webui
(app.py) hangs in INIT on a real camera (it runs an AirSim reset/takeoff at startup regardless of
the video source). This is the pure perception view for real-world testing.

Run:
    python webui/live_detect.py --cam 1 --model runs/train/uav_diverse/weights/best.pt
    # then open http://localhost:5001
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, render_template_string
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
# ByteTrack: fast (~16 fps). Once the model is fine-tuned on THIS drone it detects nearly every frame,
# so the id stays stable without the heavy ReID. (BoT-SORT+ReID is ~4 fps — only needed when detection
# is sparse.) Swap to botsort_uav.yaml if you still see id breaks after fine-tuning.
TRACKER = str(REPO / "perception" / "trackers" / "bytetrack_uav.yaml")

app = Flask(__name__)
STATE = {"jpeg": None, "n": 0, "best": 0.0, "fps": 0.0, "hit_pct": 0.0}
CAP = {"on": False, "n": 0, "thresh": 0.15}     # dataset capture: auto-label this drone while rotating
CAP_DIR = REPO / "datasets" / "realdrone"

PAGE = """
<!doctype html><html><head><title>Live Drone Detection</title>
<style>body{margin:0;background:#0d1321;color:#e6edf6;font-family:Segoe UI,system-ui,sans-serif}
.bar{padding:10px 16px;font-size:15px;letter-spacing:.5px}
.bar b{color:#39e6a3}.wrap{display:flex;justify-content:center}img{max-width:100%;height:auto}
.k{color:#7fa7d8}</style></head><body>
<div class="bar"><b>LIVE</b> drone detection + ByteTrack &nbsp;|&nbsp;
<span class="k">detections:</span> <span id="n">0</span> &nbsp;
<span class="k">best conf:</span> <span id="b">0.00</span> &nbsp;
<span class="k">hit-rate:</span> <span id="h">0</span>% &nbsp;
<span class="k">fps:</span> <span id="f">0</span></div>
<div class="wrap"><img src="/stream"></div>
<script>setInterval(async()=>{let r=await fetch('/stat');let s=await r.json();
n.textContent=s.n;b.textContent=s.best.toFixed(2);h.textContent=Math.round(s.hit_pct);f.textContent=s.fps.toFixed(1);},500);</script>
</body></html>
"""


def capture_loop(cam_index, model_path, imgsz, conf):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    for _ in range(15):                       # X5 / UVC warm-up — first reads return nothing
        cap.read(); time.sleep(0.05)
    print(f"[live] camera {cam_index} ready, model={model_path}")
    last = time.time(); hits = []
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.02); continue
        res = model.track(frame, persist=True, imgsz=imgsz, conf=conf,
                          tracker=TRACKER, verbose=False)[0]
        b = res.boxes
        n = 0 if b is None else len(b)
        STATE["n"] = n
        STATE["best"] = float(b.conf.max()) if n else 0.0
        hits.append(1 if n else 0); hits = hits[-60:]
        STATE["hit_pct"] = 100.0 * sum(hits) / max(1, len(hits))
        now = time.time(); STATE["fps"] = 0.9 * STATE["fps"] + 0.1 / max(1e-3, now - last); last = now
        # dataset capture: when ON, save (frame, YOLO label) for every confident box -> teaches THIS
        # drone. Rotate it slowly through all angles + move it around to cover the full pose space.
        if CAP["on"] and n > 0:
            H, W = frame.shape[:2]
            # keep confident-enough boxes that are NOT whole-frame garbage (real drone is small in frame)
            keep = [(xy, c) for xy, c in zip(b.xyxy.tolist(), b.conf.tolist())
                    if c >= CAP["thresh"] and (xy[2]-xy[0]) < 0.55*W and (xy[3]-xy[1]) < 0.60*H]
            if keep:
                (CAP_DIR / "images").mkdir(parents=True, exist_ok=True)
                (CAP_DIR / "labels").mkdir(parents=True, exist_ok=True)
                fn = f"x5_{CAP['n']:05d}"
                cv2.imwrite(str(CAP_DIR / "images" / (fn + ".jpg")), frame)
                lines = []
                for (x1, y1, x2, y2), c in keep:
                    cx = ((x1 + x2) / 2) / W; cy = ((y1 + y2) / 2) / H
                    lines.append(f"0 {cx:.6f} {cy:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
                (CAP_DIR / "labels" / (fn + ".txt")).write_text("\n".join(lines) + "\n")
                CAP["n"] += 1
        ann = res.plot()
        ok2, jpg = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok2:
            STATE["jpeg"] = jpg.tobytes()


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/stat")
def stat():
    from flask import jsonify
    d = {k: STATE[k] for k in ("n", "best", "hit_pct", "fps")}
    d["cap_on"] = CAP["on"]; d["cap_n"] = CAP["n"]
    return jsonify(d)


@app.route("/capture/<int:on>")
def capture(on):
    from flask import jsonify
    CAP["on"] = bool(on)
    return jsonify({"capture": CAP["on"], "cap_n": CAP["n"]})


@app.route("/stream")
def stream():
    def gen():
        while True:
            j = STATE["jpeg"]
            if j is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n")
            time.sleep(0.03)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=1)
    ap.add_argument("--model", default=str(REPO / "runs/train/uav_diverse/weights/best.pt"))
    ap.add_argument("--imgsz", type=int, default=1280)   # bigger -> catches small/odd-angle poses
    ap.add_argument("--conf", type=float, default=0.12)   # low -> weak-angle detections still fire
    ap.add_argument("--port", type=int, default=5001)
    args = ap.parse_args()
    threading.Thread(target=capture_loop,
                     args=(args.cam, args.model, args.imgsz, args.conf), daemon=True).start()
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
