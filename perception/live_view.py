#!/usr/bin/env python3
"""LIVE looping viewer for a rendered engagement. Serves a video as JS-polled JPEG snapshots so it
plays in the browser reliably. Default: the Isaac chaser-vs-target combined view.

  python perception/live_view.py [--src runs/videos/chase_combined.mp4] [--port 5071] [--fps 20]
"""
import argparse, time, threading
import cv2
from flask import Flask, Response

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="runs/videos/chase_combined.mp4")
ap.add_argument("--port", type=int, default=5071)
ap.add_argument("--fps", type=float, default=20.0)
ap.add_argument("--title", default="JetRay - Isaac Sim: CHASER vs TARGET (onboard vision-guided)")
args = ap.parse_args()

latest = {"jpg": None}

def worker():
    while True:
        cap = cv2.VideoCapture(args.src)
        if not cap.isOpened():
            time.sleep(1); continue
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            latest["jpg"] = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
            time.sleep(1.0 / args.fps)
        cap.release()

app = Flask(__name__)

@app.route("/")
def index():
    return ('<html><head><title>' + args.title + '</title></head>'
            '<body style="margin:0;background:#111;text-align:center;font-family:sans-serif">'
            '<h2 style="color:#0ff;margin:8px">' + args.title + '</h2>'
            '<img id="v" style="max-width:99vw;border:1px solid #333">'
            '<p id="s" style="color:#888">connecting...</p>'
            '<script>var im=document.getElementById("v"),s=document.getElementById("s"),k=0;'
            'function tick(){var n=new Image();'
            'n.onload=function(){im.src=n.src;k++;s.textContent="live  frame "+k;setTimeout(tick,50);};'
            'n.onerror=function(){s.textContent="waiting...";setTimeout(tick,400);};'
            'n.src="/snapshot?t="+Date.now();}tick();</script></body></html>')

@app.route("/snapshot")
def snapshot():
    j = latest["jpg"]
    return Response(j, mimetype="image/jpeg") if j is not None else Response(b"", status=503)

threading.Thread(target=worker, daemon=True).start()
print(f"LIVE_VIEW_UP -> http://localhost:{args.port}")
app.run(host="0.0.0.0", port=args.port, threaded=True)
