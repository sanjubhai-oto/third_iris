#!/usr/bin/env python3
"""3-drone air-to-air dashboard: live CHASER POV feed + a 3-D trajectory map of ALL drones (chaser +
two targets), the AirSim-style live path view. Reads files written by the a2a engagement:
  runs/videos/latest_chase.jpg       -> /video_feed (chaser POV)
  runs/videos/latest_telem_a2a.json  -> /telemetry  (per-drone world pose -> 3D paths)

Run:  .venv\\Scripts\\python.exe webui\\app_a2a.py     # open http://localhost:5062
"""
import os, time
from flask import Flask, render_template, Response, jsonify

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "runs", "videos")
JPG = os.path.join(OUT, "latest_chase.jpg")
TELEM = os.path.join(OUT, "latest_telem_a2a.json")
PORT = int(os.environ.get("A2A_WEBUI_PORT", "5062"))

app = Flask(__name__, template_folder=os.path.join(HERE, "templates"))


@app.route("/")
def index():
    return render_template("a2a.html")


@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            try:
                with open(JPG, "rb") as f:
                    frame = f.read()
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            except Exception:
                pass
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/telemetry")
def telemetry():
    try:
        with open(TELEM) as f:
            return Response(f.read(), mimetype="application/json")
    except Exception:
        return jsonify({"t": 0, "drones": []})


if __name__ == "__main__":
    print(f"[app_a2a] open http://localhost:{PORT}", flush=True)
    app.run(host="0.0.0.0", port=PORT, threaded=True)
