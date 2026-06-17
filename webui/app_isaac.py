#!/usr/bin/env python3
"""Isaac-strike webui: same dashboard front-end (templates/index.html) as the GZ build, but fed by the
Isaac Sim top-attack strike. Shows the live CHASER POV (left video panel) + the dual-drone 3-D iso MAP
(chaser blue vs target green trajectories) — the same 3D mapping we had on the GZ webui.

Reads files written by sim/isaac/chase_sim6.py (run it with ISAAC_LOOP=1 for a live repeating feed):
  runs/videos/latest_chase.jpg   -> /video_feed
  runs/videos/latest_telem.json  -> /telemetry  (ego/target world pos -> 3D map)

Run:  .venv\\Scripts\\python.exe webui\\app_isaac.py        # open http://localhost:5060
"""
import os, json, time
from flask import Flask, render_template, Response, request, jsonify

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "runs", "videos")
JPG = os.path.join(OUT, "latest_chase.jpg")
SPEC = os.path.join(OUT, "latest_spec.jpg")
TELEM = os.path.join(OUT, "latest_telem.json")
PORT = int(os.environ.get("ISAAC_WEBUI_PORT", "5060"))

app = Flask(__name__, template_folder=os.path.join(HERE, "templates"),
            static_folder=os.path.join(HERE, "static"))


def _read(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None


@app.route("/")
def index():
    return render_template("index.html")


def _feed(path):
    def gen():
        while True:
            f = _read(path)
            if f:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n")
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed")
def video_feed(): return _feed(JPG)          # chaser POV (primary panel)

@app.route("/spec_feed")
def spec_feed(): return _feed(SPEC)          # spectator (both drones) if a panel wants it


@app.route("/telemetry")
def telemetry():
    try:
        with open(TELEM) as f:
            t = json.load(f)
    except Exception:
        t = None
    if not t:
        return jsonify({"state": "DETECT", "locked": False, "nav_source": "VISION+PN",
                        "autolock": True, "search": False, "manual_mode": False, "jammed": False,
                        "fps": 0.0, "gap": 0.0, "speed": 10.0, "strike": True, "last_hit": "--"})
    ego, tgt = t["ego"], t["tgt"]
    hit = t.get("hit"); rng = t.get("range", 0.0)
    # ENU map fields: n=x, e=y, d=-z (so altitude reads up). state TRACK -> the 3D map records both paths.
    return jsonify({
        "state": "STRIKE" if hit else "TRACK",
        "autolock": True, "search": False, "manual_mode": False,
        "locked": bool(t.get("locked")), "nav_source": "VISION+PN", "jammed": False,
        "fps": 20.0,
        "ego_n": ego[0], "ego_e": ego[1], "ego_d": -ego[2],
        "tgt_n": tgt[0], "tgt_e": tgt[1], "tgt_d": -tgt[2],
        "gap": round(rng, 1), "speed": 10.0, "strike": True,
        "last_hit": (f"HIT @ {rng:.2f}m" if hit else f"closing  tgo {t.get('tgo',0):.1f}s"),
    })


# ---- control routes the dashboard calls: accepted as no-ops (the strike is autonomous PN guidance) ----
def _ok(**k): return jsonify(ok=True, **k)
for _r in ("/set_autolock", "/set_search", "/set_manual", "/set_jam", "/set_speed", "/set_gap",
           "/set_mode", "/set_avoid", "/set_strike_mode", "/strike", "/abort_strike",
           "/set_video_source", "/set_telem_source", "/set_capture", "/land",
           "/select", "/strike_select", "/clear"):
    app.add_url_rule(_r, _r, (lambda: _ok()), methods=["POST"])

if __name__ == "__main__":
    print(f"[app_isaac] open http://localhost:{PORT}", flush=True)
    app.run(host="0.0.0.0", port=PORT, threaded=True)
