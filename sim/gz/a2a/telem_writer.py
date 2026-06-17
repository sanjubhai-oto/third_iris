#!/usr/bin/env python3
"""Publish live world poses of ALL drones for the 3-drone webui 3D map. Subscribes gz /pose/info and
writes runs/videos/latest_telem_a2a.json continuously:

  {"t": <step>, "drones": [{"name","role","color","x","y","z"}, ...]}

Roles/colors: chaser (blue), target1 (green), target2 (orange). The webui (app_a2a.py) draws one
trajectory per drone. Run alongside the engagement.

  python3 sim/gz/a2a/telem_writer.py
"""
import argparse, json, os, time
os.environ.setdefault("GZ_IP", "127.0.0.1")
from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V

OUT = "/mnt/c/Users/admin/uav-vio-track/runs/videos"
ROLES = {                                   # gz model name -> (role label, hex color)
    "jetray_chaser_0": ("chaser",  "#3aa0ff"),
    "jetray_runner_1": ("target1", "#39e6a3"),
    "jetray_runner_2": ("target2", "#ffae3a"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="uav_a2a")
    ap.add_argument("--hz", type=float, default=10.0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    S = {}
    def on_pose(m):
        for p in m.pose:
            if p.name in ROLES:
                S[p.name] = (p.position.x, p.position.y, p.position.z)
    node = Node(); node.subscribe(Pose_V, f"/world/{args.world}/pose/info", on_pose)
    print(f"[telem_writer] subscribed /world/{args.world}/pose/info", flush=True)

    out = os.path.join(OUT, "latest_telem_a2a.json")
    n = 0
    while True:
        n += 1
        drones = []
        for name, (role, color) in ROLES.items():
            if name in S:
                x, y, z = S[name]
                drones.append({"name": name, "role": role, "color": color,
                               "x": round(x, 3), "y": round(y, 3), "z": round(z, 3)})
        if drones:
            with open(out + ".tmp", "w") as f:
                json.dump({"t": n, "drones": drones}, f)
            os.replace(out + ".tmp", out)
        time.sleep(1.0 / args.hz)


if __name__ == "__main__":
    main()
