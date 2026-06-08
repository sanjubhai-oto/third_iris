#!/usr/bin/env python3
"""'Do all': fly the AirSim/PX4 drone offboard (GPS state) with live vision tracking.

Launches the YOLO26 + ByteTrack vision tracker on the drone camera, then flies an offboard
box via MAVSDK. AirSim renders + simulates, PX4 (WSL2) is the autopilot, QGC monitors.

Prereqs already running:
  - AirSim binary with the PX4 settings.json (Vehicles.PX4 = PX4Multirotor)
  - PX4 SITL:  PX4_SIM_HOSTNAME=<shared-ip> make px4_sitl none_iris   (in WSL2)
  - QGroundControl (optional, for monitoring)

  python sim/fly_with_vision.py
  python sim/fly_with_vision.py --alt 6 --size 10 --no-vision   # flight only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle", default="PX4")
    ap.add_argument("--camera", default="front_center")
    ap.add_argument("--alt", type=float, default=6.0)
    ap.add_argument("--size", type=float, default=10.0)
    ap.add_argument("--dwell", type=float, default=7.0)
    ap.add_argument("--task", choices=["detect", "segment"], default="detect")
    ap.add_argument("--model", default="yolo26n.pt",
                    help="use a UAV-fine-tuned model to actually detect the target drone")
    ap.add_argument("--no-vision", action="store_true")
    args = ap.parse_args()

    vis = None
    if not args.no_vision:
        print("[do-all] starting live vision tracker ...")
        vis = subprocess.Popen([
            PY, str(REPO / "sim" / "airsim" / "airsim_yolo_track.py"),
            "--vehicle", args.vehicle, "--camera", args.camera,
            "--task", args.task, "--model", args.model, "--show", "--device", "0",
        ])
        time.sleep(5)  # let it connect + open the window

    print("[do-all] flying offboard box (GPS state) ...")
    flight = subprocess.run([
        PY, str(REPO / "sim" / "px4" / "offboard_demo.py"),
        "--require", "global", "--alt", str(args.alt),
        "--size", str(args.size), "--dwell", str(args.dwell),
    ])

    print(f"[do-all] flight finished (rc={flight.returncode}).")
    if vis is not None:
        print("[do-all] vision tracker still running; Ctrl-C or close its window to stop.")
        try:
            vis.wait()
        except KeyboardInterrupt:
            vis.terminate()


if __name__ == "__main__":
    main()
