#!/usr/bin/env python3
"""Configure PX4 EKF2 for external-vision (VIO) navigation and disable GPS, then reboot.

  python sim/px4/set_vio_params.py            # switch to vision/GPS-denied
  python sim/px4/set_vio_params.py --restore  # back to GPS

EKF2 aiding-source params take effect after an autopilot reboot (done here).
"""
from __future__ import annotations

import argparse
import time

from pymavlink import mavutil

INT32 = mavutil.mavlink.MAV_PARAM_TYPE_INT32
REAL32 = mavutil.mavlink.MAV_PARAM_TYPE_REAL32

VIO = [
    ("EKF2_GPS_CTRL", 0, INT32),     # disable all GPS aiding
    ("EKF2_EV_CTRL", 11, INT32),     # external vision: horiz pos + vert pos + yaw
    ("EKF2_HGT_REF", 3, INT32),      # height reference = vision
    ("EKF2_EV_DELAY", 5.0, REAL32),  # vision latency (ms); GT bridge ~ small
]
GPS = [
    ("EKF2_GPS_CTRL", 7, INT32),     # default GPS aiding
    ("EKF2_EV_CTRL", 0, INT32),
    ("EKF2_HGT_REF", 1, INT32),      # height reference = GPS
]


def setp(m, name, val, ptype):
    m.mav.param_set_send(m.target_system, m.target_component, name.encode(), float(val), ptype)
    # read back
    for _ in range(20):
        ack = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=2)
        if ack and ack.param_id == name:
            print(f"  {name} = {ack.param_value}")
            return
    print(f"  [warn] no ack for {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="udpin://0.0.0.0:14540")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--no-reboot", action="store_true")
    args = ap.parse_args()

    m = mavutil.mavlink_connection(args.url.replace("udpin://", "udpin:"))
    print("waiting for heartbeat...")
    m.wait_heartbeat()
    print(f"connected sysid={m.target_system}")

    params = GPS if args.restore else VIO
    print("setting params:", "GPS" if args.restore else "VIO/GPS-denied")
    for name, val, ptype in params:
        setp(m, name, val, ptype)

    if not args.no_reboot:
        print("rebooting autopilot to apply EKF2 aiding change...")
        m.reboot_autopilot()
        time.sleep(2)
    print("done")


if __name__ == "__main__":
    main()
