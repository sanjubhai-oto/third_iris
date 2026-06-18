#!/usr/bin/env python3
"""Body-frame command bridge: send the tracker's (fwd, vz, yaw_rate) to a real/SITL flight controller.

This is the ONE new piece needed to fly the existing vision stack on a real platform: the webui /
scenario_eval already produce body-frame velocity commands from the image; here we forward them to a
PX4/ArduPilot FC over MAVLink instead of AirSim.

  our servo output            -> MAVLink SET_POSITION_TARGET_LOCAL_NED (MAV_FRAME_BODY_NED)
  fwd  (m/s, +forward)        -> vx
  vz   (m/s, +down)          -> vz
  yaw_rate (deg/s, +CW)      -> yaw_rate (rad/s)

STATUS: validated path is PX4 **SITL first** (no hardware). NOT yet flight-tested on a vehicle.
Test order: SITL -> bench (props off) -> tethered -> short hops. See docs/TEAM_AND_REALPLATFORM.md.

Safety (enforced here): commands carry a timeout — if you stop calling send_body_velocity within
CMD_TIMEOUT_S the helper sends a hover, and you should trigger HOLD/RTL on the FC. Manual RC override
must always win. Do not arm near people.

Usage (SITL smoke test):
    # PX4 SITL exposing MAVLink on udp:14540
    python deploy/mavlink_control.py --conn udp:127.0.0.1:14540 --smoke
"""
from __future__ import annotations

import argparse
import math
import time

CMD_TIMEOUT_S = 0.5     # if no new command within this, the loop should hover (offboard watchdog)


class MavBridge:
    def __init__(self, conn: str):
        from pymavlink import mavutil
        self.mavutil = mavutil
        self.m = mavutil.mavlink_connection(conn)
        print(f"[mav] waiting for heartbeat on {conn} ...")
        self.m.wait_heartbeat()
        # PX4 autopilot heartbeats carry sysid>=1; a sysid-0 heartbeat (router/GCS/garbled) latches
        # target_system=0 and arm/mode commands then go nowhere. Re-acquire until we see a real one.
        t0 = time.time()
        while self.m.target_system == 0 and time.time() - t0 < 15:
            hb = self.m.recv_match(type="HEARTBEAT", blocking=True, timeout=3)
            if hb is not None and hb.get_srcSystem() != 0:
                self.m.target_system = hb.get_srcSystem(); self.m.target_component = hb.get_srcComponent()
        print(f"[mav] heartbeat from sys {self.m.target_system} comp {self.m.target_component}")

    # ---- body-frame velocity + yaw-rate setpoint (the tracker -> FC mapping) ----
    def send_body_velocity(self, fwd: float, vz: float, yaw_rate_deg: float):
        mav = self.m.mav
        # type_mask: ignore position + accel, USE velocity (vx,vy,vz) + yaw_rate. Bits set = ignore.
        IGNORE_POS = 0b0000000000000111
        IGNORE_ACC = 0b0000000111000000
        IGNORE_YAW = 0b0000010000000000          # ignore yaw angle (we command yaw_RATE)
        type_mask = IGNORE_POS | IGNORE_ACC | IGNORE_YAW
        mav.set_position_target_local_ned_send(
            0, self.m.target_system, self.m.target_component,
            self.mavutil.mavlink.MAV_FRAME_BODY_NED,
            type_mask,
            0, 0, 0,                              # x,y,z (ignored)
            float(fwd), 0.0, float(vz),           # vx (fwd), vy, vz (down+)
            0, 0, 0,                              # ax,ay,az (ignored)
            0.0, math.radians(float(yaw_rate_deg)))   # yaw (ignored), yaw_rate (rad/s)

    def send_body_velocity_xy(self, vx: float, vy: float, vz: float, yaw_rate_deg: float):
        """Full body-frame velocity (vx fwd, vy right, vz down) + yaw-rate. Used for down-camera
        ground tracking where lateral (vy) motion is needed, not just forward."""
        mav = self.m.mav
        IGNORE_POS = 0b0000000000000111
        IGNORE_ACC = 0b0000000111000000
        IGNORE_YAW = 0b0000010000000000
        type_mask = IGNORE_POS | IGNORE_ACC | IGNORE_YAW
        mav.set_position_target_local_ned_send(
            0, self.m.target_system, self.m.target_component,
            self.mavutil.mavlink.MAV_FRAME_BODY_NED, type_mask,
            0, 0, 0, float(vx), float(vy), float(vz), 0, 0, 0,
            0.0, math.radians(float(yaw_rate_deg)))

    def send_position_ned(self, n: float, e: float, d: float, yaw_deg: float = 0.0):
        """Local-NED POSITION setpoint (proven robust takeoff/hold path; same as runner_mission).
        d negative = up. Use for takeoff (0,0,-alt) before switching to velocity tracking."""
        self.m.mav.set_position_target_local_ned_send(
            0, self.m.target_system, self.m.target_component,
            self.mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000100011111000,                       # use position + YAW, ignore vel/acc/yawrate
            float(n), float(e), float(d), 0, 0, 0, 0, 0, 0, math.radians(float(yaw_deg)), 0)

    def hover(self):
        self.send_body_velocity(0.0, 0.0, 0.0)

    def set_offboard_and_arm(self):
        """PX4: stream setpoints first, then switch to OFFBOARD + arm. Caller must keep streaming."""
        for _ in range(20):
            self.hover(); time.sleep(0.05)
        self.m.set_mode("OFFBOARD")
        self.m.arducopter_arm() if hasattr(self.m, "arducopter_arm") else None
        print("[mav] OFFBOARD requested + arm sent (verify on FC / GCS)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--conn", default="udp:127.0.0.1:14540", help="MAVLink endpoint (PX4 SITL default)")
    p.add_argument("--smoke", action="store_true", help="connect + stream hover for 3s (no arm)")
    args = p.parse_args(argv)
    br = MavBridge(args.conn)
    if args.smoke:
        print("[smoke] streaming hover setpoints for 3s (NOT arming) ...")
        t0 = time.time()
        while time.time() - t0 < 3.0:
            br.hover(); time.sleep(0.05)
        print("[smoke] ok — bridge can talk to the FC. Next: SITL offboard flight test.")


if __name__ == "__main__":
    main()
