#!/usr/bin/env python3
"""Shared PX4-SITL offboard helpers for the air-to-air scripts. The key robustness fix vs the one-shot
vio_sitl pattern: in a cold multi-vehicle EKF, the vehicle isn't "Ready for takeoff" for ~10-20 s, so a
single arm command at t=1.5 s is denied. arm_offboard() RETRIES OFFBOARD+arm until the heartbeat reports
armed (caller must already be streaming setpoints at >=2 Hz on a daemon thread)."""
import time
from pymavlink import mavutil


def set_offboard(m):
    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0, 1, 6, 0, 0, 0, 0, 0)

def send_arm(m):
    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)

def arm_offboard(m, timeout=60.0, label="veh"):
    """Drain heartbeats, retry OFFBOARD+arm ~1 Hz until armed or timeout. Returns True if armed."""
    t0 = time.time(); last = 0.0
    while time.time() - t0 < timeout:
        m.recv_match(type="HEARTBEAT", blocking=False)
        if m.motors_armed():
            print(f"[{label}] ARMED + OFFBOARD ({time.time()-t0:.0f}s)", flush=True)
            return True
        if time.time() - last > 1.0:
            set_offboard(m); send_arm(m); last = time.time()
        time.sleep(0.1)
    print(f"[{label}] WARN: not armed after {timeout:.0f}s", flush=True)
    return m.motors_armed()
