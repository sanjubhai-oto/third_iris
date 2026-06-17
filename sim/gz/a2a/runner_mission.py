#!/usr/bin/env python3
"""RUNNER (target) premade route via PX4 offboard waypoint following. Reuses the proven offboard
pattern from sim/gz/vio_sitl.py: stream local-NED position setpoints at 20 Hz on a daemon thread, switch
to OFFBOARD, arm, then walk a fixed waypoint list. Flies on instance 1 (udp 14541) and DOES NOT touch
the chaser. World truth (gz /pose/info) is printed so you can confirm it actually flew the route.

Frames: PX4 local NED is relative to THIS vehicle's spawn/EKF origin. With the gz<->PX4 mapping
(N=worldY, E=worldX), +E moves the runner along world +X = away from the chaser's forward camera, so the
route below recedes (+E) while weaving in N -> a clean tail-chase target.

  python3 sim/gz/a2a/runner_mission.py                  # default route, loops once
  python3 sim/gz/a2a/runner_mission.py --loops 3 --speed-cruise 6
"""
import argparse, math, os, sys, threading, time
os.environ.setdefault("GZ_IP", "127.0.0.1")          # match the gz server's transport scope
from pymavlink import mavutil
sys.path.insert(0, os.path.dirname(__file__))
from px4_util import arm_offboard

# NED waypoints (n, e, d) relative to runner spawn. d negative = up. Climb, then recede (+E) with a
# gentle serpentine in N. Kept modest so a faster chaser can close.
def default_route(alt):
    # Stay close + slow so the chaser can ACQUIRE at the start (target big in frame), then recede only
    # moderately while weaving. A pursuing chaser keeps the separation small the rest of the way.
    d = -float(alt)
    return [
        (0.0,  0.0, d),          # climb in place (world ~12m ahead of chaser -> acquisition)
        (0.0,  0.0, d),          # hold close (give the chaser time to lock at short range)
        (0.0,  6.0, d),          # ease away slowly
        (4.0, 12.0, d),
        (-4.0, 18.0, d),
        (4.0, 24.0, d - 1.0),
        (-4.0, 30.0, d),
        (0.0, 24.0, d),          # loop back so it never gets too far
        (0.0, 12.0, d),
    ]

def gz_pose_monitor(world, model, state):
    try:
        from gz.transport13 import Node
        from gz.msgs10.pose_v_pb2 import Pose_V
    except Exception as e:
        print(f"[runner] gz pose monitor off ({e})"); return
    node = Node()
    def on_pose(m):
        for p in m.pose:
            if p.name == model:
                state["xyz"] = (p.position.x, p.position.y, p.position.z); return
    node.subscribe(Pose_V, f"/world/{world}/pose/info", on_pose)
    while True:
        time.sleep(0.2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=14541)
    ap.add_argument("--world", default="uav_a2a")
    ap.add_argument("--model", default="jetray_runner_1")
    ap.add_argument("--alt", type=float, default=8.0)
    ap.add_argument("--dwell", type=float, default=5.0, help="seconds per waypoint")
    ap.add_argument("--cruise", type=float, default=4.0, help="MPC_XY_CRUISE (slow so the chaser can catch)")
    ap.add_argument("--loops", type=int, default=1)
    args = ap.parse_args()

    state = {"xyz": None}
    threading.Thread(target=gz_pose_monitor, args=(args.world, args.model, state), daemon=True).start()

    m = mavutil.mavlink_connection(f"udpin:0.0.0.0:{args.port}")
    m.wait_heartbeat(); ts, tc = m.target_system, m.target_component
    print(f"[runner] heartbeat sys {ts} on udp:{args.port}", flush=True)
    # slow the runner so a faster chaser can close (real targets aren't necessarily slow; this keeps the
    # first integration test clean -- raise --cruise later for a harder engagement)
    for name, val in [("MPC_XY_CRUISE", args.cruise), ("MPC_XY_VEL_MAX", args.cruise + 1.0)]:
        m.mav.param_set_send(ts, tc, name.encode(), float(val), mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    time.sleep(0.5)

    SPT = {"n": 0.0, "e": 0.0, "d": -args.alt}
    def send():
        m.mav.set_position_target_local_ned_send(
            0, ts, tc, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111111000, SPT["n"], SPT["e"], SPT["d"], 0, 0, 0, 0, 0, 0, 0, 0)
    def streamer():
        while True:
            send(); time.sleep(0.05)
    threading.Thread(target=streamer, daemon=True).start()

    time.sleep(1.5)                                       # let setpoints flow before mode switch
    arm_offboard(m, timeout=60.0, label="runner")         # retries until EKF ready + armed
    print("[runner] climbing", flush=True)
    time.sleep(8.0)                                       # climb to alt

    route = default_route(args.alt)
    for lap in range(args.loops):
        for (n, e, d) in route:
            SPT["n"], SPT["e"], SPT["d"] = n, e, d
            tw = state["xyz"]
            tws = f" world=({tw[0]:.1f},{tw[1]:.1f},{tw[2]:.1f})" if tw else ""
            print(f"[runner] lap {lap} -> NED({n:.0f},{e:.0f},{d:.0f}){tws}", flush=True)
            time.sleep(args.dwell)
    print("[runner] route done; holding last setpoint (streamer keeps running)", flush=True)
    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    main()
