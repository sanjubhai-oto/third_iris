#!/usr/bin/env python3
"""PX4 offboard navigation demo via MAVSDK.

Connects to PX4 SITL, arms, enters OFFBOARD mode, flies a small NED square, and lands.
Works against the running PX4 SITL over MAVLink (onboard port 14540). With WSL2 mirrored
networking this is reachable from Windows on localhost.

  python sim/px4/offboard_demo.py                       # GPS baseline
  python sim/px4/offboard_demo.py --require local       # GPS-denied / VIO (needs vision pose)

Watch it move live in QGroundControl.
"""
from __future__ import annotations

import argparse
import asyncio

from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw


async def wait_connected(drone):
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[ok] connected to PX4")
            return


async def wait_ready(drone, require):
    """require: 'global' (GPS) or 'local' (vision/VIO)."""
    async for h in drone.telemetry.health():
        pos_ok = h.is_global_position_ok if require == "global" else h.is_local_position_ok
        if h.is_armable and pos_ok:
            print(f"[ok] armable, {require} position OK")
            return
        print(f"  waiting: armable={h.is_armable} global_ok={h.is_global_position_ok} "
              f"local_ok={h.is_local_position_ok} home={h.is_home_position_ok}")
        await asyncio.sleep(1)


async def run(args):
    drone = System()
    await drone.connect(system_address=args.url)
    print(f"[info] connecting via {args.url} ...")
    await asyncio.wait_for(wait_connected(drone), timeout=30)
    await asyncio.wait_for(wait_ready(drone, args.require), timeout=60)

    print("[info] arming")
    await drone.action.arm()

    # OFFBOARD requires a setpoint stream before/at start.
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"[err] offboard start failed: {e._result.result}; disarming")
        await drone.action.disarm()
        return
    print("[ok] OFFBOARD engaged")

    alt = -args.alt  # NED: down is negative
    legs = [(0, 0, "climb"), (args.size, 0, "north"), (args.size, args.size, "east"),
            (0, args.size, "south"), (0, 0, "home")]
    for n, e, name in legs:
        print(f"[nav] -> {name} N={n} E={e} D={alt}")
        await drone.offboard.set_position_ned(PositionNedYaw(float(n), float(e), alt, 0.0))
        await asyncio.sleep(args.dwell)

    print("[info] landing")
    await drone.offboard.stop()
    await drone.action.land()
    async for in_air in drone.telemetry.in_air():
        if not in_air:
            break
    print("[done] landed & disarmed")


def main():
    p = argparse.ArgumentParser(description="PX4 offboard square flight (MAVSDK)")
    p.add_argument("--url", default="udpin://0.0.0.0:14540", help="PX4 MAVLink onboard endpoint")
    p.add_argument("--require", choices=["global", "local"], default="global",
                   help="position source gate: global=GPS, local=vision/VIO")
    p.add_argument("--alt", type=float, default=3.0, help="flight altitude (m AGL)")
    p.add_argument("--size", type=float, default=5.0, help="square side length (m)")
    p.add_argument("--dwell", type=float, default=7.0, help="seconds per leg")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
