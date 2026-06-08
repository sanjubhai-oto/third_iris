#!/usr/bin/env python3
"""Take off the PX4-backed chaser (Ego) to a set altitude, then hold for manual/virtual-RC flight.

  python sim/px4/ego_takeoff.py --alt 10
"""
import argparse
import asyncio
from mavsdk import System


async def run(alt):
    d = System()
    await d.connect(system_address="udpin://0.0.0.0:14540")
    print("[info] connecting to PX4 (Ego)...")
    async for s in d.core.connection_state():
        if s.is_connected:
            break
    async for h in d.telemetry.health():
        if h.is_armable and h.is_global_position_ok:
            break
        await asyncio.sleep(1)
    await d.action.set_takeoff_altitude(float(alt))
    await d.action.arm()
    await d.action.takeoff()
    print(f"[ok] taking off to {alt} m")
    await asyncio.sleep(14)
    print("[done] at altitude, holding. Use QGC Virtual Joystick (Position mode) to fly the chaser.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt", type=float, default=10.0)
    asyncio.run(run(ap.parse_args().alt))
