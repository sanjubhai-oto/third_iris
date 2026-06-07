"""Throwaway: isolate whether AirSim crashes only on image capture (GPU render path)."""
import time
import traceback

import cosysairsim as airsim


def try_call(c, label, method, *args):
    try:
        r = c.client.call(method, *args)
        print(f"{label} OK -> {str(r)[:80]}", flush=True)
        return True
    except Exception as e:
        print(f"{label} FAILED: {e}", flush=True)
        return False


def main():
    c = airsim.MultirotorClient(ip="127.0.0.1")
    time.sleep(4)
    # Non-rendering RPCs first
    try_call(c, "ping", "ping")
    try_call(c, "listVehicles", "listVehicles")
    try_call(c, "simGetVehiclePose", "simGetVehiclePose", "Ego")
    # Now the rendering RPC that we suspect crashes the binary
    reqs = [airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)]
    try_call(c, "simGetImages", "simGetImages", reqs, "Ego", False)
    # If we get here, check whether the binary is still alive
    try_call(c, "ping_after_image", "ping")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
