# Gotchas & lessons learned

Hard-won, practical notes from building the AirSim air-to-air tracking stack. Most of these cost
real debugging time — read this before changing the sim, control, or recording code.

---

## AirSim setup

### `settings.json` must be in the OneDrive Documents path
On this machine the Documents folder is redirected to OneDrive. AirSim looks for
`Documents\AirSim\settings.json`, which resolves to:

```
C:\Users\admin\OneDrive\Documents\AirSim\settings.json
```

A copy at the non-OneDrive `C:\Users\admin\Documents\AirSim\settings.json` may be **ignored**. Make
sure the **OneDrive** copy is the one with the `Ego` + `Target` vehicles, or AirSim will launch with
the wrong (or default) vehicle config. Symptom: vehicle names from the scripts don't exist and the
RPC hard-crashes (see below).

### Never `simSetVehiclePose`/`reset` the Ego — it corrupts the segmentation buffer
Moving the Ego with `simSetVehiclePose`, or calling `reset()`, **corrupts AirSim's segmentation
buffer** — the seg image comes back black/garbage, which silently ruins dataset generation. Keep the
Ego at its spawn `(0,0,0)` facing +X and **teleport only the Target**. Dataset generation requires a
**freshly launched** AirSim so the Ego is untouched.

### Segmentation colormap changes every session — detect "non-background", not a fixed color
AirSim assigns segmentation **colors per session**, so the Target's color is not stable across runs.
Do **not** hard-code a target color. Instead: set a distinct seg **id** on `Target.*`, then at
capture time treat the **most common color as background** (sky/ground) and take the target as the
non-background region. This is what `datasets/generate_airsim_dataset.py::target_mask` does.

### A wrong vehicle name hard-crashes the AirSim binary
`simGetImages(..., vehicle, ...)` with a vehicle name that is **not** in `settings.json` makes the
AirSim server crash, not raise a clean error. Always pass a name that exists (`Ego`, `Target`).

### Use the raw `simGetImages` RPC signature for the 1.8.1 server
The Cosys-AirSim 3.x client wrapper sends an extra `external` arg that the older AirSim 1.8.1 server
rejects. The scripts call `client.client.call("simGetImages", reqs, vehicle, False)` directly (4-arg
form). AirSim structs use `MSGPACK_DEFINE_MAP`, so msgpack keys can arrive as **`str` or `bytes`** —
the decode helpers (`g = lambda d,k: d[k] if k in d else d.get(k.encode())`) handle both. Don't
assume one or the other.

---

## Networking (PX4 in WSL2)

### WSL2 mirrored networking + shared LAN IP for PX4 TCP
PX4 SITL runs in WSL2 while AirSim / Python tools run on Windows. Use WSL2 **mirrored networking
mode** so Windows and WSL2 share the LAN IP and `localhost`; this lets the Windows side reach PX4's
MAVLink TCP/UDP endpoints (e.g. `udpin:0.0.0.0:14540`) without manual port forwarding. Matching
`ROS_DOMAIN_ID` and a WSL-friendly DDS config are needed if you bridge ROS 2 across the boundary —
this is the #1 friction point of the sim-in-the-loop setup.

---

## Video recording

### Record AVI/MJPG with a clean exit — force-kill corrupts MP4
Always write **AVI with the MJPG codec** (`cv2.VideoWriter_fourcc(*"MJPG")`, `.avi`) and ensure the
writer is **`release()`d** in a `finally` block. If the process is **force-killed**, an MP4 (`mp4v`)
file is left with no moov atom and is **corrupt/unplayable**; MJPG/AVI survives an abrupt stop far
better. `acquire_track.py` uses AVI/MJPG for exactly this reason. (Some older scripts still write
mp4 — prefer AVI for anything you might Ctrl-C.)

---

## Control software

### MAVSDK (asyncio) vs cosysairsim (tornado) event-loop conflict → use pymavlink
MAVSDK runs on **asyncio**; cosysairsim's RPC uses **tornado**, and the two event loops clash when
used in the same process — the loop hangs or throws. For any script that talks to **both** AirSim and
PX4 (e.g. `acquire_track.py`), drive PX4 with **pymavlink** (plain blocking sends) instead of MAVSDK.
Use MAVSDK only in standalone PX4-only scripts (`sim/px4/offboard_demo.py`).

### PX4 needs `NAV_DLL_ACT=0` to arm without a GCS
Offboard control from a script with no ground control station connected: set **`NAV_DLL_ACT=0`** so
PX4 doesn't trigger a data-link-loss failsafe and refuses to arm / kicks out of OFFBOARD. Also stream
setpoints **before** requesting OFFBOARD (a few at low rate) so the mode switch is accepted.

### Integral on a heading-relative yaw setpoint causes a spin — use proportional yaw-rate
Putting an **integral** term on a yaw command that is **relative to the current heading** winds up
and makes the drone **spin continuously**. Command yaw as a **proportional rate**
(`yaw_rate = K · bearing_error`, clamped), with no integral on the relative term. The Web UI uses
`K_YR` proportional yaw-rate only. (`acquire_track.py` uses a small bounded integral on the *absolute
bearing* error with a tight `I_MAX` clamp — bounded and on an absolute quantity, which is safe; the
dangerous case is an unbounded integral on a heading-relative setpoint.)

### Small gap = high angular rate
The closer the standoff gap, the **faster the target sweeps across the field of view** for the same
target speed, so the required yaw rate climbs sharply. Very small gaps make tracking jittery and can
saturate the yaw-rate limit. Prefer a larger standoff (e.g. 12–18 m) for steady tracking; tighten
the gap only when you need it. This is also why `acquire_track.py` defaults to an 18 m chase
distance.

---

## Detection / tracking

### Match runtime `imgsz` to the model, keep ByteTrack `persist=True`
Live loops run inference at `imgsz=960` (speed); training is at `imgsz=1280` (accuracy on tiny
targets). Always pass **`persist=True`** to `model.track(...)` so ByteTrack keeps its Kalman state
and track IDs across frames — without it every frame starts fresh and IDs are useless.

### Keep the lock even when ByteTrack reassigns the ID
ByteTrack can swap a target's ID after an occlusion. The Web UI tolerates this: if the locked ID
disappears, it falls back to the **most-centered real detection** rather than dropping the lock, and
only formally re-acquires after the target is gone for a sustained period.
