# Architecture — air-to-air visual detection, tracking, and following

This document describes the full closed loop that the project runs today: a simulated **chaser**
drone (the *Ego*) that visually detects, locks, and follows another **target** drone in the air.
The current, working stack runs on **AirSim** (Cosys-AirSim fork) on Windows, with perception by
**YOLO26-seg + ByteTrack** and guidance/control that mixes a Kalman image filter, GPS-assisted
bearing, depth ranging, and FPV body-yaw steering. PX4 / `simple_flight` provide actuation.

> The README also describes an earlier/parallel Isaac Sim + Pegasus + PX4 + VIO plan (see
> `sim/README.md`, `vio/README.md`). That plumbing is documented but the live demos below run on
> AirSim. This file focuses on the AirSim air-to-air pipeline because that is what the Web UI and
> the `sim/airsim/*` scripts exercise.

---

## 1. Pipeline at a glance

```
                          ┌──────────────────────────────────────────────────────────┐
                          │                    AirSim (Unreal: Blocks)                 │
                          │                                                            │
   Target drone  ◄────────┤  Target vehicle (simple_flight)                           │
   flies dynamic          │    moveOnPathAsync( trajectory waypoints )                 │
   trajectory             │                                                            │
                          │  Ego vehicle  ── front_center camera ──┐                   │
                          │    (simple_flight OR PX4 backend)      │                   │
                          └────────────────────────────────────────┼──────────────────┘
                                                                    │ RPC: simGetImages
                          Scene (RGB) + DepthPlanar + Segmentation  │ (one call, same frame)
                                                                    ▼
                          ┌──────────────────────────────────────────────────────────┐
                          │  PERCEPTION                                                │
                          │   YOLO26-seg  best.pt  (fine-tuned on the target drone)    │
                          │   model.track(..., tracker=bytetrack_uav.yaml, persist)    │
                          │   -> boxes + masks + stable track IDs                      │
                          └───────────────┬──────────────────────────────────────────┘
                                          │  detections {cx,cy,conf,id, depth}
                                          ▼
                          ┌──────────────────────────────────────────────────────────┐
                          │  GUIDANCE                                                  │
                          │   • depth shadow filter (is_real)                          │
                          │   • image Kalman (smooth + lead/coast)  smooth_control.py  │
                          │   • GPS-assisted azimuth (keep target in frame)            │
                          │   • approach-to-gap range control (depth or GPS range)     │
                          │   • fusion of vision bearing + known location  fusion.py   │
                          └───────────────┬──────────────────────────────────────────┘
                                          │  vn, ve, vd  +  yaw / yaw-rate
                                          ▼
                          ┌──────────────────────────────────────────────────────────┐
                          │  CONTROL / ACTUATION                                       │
                          │   simple_flight:  moveByVelocityAsync(+ YawMode)           │
                          │   PX4 (offboard): SET_POSITION_TARGET_LOCAL_NED via        │
                          │                   pymavlink (vel + yaw-rate)               │
                          └───────────────┬──────────────────────────────────────────┘
                                          │
                                          └──► Ego flies toward target, holds the GAP, yaws to center
```

The loop runs at roughly 10–15 Hz (camera RPC + inference bound) on the RTX 5070 Ti laptop.

---

## 2. Stage-by-stage

### Stage A — Simulation & sensing (AirSim)

* Two vehicles are defined in `settings.json`: **Ego** (chaser, carries the camera) and
  **Target** (the drone being chased). Each spawns at a fixed offset (see frames below).
* The **Target** is flown along scripted 3-D paths from `sim/airsim/trajectories.py`
  (`circle`, `figure8`, `spiral`, `helix`, `zigzag`, `random`, `line`). Trajectories are pure-math
  generators returning world-NED `(n, e, d)` waypoint lists, fed to `moveOnPathAsync`.
* The **Ego camera** is read with a single `simGetImages` RPC that returns up to three aligned
  buffers from the *same rendered frame*:
  * `Scene` — RGB (returned BGR) for detection,
  * `DepthPlanar` — float depth for ranging and shadow rejection,
  * `Segmentation` — instance ground truth (used only for **dataset generation**, not at runtime).
* Capture helpers live in each script as `grab()` / `get_frame()`; they tolerate msgpack keys
  arriving as `str` or `bytes` and call the raw 4-arg `simGetImages` signature that the
  AirSim 1.8.1 server expects.

### Stage B — Detection + tracking (perception)

* Model: **YOLO26-seg**, fine-tuned on an AirSim auto-labeled drone dataset. Weights:
  `runs/train/airsim_drone/weights/best.pt` (single class `drone`).
* Inference + tracking in one call: `model.track(frame, tracker=bytetrack_uav.yaml, persist=True,
  imgsz=960, conf=0.3)`. `persist=True` keeps ByteTrack's Kalman state across frames so each
  detection carries a **stable track ID**.
* The tracker config (`perception/trackers/bytetrack_uav.yaml`) is tuned for small, often-occluded
  air-to-air targets: low confidence thresholds (`track_high_thresh: 0.15`), a long `track_buffer`
  (60 frames) so a drone that vanishes behind terrain keeps its ID, and a loosened `match_thresh`
  for fast inter-frame motion.
* The CLI front-end `perception/detect_track.py` runs the same pipeline on video/images/webcam/RTSP
  and can dump per-frame `tracks.jsonl` (frame, track_id, cls, conf, xyxy, optional mask RLE).

### Stage C — Guidance

Guidance turns detections into a velocity + heading command. Several strategies are implemented and
share these building blocks:

* **Depth shadow filter** (`is_real` / `is_real_object`): compares the median depth *inside* the
  box to the depth of a surrounding ring. A detection that is *coplanar* with a finite background
  surface AND has near-uniform depth is treated as a painted **shadow** and rejected; anything
  against the sky, or where depth is sparse/uncertain, is accepted (so real distant drones are
  never dropped). The current Web UI keeps depth for *range only* and has shadow rejection disabled
  so it will lock onto any UAV.
* **Image Kalman** (`sim/airsim/smooth_control.py::ImageKalman`): a constant-velocity filter on the
  *normalized* image error `(ex, ey) ∈ [-1,1]`. It smooths bbox jitter, can **lead** (predict a few
  hundred ms ahead to cancel detection+actuation latency), and can **coast** on the prediction
  through brief detection gaps.
* **GPS-assisted azimuth**: the absolute bearing to the target computed from the known world
  positions, `az = atan2(de, dn)`. This provides a steady, jitter-free heading to keep the camera
  pointed at the target so vision can do *fine* centering — and a fallback when vision is lost.
* **Approach-to-gap range control**: closing speed along the bearing is proportional to
  `(range − gap)`, with a deadband and EMA/rate-limiting so the Ego closes decisively to the
  requested standoff and then holds it. Range comes from depth when available, else GPS range.
* **Vision↔location fusion** (`sim/airsim/fusion.py`): pure functions that convert a bbox center to
  a camera bearing (`bbox_to_bearing`, pinhole model), compute relative geometry
  (`relative_location`), and blend the vision heading with the known-location azimuth via a
  **confidence-weighted circular mean** (`fuse_guidance`). It degrades gracefully:
  vision-only → location-only as confidence drops, and either-only when one source is missing.

The guidance modes you will encounter across scripts:

| Mode | Heading source | Range source | Re-acquire | Script |
|------|----------------|--------------|------------|--------|
| Fused | vision + location (weighted) | location | location azimuth | `multi_traj_track.py` |
| Vision-only | bbox bearing | bbox-size estimate | visual yaw search | `servo_track.py --mode visual` |
| Fused servo | bbox bearing | sim pose | location | `servo_track.py --mode fused` |
| GPS-acquire → vision | GPS until in frame, then KF vision | depth, else GPS | GPS slew | `acquire_track.py` |
| Web UI | GPS azimuth (steady) + vision lock | depth, else GPS | GPS yaw | `webui/app.py` |

### Stage D — Control / actuation

Two actuation paths exist; both consume a world-NED velocity plus a heading command:

* **`simple_flight` (AirSim built-in)** — used by the Web UI, `follow_demo.py`, `servo_track.py`,
  `multi_traj_track.py`. Commands go through `moveByVelocityAsync(vn, ve, vd, dt, yaw_mode=…)`.
  `YawMode(is_rate=True, yaw_or_rate)` gives a **yaw-rate** command (used while tracking so the
  whole FPV body slews smoothly to keep the target centered); `YawMode(False, yaw_deg)` gives an
  absolute yaw setpoint.
* **PX4 offboard (`acquire_track.py`)** — the Ego is a PX4-backed vehicle. Setpoints are sent with
  **pymavlink** `SET_POSITION_TARGET_LOCAL_NED` using a type-mask for *velocity + yaw* (acquire/
  climb) or *velocity + yaw-rate* (smooth track). pymavlink is used instead of MAVSDK to avoid an
  event-loop clash (see `GOTCHAS.md`). PX4 must be armed in OFFBOARD with `NAV_DLL_ACT=0`.

Because the camera is **body-fixed** (no gimbal), "centering the target" means **yawing the entire
aircraft** — this is the FPV body-yaw behavior. Yaw is commanded as a *rate* (proportional, no
integral on a heading-relative setpoint) to avoid a runaway spin (see `GOTCHAS.md`).

---

## 3. Coordinate frames

All math is in **world NED** (North-East-Down), meters:

```
x = North      y = East      z = Down
altitude H above ground  ==  z = -H   (up is negative Down)
yaw 0 = North, +yaw toward East   (yaw = atan2(East, North))
```

### Per-vehicle spawn offsets (HOMES)

`simGetVehiclePose(name)` returns each vehicle's pose **relative to that vehicle's own spawn
point**, not a shared world origin. To compare Ego and Target positions you must add back each
vehicle's spawn offset. The scripts encode these as constants that mirror `settings.json`:

```python
EGO_HOME    = (0.0, 0.0, 0.0)   # Ego spawns at the world origin, facing +X (North)
TARGET_HOME = (8.0, 0.0, 0.0)   # Target spawns 8 m North of Ego
# world_pos(name) = HOMES[name] + simGetVehiclePose(name).position
```

`follow_demo.py` exposes these as a `HOMES` dict; the other scripts use `EGO_HOME` / `TARGET_HOME`
arrays. **If you change spawn positions in `settings.json`, update these constants** or the
GPS-assisted azimuth and range will be wrong.

### Image frame and bearings

* Pixel origin is top-left; `cx` increases to the right, `cy` increases downward.
* Normalized image error: `ex = (cx − W/2)/(W/2)`, `ey = (cy − H/2)/(H/2)`.
* Camera bearing from a bbox uses a pinhole model with `HFOV = 90°`; vertical FOV is derived from
  the aspect ratio (`tan(vfov/2) = tan(hfov/2)·H/W`). `bbox_to_bearing` returns
  `(yaw_err, pitch_err)` where `yaw_err > 0` ⇒ target to the **right**, `pitch_err > 0` ⇒ target
  **above** center.

---

## 4. Data flow summary

1. Target flies a `trajectories.py` path via `moveOnPathAsync`.
2. Ego camera → `simGetImages` → `Scene` (+ `DepthPlanar`, + `Segmentation` for dataset gen).
3. `YOLO26-seg.track()` → boxes/masks + stable track IDs.
4. Depth filter → keep real detections; attach per-box depth (range).
5. Image Kalman smooths/leads the chosen detection's image error.
6. GPS azimuth + depth/GPS range + (optionally) fused vision bearing → desired heading + closing
   speed + climb/descend rate.
7. `simple_flight` velocity+yaw command, or PX4 offboard NED setpoint via pymavlink.
8. Ego converges to the set **gap** behind the target and holds, re-centering by body yaw; loop.

---

## 5. Where each piece lives

| Concern | File |
|---------|------|
| Trajectory generators (NED) | `sim/airsim/trajectories.py` |
| Smoothing / Kalman / deadband / rate-limit | `sim/airsim/smooth_control.py` |
| Vision↔location fusion (pure math) | `sim/airsim/fusion.py` |
| Detection + ByteTrack CLI | `perception/detect_track.py` |
| Tracker tuning | `perception/trackers/bytetrack_uav.yaml` |
| Fine-tuned model | `runs/train/airsim_drone/weights/best.pt` |
| Web UI (current best entrypoint) | `webui/app.py`, `webui/templates/index.html` |
| GPS-acquire → vision-lock (PX4) | `sim/airsim/acquire_track.py` |
| Multi-trajectory fused follow | `sim/airsim/multi_traj_track.py` |
| Visual-servo (vision-only / fused) | `sim/airsim/servo_track.py` |
| Simple 2-drone follow demo | `sim/airsim/follow_demo.py` |
| PX4 offboard square (MAVSDK) | `sim/px4/offboard_demo.py` |
| EKF2 external-vision params (Phase 3) | `px4_config/ekf2_vision.params` |
