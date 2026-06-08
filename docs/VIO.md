# VIO — GPS-denied self-localization & tracking

This is the **VIO half** of `uav-vio-track`: the chaser estimates its **own** pose from camera + IMU
so it can keep tracking and navigating when **GPS is jammed**.

## Operational concept

| GPS / telemetry | What the system does |
|-----------------|----------------------|
| **Available** | Select the target and track with the world-frame **standoff** guidance (GPS-aided). VIO runs in the background, continuously **anchored to GPS** (zero drift) so it's primed. |
| **Jammed** | The onboard computer switches to **VIO** for absolute position, and holds the target lock with a **pure body-frame visual servo** (drift-immune). Toggle in the UI: *"Simulate GPS jamming → navigate on VIO"*. |

## The estimator — `vio/vio_estimator.py`

Loosely-coupled **RGB-D visual odometry + IMU** (depth gives metric scale → no monocular scale drift):

1. **Visual odometry** — Shi-Tomasi features + Lucas-Kanade flow (with forward-backward consistency
   check) → back-project to 3-D with the DepthPlanar image (`fx=fy=W/2` for HFOV 90) → **Umeyama SVD
   rigid transform (with det reflection fix) inside RANSAC** for the frame-to-frame camera translation.
2. **IMU** — `getImuData` gyro integrates attitude every frame; the accelerometer (gravity-compensated,
   AirSim accel *includes* gravity so `g_ned=[0,0,+9.81]`) **coasts** position through frames where VO
   fails (sky / low texture).
3. **Anchoring** — `anchor(pos, quat)` re-aligns to GPS while it's available; when jammed the estimate
   **free-runs** and drift accumulates.

Frames: camera optical (x-right, y-down, z-fwd) → body via `R_BC=[[0,0,1],[1,0,0],[0,1,0]]`; world NED.

### Measured drift — `vio/vio_test.py`
Prime from GPS for 8 s, then **jam GPS for 22 s** and dead-reckon:
```
VO success : 98 % of jam frames
drift      : 4.5 m after 22 s GPS-denied / 38 m flown  (plateaus mid-jam — bounded, not divergence)
```
Run: `python -u vio/vio_test.py`

## Why the lock survives drift — body-frame visual servo

The critical lesson: while jammed, do **not** reconstruct the target in the (drifting) VIO world frame
and feed it to the velocity-feed-forward Kalman — VIO position jitter becomes phantom *target* velocity
and the chaser flies off (we saw range blow to 115 m).

Instead, when jammed the chaser uses a **pure body-frame visual servo** (`moveByVelocityBodyFrame`):
- **forward/back** = `0.8·(depth_range − gap)` → holds the gap
- **vertical** = `2.2·ey` → climbs/descends to center the target
- **yaw rate** = from the image bearing → centers horizontally

This uses **only the camera** (depth + image), commanded in the body frame, so it needs no world
position or heading and is **immune to VIO drift**. VIO then serves only absolute position awareness /
navigation, not the lock.

### Robust range + loss-aware control (`sim/airsim/range_filter.py`)
A momentary occlusion used to make the depth at the bbox read the far **background** (e.g. 125 m → a
forward surge). Fixed with a research-grounded `RangeFilter` (see [RESEARCH.md](RESEARCH.md)):
- **Foreground depth** — near-cluster percentile over the bbox *interior*, not one pixel
  (ForeSeE / NOVA histogram-mode depth, stable through occlusion).
- **Independent size-range** `r = fy·H/h_px` with the target height **H calibrated against depth while
  GPS is on** — it doesn't jump to background on occlusion, so it cross-checks the depth.
- **Plausibility gate** — reject any range jump beyond the max closing rate; median-of-K backstop.
- **Loss-aware control** — when no trustworthy range this frame, **freeze forward velocity** (the
  range channel is least reliable at loss) and coast on yaw, instead of surging.

Result: in a live 30 s jam the range stayed bounded (**max 22 m vs 125 m before**) around the 12 m gap,
lock maintained.

Measured under sustained jam: **tracked 9/11 samples over ~30 s**, range held 9–16 m around a 12 m gap,
centering error 0.10–0.42 (vs 2/10 before the fix).

## Telemetry / UI
`nav_source` (GPS / VIO), `vio_drift` (m, vs truth — for the demo), an amber **GPS JAMMED — VIO NAV**
banner on the video, and the jam toggle. Backend route `POST /set_jam {jammed: true|false}`.

## GPS-denied waypoint navigation — `vio/vio_nav.py`

Beyond *following a visible target*, the drone can **fly to coordinates with no GPS and no target in
view**, steering purely on the VIO pose. `vio_nav.py` primes + calibrates the compass with GPS on,
then **jams GPS** and flies a 4-waypoint square (incl. an altitude change), logging the *true*
position error vs ground truth at each waypoint.

### Magnetometer yaw-aid (essential)
Pure gyro yaw drifts with no absolute reference, which compounds into position divergence. The
estimator fuses the **magnetometer** (`getMagnetometerData`, tilt-compensated heading, offset
**calibrated against truth at the GPS anchor**) to bound yaw drift (`VIOEstimator.update(mag=...)`).
The comparison is decisive:

| GPS-denied nav (VIO only), ~56 m square | mean error | max error | verdict |
|---|---|---|---|
| **with magnetometer aid** | **4.37 m** | 7.02 m | ✅ USABLE |
| without magnetometer aid | 20.63 m | 40.03 m (diverged) | ❌ HIGH DRIFT |

So GPS-denied point-to-point navigation works (a few metres of error, growing with distance as
odometry drift accumulates), and the magnetometer aid is what keeps it bounded. Run:
`python -u vio/vio_nav.py` (add `--no-mag` to see the divergence).

## Limitations & next steps
- Position error still **grows with distance** (it's odometry — no loop closure); good for tens of
  metres / tens of seconds of GPS denial, not unbounded. Loop closure (ORB-SLAM3) or PX4 EKF2 fusion
  would bound it further.
- **Aggressive chase degrades VO** (large inter-frame motion at ~5 fps + YOLO sharing the CPU) → more
  drift than the gentle standalone test. Running VIO in its own thread at higher rate would help.
- This is a self-contained estimator, **not** PX4 EKF2 fusion. The original Phase-3 plan
  (OpenVINS → `vehicle_visual_odometry` → EKF2) in `vio/README.md` remains the route for on-autopilot
  GPS-denied flight; this module proves the perception + control concept inside AirSim.
