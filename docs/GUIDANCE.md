# Guidance — standoff air-to-air follow (no spin, speed-matched)

`sim/airsim/guidance.py` is the controller shared by the Web UI (`webui/app.py`) and the scenario
tester (`sim/airsim/scenario_eval.py`). It replaced an earlier radial+yaw law that **spun in a
circle, lagged a moving target, and lost it**.

## Why the old law failed

The old controller only moved the chaser **radially** (toward/away along the bearing) and **yawed**
to center the target. A target moving *across* the frame could then only be followed by yawing the
whole aircraft — so the chaser traced a **pursuit spiral** and always lagged by an error
proportional to the target's speed. There was no lateral (cross-track) velocity, no velocity
feed-forward, and the vertical channel used GPS (not something a real drone has for the target).

## The standoff / leader-follower law

1. **Estimate the target's 3-D position from the camera alone** — `target_from_vision()` turns the
   bbox image error (`ex`, `ey`) and the **DepthPlanar** range into a world-NED point using the
   pinhole relation (`lateral/forward = ex·tan(HFOV/2)`, `vertical/forward = ey·tan(VFOV/2)`) rotated
   by the ego heading. (Modes can instead use GPS, or a confidence blend — see below.)
2. **Kalman position + velocity** — `Vec3KF` (constant-velocity, per-axis) smooths the estimate,
   produces a **velocity** estimate, and **coasts** (predicts) through frames where the target is
   momentarily not detected. The velocity is hard-clamped (`vmax`) so it can never run away.
3. **Velocity command = position-P + target-velocity feed-forward** — `standoff_command()`:
   `v_xy = kp_pos·(p_des − ego) + target_velocity`, where `p_des = target − gap·LOŜ`.
   The feed-forward makes the chaser **fly alongside the target, matching its speed**; only the
   position error is closed by the P term, so it no longer spirals. Vertical:
   `vd = kp_z·(target_lead.z − ego.z) + target_vz` (the altitude is **led**, because vertical lag is
   the biggest centering error for a yaw-only FPV camera).
4. **Yaw = absolute, lead-compensated heading** toward the target (`YawMode(False, az)`). It is an
   absolute world heading with **no integral and no relative accumulation**, so it physically cannot
   spin (the earlier spin came from an integral on a heading-relative setpoint).

### Robustness (what keeps it durable)

* **Close-range brake** — if the chaser is already inside the gap, any *inward* radial velocity is
  cancelled, so a velocity spike can't surge it in and throw the target out of the vertical FOV.
* **Lost-target decay** — when the target isn't seen, the feed-forward is decayed
  (`tv ·= 1 − 0.2·misses`) and the speed cap is reduced, so the chaser **holds** instead of flying
  off chasing a ghost. Combined with the velocity clamp, range stays bounded and it re-acquires.

## Guidance modes (Web UI dropdown / `/set_mode`)

All modes run the **same** standoff law; they only choose the *measurement source* fed to the KF:

| Mode | Measurement |
|------|-------------|
| `fused` (default) | confidence-weighted blend of the vision estimate and GPS |
| `vision_after_arrival` | GPS while approaching, switches to **camera-only** once within the gap |
| `vision` | **camera-only** (bearing + depth) — real-drone-faithful, no target GPS |
| `location` | GPS only (baseline) |

## Scenario tester & measured results

`python sim/airsim/scenario_eval.py --gap 12 --record` launches the ego at 5 m, climbs to a clear
operating altitude (Blocks has structures below ~20 m), places the target 30 m ahead facing the ego,
**approaches under GPS, then tracks PURE VISION-ONLY** through a sequence of patterns (orbit /
figure-8+altitude / climb-descend / zigzag / recede-approach) while recording
`runs/videos/vision_only_track.avi` and writing `runs/videos/scenario_metrics.txt`. The target is
held inside a safe leash box with a collision guard so it can't fly into a wall.

Measured improvement from the re-engineering (gap 12 m):

| metric | before | after |
|--------|--------|-------|
| target in-frame | 27.6 % | **87.1 %** |
| range hold | blew to **54 m** | **10.8–17 m** (gap 12) |
| altitude-track RMS | 8.5 m | **2.5 m** |
| collisions | 0 | **0** |
| orbit centering error | — | **0.135** (no spin) |

Remaining work: centering error ~0.31 during hard maneuvers (vertical lag — inherent to a yaw-only
FPV camera at ~5 fps detection); a gimbal or higher detection rate would close this.

## Platform steadiness (how much the camera shifts / tilts / yaws)

A body-fixed (FPV) camera moves with the airframe, so every pitch/roll/yaw shows up as image motion.
`scenario_eval.py` logs the ego's full 3-D path **and attitude** to `runs/videos/ego_trajectory.csv`
and prints roll/pitch/yaw-rate rms+max. To keep the image steady the controller
**acceleration-limits** the velocity command (`A_MAX` — caps tilt to ~`atan(A_MAX/g)`) and
**slew-limits the yaw setpoint** (`YAW_SLEW`, deg/s — smooth yaw instead of snapping), in both the
Web UI and the tester.

For an **erratic/random** target with no gimbal at ~5 fps this is a genuine trade-off — steadier
settings lose a fast target, tracking settings tilt more:

| setting | in-frame | roll rms/max | pitch rms/max | yaw-rate rms/max |
|---------|----------|--------------|---------------|------------------|
| aggressive | 86 % | 3.3 / 13 | 8.4 / 28 | 14 / 54 |
| very smooth (slew 45) | 52 % | **1.3 / 5.4** | 6.2 | **3.7 / 13** |
| balanced (**default**: kp_pos 0.8, A_MAX 2.4, YAW_SLEW 65) | ~84 % | 2.3 / 7.7 | 9.4 / 31 | 15 / 65 |

Roll and yaw smooth out easily; **pitch is stubborn** because matching a radially-moving target needs
fore/aft acceleration, which a multirotor produces by pitching. The clean fix for a steady image while
still maneuvering hard is a **gimbal** (stabilised camera in `settings.json`) — airframe chases, gimbal
keeps the target locked.

## Obstacle avoidance (depth as a LiDAR-like sensor)

`avoidance.py` is a reactive safety layer that keeps the chaser from flying into the scene while it
pursues the target. Concepts adapted from **simondlevy/AirSimTensorFlow** (use the forward camera to
predict an imminent collision and brake — it calls the depth image "LiDAR-like") and
**mrhosseini75/Semi_Autonomous_Drone_Nav** (use depth/LiDAR free-space to steer).

* `clearance_and_escape(depth)` — the 20th-percentile depth in a central window is the **clearance
  ahead**; per-sector (left/right/up/down) free space picks an **escape direction** (steer to the
  freer side, climb if both sides are blocked).
* `apply_avoidance(v, depth, ego_yaw, speed)` — blends the escape into the standoff velocity (gentle
  at `brake_dist`=9 m, full escape at `crit_dist`=4.5 m) and cancels any velocity heading *into* the
  obstacle. Far from obstacles it does nothing.

In the Web UI it's a checkbox (`/set_avoid`); telemetry shows **Obstacle clearance** and a red
on-video banner appears while avoiding. It needs the AirSim depth channel (off for external video).
It deliberately does **not** brake for the target drone itself — the target is small, so the depth
percentile sees past it; only large near obstacles (walls/trees) trigger it.

## Camera: FPV by default, gimbal optional

Tracking runs in **FPV mode** (body-fixed camera, the whole aircraft yaws to point) by default — this
is the validated path. A software **gimbal** (point the camera at the target each frame, decoupling it
from airframe tilt) is implemented in `guidance.py`/`scenario_eval.py` as an opt-in (`--gimbal`) but
is **experimental**: `simSetCameraPose` did not reliably move the camera in the test build, so it is
off by default. If a working gimbal is available, enabling it gives rock-steady centering; otherwise
FPV mode is used.

> Only one program (Web UI **or** scenario tester) may drive `Ego`/`Target` at a time — stop the
> other first. Run the tester with `python -u` so its progress isn't hidden by stdout buffering.
