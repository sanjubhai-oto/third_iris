# Two-drone PX4-SITL air-to-air engagement (jetray runner vs vision chaser)

Realistic flight: **two real PX4 SITL multirotors** in one Gazebo world. The **runner** flies a premade
route (offboard waypoints); the **chaser** detects it with our **Hybrid YOLO+KLT** detector on a forward
camera and pursues via **MAVLink offboard** body-velocity setpoints, declaring a HIT on true range < 1.5 m.
Runs in WSL2 Ubuntu (software GL — hardware ogre2 core-dumps on camera render under WSLg).

## Files (`sim/gz/a2a/`)
- `env_a2a.sh` — detects PX4 dir, exports gz paths, geo-origin, software-GL.
- `make_models.py` — generates `jetray_chaser` (forward cam `/chaser/camera_front`, tilted up 10°) and
  `jetray_runner` (no cameras) PX4 gz model variants (additive; base jetray untouched).
- `launch_a2a.sh` — standalone gz server (`sim/gz/uav_a2a.sdf`) + 2 PX4 instances (chaser i=0 @14540,
  runner i=1 @14541). Gates on `scene/info` ready; sets `PX4_GZ_WORLD`.
- `runner_mission.py` — runner offboard waypoint follower (udp 14541), slow cruise so it's catchable.
- `detector.py` — `HybridDetector` (YOLO every N + KLT between; dark-blob fallback).
- `cam_check.py` — Stage-3 offline detection check (subscribe cam, annotate, report lock%/Hz).
- `chaser_strike.py` — chaser vision-servo strike (udp 14540 via `deploy/mavlink_control.py` MavBridge):
  detect → image-based servo (yaw/vz/fwd) → offboard → HIT on true range.
- `px4_util.py` — `arm_offboard()` retry (cold multi-vehicle EKF isn't "ready" for ~10-20 s; one-shot arm fails).

## Run (4 shells in WSL)
```bash
cd /mnt/c/Users/admin/uav-vio-track
bash sim/gz/a2a/launch_a2a.sh                     # 1) world + both vehicles (both heartbeats + /chaser/camera_front)
python3 sim/gz/a2a/chaser_strike.py --secs 120    # 2) chaser (start FIRST; climbs, then engages)
# ~15 s later:
python3 sim/gz/a2a/runner_mission.py --cruise 3   # 3) runner flies the route
# Windows: webui dashboard (POV + 3D map)
.venv\Scripts\python.exe webui\app_isaac.py       # 4) open http://localhost:5060
```
Backgrounding inside WSL: wrap with `setsid bash -c "exec python3 ... >log 2>&1" </dev/null &` (a plain
`nohup ... &` from a transient `wsl.exe` shell gets killed on shell exit before it reparents).

## Result (first integration, honest)
- Both vehicles arm/fly; runner flies the serpentine; chaser acquires + pursues + **HIT at ~1.35-1.42 m**.
- Vision **lock 17-49%** (varies run-to-run): YOLO+KLT steers when the target is detected; between
  detections the chaser **coasts/creeps** forward down the corridor. So the strike is **vision-initiated
  and vision-assisted, NOT continuously vision-tracked** — detection of a small drone at 15-40 m against
  the sky/horizon is intermittent, and the multirotor's nose-down pitch in forward flight pushes the
  target toward the frame edge (mitigated by the 10° camera up-tilt). HIT is decided on TRUE range.
- Camera ~8-11 Hz under software GL.

## Known limits / next steps
- Detection rate is the bottleneck. Improve with: weights tuned for small/distant UAVs, larger imgsz,
  a gimbal (decouple camera from body pitch), or proportional-navigation guidance on a Kalman track
  (`sim/airsim/target_tracker.py` TargetCA) instead of creep/coast on loss.
- Geometry is tail-chase + forward camera (per request). Top-attack variant is future work.
