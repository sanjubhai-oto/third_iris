# GZ Sim up-facing-camera VIO tracking — session results (2026-06-13)

Autonomous session. Goal: the sky-facing-camera air-to-air tracking that AirSim could not render, +
real-world research, in Gazebo. Brutal-honest: proven vs partial.

## Headline
**The up-facing sky camera works in Gazebo and tracks an enemy drone overhead at ~99% in-frame using
the universal YOLO detector + no-yaw strafe/climb law.** This is the JetRay nose-to-sky scenario AirSim
could not produce (that build has no up camera).

## What was proven (metrics)

**Up-cam VIO tracking — SOLVED**
| run | detector | in-frame | center_err | vertical-gap RMS | horiz-sep | notes |
|---|---|---|---|---|---|---|
| orbit_climb | YOLO uav_diverse @960 + depth | **99.3%** | 0.265 | **1.77 m** (through climb 18→40 m) | 4.1 m | no divergence |
| orbit (const alt) | YOLO @960 + depth | 100% | 0.25 | **1.28 m** | 3.9 m | |
| zigzag (hard lateral) | YOLO @960 + depth | **100%** | 0.225 | **1.01 m** | 3.2 m | robust to fast lateral |
| orbit_climb | classical blob (clear sky) | 99% | 0.18 | — | 5 m | first proof |

- Control law: HOLD heading (no yaw — a fixed up-cam only rotates the image when yawed; this killed the
  AirSim yaw-shake), STRAFE N/E to keep the target centred, move vertically to hold the gap.
- Range: up DEPTH camera (true range). Size-proxy range failed (fooled by target scale → chaser sank).
- Detector: YOLO at **imgsz 960** needed (640 → 15% recall on small targets). Classical blob is fast but
  DIVERGES (locks cloud/horizon dark regions); a temporal gate helps but YOLO is the reliable choice.

**Terminal intercept (track-then-commit) — PARTIAL, honest miss**
- Pattern: track at gap 12 m, then commit → dash onto the predicted-intercept-point (PIP/PN lead),
  coasting blind through point-blank vision saturation.
- Result: estimate declares HIT (1.3 m) but **TRUE closest approach = 5.4 m** (was 8.0 m before the
  windowed-velocity-fit fix). NOT a reliable kill. Bottleneck = target world-position estimate error
  (~2 m depth/ray) growing during the blind coast on a curving target — the known hard point-blank
  problem. Reported as TRUE miss so a false-positive estimate-HIT isn't mistaken for a real kill.

## WSL/Gazebo engineering cracked (so this is reproducible)
- Camera never rendered headless until: (1) `GZ_SIM_SERVER_CONFIG_PATH` = PX4 gz_bridge/server.config
  (default config lacks the **Sensors** system); (2) `LIBGL_ALWAYS_SOFTWARE=1` (ogre2 over WSLg/D3D12
  hardware GL **core-dumps**; llvmpipe software GL renders fine). In-SDF `<plugin>` tags are ignored.
- Reproduce: `bash sim/gz/launch_gz.sh` then
  `LIBGL_ALWAYS_SOFTWARE=1 python3 sim/gz/track_gz.py --pattern orbit_climb --yolo <weights> --imgsz 960`.

## Real-world research → see docs/REALWORLD_UAV_INTERCEPT_RESEARCH.md
Our stack (single camera + YOLO + Kalman tracker + PN terminal) matches the published air-to-air MAV
interceptor and Ukraine/Israel field practice (Wild Hornets Sting: thermal + AI terminal guidance,
300 km/h; Smart Shooter: auto detect+track+lead). ProNav > pure pursuit for terminal. C2FDrone
coarse-to-fine ViT for tiny/distant drones. Collaborative multi-UAV capture = the "team monitor" idea.

## Honest open items
- Terminal kill: needs better close-range state estimation (CA Kalman on the world track + keep depth
  valid deeper into the dash, or a proximity model) to get true miss < 1.5 m. Currently 5.4 m.
- Thermal camera (real HW) for day/night + hot-target contrast — sim uses RGB.
- Small/distant-target detection (coarse-to-fine) for long-range acquisition.
- Multi-UAV team (shared fusion + role handoff) — the "team to monitor live".
- Real flight / real-HW VIO — still sim only.
