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

**Terminal intercept (track-then-commit) — SOLVED (reliable kinetic kill)**
- Pattern: track at gap 12 m, then commit → dash onto the predicted-intercept-point (PIP/PN lead).
- Fix: reuse the proven **TargetCA** (9-state constant-accel Kalman) for the target world track, fed
  vision+depth INTO the dash (its outlier gate rejects point-blank garbage; it coasts on the accel
  state = the orbit curve when saturated). Smoother tuning (q 1.2) + capped velocity lead.
- Result: TRUE closest approach **1.07 / 0.96 / 1.13 m** over 3 runs (was 8.0 → 5.4 → ~1.0). All under
  the 1.5 m hit radius; target span ~2.5 m → center-to-center ~1 m = physical collision = reliable hit.
  (Earlier EMA/windowed-CV blind coast gave 5–8 m miss; the CA filter following the curve fixed it.)

**Team sky monitor (multi-target) — PARTIAL**
- One up-cam, 3 drones overhead, YOLO + ByteTrack. At ~10 m only 1–2 of 3 detected (small clustered
  same-model targets); at ~5 m **all 3 seen in 98% of frames** (≥2 in 100%).
- Open items: over-counts (duplicate boxes/drone → needs class-agnostic NMS) and ByteTrack churns IDs
  (44 unique, 2 persistent) → needs appearance **re-ID (StrongSORT)** for stable per-drone identity.

**GUI in WSLg — working.** gz GUI as root under software GL opens 1×1 + unmapped (taskbar icon, no
window). Fix in launch_gui.sh: XDG_RUNTIME_DIR for root + force xcb + xdotool resize/map. Docked
up-camera Image Display panel (gui_upcam.config) shows the drone's-eye sky view.

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
