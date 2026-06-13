# Real-world UAV-vs-UAV sky interception — research → our system

Researched 2026-06-13 (Ukraine/Israel field use, academic papers, industry autopilots) to validate and
upgrade our vision-guided air-to-air tracking/intercept. Brutal-honest mapping: what real systems do,
what we already have, what to add.

## What the real operators do

**Ukraine (FPV interceptors — the dominant air-to-air drone-kill practice today)**
- 100k+ interceptors built in 2025; doctrine = cheap kinetic FPV that *launches on detection, climbs fast,
  rams or net-captures* enemy strike/recon drones before they reach targets.
- Wild Hornets **"Sting"**: ~$2.5k FPV, **195 mph (~314 km/h)**, **thermal camera + AI-assisted terminal
  guidance**, engages targets km out; thousands of drone kills. SkyFall P1-SUN: $1k fiber-optic Shahed hunter.
- Takeaway: SPEED (300+ km/h), FAST CLIMB, THERMAL for day/night + small-hot-target contrast, AI handles the
  TERMINAL phase (human picks target, autonomy flies the kill). Intercept = collide/net, not standoff.

**Israel (fire-control + RWS, mostly ground-based C-UAS but same perception stack)**
- Smart Shooter SMASH / Smash Dome / Hopper RWS: **auto detect + track + ballistic/lead solution**, AI vision,
  day/night, "every round finds the target regardless of target motion"; C2-integrated for auto detect+track.
- Takeaway: detect → lock → **lead/predict (PIP)** → commit, with IFF (don't shoot birds), all automated.

**Industry autopilots (UAV Navigation VECTOR-300, etc.)**
- C-UAS/loitering-munition autopilots fold **AI target ID + optical data directly into the GNC loop**;
  **real-time trajectory adaptation in pursuit AND terminal phases**; camera provides positional data to the
  autopilot through impact. = exactly the "vision in the control loop" we build.

## What the papers say (algorithm-level)

- **Air-to-air MAV interceptor** (ScienceDirect S1270963823000895): single onboard camera + **DL detector +
  Kalman tracker**, autonomous track-and-engage. ← our exact stack (YOLO + Kalman + servo).
- **Proportional Navigation (ProNav)** is the standard terminal guidance and beats pure pursuit for a moving
  target (missile heritage; now in C-UAS). ← we have true-PN + PIP lead in `strike.py`.
- **C2FDrone** (arXiv 2404.19276): coarse-to-fine **Vision-Transformer** detection for *small/distant/occluded*
  drones; +1–7% F1 on FL-Drones/AOT/NPS; runs on edge. Ideas: multi-scale attention, **temporal/motion fusion**,
  synthetic blur aug, **adaptive region proposals by range**.
- **Bird-vs-drone IFF** (tandfonline 2318672): classify drone vs bird before engaging. ← we already drop birds
  (class-aware dataset → birds become negatives).
- **Collaborative multi-UAV tracking & capture** (arXiv 2010.01588, 2405.13542): a **team** tracks/captures one
  aerial target — multi-view fusion, role assignment (primary/secondary), **handoff when one loses sight**,
  encirclement approach. ← the "team to monitor live" you asked for.
- **Vision UAV detect+track + Kalman** (Wiley coin.70026): YOLO-class detector + Kalman for robust track. ← ours.

## Map to our system (proven vs to-do)

| Real-world technique | Our status |
|---|---|
| Single cam + DL detector + Kalman tracker, autonomous | ✅ YOLO/ByteTrack + TargetCA Kalman + body-servo |
| Predict-through-occlusion (coast on Kalman) | ✅ TargetCA `predict_only` coast-decay (1.3 m through loss) |
| Proportional Navigation terminal guidance | ✅ in `strike.py` (true-PN + PIP); ⚠️ not yet wired into tracking |
| Lead / predicted-intercept-point (PIP) | ✅ DJI-style lead in servo + PIP in strike |
| Bird-vs-drone IFF | ✅ class-aware (birds → negatives) |
| Speed 300 km/h, fast climb | ✅ speed range 5–83 m/s; ✅ skyward climb-track (95% in sim) |
| Up/sky-facing camera (target overhead) | ✅ **GZ Sim** up-cam tracking 99% in-frame (AirSim couldn't render it) |
| Thermal camera for day/night + contrast | ⚠️ real-HW; sim uses RGB. NOTE for real build |
| Small/distant-drone detection (coarse-to-fine, temporal) | ⚠️ to add: tiling + temporal/motion cue for far targets |
| Multi-UAV team: shared fusion + role handoff | ⚠️ partial (multi-target world exists); team handoff to build |
| Track-then-commit terminal intercept (gap→0) | ⚠️ building now in GZ (PN commit on the world tracker) |

## Integration priorities (doing in this session)
1. **GZ up-cam vertical-gap fix** via true depth range (done — depth camera added).
2. **Track-then-commit PN terminal intercept** in GZ: track at standoff with the up-cam, then PN-commit to
   gap→0 driving on the world tracker (the Ukraine/Israel "AI terminal" pattern).
3. **Team monitor** (stretch): track all drones in the sky view (multi-target), role handoff concept.

Sources: Ukraine interceptors (DefenseNews/MilitaryTimes/TWZ 2026), Wild Hornets Sting; Israel Smart Shooter
SMASH/Smash Dome (C4ISRNET/Euro-SD); UAV Navigation VECTOR-300 (UASVision); arXiv 2404.19276, 2010.01588,
2405.13542; ScienceDirect S1270963823000895; Wiley coin.70026; tandfonline 2024.2318672.
