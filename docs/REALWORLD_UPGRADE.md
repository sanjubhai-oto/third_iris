# Real-World Upgrade Roadmap — How the Leaders Do It, and Our Path

> Honest gap analysis + prioritized roadmap to take `uav-vio-track` from an AirSim demo to a
> real-world-deployable air-to-air tracking + GPS-denied VIO system. Grounded in public research on
> Skydio, DJI, Anduril, and Shield AI (June 2026), and benchmarked against open-source VIO.

## 1. How the leaders actually build it

### Skydio (best-published consumer/defense autonomy)
- **6 nav cameras, true 360°** (X10: 32 MP, ~200° FOV each), full **multi-camera VIO + real-time 3D map**.
  GPS-denied hover **±10 cm**; claims VIO nav to km altitude with vision-based map placement (terrain-relative).
- Perception = **imitation-learned "neural pilot" (CEILing)**: distil the classical motion planner into a
  fast NN using ~3 h of logged flight, exploiting future-data labels to anticipate hazards.
- **Shadow** tracking = motion prediction + per-subject **appearance re-ID**, auto-reacquire after occlusion
  (works in color + thermal). Compute: **Jetson Orin + Snapdragon 865**.
- Sources: skydio.com/x10/technical-specs, skydio.com/blog/deep-neural-pilot-skydio-2, spectrum.ieee.org/skydio-r1-drone

### DJI
- GPS-denied = **downward optical-flow + ToF** hover hold (±0.3 m, low altitude), *not* full 6-DOF VIO.
  Omnidirectional fish-eyes (APAS 5.0) are for **obstacle sensing**, not full-state SLAM.
- ActiveTrack = undisclosed on-SoC CNN detector/tracker. Proprietary vision SoC (not Jetson).
- Lesson for air-to-air at altitude: DJI's downward-flow trick is **useless** (no ground texture) — we need
  Skydio-style forward/omni VIO.
- Sources: DJI APAS support docs, dji.com/newsroom/news/inside-a-drone-computer-vision.

### Shield AI — Hivemind / Nova 2 (the most concrete public defense-VIO recipe)
- **State estimation:** dense depth-camera matching + feature tracking on **4 RGBD cameras**, VIO @ **10 Hz**.
- **Drift control = explicit LOOP CLOSURE** (place recognition of previously-visited locations). This is the
  headline feature that bounds long-mission drift — pure VO/VIO cannot.
- 15-camera array, IMU, magnetometer; GNSS only for pre-deployment localization. **IR illuminators** for
  full-blackout nav. **Dense 3D mapping @ 5 cm**, next-best-view exploration. **Local planner 20 Hz / global 2 Hz.**
- RL "AI pilot" lineage from Heron Systems (DARPA AlphaDogfight winner). Compute: **Jetson Xavier AGX**.
- Source: shield.ai/autonomy-for-the-world-indoor-exploration-with-nova-2/

### Anduril — Lattice
- Not a published VIO company; the value is **multi-sensor track fusion with handoff** (Lattice Mesh
  pub/sub, sensor-to-sensor cueing — a track survives one sensor losing line-of-sight). Ghost: onboard
  CV detect/classify/track, **~32 TOPS** Lattice AI Core. GPS-denied *method* not disclosed.
- Lesson: don't trust one lock — fuse detections across frames/sensors and coast through dropouts.
- Source: technologyreview.com/2024/12/10 demo, anduril.com/lattice/mission-autonomy.

### Defense GPS-denied playbook (standard, openly documented)
- **All-source / ASPN philosophy**: a fusion engine that extracts position from *any* available sensor, so
  there's no single jammable point of failure. Pillars: high-grade **IMU dead-reckon** + **external aiding**
  (vision, magnetometer, terrain-relative, signals-of-opportunity). INS drift is bounded by an **absolute
  reference** (terrain-relative nav / loop closure). Hybrid multi-method fusion wins.

## 2. Our system vs. them (gap)

| Dimension | uav-vio-track (now) | Leader practice | Verdict |
|---|---|---|---|
| VIO core | Hand-rolled loose RGB-D VO + IMU coast | Tightly-coupled multi-cam VIO + map | 100–400× worse than benchmark |
| Drift control | None (unbounded; 4–20 m measured) | Loop closure / relocalization | Missing the key piece |
| Odometry sensor | RGB-D depth (dies >6 m) | Stereo IR + IMU | Wrong sensor for range |
| Detector training | AirSim synthetic only | Domain randomization + real data | Never seen a real drone |
| Track persistence | Image-space coast ✓ | Motion-predict + appearance re-ID | On the right track |
| GPS/VIO dual-mode | Jam toggle + body-servo ✓ | All-source graceful degrade | Right architecture |

## 3. Recommended VIO: adopt OpenVINS (stop hand-rolling)

- **OpenVINS** (MSCKF filter, GPLv3) — the only production VIO with documented repeated **Jetson + RealSense
  D435i UAV** deployment incl. position-hold; filter-based = bounded latency. Best-documented, most active.
- Benchmark ceiling / clean-license alternative: **OKVIS2** (BSD-3, EuRoC ATE **0.031 m**) or **Basalt** (BSD).
- **Use stereo+IMU, not RGB-D**, for odometry (depth degrades past ~6 m; air-to-air ranges are far).
- **Targets to beat:** EuRoC ATE **< 0.1–0.5 m**, then survive **UZH-FPV** aggressive-motion (closest to an
  intercept maneuver). Even bounded ~1 m would be a 4–20× win over today.
- Integration: OpenVINS → MAVROS `/mavros/vision_pose/pose` → PX4 **EKF2** (`EKF2_EV_CTRL`, `EKF2_HGT_REF`,
  `EKF2_EV_DELAY`). Needs **ROS 2 (WSL2/Linux)** — this is the flagged P2/P3 dependency.
- Sources: docs.openvins.com, github.com/engcang/vins-application, arxiv.org/pdf/2202.09199 (OKVIS2).

## 4. Detector: train on REAL air-to-air data (highest-impact, executable now on Windows)

Current detector = synthetic only. Fix by fine-tuning YOLO26-seg on real air-to-air datasets:

| Priority | Dataset | Size | Annotation | License | Link |
|---|---|---|---|---|---|
| 1 | **Det-Fly** | 13,271 img, drone-from-drone | Box | CC BY 4.0 | github.com/Jake-WU/Det-Fly |
| 1 | **AOT** (Amazon Airborne) | 5.9 M img, track IDs | Box + track | CDLA-Permissive | registry.opendata.aws/airborne-object-tracking |
| 2 | **Anti-UAV** (300/410/600) | RGB **+ thermal** | Box + track | MIT | github.com/ZhaoJ9014/Anti-UAV |
| 2 | **NPS-Drones** | 70,250 frames | Box | research | engineering.purdue.edu/~bouman/UAV_Dataset |
| 3 | **Drone-vs-Bird** | 77+ seq | Box | non-commercial | github.com/wosdetc/challenge |

Plan: **Det-Fly first** (permissive, true air-to-air geometry) → mix with our AirSim synthetic + heavy
**domain randomization** (sky/cloud/terrain backgrounds, sun angle, motion blur, target scale to sub-10 px).
Proven recipe: UZH "Deep Drone Racing: Sim-to-Reality with Domain Randomization" (arXiv 1905.09727),
SynDroneVision (arXiv 2411.05633).

## 5. Prioritized roadmap

**Track A — Real-data detector (now, Windows, measurable mAP):** download Det-Fly → convert to YOLO →
fine-tune YOLO26-seg (AirSim + Det-Fly mix) → eval mAP on held-out real frames. Closes the biggest
real-world gap; no new hardware.

**Track B — Production VIO (WSL2):** stand up OpenVINS in ROS 2, validate on **EuRoC** (hit <0.1 m ATE),
then **UZH-FPV**; wire OpenVINS → MAVROS vision_pose → PX4 EKF2; measure GPS-denied drift on a closed loop.

**Track C — In-sim VIO hardening (now, Windows, bounded value):** replace ad-hoc coast with an error-state
EKF; add keyframe-to-frame matching to cut integration drift; re-benchmark with `scenario_eval.py` jam
phase. Useful but a stopgap — Track B is the real answer.

**Track D — Real hardware bring-up (later):** RealSense **D455** (wide baseline, outdoor) + Jetson Orin +
gimbal; PX4 + MAVROS; field-calibrate camera/IMU (expect the D435i Jetson frame-rate bug — budget plumbing time).

**Track E — Agentic supervisor (LAST, per plan):** lightweight onboard policy distilled from the classical
guidance/avoidance planner (Skydio CEILing pattern) — only after A–C are solid.

## 6. Honesty caveats
- Leader estimator math, IMU grades, and frame rates are **not public** — "±10 cm" / "km-altitude VIO" are
  marketing without independent benchmarks.
- Anduril Ghost and V-BAT GPS-denied *methods* are inferred, not documented.
- OpenVINS ROS 2 bindings are community-maintained; GPLv3 is viral (use OKVIS2/Basalt BSD to ship closed).
