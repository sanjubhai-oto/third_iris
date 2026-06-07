# Research & reuse decisions

This project deliberately **reuses proven open-source components** instead of reinventing them. This
doc records the literature we drew on and, for each subsystem, the *reuse-vs-build* decision.

---

## 1. Air-to-air detection, segmentation & tracking

**Datasets / benchmarks**
- **Det-Fly** — drone-shot-from-drone detection set, varied scenes. https://github.com/Jake-WU/Det-Fly
- **ARD100** — 202k frames, smallest average object size of any drone set; released with YOLOMG. https://github.com/Irisky123/YOLOMG
- **Anti-UAV / Anti-UAV410** — large single-target UAV *tracking* benchmarks (IEEE TMM 2023; TPAMI 2023).
- **Drone-vs-Bird (WOSDETC)** — distractor (bird) rejection. https://github.com/wosdetc/challenge

**Methods**
- **SAHI — Slicing Aided Hyper Inference** (Akyon et al., ICIP 2022): tile the frame → big small-object mAP gains. https://github.com/obss/sahi
- **BoT-SORT** (Aharon et al., 2022): tracker with **camera-motion compensation (GMC)** + ReID — directly counters a moving ego-camera. Built into Ultralytics.
- **OC-SORT / Deep OC-SORT** (CVPR 2023): observation-centric, strong on non-linear motion/occlusion; via `boxmot`. https://github.com/mikel-brostrom/boxmot
- **YOLOv12 + BoT-SORT-ReID** (4th Anti-UAV Challenge, CVPRW 2025): YOLO+BoT-SORT-ReID ≈ doubled the MOT baseline for multi-UAV. https://github.com/wish44165/YOLOv12-BoT-SORT-ReID
- **Siam R-CNN: tracking by re-detection** (CVPR 2020): the reference design for recovering a lost single target. https://arxiv.org/abs/1911.12836
- **TransVisDrone** (ICRA 2023, temporal) and **YOLOMG** (motion-fusion, 2025) — higher-accuracy ceilings for tiny targets; heavier integration.

**Decision**
| Item | Decision |
|------|----------|
| **BoT-SORT + native ReID** | ✅ **ADOPTED** — `perception/trackers/botsort_uav.yaml`, GMC + `with_reid`, larger `track_buffer`. Zero new deps (Ultralytics ≥ 8.3.114). Fixes moving-camera ID swaps. |
| SAHI sliced inference | 📋 documented option (`pip install sahi`) — enable when targets are very distant; adds latency. |
| OC-SORT / Deep OC-SORT | 📋 documented (`pip install boxmot`) — escalation if a maneuvering target still breaks tracking. |
| YOLOMG / TransVisDrone | 📋 roadmap — retrain/temporal, for sub-10px targets. |
| Re-acquisition by embedding match | 📋 roadmap — cache last-good crop embedding, full-frame re-detect on loss. |

We keep our fine-tuned **YOLO26-seg** detector (mAP50 0.97) and ByteTrack remains available; BoT-SORT
is the new default.

---

## 2. Visual-Inertial Odometry (GPS-denied self-localization)

**Open-source systems**
- **OpenVINS** (Geneva et al., ICRA 2020) — MSCKF filter VIO, ROS2, the research standard. https://github.com/rpng/open_vins
- **VINS-Fusion / VINS-Mono** (HKUST, 2018-19) — optimization VIO (+GPS fusion). https://github.com/HKUST-Aerial-Robotics/VINS-Fusion
- **ORB-SLAM3** (Campos et al., 2020) — feature SLAM, loop closure; most accurate stereo-inertial. https://github.com/UZ-SLAMLab/ORB_SLAM3
- **Basalt** (Usenko et al., 2019) — BSD, no-ROS, very accurate. https://gitlab.com/VladyslavUsenko/basalt
- **Kimera** (MIT-SPARK, 2020, BSD); **ROVIO/MSCKF-VIO** (older/ROS1).

**Foundational papers (techniques we use)**
- Mourikis & Roumeliotis, *MSCKF*, ICRA 2007.
- Horn 1987 / **Umeyama 1991** — closed-form 3-D↔3-D rigid alignment (our RGB-D pose step).
- Steinbrücker et al., ICCV-W 2011 / Park et al., ICCV 2017 — **RGB-D direct odometry** (Open3D backend).
- Mahony 2008 / Madgwick 2010 / **Solà 2017 (ESKF)** — IMU complementary / error-state fusion.

**Reality of integration:** all the full systems are C++/ROS. The supported path is **AirSim on Windows
↔ ROS2 VIO in WSL2 → MAVROS `vision_pose` → PX4 EKF2 (GPS disabled)**, streaming odometry at 30–50 Hz.

**Decision**
| Item | Decision |
|------|----------|
| **Open3D `compute_rgbd_odometry`** | ✅ **ADOPTED (optional backend)** — proven RGB-D odometry with a 6×6 information matrix; used as the VO front-end in `vio/vio_estimator.py` when `open3d` is installed, falling back to our Umeyama+RANSAC otherwise. Keeps our IMU fusion + GPS-anchor/jam logic. |
| Our RGB-D + IMU estimator | ✅ kept (grounded in Umeyama/Horn + ESKF papers above); validated 98% VO, ~4.5 m drift / 22 s jam. |
| **OpenVINS → EKF2 (WSL2/ROS2)** | 📋 the *proper* production path, documented in [vio/README.md](../vio/README.md) and [VIO.md](VIO.md). We already have PX4/EKF2/MAVROS + WSL2 networking working; remaining work is the Cosys-AirSim ROS2 camera+IMU bridge, calibration, time sync. |

No pip-installable MSCKF VIO exists for Python — hence the two-tier story (Open3D now, OpenVINS later).

---

## 3. Obstacle avoidance

**Methods**
- **Pushbroom Stereo** (Barry & Tedrake, JFR 2018) — the canonical "depth as a forward LiDAR" idea. https://arxiv.org/abs/1407.7091
- **VFH / 3DVFH+** (Borenstein & Koren 1991; Vanneste et al. 2014) — polar free-space histogram → steer to the freest direction.
- **Artificial Potential Fields** — attractive target + repulsive obstacles; APF for *moving-target* following (Sensors 2024); velocity-adaptive/rotational APF to escape local minima (arXiv:2512.07609).
- **EGO-Planner / Fast-Planner** (ZJU/HKUST) — gradient local planners (ROS/C++, ESDF maps). https://github.com/ZJU-FAST-Lab/ego-planner
- **Learning High-Speed Flight in the Wild** (Loquercio et al., Science Robotics 2021). https://github.com/uzh-rpg/agile_autonomy
- **Perception-/occlusion-aware NMPC** (arXiv:2302.04708, 2112.12177) and **NOVA CBF** (arXiv:2506.18689) — co-optimize *keep target visible* + *avoid* so avoidance doesn't drop the lock.
- AirSim's own `PythonClient/multirotor/navigate.py` (depth-sector steering); simondlevy/AirSimTensorFlow (collision classifier — demo only).

**Composition principle (key):** avoidance must not throw the target out of frame. Best practice —
blend the safety contribution by **time-to-collision**, route avoidance into **lateral/vertical**
motion, and keep the attractive (target) term always on.

**Decision**
| Item | Decision |
|------|----------|
| **APF + tangential repulsion, TTC-blended** | ✅ **ADOPTED** — `sim/airsim/avoidance.py` upgraded from a plain depth threshold to attractive(target)+repulsive+**tangential** depth force, blended by time-to-collision (graceful, keeps forward progress). |
| Target-biased depth histogram (VFH-lite) | ✅ retained — per-sector free-space pick already steers toward the open side. |
| CBF safety filter | 📋 upgrade if APF oscillates near structures (NOVA recipe). |
| Perception-aware NMPC | 📋 roadmap (co-optimize visibility + avoidance + dynamics). |
| EGO-Planner / learning-based | 📋 deferred — ROS/C++/GPU, overkill for a forward-camera reactive chaser. |

---

## Summary — what we reuse vs build

- **Reused (adopted in-repo):** BoT-SORT+ReID (Ultralytics), Open3D RGB-D odometry, APF/VFH avoidance,
  YOLO26-seg + ByteTrack (Ultralytics), AirSim/Cosys-AirSim, PX4/MAVSDK/pymavlink, OpenCV LK/Umeyama.
- **Built (thin glue, grounded in papers):** the standoff air-to-air guidance law, the GPS-anchor/jam
  dual-mode + body-frame visual servo, the AirSim auto-labeled dataset pipeline, the Web UI.
- **Documented as the proper heavyweight path:** OpenVINS/Basalt → PX4 EKF2 in WSL2/ROS2.
