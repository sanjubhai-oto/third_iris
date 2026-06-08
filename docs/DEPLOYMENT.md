# Real-world deployment roadmap

This project is a **simulation prototype** (AirSim) of the perception + guidance logic for an
air-to-air tracking / GPS-denied / strike drone. This document is the honest plan to take it to a
**real aircraft**, what carries over, what gets replaced, and the phases + effort.

> Honesty note: in the sim, normal (non-jammed) tracking uses AirSim **ground-truth** ego pose
> (standing in for GPS/telemetry); the hand-rolled camera+IMU **VIO** only takes over on the GPS-jam
> toggle and drifts 4–20 m. It is **not** VINS/OpenVINS and is **not** flight-grade. See below.

---

## What carries over vs. what gets replaced

| Carries over (the real value built here) | Replaced for real hardware |
|------------------------------------------|----------------------------|
| YOLO26-seg detection/segmentation pipeline | Retrain on **real** footage (Det-Fly/ARD100/own) — sim weights won't transfer |
| BoT-SORT + ReID tracking | (keep) |
| Standoff guidance law (`guidance.py`) | (keep) |
| Obstacle avoidance (`avoidance.py`) | Re-tune for **noisy real depth** |
| Strike / PN terminal guidance (`strike.py`) | (keep) |
| Robust range + loss-aware control (`range_filter.py`) | Re-tune for real sensor noise |
| Web UI + architecture + link-protocol layer (`sources.py`) | (keep) |
| **Hand-rolled RGB-D VIO** (`vio/vio_estimator.py`) | **Replace with VINS-Fusion / OpenVINS** |
| AirSim ground-truth ego pose | **PX4 EKF2** state estimate (GPS/baro/IMU + vision) |

---

## Hardware (bill of materials)

| Part | Role |
|------|------|
| **Flight controller** (Pixhawk 6C / Cube Orange) running **PX4** (or ArduPilot) | EKF2 fuses vision when GPS is denied |
| **Companion computer** (Jetson Orin Nano/NX, or RPi 5) | Runs VIO + detection + guidance onboard, talks MAVLink to the FC |
| **Stereo + IMU camera** — Intel RealSense **D435i / D455** (T265 is **discontinued**) or Luxonis **OAK-D** | Real depth + IMU for VIO |
| **2–3 axis gimbal** | The real fix for the fore/aft pitch shake — decouples the camera from airframe tilt |
| GPS + telemetry radio | The "GPS available" half + GPS→jam handover demo |

---

## Software to reuse (do NOT reinvent — see [RESEARCH.md](RESEARCH.md))

- **VIO:** **VINS-Fusion** (stereo+IMU, loop closure) or **OpenVINS** (MSCKF, very robust). Both ROS2.
  *This is the single biggest upgrade* — it replaces the drift-prone hand-rolled estimator.
- **Autopilot fusion:** PX4 **EKF2** external-vision path (already standard).
- **Tracking/guidance:** keep BoT-SORT+ReID, the standoff law, PN strike, avoidance.

---

## Integration — the glue (where the real work is)

1. **VIO → autopilot:** publish VIO odometry to MAVROS **`/mavros/vision_pose/pose`** (or
   `vehicle_visual_odometry`) at **30–50 Hz** with covariances.
2. **EKF2 params:** `EKF2_GPS_CTRL=0` (simulate jam) · `EKF2_EV_CTRL` (enable vision) ·
   `EKF2_HGT_REF=Vision` · `EKF2_EV_DELAY` (tune to camera↔IMU latency — top reason vision is rejected)
   · `EKF2_EV_POS_X/Y/Z` (camera offset from IMU). See `px4_config/ekf2_vision.params`.
3. **Frame conventions:** ROS ENU/FLU vs PX4 NED/FRD — MAVROS converts, but align body-x at init or
   yaw fusion silently fails.
4. **Calibration + time sync:** **Kalibr** for camera↔IMU extrinsics/intrinsics; hardware-triggered
   time sync. *This is the #1 make-or-break for real VIO.*

---

## Sim-to-real reality checks
- Real depth is noisy / limited range → re-tune `range_filter.py` and avoidance thresholds.
- Vibration isolation for camera/IMU (props shake everything).
- **Failsafe when VIO diverges** → loiter / RTL, never fly blind.
- Detector robustness to real lighting/weather; retrain on real data.

---

## Phases (recommended order)

**Phase A — Real VIO → EKF2, in sim (NO hardware).** Bring up WSL2/ROS2 + PX4 SITL + **VINS-Fusion**
(or OpenVINS) against AirSim; bridge its odometry into EKF2; disable GPS and confirm EKF2 holds
position on vision alone. *Swaps the fake VIO for a real one with no parts.* **Highest-value next
step.** Effort: days–2 weeks. (We already have PX4/MAVROS/WSL2 groundwork — see `vio/README.md`,
`sim/README.md`.)

**Phase B — Hardware bench bring-up.** Camera + Jetson + FC on the bench; Kalibr calibration; verify
EKF2 holds position on vision with GPS off. Effort: 1–2 weeks.

**Phase C — Flight.** Tethered → short hops → GPS-jam handover → target tracking → (optionally)
strike-intercept on a safe range. Effort: weeks–months (integration/tuning/**safety** is the long
pole; the algorithms are mature).

---

## Effort summary (honest)
- Phase A (real VIO→EKF2 in sim): **days to ~2 weeks**.
- Phases B+C (hardware + safe flight): **weeks to months** — dominated by calibration, time-sync,
  vibration, failsafes, and field testing, not by the algorithms.

The algorithms you'd reuse (VINS/OpenVINS, EKF2, YOLO, BoT-SORT, PN guidance) are all mature
open-source. The work is the **integration, calibration, and safety**, not the math.
