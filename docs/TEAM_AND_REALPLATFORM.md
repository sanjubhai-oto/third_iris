# Roadmap: Multi-target "team" tracking & real-platform navigation

Two extensions beyond the current single-target, desk-camera setup. Status as of 2026-06-11.

---

## 1. Multi-target ("team") tracking

**What already works:** the detector (YOLO) + **ByteTrack** already track *every* drone in
frame each frame with persistent IDs, and the webui draws them all (yellow boxes + id, green =
locked). So multi-target *detection + tracking + display* is already there for any drone the
detector finds. Telemetry now exposes the full set: `n_targets` + `targets[]` (id, conf, norm cx/cy).

**Key design constraint:** one chaser airframe can pursue **one** target at a time — multi-target
is about **situational awareness + target selection/priority**, not commanding the drone toward
several at once.

**Remaining work (in priority order):**
1. **Multi manual-lock** — today one manual `TemplateTracker`. Make it a list so the operator can
   lock several detector-missed targets. Each frame: update all, drop trackers whose NCC < 0.35 for
   >30 frames, draw all ("M1, M2…"). *Control* still uses one **primary** (most-centered or
   operator-chosen). Refactor: `manual_tk` → `manual_tks: list`, update the TRACK block + draw + clear.
2. **Primary selection / priority** — pick the pursued target by a score: centering, size (closer =
   bigger = higher threat), confidence, or operator tap. Expose `primary_id`; let the operator cycle
   targets (UI button → `/select_next`).
3. **Frontend team panel** — list active IDs with thumbnails + a "lock this" button per row
   (index.html). Needs UI work; backend `targets[]` already feeds it.
4. **Re-ID across occlusion** — BoT-SORT already has ReID; tune `with_reid`/appearance thresh so an
   ID survives a drone passing behind an obstacle (matters when juggling a team).

**Effort:** multi manual-lock ~half day; full team panel + priority ~1–2 days.

---

## 2. Navigation on a real flying platform (MAVLink)

Everything tested on the desk camera is **detection + tracking only** — there is no airframe to
move. Real navigation = the camera **on a drone** with a flight controller the code can command.

**Architecture (reuses the existing stack):**
- **Perception/track/servo:** unchanged — `webui` already computes body-frame commands
  (`fwd, vz, yaw_rate`) from the image. That math is platform-agnostic.
- **Video in:** `sources.py` already supports `rtsp/udp/http/device/file` — point it at the
  aircraft's camera downlink (or run the stack onboard a companion computer reading the camera直接).
- **Command out (the new piece):** replace the AirSim `moveByVelocityBodyFrameAsync` calls with
  **MAVLink** `SET_POSITION_TARGET_LOCAL_NED` in body frame (`MAV_FRAME_BODY_NED`) via pymavlink
  (already a dependency) or MAVROS. Map our `(fwd, vz, yaw_rate)` → vx, vz, yaw_rate fields.
- **Telemetry in:** `sources.py` `MavlinkReceiver` (mavlink_udp/serial/tcp) already parses attitude/
  position — feed it as the nav source instead of AirSim truth.
- **Localization (GPS-denied):** the proven VIO/EKF2 path (see [[vio-nav-tuning]]) — publish VIO pose
  to the FC's EKF2 external-vision input (`px4_config/ekf2_vision.params`); OpenVINS for production.

**Hardware (from docs/DEPLOYMENT.md):** Pixhawk/Cube FC (PX4), Jetson Orin companion, RealSense/OAK
depth+IMU, gimbal, telemetry radio.

**Safety gates (must-have before any armed flight):**
- Offboard command timeout → auto-hover/RTL if the vision loop stalls.
- Geofence + max-speed/tilt caps (already accel-limited in the servo).
- Manual RC override always wins; software is advisory until proven.
- Bench test (props off) → tethered → short hops → GPS-jam handover → tracking, per the deployment
  phases. Algorithms are mature; integration + safety dominate the timeline.

**Effort:** MAVLink command bridge + bench bring-up ~1–2 weeks; flight tuning weeks–months (safety).

**Next concrete step:** a `mavlink_control.py` that mirrors the webui body-servo outputs to
`SET_POSITION_TARGET_LOCAL_NED`, tested against PX4 SITL (no hardware) first.
