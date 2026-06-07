# Web UI — click-to-lock visual tracking

The Web UI (`webui/app.py`) is the **recommended way to drive the system**. It shows the live Ego
camera with YOLO detections, lets you click a detected drone to lock it, set the standoff gap, and
watch live telemetry while the Ego flies toward the target and holds the gap — all in a browser.

It uses AirSim `simple_flight` for both Ego and Target (no PX4 / QGC needed for this entrypoint).

---

## 1. Run it

Prereqs: AirSim binary running with the project `settings.json` (Ego + Target), and the venv with
`cosysairsim`, `ultralytics`, `torch` (CUDA), `opencv`, `flask` installed.

```powershell
# 1. Launch AirSim (Blocks). Wait for it to be running before starting the UI.
sim\airsim\Blocks\WindowsNoEditor\Blocks.exe

# 2. Activate the env and start the UI.
.\.venv\Scripts\Activate.ps1
python webui\app.py

# 3. Open the browser:
#    http://localhost:5000
```

On start, the backend connects to AirSim, arms **both** vehicles, takes them off, climbs to ~18 m,
and begins flying the Target through randomized trajectories (from `trajectories.py`) at varying
altitudes so you can watch the Ego climb/descend to follow. The page serves at port **5000** on all
interfaces (`0.0.0.0`), so it is also reachable from other devices on the LAN.

---

## 2. The screen

```
┌───────────────────────────────────────────────┬─────────────────────┐
│  Header:  title ............ [offline pill] [STATE badge]            │
├───────────────────────────────────────────────┼─────────────────────┤
│                                                │  TELEMETRY          │
│   LIVE VIDEO (click to lock)                   │  (live fields)      │
│     • center reticle (cross + ring)            │                     │
│     • detection boxes                          │  CONTROLS           │
│     • depth thumbnail (top-right)              │   • Set gap slider  │
│     • status banner (DETECT / TRACK …)         │   • Clear Lock      │
│                                                │   • Land            │
│                                                │                     │
│                                                │  ON-VIDEO LEGEND    │
└───────────────────────────────────────────────┴─────────────────────┘
```

### State badge (top-right)

| State | Color | Meaning |
|-------|-------|---------|
| `DETECT` | orange | No lock. Ego gently yaws (GPS-assisted) to keep the target in view so you can see and click it. |
| `TRACK` | green | Locked. Ego flies toward the target, holds the set gap, and yaws to keep it centered. |
| `LANDED` | gray | Land was commanded; vehicles are landing/landed. |

An **"backend offline"** pill appears if the browser cannot reach the Flask backend (telemetry
poll failing).

---

## 3. Controls

### Click-to-lock
Click anywhere on the video. If you click **inside** a detection box, that drone is locked. If you
click empty space, the **nearest** detection to your click is locked. On lock the state goes to
`TRACK`, the Kalman filter resets, and the box turns green with a `LOCKED id… <conf>` label plus a
centering line from frame center to the target.

Click coordinates are sent normalized (0–1) to `POST /select`, so locking works regardless of how
the video is scaled in your browser window.

### Set gap (meters)
The standoff distance the Ego holds behind the target. Adjust with the number box or the slider
(synced), range **0–50 m**, sent via `POST /set_gap`. The regulation is **bidirectional**: if the
range exceeds the gap the Ego closes in (forward along the GPS bearing); if it drops below the gap
the Ego backs off (reverse); within a small deadband (±0.3 m) it holds. The closing/backing speed
is EMA-smoothed and capped by the **Max speed** control, so it is steady rather than jerky and
survives the target maneuvering. A **small gap means a high angular rate** (the target sweeps across
the FOV faster the closer you are) — see `GOTCHAS.md`.

### Max speed (m/s)
The cap on the Ego's approach/back-off velocity along the bearing. Slider range **0.5–12 m/s**, sent
via `POST /set_speed` (the backend clamps to this range). Lower it for gentle, cinematic closes;
raise it to catch a fast-departing target sooner.

### Guidance mode
Selects what drives the yaw (centering) while tracking, via `POST /set_mode`:

| Mode | Behavior |
|------|----------|
| **Fused** (default) | Confidence-weighted blend of vision bearing and GPS azimuth — smooth when detections flicker. |
| **Vision after arrival** | GPS azimuth while approaching, switches to pure vision once within the gap (+2.5 m). |
| **Vision** | Pure image-based bearing (falls back to GPS only if no detection). |
| **Location** | GPS azimuth only — ignores vision for yaw (useful as a baseline). |

### Sources (link protocol) — real-world video + telemetry receivers
On a real aircraft you have two independent downlinks: a **video receiver** and a **telemetry
receiver**. The *Sources* panel lets you pick the protocol for each (AirSim is the default so the sim
keeps working):

* **Video feed** — `AirSim` · `RTSP` · `UDP` · `HTTP/MJPEG` · `Capture device` · `File`, plus an
  endpoint (e.g. `rtsp://192.168.1.10:8554/fpv`, or `0` for the first capture card). Backend opens it
  with OpenCV (`POST /set_video_source`). External video has no depth channel, so vision range falls
  back to GPS/coast in sim; on a real drone range comes from the detector or a depth payload.
* **Telemetry** — `AirSim` · `MAVLink (UDP)` · `MAVLink (serial)`, plus an endpoint
  (`udpin:0.0.0.0:14550` or `COM5,57600`). A pymavlink thread reads attitude/position/speed
  (`POST /set_telem_source`); the panel shows the connection status.

The active video/telemetry source is shown in the telemetry list.

### Clear Lock
`POST /clear` — drops the lock, returns to `DETECT`, resets the Kalman filter and the path trail.

### Land
`POST /land` (with a browser confirm) — commands both vehicles to land. State becomes `LANDED`.

---

## 4. Telemetry fields

Polled every 200 ms from `GET /telemetry`:

| Field | Meaning |
|-------|---------|
| **State** | `INIT` / `DETECT` / `TRACK` / `LANDED`. |
| **Locked** | Yes/No — whether a target is currently locked. |
| **Target ID** | ByteTrack track ID of the locked drone (null when not tracking). |
| **Confidence** | Detection confidence of the locked/centered detection (0–1). |
| **Range to target** | Best range estimate in meters — depth-derived when available, else GPS range. |
| **Set gap** | Current standoff setpoint (m). |
| **Ego altitude** | Height above ground (m); shown as `-ego_d`. |
| **Yaw rate** | Commanded body yaw rate (°/s). |
| **Ego N / E / D** | Ego world-NED position (m), including the spawn offset (`EGO_HOME`). |
| **Detections** | Count of *real* (post-filter) detections this frame. |
| **Vision rate** | Percent of recent frames (~5 s window) that had a valid detection — a tracking-health gauge. |
| **Shadows rejected** | Cumulative count of depth-rejected shadow detections (0 in the current UI, which has shadow rejection off). |
| **FPS** | Smoothed loop rate of the tracking thread. |
| **Mode** | Active guidance mode (`fused` / `vision_after_arrival` / `vision` / `location`). |
| **Source** | What is actually driving yaw this frame (`fused` / `vision` / `gps(approach)` / `vision(arrived)` …). |
| **Reached** | Whether the Ego is within the gap (+2.5 m) of the target. |
| **Speed** | Current max-speed cap (m/s). |

---

## 5. On-video overlays

| Overlay | Meaning |
|---------|---------|
| **Green box + label + line** | The locked / currently-centered target; the line runs from frame center to the target center. |
| **Yellow/amber boxes** | Other detections (`id<n> <conf>`), candidates you can click to lock. |
| **Gray box labeled "shadow"** | A detection rejected by the depth shadow filter (only if that filter is enabled). |
| **White cross (center)** | Frame center / camera boresight — the point the Ego yaws to put the target on. |
| **Yellow trail** | The locked target's path over the last ~5 seconds. |
| **Top-right thumbnail** | Depth map (TURBO colormap, clipped to ~40 m) for the current frame. |
| **Top banner** | `DETECT - click a target box to LOCK`, or `TRACK id… gap=…m range=…m`. |
| **Reticle (HTML)** | A faint cross + ring drawn by the page over the video for aiming reference. |

---

## 6. HTTP API (for reference / scripting)

| Route | Method | Body | Purpose |
|-------|--------|------|---------|
| `/` | GET | — | The UI page. |
| `/video_feed` | GET | — | MJPEG stream (`multipart/x-mixed-replace`). |
| `/telemetry` | GET | — | JSON telemetry (see table above). |
| `/select` | POST | `{"x":0..1,"y":0..1}` | Lock the detection at/near a normalized click. |
| `/clear` | POST | — | Clear the lock. |
| `/set_gap` | POST | `{"gap": meters}` | Set the standoff gap (0–50 m). |
| `/set_speed` | POST | `{"speed": m/s}` | Set the max approach/back-off speed (clamped 0.5–12). |
| `/set_mode` | POST | `{"mode": "fused"\|"vision_after_arrival"\|"vision"\|"location"}` | Set guidance mode. |
| `/set_video_source` | POST | `{"proto": "airsim"\|"rtsp"\|"udp"\|"http"\|"device"\|"file", "endpoint": "..."}` | Switch the video feed to a real-world receiver (OpenCV). |
| `/set_telem_source` | POST | `{"proto": "airsim"\|"mavlink_serial"\|"mavlink_udp"\|"mavlink_tcp", "endpoint": "..."}` | Switch telemetry to a MAVLink receiver (pymavlink: serial/UDP/TCP). |
| `/set_avoid` | POST | `{"avoid": true\|false}` | Toggle depth-based obstacle avoidance. |
| `/land` | POST | — | Command both vehicles to land. |

All responses send `Cache-Control: no-store` so the browser never serves a stale UI (this previously
masked the gap-below-3 fix behind a cached page).

---

## 7. Tuning knobs (in `webui/app.py`)

These constants at the top of the file control the feel:

| Constant | Default | Effect |
|----------|---------|--------|
| `MODEL` | `runs/train/airsim_drone/weights/best.pt` | Detector weights. |
| `EGO_HOME` / `TARGET_HOME` | `(0,0,0)` / `(8,0,0)` | Spawn offsets — must match `settings.json`. |
| `HFOV` | `90.0` | Camera horizontal FOV (deg). |
| `K_YR` | `2.0` | Proportional gain, bearing error → yaw rate. |
| `YR_MAX` | `40.0` | Max yaw rate (°/s). |
| `DB` | `0.03` | Deadband on normalized image error. |
| `LEAD_T` | `0.2` | Kalman lead time (s). |

Yaw is a **proportional rate** with no integral, on purpose — an integral on a heading-relative yaw
setpoint causes a spin (see `GOTCHAS.md`).
