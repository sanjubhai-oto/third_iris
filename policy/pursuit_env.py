#!/usr/bin/env python3
"""Fast kinematic air-to-air pursuit environment + classical expert.

The vision tracking/navigation POLICY never sees pixels -- it consumes the abstract output of the
(separately trained) detector + image-space Kalman tracker:

    obs = [ex, ey, vex, vey, range_err, closing]   (all normalized ~[-1,1])
      ex,ey     target bbox offset from image center (right+, down+), /half-FOV
      vex,vey   image-space target velocity (per second)
      range_err (range - gap) / gap        (>0 too far, <0 too close)
      closing   d(range)/dt / vmax         (>0 opening, <0 closing)

    act = [fwd, vz, yaw_rate]                       (body-frame command)
      fwd       forward velocity   [-vmax, vmax]  m/s
      vz        vertical velocity  [-2.8, 2.8]    m/s (down+)
      yaw_rate  [-60, 60]                          deg/s

Because obs/act are abstract, this runs thousands of episodes instantly with NO rendering -- ideal
for behavior-cloning the classical guidance (Skydio CEILing recipe) before PyBullet RL fine-tune.

Geometry: ego at (ex,ey,ez) world with heading psi (level flight). Camera bore = +body_x. Target
flies a randomized trajectory. We project the true relative geometry into image space for the EXPERT,
and a noisy/dropout-corrupted copy for the POLICY observation.
"""
from __future__ import annotations

import math
import numpy as np

HFOV = 90.0           # deg, matches the AirSim forward cam
VFOV = 58.0           # deg (16:9-ish)
VMAX = 8.0            # m/s forward cap
VZ_MAX = 2.8
YAW_MAX = 60.0       # deg/s
DT = 0.1             # 10 Hz control

TRAJ = ("orbit", "zigzag", "straight", "climb", "dive", "weave")


class PursuitEnv:
    """One air-to-air pursuit episode. Reset randomizes target trajectory, speed, gap, geometry."""

    def __init__(self, seed=0, noise=0.03, dropout=0.15, max_steps=400, dyn=False, tau=0.25,
                 fpv_pitch=True, k_pitch=0.020, tau_pitch=0.30, range_noise=0.06, range_glitch=0.04):
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.dropout = dropout
        self.max_steps = max_steps
        self.dyn = dyn                # first-order actuator lag (approx real quad command response)
        self.tau = tau               # lag time-constant (s); alpha = dt/tau per step
        self._appl = np.zeros(3)     # currently-applied [fwd, vz, yaw]
        self._prev_appl = np.zeros(3)  # previous applied cmd -> jerk (smoothness) penalty
        # ---- AirSim-realism: FPV body-fixed camera pitch coupling + depth noise ----
        self.fpv_pitch = fpv_pitch   # multirotor pitches to translate -> tilts the body cam -> biases ey
        self.k_pitch = k_pitch       # steady nose-down pitch (rad) per m/s of forward speed
        self.tau_pitch = tau_pitch   # pitch first-order response time-constant (s)
        self.pitch = 0.0             # current body pitch (rad, nose-down +)
        self.range_noise = range_noise   # multiplicative depth noise (AirSim depth is imperfect on a small target)
        self.range_glitch = range_glitch # prob of a badly-wrong depth reading this frame

    # ---------------- episode lifecycle ----------------
    def reset(self, cfg=None):
        r = self.rng
        self.gap = float(cfg.get("gap") if cfg else r.uniform(8, 16))
        self.tspeed = float(cfg.get("tspeed") if cfg else r.uniform(2, 9))
        self.traj = (cfg.get("traj") if cfg else TRAJ[r.integers(len(TRAJ))])
        self.t = 0
        # target starts ahead within FOV, ego at origin facing +x
        rng0 = r.uniform(12, 28)
        az0 = math.radians(r.uniform(-25, 25))
        self.ego = np.zeros(3)
        self.psi = 0.0
        self.tgt = np.array([rng0 * math.cos(az0), rng0 * math.sin(az0), r.uniform(-3, 3)])
        self.tgt0 = self.tgt.copy()
        self._heading = r.uniform(-math.pi, math.pi)
        self._phase = r.uniform(0, 2 * math.pi)
        self.prev_range = float(np.linalg.norm(self.tgt - self.ego))
        self.prev_img = None
        self._appl[:] = 0.0
        self._prev_appl[:] = 0.0
        self.pitch = 0.0
        return self._obs()

    def _move_target(self):
        s, t = self.tspeed, self.t * DT
        h, ph = self._heading, self._phase
        v = np.zeros(3)
        if self.traj == "orbit":
            c = np.array([self.tgt0[0], self.tgt0[1], self.tgt0[2]])
            ang = 0.4 * t + ph
            rad = 10.0
            nxt = c + np.array([rad * math.cos(ang), rad * math.sin(ang), 0.0])
            v = (nxt - self.tgt) / DT
        elif self.traj == "zigzag":
            v = s * np.array([math.cos(h), math.sin(h), 0.0])
            v[:2] += 4.0 * math.sin(2.0 * t + ph) * np.array([-math.sin(h), math.cos(h)])
        elif self.traj == "weave":
            v = s * np.array([math.cos(h), math.sin(h), 0.0])
            v[2] = 1.5 * math.sin(1.5 * t + ph)
            v[:2] += 3.0 * math.sin(3.0 * t) * np.array([-math.sin(h), math.cos(h)])
        elif self.traj == "climb":
            v = s * np.array([math.cos(h), math.sin(h), 0.0]); v[2] = -1.5
        elif self.traj == "dive":
            v = s * np.array([math.cos(h), math.sin(h), 0.0]); v[2] = +1.5
        else:  # straight
            v = s * np.array([math.cos(h), math.sin(h), 0.0])
        self.tgt = self.tgt + v * DT

    # ---------------- observation (true geometry -> image space) ----------------
    def _geom(self):
        rel = self.tgt - self.ego
        cp, sp = math.cos(self.psi), math.sin(self.psi)
        bx = cp * rel[0] + sp * rel[1]          # forward
        by = -sp * rel[0] + cp * rel[1]         # right
        bz = -rel[2]                             # up(+); world z is NED (down+), so up = -z
        rng = float(np.linalg.norm(rel))
        az = math.atan2(by, max(1e-3, bx))       # rad, right+
        el = math.atan2(bz, max(1e-3, math.hypot(bx, by)))
        ex = az / math.radians(HFOV / 2)         # normalized; |ex|<1 in FOV
        ey = -el / math.radians(VFOV / 2)        # image down+ => target above gives ey<0
        behind = bx <= 0
        return ex, ey, rng, behind

    def _obs(self, noisy=True):
        ex_t, ey_t, rng_t, behind = self._geom()
        in_frame = (abs(ex_t) < 1.0 and abs(ey_t) < 1.0 and not behind)
        self._true = (ex_t, ey_t, rng_t, in_frame)   # expert keeps TRUE (privileged) geometry
        # ---- what the on-board detector actually SEES (corrupted) ----
        # FPV body-fixed camera: nose-down pitch lifts the target in the image -> ey biased upward.
        ey_o = ey_t - (self.pitch / math.radians(VFOV / 2.0) if self.fpv_pitch else 0.0)
        ex_o = ex_t
        rng_o = rng_t
        if noisy and self.range_glitch and self.rng.random() < self.range_glitch:
            rng_o = rng_t * float(self.rng.uniform(0.4, 1.8))     # occasional bad depth read
        elif noisy:
            rng_o = rng_t * (1.0 + self.rng.normal(0, self.range_noise))
        # image-space velocity from the OBSERVED history (what the tracker would output)
        if self.prev_img is None:
            vex = vey = 0.0
        else:
            vex = (ex_o - self.prev_img[0]) / DT
            vey = (ey_o - self.prev_img[1]) / DT
        self.prev_img = (ex_o, ey_o)
        range_err = (rng_o - self.gap) / self.gap
        closing = (rng_o - self.prev_range) / DT / VMAX
        self.prev_range = rng_o
        fwd_prop = float(self._appl[0]) / VMAX           # proprioception: lets policy infer its pitch
        obs = np.array([ex_o, ey_o, vex, vey, range_err, closing, fwd_prop], dtype=np.float32)
        if noisy:
            drop = self.rng.random() < self.dropout
            if drop or not in_frame:
                obs = obs.copy(); obs[0] = obs[1] = 0.0   # detector miss -> no bearing, policy coasts
            obs[:4] += self.rng.normal(0, self.noise, 4).astype(np.float32)
        return obs

    # ---------------- classical EXPERT (the law we distill) ----------------
    def expert(self):
        ex, ey, rng, in_frame = self._true
        fwd = float(np.clip(0.8 * (rng - self.gap), -VMAX, VMAX))
        vz = float(np.clip(2.2 * ey, -VZ_MAX, VZ_MAX))
        bearing_deg = math.degrees(math.atan(ex * math.tan(math.radians(HFOV / 2))))
        yaw = float(np.clip(2.2 * bearing_deg, -YAW_MAX, YAW_MAX))
        if not in_frame:                              # target lost -> scan toward last side
            yaw = YAW_MAX * (1.0 if ex >= 0 else -1.0); fwd *= 0.3; vz = 0.0
        return np.array([fwd, vz, yaw], dtype=np.float32)

    # ---------------- dynamics step (kinematic) ----------------
    def step(self, act):
        cmd = np.array([np.clip(act[0], -VMAX, VMAX), np.clip(act[1], -VZ_MAX, VZ_MAX),
                        np.clip(act[2], -YAW_MAX, YAW_MAX)], dtype=float)
        if self.dyn:                                   # first-order lag toward the command
            accel_demand = cmd[0] - self._appl[0]
            self._appl += (DT / self.tau) * (cmd - self._appl)
        else:
            accel_demand = 0.0
            self._appl = cmd
        fwd, vz, yaw = float(self._appl[0]), float(self._appl[1]), float(self._appl[2])
        if self.fpv_pitch:                             # body pitch: steady (speed) + transient (accel)
            pitch_target = self.k_pitch * fwd + 0.10 * float(accel_demand)
            self.pitch += (DT / self.tau_pitch) * (pitch_target - self.pitch)
        self.psi += math.radians(yaw) * DT
        self.ego = self.ego + np.array([fwd * math.cos(self.psi), fwd * math.sin(self.psi), vz]) * DT
        self._move_target()
        self.t += 1
        obs = self._obs()
        ex, ey, rng, in_frame = self._true
        rerr = abs(rng - self.gap)
        # jerk (action-change) penalty -> smooth, steady commands (no twitchy roll/pitch/yaw). Normalized
        # per axis so each term is ~[0,1] per step; coeff kept small so tracking still dominates.
        d = self._appl - self._prev_appl
        jerk = abs(d[0]) / VMAX + abs(d[1]) / VZ_MAX + abs(d[2]) / YAW_MAX
        self._prev_appl = self._appl.copy()
        reward = (1.0 if in_frame else -1.0) - 0.5 * (abs(ex) + abs(ey)) - 0.05 * rerr \
                 - 0.01 * (abs(fwd) + abs(vz) + abs(yaw) / 20.0) - 0.04 * jerk
        done = self.t >= self.max_steps or rng > 60.0
        info = {"in_frame": in_frame, "center_err": math.hypot(ex, ey), "range_err": rerr, "rng": rng}
        return obs, reward, done, info


def rollout(env, policy_fn, cfg=None):
    """Run one episode under policy_fn(obs)->act. Returns per-step metric arrays."""
    obs = env.reset(cfg)
    inf, cen, rer = [], [], []
    done = False
    while not done:
        act = policy_fn(obs)
        obs, _, done, info = env.step(act)
        inf.append(info["in_frame"]); cen.append(info["center_err"]); rer.append(info["range_err"])
    return {"in_frame": float(np.mean(inf)), "center_err": float(np.mean(cen)),
            "range_err": float(np.mean(rer))}
