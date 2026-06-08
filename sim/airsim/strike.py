#!/usr/bin/env python3
"""Terminal-guidance INTERCEPT for a selected ground target (simulated strike).

Classic homing guidance, grounded in the literature (Zarchan; JHU APL Palumbo; see docs/RESEARCH.md):
  * Pure pursuit  : aim the velocity straight at the target (good for slow/static targets).
  * PIP lead      : aim at the Predicted Intercept Point  PIP = p_t + v_t * t_go  (kills the tail
                    chase against a moving target); t_go from the constant-velocity intercept quadratic.
  * True PN       : a = N * (V_r x Omega),  Omega = (R x V_r)/(R.R) — drives the LOS rate to zero for a
                    near-straight intercept; layered as a lateral correction.
Plus a closing-speed schedule (accelerate as range closes -> terminal dive) and an impact test
(kill-radius OR closing-rate zero-crossing = point of closest approach).

Everything is in world NED. This is a simulation intercept (the chaser collides with the target in
AirSim); the same guidance family is used for counter-UAV interception.
"""
from __future__ import annotations

import math

import numpy as np


def _t_go(R, v_t, s):
    """Smallest positive time-to-go for a chaser of speed s to intercept a constant-velocity target.
    Solves (|v_t|^2 - s^2) t^2 + 2 (R.v_t) t + |R|^2 = 0."""
    a = float(np.dot(v_t, v_t)) - s * s
    b = 2.0 * float(np.dot(R, v_t))
    c = float(np.dot(R, R))
    if abs(a) < 1e-6:
        return (-c / b) if abs(b) > 1e-6 and -c / b > 0 else None
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sq = math.sqrt(disc)
    cands = [t for t in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)) if t > 0]
    return min(cands) if cands else None


def intercept_command(ego_pos, ego_vel, tgt_pos, tgt_vel, v_max=8.0, v_min=3.0,
                      N=3.0, kill_radius=2.5, term_range=12.0):
    """Return (v_cmd_world(3), range_m, closing_speed, hit, info).

    PIP lead-pursuit base + PN lateral correction + closing-speed schedule (faster as range closes).
    ``hit`` True when within the kill radius.
    """
    ego_pos = np.asarray(ego_pos, float); ego_vel = np.asarray(ego_vel, float)
    tgt_pos = np.asarray(tgt_pos, float); tgt_vel = np.asarray(tgt_vel, float)
    R = tgt_pos - ego_pos
    r = float(np.linalg.norm(R))
    if r < 1e-3:
        return np.zeros(3), r, 0.0, True, {}
    rhat = R / r
    V_r = tgt_vel - ego_vel
    Vc = -float(np.dot(rhat, V_r))                     # closing speed (+ = closing)

    # closing-speed schedule: accelerate as range closes (terminal dive inside term_range)
    speed = float(np.clip(v_min + (v_max - v_min) * (1.0 - min(1.0, r / max(1.0, 3 * term_range))),
                          v_min, v_max))
    if r < term_range:
        speed = v_max                                   # full speed in the terminal phase

    # PIP lead toward where a constant-velocity target will be
    tg = _t_go(R, tgt_vel, speed)
    pip = tgt_pos + tgt_vel * tg if tg is not None else tgt_pos
    dirv = pip - ego_pos
    dn = float(np.linalg.norm(dirv))
    vdir = dirv / dn if dn > 1e-3 else rhat
    v_cmd = vdir * speed

    # PN lateral correction (drives LOS rate to zero) for a maneuvering target
    rr = float(np.dot(R, R))
    if rr > 1e-6:
        Omega = np.cross(R, V_r) / rr                   # LOS rotation-rate vector
        a_pn = N * np.cross(V_r, Omega)                 # ~ perpendicular to closing velocity
        v_cmd = v_cmd + a_pn * 0.3                       # small dt-scaled blend

    sp = float(np.linalg.norm(v_cmd))
    if sp > v_max:
        v_cmd = v_cmd / sp * v_max
    hit = r < kill_radius
    return v_cmd, r, Vc, hit, {"t_go": tg, "speed": speed}
