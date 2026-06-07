#!/usr/bin/env python3
"""Depth-based reactive obstacle avoidance — a safety layer for the chaser.

Concepts adapted from two AirSim collision-avoidance projects:
  * simondlevy/AirSimTensorFlow — predict an imminent collision from the forward camera and brake
    (it notes the depth image is "like the LiDAR on self-driving cars").
  * mrhosseini75/Semi_Autonomous_Drone_Nav — use depth/LiDAR free-space to perceive and steer.

Here we treat AirSim's **DepthPlanar** image as a forward range sensor: estimate the clearance
straight ahead and the free space per sector, and when an obstacle gets close, brake the forward
motion and steer toward the most open direction (or climb). It runs ON TOP of the standoff guidance —
far from obstacles it does nothing; only when something is near the flight path does it intervene, so
the chaser keeps following the target but won't fly into a building/tree.
"""
from __future__ import annotations

import math

import numpy as np


def _p20(region):
    """20th-percentile valid depth in a region (robust 'nearest stuff' estimate)."""
    v = region[(region > 0.3) & (region < 1e4)]
    return float(np.percentile(v, 20)) if v.size > 30 else float("inf")


def clearance_and_escape(depth, brake_dist=9.0, crit_dist=4.5):
    """From a DepthPlanar image return (ahead_m, severity 0..1, escape_body).

    ``ahead_m`` = clearance straight ahead; ``severity`` ramps 0->1 between brake_dist and crit_dist;
    ``escape_body`` = unit (forward, right, up) suggestion in the BODY frame to get clear (brake +
    steer toward the freer side, climb if both sides are blocked).
    """
    if depth is None:
        return float("inf"), 0.0, (0.0, 0.0, 0.0)
    h, w = depth.shape
    ahead = _p20(depth[int(h * 0.30):int(h * 0.70), int(w * 0.35):int(w * 0.65)])
    if not math.isfinite(ahead) or ahead >= brake_dist:
        return ahead, 0.0, (0.0, 0.0, 0.0)
    sev = float(np.clip((brake_dist - ahead) / max(0.1, brake_dist - crit_dist), 0.0, 1.0))
    left = _p20(depth[:, :int(w * 0.40)])
    right = _p20(depth[:, int(w * 0.60):])
    up = _p20(depth[:int(h * 0.40), :])
    down = _p20(depth[int(h * 0.60):, :])
    fin = lambda x: x if math.isfinite(x) else 1e4
    lateral = 1.0 if fin(right) >= fin(left) else -1.0          # steer to the more open side
    vert = 0.0
    if max(fin(left), fin(right)) < brake_dist:                 # both sides blocked -> go vertical
        vert = 1.0 if fin(up) >= fin(down) else -1.0
    esc = np.array([-1.0, lateral, 0.8 * vert])                 # brake fwd + steer + maybe climb
    n = float(np.linalg.norm(esc))
    esc = esc / n if n > 0 else esc
    return ahead, sev, (float(esc[0]), float(esc[1]), float(esc[2]))


def _body_to_world(vec_body, yaw):
    """(forward, right, up) body -> world-NED direction."""
    f, r, u = vec_body
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([f * cy - r * sy, f * sy + r * cy, -u])     # up = -down


def apply_avoidance(v_world, depth, ego_yaw, max_speed, brake_dist=9.0, crit_dist=4.5):
    """Blend obstacle avoidance into the guidance velocity.

    Returns (v_world_out, ahead_m, avoiding). Gentle at ``brake_dist`` and a full escape maneuver at
    ``crit_dist``; the inward (toward-obstacle) component is cancelled so the chaser can't drive into
    something even while the target is dead ahead.
    """
    ahead, sev, esc_body = clearance_and_escape(depth, brake_dist, crit_dist)
    if sev <= 0.0:
        return np.asarray(v_world, float), ahead, False
    esc_world = _body_to_world(esc_body, ego_yaw) * max_speed
    v = np.asarray(v_world, float)
    # cancel any remaining velocity straight ahead (into the obstacle), then blend in the escape
    fwd = np.array([math.cos(ego_yaw), math.sin(ego_yaw), 0.0])
    into = float(np.dot(v, fwd))
    if into > 0:
        v = v - sev * into * fwd
    v = (1.0 - sev) * v + sev * esc_world
    return v, ahead, True
