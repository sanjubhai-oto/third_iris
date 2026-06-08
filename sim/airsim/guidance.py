#!/usr/bin/env python3
"""Standoff air-to-air guidance for an FPV (body-fixed camera) chaser.

The old controller only moved *radially* (toward/away along the bearing) and *yawed* to center the
target. A target moving across the frame could then only be followed by spinning the whole aircraft,
so it traced a pursuit circle and lagged behind by an error proportional to the target's speed.

This module replaces that with a proper **standoff / leader-follower** law that is faithful to what
a real drone can do:

  1. Estimate the target's 3-D position from the camera (bearing from the bbox + range from depth),
     OR from GPS, OR a blend.  -> ``target_from_vision`` / caller supplies GPS.
  2. Run a constant-velocity Kalman (``Vec3KF``) on that position to get a *smoothed position AND
     velocity*, and to *coast* (predict) through frames where the target is momentarily lost.
  3. Command the FULL horizontal velocity = position-P toward the standoff point + the target's
     velocity as **feed-forward**, so the chaser flies *alongside* the target (matches its speed)
     instead of chasing its tail. Vertical velocity tracks the target's altitude + climb/descent FF.
  4. Point the camera with an **absolute, lead-compensated yaw setpoint** (not a relative + integral
     setpoint, which spins) -> the target stays centered even while it maneuvers.

All positions are world-NED (north, east, down[+]). Yaw is radians from North toward East.
"""
from __future__ import annotations

import math

import numpy as np


class Vec3KF:
    """Constant-velocity Kalman filter on a 3-D position measurement.

    State = [pn, pe, pd, vn, ve, vd]. ``update`` fuses a position measurement and returns the
    smoothed (position, velocity); ``predict_only`` coasts on the last estimate when the target is
    not seen this frame.
    """

    def __init__(self, q: float = 4.0, r: float = 0.9, vmax: float = 9.0):
        self.q = float(q)        # process noise (accel uncertainty) — higher tracks maneuvers faster
        self.r = float(r)        # measurement noise (m) — higher trusts the model more (smoother)
        self.vmax = float(vmax)  # hard clamp on the velocity estimate so it can never run away
        self.x = None            # state vector (6,)
        self.P = None

    def _clamp_v(self):
        v = self.x[3:]
        s = float(np.linalg.norm(v))
        if s > self.vmax:
            self.x[3:] = v / s * self.vmax

    def reset(self):
        self.x = None
        self.P = None

    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(6)
        for i in range(3):
            F[i, i + 3] = dt
        return F

    def predict_only(self, dt: float):
        if self.x is None:
            return None, None
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + np.eye(6) * self.q * dt
        self._clamp_v()
        return self.x[:3].copy(), self.x[3:].copy()

    def update(self, z, dt: float):
        z = np.asarray(z, dtype=float)
        if self.x is None:
            self.x = np.concatenate([z, np.zeros(3)])
            self.P = np.eye(6) * 10.0
            return self.x[:3].copy(), self.x[3:].copy()
        # predict
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + np.eye(6) * self.q * dt
        # update (measure position only)
        H = np.zeros((3, 6))
        for i in range(3):
            H[i, i] = 1.0
        R = np.eye(3) * self.r
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        self._clamp_v()
        return self.x[:3].copy(), self.x[3:].copy()


def vfov_from_hfov(hfov_deg: float, width: int, height: int) -> float:
    """Vertical FOV (deg) for a rectilinear camera given horizontal FOV and image aspect."""
    return math.degrees(2.0 * math.atan(math.tan(math.radians(hfov_deg) / 2.0) * height / max(1, width)))


def target_from_vision(ego, ego_yaw, ex, ey, depth, hfov_deg, vfov_deg):
    """Reconstruct the target's world-NED position from a monocular detection.

    ``ex``, ``ey`` are the bbox-center image errors normalized to [-1, 1] (ex>0 right, ey>0 down).
    ``depth`` is the planar (optical-axis) depth in metres at the target. Uses the pinhole relation
    lateral/forward = ex*tan(HFOV/2), vertical/forward = ey*tan(VFOV/2), then rotates the
    camera-frame offset into world NED by the ego heading. Returns np.array([n, e, d]).
    """
    ego = np.asarray(ego, dtype=float)
    thx = ex * math.tan(math.radians(hfov_deg) / 2.0)   # tan of horizontal angle
    thy = ey * math.tan(math.radians(vfov_deg) / 2.0)   # tan of vertical angle (down +)
    fwd = float(depth)
    right = float(depth) * thx
    down = float(depth) * thy
    cy, sy = math.cos(ego_yaw), math.sin(ego_yaw)
    n = ego[0] + fwd * cy - right * sy
    e = ego[1] + fwd * sy + right * cy
    d = ego[2] + down
    return np.array([n, e, d])


def look_at_angles(ego, tp):
    """World yaw, pitch (rad) to point a camera at ``tp`` from ``ego`` (pitch up positive, NED)."""
    los = np.asarray(tp, float)[:3] - np.asarray(ego, float)[:3]
    horiz = math.hypot(los[0], los[1])
    return math.atan2(los[1], los[0]), math.atan2(-los[2], horiz)


def euler_R(roll, pitch, yaw):
    """NED body->world rotation (body x-fwd, y-right, z-down)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
        [-sp,     sr * cp,                cr * cp]])


def quat_to_R(q):
    """AirSim quaternion (x,y,z,w) -> body->world rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    """Rotation matrix -> (x,y,z,w)."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s; x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return (x, y, z, w)


def camera_rel_quat(vehicle_q, yaw_cam, pitch_cam):
    """Camera orientation RELATIVE to the vehicle body that achieves the desired WORLD (yaw,pitch).

    Software gimbal stabilization: R_rel = R_vehicle^T * R_desired_world. Setting the camera to this
    each frame keeps it pointing at the target regardless of how the airframe pitches/rolls/yaws.
    """
    R_v = quat_to_R(vehicle_q)
    R_d = euler_R(0.0, pitch_cam, yaw_cam)
    return R_to_quat(R_v.T @ R_d)


def target_from_vision_cam(ego, yaw_cam, pitch_cam, ex, ey, depth, hfov_deg, vfov_deg):
    """Reconstruct target world pos using the CAMERA's world orientation (gimbal-aware variant)."""
    R = euler_R(0.0, pitch_cam, yaw_cam)
    pt_body = float(depth) * np.array([1.0,
                                       ex * math.tan(math.radians(hfov_deg) / 2.0),
                                       ey * math.tan(math.radians(vfov_deg) / 2.0)])
    return np.asarray(ego, float)[:3] + R @ pt_body


def standoff_command(ego, ego_yaw, tp, tv, gap, max_speed,
                     kp_pos=0.8, kp_z=1.15, vz_max=3.3, lead_t=0.3,
                     yaw_lead_t=0.45, deadband_xy=0.5):
    """Compute (vn, ve, vd, yaw_deg, range_m, yaw_err_deg) to hold ``gap`` behind a moving target.

    Horizontal: velocity = kp_pos * (standoff_point - ego) + target_velocity (feed-forward), so the
    chaser *matches the target's speed* and only the position error is closed by the P term -> flies
    alongside instead of spiraling. Vertical: tracks target altitude + vertical-velocity FF.
    Yaw: an ABSOLUTE world heading toward the *lead-predicted* target (no integral, cannot spin).
    """
    ego = np.asarray(ego, dtype=float)
    tp = np.asarray(tp, dtype=float)
    tv = np.asarray(tv, dtype=float)

    tp_lead = tp + tv * lead_t                      # anticipate where the target will be
    los = tp_lead[:2] - ego[:2]
    rng = float(np.hypot(los[0], los[1]))
    los_hat = los / rng if rng > 1e-3 else np.array([1.0, 0.0])

    # standoff point = gap behind the (lead) target along the line of sight
    p_des = tp_lead[:2] - gap * los_hat
    err_xy = p_des - ego[:2]
    err_n = float(np.hypot(err_xy[0], err_xy[1]))
    # deadband the position term so it doesn't jitter at the setpoint; velocity FF still applies
    scale = 0.0 if err_n < deadband_xy else (err_n - deadband_xy) / err_n
    v_xy = kp_pos * err_xy * scale + tv[:2]
    # CLOSE-RANGE BRAKE: never let the feed-forward surge the chaser inside the gap (that throws the
    # target out of the vertical FOV). When already closer than the gap, cancel any inward radial vel.
    radial = float(np.dot(v_xy, los_hat))
    if rng < gap and radial > 0.0:
        v_xy = v_xy - radial * los_hat
    sp = float(np.hypot(v_xy[0], v_xy[1]))
    if sp > max_speed:
        v_xy = v_xy / sp * max_speed

    # vertical: anticipate the target's altitude (lead) + climb/descent feed-forward. Vertical lag
    # is the dominant centering error for an FPV (yaw-only) camera, so we lead it like the heading.
    vd = float(np.clip(kp_z * (tp_lead[2] - ego[2]) + tv[2], -vz_max, vz_max))

    # absolute, lead-compensated yaw setpoint toward the target
    tp_ylead = tp + tv * yaw_lead_t
    los_y = tp_ylead[:2] - ego[:2]
    az = math.atan2(los_y[1], los_y[0])
    yaw_err = math.atan2(math.sin(az - ego_yaw), math.cos(az - ego_yaw))
    return float(v_xy[0]), float(v_xy[1]), vd, math.degrees(az), rng, math.degrees(yaw_err)
