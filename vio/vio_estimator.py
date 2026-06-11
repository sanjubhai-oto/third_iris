#!/usr/bin/env python3
"""Visual-Inertial Odometry (VIO) for a GPS-denied / jammed chaser.

Estimates the drone's OWN pose (position + attitude) from its forward RGB-D camera + IMU, so when
GPS is jammed the onboard computer can keep localizing itself and navigating to the target.

Design (loosely-coupled RGB-D VO + IMU, per research):
  * RGB-D visual odometry — Shi-Tomasi features + Lucas-Kanade optical flow (with a forward-backward
    consistency check), back-projected to 3-D using the DepthPlanar image (metric scale, so NO
    monocular scale drift), then a 3-D<->3-D rigid transform via Umeyama SVD inside RANSAC gives the
    frame-to-frame camera translation.
  * IMU — the gyro propagates attitude every frame; the accelerometer (gravity-compensated) coasts
    velocity/position through frames where VO fails (sky / low texture). AirSim's accel INCLUDES
    gravity, so a_world = R_wb @ accel_body + g_ned with g_ned=[0,0,+9.81] reads ~0 at hover.

Operational model (matches the GPS-available / jammed concept):
  * GPS available  -> ``anchor(pos, quat)`` is called every frame, so the estimate tracks truth with
    zero drift (VIO is "primed").
  * GPS jammed     -> ``update(...)`` free-runs from the last anchor; drift accumulates and is what we
    measure against ground truth.

Camera optical frame: x=right, y=down, z=forward. Body: x=fwd, y=right, z=down (NED). World: NED.
R_bc (camera->body) = [[0,0,1],[1,0,0],[0,1,0]].
"""
from __future__ import annotations

import math

import cv2
import numpy as np

try:
    import open3d as _o3d                      # optional proven RGB-D odometry backend (Steinbrücker/Park)
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False

R_BC = np.array([[0.0, 0.0, 1.0],
                 [1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0]])      # camera-optical -> body (NED)
G_NED = np.array([0.0, 0.0, 9.81])      # gravity in NED (down +), to cancel accel's gravity term


# ----------------------------------------------------------------- quaternion utils (body->world)
def quat_mul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz])


def quat_norm(q):
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([0.0, 0.0, 0.0, 1.0])


def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]])


def quat_from_gyro(omega, dt):
    """Small-angle quaternion increment from a body angular rate (rad/s)."""
    ang = float(np.linalg.norm(omega)) * dt
    if ang < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = omega / np.linalg.norm(omega)
    s = math.sin(ang / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(ang / 2.0)])


def yaw_of(q):
    R = quat_to_R(q)
    return math.atan2(R[1, 0], R[0, 0])


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def rpy_of(q):
    """roll, pitch, yaw (rad) from a body->world quaternion (NED ZYX)."""
    R = quat_to_R(q)
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def euler_to_quat(roll, pitch, yaw):
    """(roll,pitch,yaw) rad -> (x,y,z,w) body->world quaternion."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy,
                     cr * cp * cy + sr * sp * sy])


def mag_heading(mag_body, roll, pitch):
    """Tilt-compensated magnetic heading (rad) from a body-frame magnetometer vector."""
    mx, my, mz = float(mag_body[0]), float(mag_body[1]), float(mag_body[2])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    mxh = mx * cp + mz * sp
    myh = mx * sr * sp + my * cr - mz * sr * cp
    return math.atan2(-myh, mxh)


# ----------------------------------------------------------------- rigid 3D-3D (Umeyama + RANSAC)
def umeyama(P, Q):
    """Least-squares rigid R,t with q_i ~= R p_i + t (no scale). Returns (R, t)."""
    cp = P.mean(axis=0); cq = Q.mean(axis=0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cq - R @ cp
    return R, t


def ransac_rigid(P, Q, iters=60, tau=0.10, rng=None):
    """RANSAC over umeyama. Returns (R, t, n_inliers) or (None, None, 0)."""
    n = len(P)
    if n < 6:
        return None, None, 0
    rng = rng or np.random
    best_in = None; best_cnt = 0
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            R, t = umeyama(P[idx], Q[idx])
        except np.linalg.LinAlgError:
            continue
        resid = np.linalg.norm(Q - (P @ R.T + t), axis=1)
        inl = resid < tau
        c = int(inl.sum())
        if c > best_cnt:
            best_cnt = c; best_in = inl
    if best_in is None or best_cnt < 8:
        return None, None, best_cnt
    R, t = umeyama(P[best_in], Q[best_in])     # refit on inliers
    return R, t, best_cnt


class VIOEstimator:
    """RGB-D + IMU odometry. Anchor to GPS when available; free-run (dead-reckon) when jammed."""

    def __init__(self, hfov_deg=90.0, width=1280, height=720, max_depth=45.0, backend="umeyama",
                 vo_depth_max=35.0):
        self.fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        self.fy = self.fx
        self.cx = width / 2.0; self.cy = height / 2.0
        self.max_depth = max_depth
        # VO correspondences use only nearer depth: depth noise grows ~quadratically with range, so far
        # points poison the 3D-3D fit. Detection/range can still use the full max_depth elsewhere.
        self.vo_depth_max = min(vo_depth_max, max_depth)
        # VO front-end: "umeyama" = our LK+RANSAC 3D-3D (validated); "open3d" = proven Open3D RGB-D
        # direct odometry (Steinbrücker/Park) if installed. Falls back to umeyama if open3d missing.
        self.backend = backend if (backend != "open3d" or _HAS_O3D) else "umeyama"
        self._o3d_intr = None
        self._o3d_init = np.eye(4)            # odo_init seed (IMU-predicted relative pose; identity default)
        if self.backend == "open3d":
            self._o3d_intr = _o3d.camera.PinholeCameraIntrinsic(
                int(width), int(height), self.fx, self.fy, self.cx, self.cy)
        # state (world NED)
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([0.0, 0.0, 0.0, 1.0])      # body->world
        self.prev_gray = None
        self.prev_depth = None
        self.last_vo_ok = False
        self.coast = 0                                # consecutive VO-failure frames
        self.rng = np.random.default_rng(0)
        self.mag_offset = None                        # calibrated (true_yaw - raw_mag_heading) at anchor
        self.gyro_bias = np.zeros(3)                  # estimated during near-static frames (ZUPT-style)
        self.baro_ref = None                          # baro altitude captured at last anchor
        self.anchor_z = 0.0                           # world-NED z (down+) at last anchor

    # ---- anchoring (called while GPS is available) ----
    def anchor(self, pos, quat):
        self.p = np.asarray(pos, float).copy()
        self.q = quat_norm(np.asarray(quat, float))
        self.v[:] = 0.0
        self.anchor_z = float(self.p[2])

    def calibrate_mag(self, mag_body):
        """Calibrate the compass offset against the (truth-anchored) attitude — call while GPS is on."""
        r, p, y = rpy_of(self.q)
        self.mag_offset = _wrap(y - mag_heading(mag_body, r, p))

    def calibrate_baro(self, baro_alt):
        """Pin the barometric reference to the (truth-anchored) altitude — call while GPS is on.

        Afterwards a baro altitude reading constrains world z (down+) to anchor_z - (alt - baro_ref),
        which removes the vertical channel from the dead-reckoning drift (gravity is observable, so this
        is a cheap, high-value 1-D update — see VINS observability)."""
        self.baro_ref = float(baro_alt)
        self.anchor_z = float(self.p[2])

    def _apply_gravity(self, accel, k=0.04):
        """Pin roll/pitch to the accelerometer gravity vector when linear accel is small.

        AirSim accel includes gravity (reads ~ -g in body at hover), so at low net acceleration the
        measured specific force points along world-up in body. Roll/pitch are observable from it; yaw is
        not. Blending bounds tilt drift and stops accel error leaking into the visual yaw channel."""
        a = np.asarray(accel, float)
        an = float(np.linalg.norm(a))
        if an < 1e-3 or abs(an - 9.81) > 1.8:        # only trust accel as gravity near-static
            return
        g_body = -a / an                              # world-down direction expressed in body
        roll_m = math.atan2(g_body[1], g_body[2])
        pitch_m = math.atan2(-g_body[0], math.hypot(g_body[1], g_body[2]))
        r, p, y = rpy_of(self.q)
        self.q = quat_norm(euler_to_quat(r + k * _wrap(roll_m - r),
                                         p + k * _wrap(pitch_m - p), y))

    def _apply_mag(self, mag_body, k=0.06):
        """Correct yaw toward the magnetometer heading (bounds gyro yaw drift). roll/pitch untouched."""
        if self.mag_offset is None:
            return
        r, p, y = rpy_of(self.q)
        mag_y = _wrap(mag_heading(mag_body, r, p) + self.mag_offset)
        self.q = quat_norm(euler_to_quat(r, p, y + k * _wrap(mag_y - y)))

    def _backproject(self, pts, depth, zmax=None):
        """pts (N,2) pixel -> (valid_mask, P (M,3) camera-frame 3D)."""
        zmax = self.max_depth if zmax is None else zmax
        u = pts[:, 0]; v = pts[:, 1]
        ui = np.clip(np.round(u).astype(int), 0, depth.shape[1] - 1)
        vi = np.clip(np.round(v).astype(int), 0, depth.shape[0] - 1)
        Z = depth[vi, ui]                            # nearest-neighbour (avoid edge mixing)
        ok = (Z > 0.3) & (Z < zmax) & np.isfinite(Z)
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy
        return ok, np.stack([X, Y, Z], axis=1)

    def _vo_open3d(self, gray, depth, R_wc):
        """Frame-to-frame camera displacement (world NED) via Open3D RGB-D direct odometry."""
        try:
            to3 = lambda g: np.ascontiguousarray(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)) if g.ndim == 2 else g
            mk = lambda g, d: _o3d.geometry.RGBDImage.create_from_color_and_depth(
                _o3d.geometry.Image(to3(g)),
                _o3d.geometry.Image(np.ascontiguousarray(d.astype(np.float32))),
                depth_scale=1.0, depth_trunc=self.max_depth, convert_rgb_to_intensity=True)
            src = mk(self.prev_gray, self.prev_depth); tgt = mk(gray, depth)
            opt = _o3d.pipelines.odometry.OdometryOption()
            opt.depth_min = 0.3
            opt.depth_max = self.max_depth          # default 4 m rejects everything a drone sees -> 0% conv
            opt.depth_diff_max = 0.07               # default 0.03 too tight for sim depth
            ok, T, _info = _o3d.pipelines.odometry.compute_rgbd_odometry(
                src, tgt, self._o3d_intr, self._o3d_init,
                _o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), opt)
            if not ok:
                return None
            R = np.asarray(T)[:3, :3]; t = np.asarray(T)[:3, 3]
            return R_wc @ (-R.T @ t)                # camera displacement (prev->cur) in world NED
        except Exception:
            return None

    def update(self, gray, depth, imu, dt, mag=None, baro_alt=None):
        """Advance the estimate one frame. imu = (accel_body(3), gyro_body(3)); mag = body magnetometer
        vector (3) or None; baro_alt = barometric altitude (m) or None. Returns pose dict.

        While GPS is available the caller should still run this (to keep features warm) and then call
        anchor(); while jammed, the returned p/q is the dead-reckoned estimate. The magnetometer (when
        provided + calibrated) bounds yaw drift.
        """
        dt = float(max(1e-3, min(0.3, dt)))
        # ---- IMU: propagate attitude from (bias-corrected) gyro, pin roll/pitch to gravity, then
        #      correct yaw with the magnetometer if present ----
        if imu is not None:
            accel = np.asarray(imu[0], float); gyro = np.asarray(imu[1], float)
            # near-static -> learn gyro bias (ZUPT-style) and trust accel as gravity
            an = float(np.linalg.norm(accel)); wn = float(np.linalg.norm(gyro))
            near_static = wn < 0.06 and abs(an - 9.81) < 0.6
            if near_static:
                self.gyro_bias = 0.98 * self.gyro_bias + 0.02 * gyro
            self.q = quat_norm(quat_mul(self.q, quat_from_gyro(gyro - self.gyro_bias, dt)))
            self._apply_gravity(accel)               # bounds roll/pitch (yaw stays gyro/mag driven)
            if mag is not None:
                self._apply_mag(mag)
            rot_rate = wn                            # body angular-rate magnitude (rad/s)
        else:
            accel = None
            rot_rate = 0.0
        R_wb = quat_to_R(self.q)
        R_wc = R_wb @ R_BC                            # world <- camera-optical

        # ---- RGB-D visual odometry (translation) ----
        vo_disp_world = None
        if self.backend == "open3d" and self.prev_gray is not None and self.prev_depth is not None:
            vo_disp_world = self._vo_open3d(gray, depth, R_wc)
        elif self.prev_gray is not None and self.prev_depth is not None:
            p0 = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=500, qualityLevel=0.01,
                                         minDistance=12, blockSize=7)
            if p0 is not None and len(p0) >= 8:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None,
                                                     winSize=(21, 21), maxLevel=3)
                p0b, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, p1, None,
                                                       winSize=(21, 21), maxLevel=3)
                fb = np.linalg.norm((p0 - p0b).reshape(-1, 2), axis=1)
                good = (st.flatten() == 1) & (st2.flatten() == 1) & (fb < 1.5)
                a = p0.reshape(-1, 2)[good]; b = p1.reshape(-1, 2)[good]
                if len(a) >= 8:
                    okA, P = self._backproject(a, self.prev_depth, zmax=self.vo_depth_max)
                    okB, Q = self._backproject(b, depth, zmax=self.vo_depth_max)
                    m = okA & okB
                    P = P[m]; Q = Q[m]
                    if len(P) >= 8:
                        R, t, nin = ransac_rigid(P, Q, rng=self.rng)
                        if R is not None:
                            disp_cam = -R.T @ t      # camera displacement in prev-cam frame
                            vo_disp_world = R_wc @ disp_cam
        # ---- REJECT-AND-COAST GATE: accept the VO step only if it is finite, within a plausible
        #      per-frame speed, AND consistent with the smooth predicted motion. One bad (sky / low
        #      texture) frame otherwise corrupts the whole trajectory -> this is what bounds the drift.
        # ROTATION GATE: under fast rotation the optical flow is rotation-dominated, so the
        # translation-only 3D-3D fit returns a spurious displacement that can sail through the speed
        # gate and corrupt the trajectory. Reject VO while spinning and just hold position (turns are
        # flown as hovering yaw, so zero translation is the correct prior).
        ROT_GATE = 0.25                                          # rad/s (~14 deg/s)
        spinning = rot_rate > ROT_GATE
        gate_ok = False
        if not spinning and vo_disp_world is not None and np.all(np.isfinite(vo_disp_world)):
            implied_speed = float(np.linalg.norm(vo_disp_world)) / dt
            predicted = self.v * dt                              # where the smooth motion expects us
            dev = float(np.linalg.norm(vo_disp_world - predicted))
            gate = max(1.5, 2.5 * float(np.linalg.norm(self.v)) * dt)
            gate_ok = implied_speed < 12.0 and dev < gate
        if gate_ok:
            self.p = self.p + vo_disp_world
            self.v = 0.6 * self.v + 0.4 * (vo_disp_world / dt)  # smoothed velocity estimate
            self.last_vo_ok = True; self.coast = 0
        else:
            # VO rejected/failed -> coast on IMU accel (gravity-compensated), short horizon only;
            # but while spinning, don't dead-reckon accel either -> just hold and bleed velocity.
            self.last_vo_ok = False; self.coast += 1
            if accel is not None and not spinning and self.coast < 12:
                a_world = R_wb @ accel + G_NED
                self.v = self.v + a_world * dt
                self.p = self.p + self.v * dt
            else:
                self.v *= 0.9                                    # bleed off; don't fly away on garbage

        # ---- barometer altitude fusion: pin world z (down+) to the calibrated baro reference. Vertical
        #      is observable, so this removes the z channel from the dead-reckoning drift entirely. ----
        if baro_alt is not None and self.baro_ref is not None:
            z_meas = self.anchor_z - (float(baro_alt) - self.baro_ref)
            kz = 0.15
            self.p[2] = (1.0 - kz) * self.p[2] + kz * z_meas
            self.v[2] *= (1.0 - kz)

        self.prev_gray = gray
        self.prev_depth = depth
        return {"p": self.p.copy(), "q": self.q.copy(), "v": self.v.copy(),
                "vo_ok": self.last_vo_ok, "coast": self.coast, "yaw": yaw_of(self.q)}
