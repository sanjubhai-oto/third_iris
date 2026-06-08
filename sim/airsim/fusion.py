"""
fusion.py -- Pure-function guidance fusion for a chaser drone (Ego) following a target.

This module fuses VISION-based guidance (YOLO bounding-box bearing from a forward
camera) with LOCATION-based guidance (known target world position) to produce a
desired heading and a world-NED velocity command for a multirotor chaser.

Conventions
-----------
Coordinates are world NED:
    x = North, y = East, z = Down  (altitude up == negative z)
The Ego carries a forward-facing camera aligned with body +X; yaw rotates it.
All angles are radians unless a name explicitly ends in `_deg`.

Design note: every function here is PURE -- it only reads its arguments and returns
new values. Only `numpy` and `math` are imported, so this file has no AirSim (or
torch) dependency and can be unit-tested standalone.
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Small angle helpers
# ---------------------------------------------------------------------------

def _clamp(x, lo, hi):
    """Clamp scalar x to the inclusive range [lo, hi]."""
    return max(lo, min(hi, x))


def wrap_pi(angle):
    """Wrap an angle in radians to (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def circular_mean(angles, weights):
    """
    Weighted circular mean of a set of angles (radians).

    Averaging angles naively (e.g. mean of 179 deg and -179 deg) gives a wrong
    result because of the wrap-around at +/-pi. Instead we convert each angle to a
    unit vector, take the weighted vector sum, and read back the resultant angle.

    Parameters
    ----------
    angles  : sequence of float, angles in radians.
    weights : sequence of float, non-negative weights (need not sum to 1).

    Returns
    -------
    float : the weighted mean angle in radians, wrapped to (-pi, pi].
            Falls back to the plain (unweighted) circular mean if all weights are 0.
    """
    angles = np.asarray(angles, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if angles.size == 0:
        return 0.0

    wsum = float(np.sum(weights))
    if wsum <= 0.0:
        # Degenerate weights: fall back to equal weighting so we still return
        # something meaningful rather than dividing by zero.
        weights = np.ones_like(angles)

    s = float(np.sum(weights * np.sin(angles)))
    c = float(np.sum(weights * np.cos(angles)))

    if s == 0.0 and c == 0.0:
        # Vectors cancelled exactly (e.g. opposite angles, equal weight).
        # Direction is undefined; return the first angle as a stable choice.
        return wrap_pi(float(angles[0]))

    return math.atan2(s, c)


# ---------------------------------------------------------------------------
# 1. Vision: bounding box -> bearing
# ---------------------------------------------------------------------------

def bbox_to_bearing(cx, cy, img_w, img_h, hfov_deg=90.0):
    """
    Convert a detection bounding-box center pixel to angular offsets from the
    image center, using a pinhole camera approximation.

    Parameters
    ----------
    cx, cy        : float, bounding-box center pixel (origin top-left).
    img_w, img_h  : int/float, image width and height in pixels.
    hfov_deg      : float, horizontal field of view in degrees.

    Returns
    -------
    (yaw_err_rad, pitch_err_rad) : tuple of float.
        yaw_err  > 0  => target is to the RIGHT of image center.
        pitch_err > 0 => target is ABOVE image center.

    The vertical FOV is derived from the horizontal FOV and the aspect ratio:
        tan(vfov/2) = tan(hfov/2) * (img_h / img_w)
    For each axis we normalise the pixel offset to [-1, 1] across the half-image
    and map it through the pinhole relation:
        angle = atan( normalized_offset * tan(fov/2) )
    """
    img_w = float(img_w)
    img_h = float(img_h)

    half_hfov = math.radians(hfov_deg) / 2.0
    # Vertical FOV from aspect ratio (height / width).
    half_vfov = math.atan(math.tan(half_hfov) * (img_h / img_w))

    # Normalised offsets from center, in [-1, 1] across each half-image.
    # x: pixel increases to the right  -> positive yaw_err to the right.
    nx = (cx - img_w / 2.0) / (img_w / 2.0)
    # y: pixel increases downward; we negate so "above center" is positive pitch.
    ny = -(cy - img_h / 2.0) / (img_h / 2.0)

    yaw_err_rad = math.atan(nx * math.tan(half_hfov))
    pitch_err_rad = math.atan(ny * math.tan(half_vfov))

    return (yaw_err_rad, pitch_err_rad)


# ---------------------------------------------------------------------------
# 2. Location: relative geometry between Ego and target
# ---------------------------------------------------------------------------

def relative_location(ego_xyz, target_xyz):
    """
    Compute relative geometry from Ego to target in world NED.

    Parameters
    ----------
    ego_xyz, target_xyz : sequence (n, e, d) world NED coordinates (meters).

    Returns
    -------
    dict with keys:
        "range"   : horizontal (N-E plane) distance, meters.
        "range3d" : full 3D distance, meters.
        "az_rad"  : absolute azimuth to target = atan2(de, dn). 0 = due North,
                    +pi/2 = due East. This matches NED yaw convention.
        "el_rad"  : elevation angle; positive means the target is HIGHER than Ego.
        "dn"      : North delta (target - ego).
        "de"      : East delta.
        "dd"      : Down delta (target - ego); positive means target is lower.
    """
    en, ee, ed = float(ego_xyz[0]), float(ego_xyz[1]), float(ego_xyz[2])
    tn, te, td = float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])

    dn = tn - en
    de = te - ee
    dd = td - ed  # positive => target farther down (lower)

    rng = math.hypot(dn, de)
    rng3d = math.sqrt(dn * dn + de * de + dd * dd)

    az_rad = math.atan2(de, dn)
    # Elevation: target higher => smaller (more negative) d => -dd positive.
    el_rad = math.atan2(-dd, rng) if rng > 0.0 else math.atan2(-dd, 1e-9)

    return {
        "range": rng,
        "range3d": rng3d,
        "az_rad": az_rad,
        "el_rad": el_rad,
        "dn": dn,
        "de": de,
        "dd": dd,
    }


# ---------------------------------------------------------------------------
# 3. Core fusion of vision + location guidance
# ---------------------------------------------------------------------------

def fuse_guidance(ego_xyz, ego_yaw_rad, vision=None, target_xyz=None,
                  hfov_deg=90.0, img_wh=(1280, 720)):
    """
    Fuse vision-based and location-based guidance into a single desired heading.

    Parameters
    ----------
    ego_xyz      : (n, e, d) world NED of the Ego.
    ego_yaw_rad  : current Ego yaw (radians, NED: 0 = North, +East).
    vision       : None, or dict with keys:
                       "cx", "cy"   : detection center pixel,
                       "conf"       : detection confidence in [0, 1],
                       "track_id"   : (unused here, passed through context).
                   None means no detection this frame.
    target_xyz   : None, or (n, e, d) the target's known world location.
    hfov_deg     : camera horizontal FOV (degrees).
    img_wh       : (img_w, img_h) image size in pixels.

    Returns
    -------
    dict with keys:
        "desired_yaw_rad" : absolute world yaw to point Ego at the target.
        "range"           : best available horizontal range estimate (m) or None.
        "source"          : 'vision' | 'location' | 'fused' | 'none'.
        "yaw_err_rad"     : vision bearing error (camera-frame) if vision present,
                            else the heading error vs. ego_yaw toward the target.
        "have_vision"     : bool, whether a vision detection was used.
        "have_location"   : bool, whether a target location was used.

    Fusion logic
    ------------
    * VISION bearing: desired absolute yaw = ego_yaw + bbox_yaw_err (camera +X is
      body +X, so a horizontal bbox offset is a yaw offset).
    * LOCATION azimuth: absolute az toward the target from relative_location.
    * When BOTH are available we blend the two absolute yaws via a confidence-
      weighted circular mean with weights [w_vision, 1 - w_vision], where
      w_vision = clamp(conf, 0, 1). This degrades gracefully to vision-only as
      conf -> 1 and to location-only as conf -> 0.
    * Range prefers the location solution; with vision-only it is left as None
      (a bbox-size range estimate could be slotted in here if box dims were given).
    """
    img_w, img_h = img_wh

    have_vision = vision is not None
    have_location = target_xyz is not None

    vision_yaw = None
    yaw_err = 0.0
    w_vision = 0.0

    if have_vision:
        conf = float(vision.get("conf", 1.0))
        w_vision = _clamp(conf, 0.0, 1.0)
        yaw_err, _pitch_err = bbox_to_bearing(
            float(vision["cx"]), float(vision["cy"]), img_w, img_h, hfov_deg
        )
        # Absolute desired heading implied by the camera bearing.
        vision_yaw = wrap_pi(ego_yaw_rad + yaw_err)

    location_az = None
    rng = None
    if have_location:
        rel = relative_location(ego_xyz, target_xyz)
        location_az = rel["az_rad"]
        rng = rel["range"]

    # ----- Decide the desired yaw and tag the source -----
    if have_vision and have_location:
        # Confidence-weighted circular blend of the two absolute headings.
        desired_yaw = circular_mean(
            [vision_yaw, location_az],
            [w_vision, 1.0 - w_vision],
        )
        source = "fused"
    elif have_vision:
        desired_yaw = vision_yaw
        source = "vision"
    elif have_location:
        desired_yaw = location_az
        # With no vision, the "yaw error" is the heading error toward the target.
        yaw_err = wrap_pi(location_az - ego_yaw_rad)
        source = "location"
    else:
        # No guidance at all: hold current heading, no range.
        desired_yaw = wrap_pi(ego_yaw_rad)
        yaw_err = 0.0
        source = "none"

    return {
        "desired_yaw_rad": wrap_pi(desired_yaw),
        "range": rng,
        "source": source,
        "yaw_err_rad": yaw_err,
        "have_vision": have_vision,
        "have_location": have_location,
    }


# ---------------------------------------------------------------------------
# 4. Follow controller: produce a world-NED velocity command
# ---------------------------------------------------------------------------

def follow_velocity(ego_xyz, desired_yaw_rad, range_now, chase_dist=8.0,
                    alt_match_d=None, kp=0.8, vmax=8.0):
    """
    Produce a world-NED velocity command to chase the target.

    The controller drives the Ego along `desired_yaw_rad` so as to close the gap
    between the current range and the desired standoff `chase_dist`, while
    optionally matching the target altitude.

    Parameters
    ----------
    ego_xyz         : (n, e, d) world NED of the Ego (only d is used, for altitude).
    desired_yaw_rad : absolute heading toward the target (radians, NED).
    range_now       : current horizontal range to the target (m). If None, we
                      assume we are far away and command full closing speed.
    chase_dist      : desired standoff distance behind/at the target (m).
    alt_match_d     : None, or the target's Down coordinate to match altitude.
    kp              : proportional gain on the range error.
    vmax            : magnitude clip applied to the horizontal velocity AND,
                      separately, to the vertical velocity.

    Returns
    -------
    (vx, vy, vz, yaw_deg) :
        vx, vy, vz : world-NED velocity command (m/s), clipped to vmax.
        yaw_deg    : desired yaw setpoint in DEGREES (for a yaw command).

    Geometry
    --------
    If range_now > chase_dist we want to move FORWARD along desired_yaw (close in).
    If range_now < chase_dist we want to back off (negative speed along yaw).
    Horizontal speed = kp * (range_now - chase_dist), clipped to [-vmax, vmax].
    The horizontal velocity is then projected onto North/East using desired_yaw:
        vx = speed * cos(yaw), vy = speed * sin(yaw)   (NED yaw convention)
    Vertical speed closes the altitude gap toward alt_match_d, clipped to vmax.
    """
    # --- Horizontal closing speed along the desired heading ---
    if range_now is None:
        range_err = vmax / max(kp, 1e-6)  # forces full-speed approach
    else:
        range_err = float(range_now) - float(chase_dist)

    speed = _clamp(kp * range_err, -vmax, vmax)

    # Project onto North/East. In NED, yaw 0 = North (+x), yaw +pi/2 = East (+y).
    vx = speed * math.cos(desired_yaw_rad)
    vy = speed * math.sin(desired_yaw_rad)

    # Re-clip the horizontal vector magnitude (the components above already
    # respect vmax since |speed| <= vmax, but clip defensively for safety).
    hmag = math.hypot(vx, vy)
    if hmag > vmax and hmag > 0.0:
        scale = vmax / hmag
        vx *= scale
        vy *= scale

    # --- Vertical: match target altitude if requested ---
    if alt_match_d is None:
        vz = 0.0
    else:
        ego_d = float(ego_xyz[2])
        # Positive vz means descend (increase Down). Drive ego_d toward alt_match_d.
        vz = _clamp(kp * (float(alt_match_d) - ego_d), -vmax, vmax)

    yaw_deg = math.degrees(wrap_pi(desired_yaw_rad))

    return (vx, vy, vz, yaw_deg)


# ---------------------------------------------------------------------------
# Self-test (no AirSim dependency)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== fusion.py self-test ===\n")

    # --- circular_mean ---
    cm = circular_mean([math.radians(170), math.radians(-170)], [1.0, 1.0])
    print("circular_mean(170deg, -170deg, equal w) = "
          "{:.2f} deg (expect ~180)".format(math.degrees(cm)))
    cm2 = circular_mean([math.radians(10), math.radians(50)], [3.0, 1.0])
    print("circular_mean(10deg w=3, 50deg w=1)      = "
          "{:.2f} deg (expect ~20)".format(math.degrees(cm2)))

    # --- bbox_to_bearing ---
    print("\n-- bbox_to_bearing (1280x720, hfov=90) --")
    yc, pc = bbox_to_bearing(640, 360, 1280, 720)
    print("center pixel        -> yaw {:.2f} deg, pitch {:.2f} deg (expect 0,0)"
          .format(math.degrees(yc), math.degrees(pc)))
    yr, pr = bbox_to_bearing(1280, 360, 1280, 720)
    print("right edge          -> yaw {:.2f} deg (expect +45)"
          .format(math.degrees(yr)))
    yt, pt = bbox_to_bearing(640, 0, 1280, 720)
    print("top edge            -> pitch {:.2f} deg (expect positive, target above)"
          .format(math.degrees(pt)))

    # --- relative_location ---
    print("\n-- relative_location --")
    ego = (0.0, 0.0, -10.0)        # 10 m up
    tgt = (10.0, 10.0, -20.0)      # NE and higher (20 m up)
    rel = relative_location(ego, tgt)
    print("ego={} target={}".format(ego, tgt))
    print("  range   = {:.2f} m (expect ~14.14)".format(rel["range"]))
    print("  range3d = {:.2f} m".format(rel["range3d"]))
    print("  az      = {:.2f} deg (expect ~45, NE)".format(math.degrees(rel["az_rad"])))
    print("  el      = {:.2f} deg (expect positive, target higher)"
          .format(math.degrees(rel["el_rad"])))

    # --- fuse_guidance: vision only ---
    print("\n-- fuse_guidance: VISION only --")
    ego_yaw = math.radians(0.0)  # facing North
    vis = {"cx": 960, "cy": 360, "conf": 0.9, "track_id": 1}  # target to the right
    g = fuse_guidance(ego, ego_yaw, vision=vis, target_xyz=None)
    print("  source={}  desired_yaw={:.2f} deg  yaw_err={:.2f} deg  range={}"
          .format(g["source"], math.degrees(g["desired_yaw_rad"]),
                  math.degrees(g["yaw_err_rad"]), g["range"]))

    # --- fuse_guidance: location only ---
    print("\n-- fuse_guidance: LOCATION only --")
    g = fuse_guidance(ego, ego_yaw, vision=None, target_xyz=tgt)
    print("  source={}  desired_yaw={:.2f} deg  yaw_err={:.2f} deg  range={:.2f}"
          .format(g["source"], math.degrees(g["desired_yaw_rad"]),
                  math.degrees(g["yaw_err_rad"]), g["range"]))

    # --- fuse_guidance: both (fused) ---
    print("\n-- fuse_guidance: BOTH (fused) --")
    g = fuse_guidance(ego, ego_yaw, vision=vis, target_xyz=tgt)
    print("  source={}  desired_yaw={:.2f} deg  range={:.2f}  "
          "have_vision={} have_location={}"
          .format(g["source"], math.degrees(g["desired_yaw_rad"]),
                  g["range"], g["have_vision"], g["have_location"]))

    # --- fuse_guidance: none ---
    print("\n-- fuse_guidance: NONE --")
    g = fuse_guidance(ego, ego_yaw, vision=None, target_xyz=None)
    print("  source={}  desired_yaw={:.2f} deg  range={}"
          .format(g["source"], math.degrees(g["desired_yaw_rad"]), g["range"]))

    # --- follow_velocity ---
    print("\n-- follow_velocity --")
    g = fuse_guidance(ego, ego_yaw, vision=None, target_xyz=tgt)
    vx, vy, vz, yaw_deg = follow_velocity(
        ego, g["desired_yaw_rad"], g["range"],
        chase_dist=8.0, alt_match_d=tgt[2], kp=0.8, vmax=8.0,
    )
    print("  range_now={:.2f}  chase_dist=8.0".format(g["range"]))
    print("  vel=({:.2f}, {:.2f}, {:.2f}) m/s  yaw_set={:.2f} deg"
          .format(vx, vy, vz, yaw_deg))
    print("  (vz should be negative => climbing to match higher target)")

    print("\n=== self-test complete ===")
