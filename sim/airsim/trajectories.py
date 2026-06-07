"""Pure-math 3D trajectory library for a moving target drone.

Coordinates are world NED (North-East-Down) meters:
    x = North, y = East, z = Down.
Altitude H meters *above* ground is therefore z = -H (negative Down).

Every generator returns a plain Python list of (n, e, d) float tuples,
sampled evenly along its parametric curve. There are NO external
dependencies beyond ``numpy`` and the standard ``math`` module -- in
particular this file does NOT import airsim / cosysairsim, so it can be
used and unit-tested without a simulator running.
"""

import math

import numpy as np


# ---------------------------------------------------------------------------
# Trajectory generators
# ---------------------------------------------------------------------------

def circle(center=(0.0, 0.0, -20.0), radius=25.0, turns=2.0, points=160):
    """Horizontal circle at a fixed altitude.

    Parameters
    ----------
    center : (n, e, d) center of the circle in world NED meters.
    radius : circle radius in meters.
    turns  : number of full revolutions over the path.
    points : number of waypoints to sample.

    Returns a list of (n, e, d) tuples.
    """
    cn, ce, cd = center
    out = []
    for i in range(points):
        # t goes 0 -> 1 across the whole (possibly multi-turn) path.
        t = i / (points - 1) if points > 1 else 0.0
        ang = 2.0 * math.pi * turns * t
        n = cn + radius * math.cos(ang)
        e = ce + radius * math.sin(ang)
        d = cd
        out.append((float(n), float(e), float(d)))
    return out


def figure8(center=(0.0, 0.0, -20.0), size=25.0, turns=2.0, points=200):
    """Figure-eight (lemniscate of Gerono) at a fixed altitude.

    The lemniscate of Gerono is parametrized as:
        x = cos(t)
        y = sin(t) * cos(t)
    which traces a crisp figure-eight. ``size`` scales it in meters.
    """
    cn, ce, cd = center
    out = []
    for i in range(points):
        t = i / (points - 1) if points > 1 else 0.0
        ang = 2.0 * math.pi * turns * t
        n = cn + size * math.cos(ang)
        e = ce + size * math.sin(ang) * math.cos(ang)
        d = cd
        out.append((float(n), float(e), float(d)))
    return out


def spiral(center=(0.0, 0.0, -20.0), r0=5.0, r1=35.0, turns=3.0, points=200):
    """Planar spiral that expands from ``r0`` to ``r1`` and back to ``r0``.

    The radius grows linearly to the outer radius at the halfway point,
    then shrinks back, so the path starts and ends near the inner radius.
    Altitude is held constant.
    """
    cn, ce, cd = center
    out = []
    for i in range(points):
        t = i / (points - 1) if points > 1 else 0.0
        # Triangular radius profile: r0 -> r1 -> r0.
        tri = 1.0 - abs(2.0 * t - 1.0)  # 0 -> 1 -> 0
        radius = r0 + (r1 - r0) * tri
        ang = 2.0 * math.pi * turns * t
        n = cn + radius * math.cos(ang)
        e = ce + radius * math.sin(ang)
        d = cd
        out.append((float(n), float(e), float(d)))
    return out


def helix(center=(0.0, 0.0, -15.0), radius=20.0, turns=3.0, climb=15.0,
          points=200):
    """Circle in the horizontal plane combined with a steady altitude change.

    The drone climbs by ``climb`` meters over the whole path (climb means
    going *up*, i.e. Down decreases by ``climb``).
    """
    cn, ce, cd = center
    out = []
    for i in range(points):
        t = i / (points - 1) if points > 1 else 0.0
        ang = 2.0 * math.pi * turns * t
        n = cn + radius * math.cos(ang)
        e = ce + radius * math.sin(ang)
        # Climbing upward => Down coordinate decreases.
        d = cd - climb * t
        out.append((float(n), float(e), float(d)))
    return out


def zigzag(start=(0.0, 0.0, -20.0), length=80.0, amplitude=20.0, segments=6,
           points=200):
    """Sawtooth sweep: advance along North while oscillating in East.

    The drone marches a total of ``length`` meters in North, while the
    East coordinate follows a triangular (sawtooth) wave with the given
    ``amplitude`` and number of ``segments`` (zig + zag legs). Altitude
    is held constant.
    """
    sn, se, sd = start
    out = []
    for i in range(points):
        t = i / (points - 1) if points > 1 else 0.0
        n = sn + length * t
        # Triangular wave in East: oscillates between -amplitude and +amplitude.
        phase = (segments * t) % 1.0  # 0 -> 1 within each segment
        tri = 1.0 - abs(2.0 * phase - 1.0)  # 0 -> 1 -> 0
        e = se + amplitude * (2.0 * tri - 1.0)  # map to [-amp, +amp]
        d = sd
        out.append((float(n), float(e), float(d)))
    return out


def random_waypoints(center=(0.0, 0.0, -20.0), extent=40.0,
                     alt_range=(-25.0, -12.0), n=8, points=200, seed=0):
    """Smooth-ish random tour through ``n`` random waypoints.

    Picks ``n`` random anchor waypoints within +/-``extent`` (in North and
    East, relative to ``center``) and a random altitude within
    ``alt_range`` (Down coordinates), using a deterministic
    ``numpy.random.RandomState(seed)``. The anchors are then linearly
    interpolated to produce ``points`` continuous samples.
    """
    cn, ce, _cd = center
    rng = np.random.RandomState(seed)

    # Random anchor waypoints.
    anchors = []
    for _ in range(n):
        an = cn + rng.uniform(-extent, extent)
        ae = ce + rng.uniform(-extent, extent)
        ad = rng.uniform(alt_range[0], alt_range[1])
        anchors.append((an, ae, ad))

    if n == 1:
        return [(float(anchors[0][0]), float(anchors[0][1]),
                 float(anchors[0][2])) for _ in range(points)]

    # Linearly interpolate along the polyline of anchors. The path is
    # parametrized by segment index so that each leg gets an equal share
    # of the parameter space (simple and continuous).
    out = []
    n_segments = n - 1
    for i in range(points):
        t = i / (points - 1) if points > 1 else 0.0
        # Global position along the polyline in [0, n_segments].
        pos = t * n_segments
        seg = int(math.floor(pos))
        if seg >= n_segments:  # clamp the final endpoint
            seg = n_segments - 1
        local = pos - seg
        a = anchors[seg]
        b = anchors[seg + 1]
        nn = a[0] + (b[0] - a[0]) * local
        ee = a[1] + (b[1] - a[1]) * local
        dd = a[2] + (b[2] - a[2]) * local
        out.append((float(nn), float(ee), float(dd)))
    return out


def line_sweep(start=(0.0, 0.0, -20.0), end=(60.0, 0.0, -20.0), points=120):
    """Straight line from ``start`` to ``end`` in 3D NED space."""
    sn, se, sd = start
    en, ee, ed = end
    out = []
    for i in range(points):
        t = i / (points - 1) if points > 1 else 0.0
        n = sn + (en - sn) * t
        e = se + (ee - se) * t
        d = sd + (ed - sd) * t
        out.append((float(n), float(e), float(d)))
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TRAJECTORIES = {
    "circle": circle,
    "figure8": figure8,
    "spiral": spiral,
    "helix": helix,
    "zigzag": zigzag,
    "random": random_waypoints,
    "line": line_sweep,
}


def list_trajectories():
    """Return the sorted list of registered trajectory names."""
    return sorted(TRAJECTORIES.keys())


# ---------------------------------------------------------------------------
# Self-verification (no AirSim required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Available trajectories:", list_trajectories())
    print()
    for name in list_trajectories():
        gen = TRAJECTORIES[name]
        path = gen()  # use each generator's default parameters
        first = path[0]
        last = path[-1]
        print(
            "{:8s} | {:4d} waypoints | "
            "first=({:8.2f}, {:8.2f}, {:8.2f}) | "
            "last=({:8.2f}, {:8.2f}, {:8.2f})".format(
                name, len(path),
                first[0], first[1], first[2],
                last[0], last[1], last[2],
            )
        )
