#!/usr/bin/env python3
"""Generate FRESH brochure assets — no recycled material.

A) Real-dataset detections: scan validation labels for LARGE targets (visible aircraft, not specks),
   require sharpness (Laplacian), run our trained detector, draw a custom aerospace HUD overlay
   (corner brackets + mono telemetry), save 16:10-ish crops centered on the target.
B) RL pursuit trajectories: roll the trained PPO policy in the pursuit env, render top-down
   multi-episode chase geometry in the dark aerospace style (matplotlib).

Outputs into docs/ppt_assets/fresh/.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "ppt_assets" / "fresh"
OUT.mkdir(parents=True, exist_ok=True)

ACC = (26, 92, 255)        # BGR signal orange
CYN = (255, 210, 63)       # BGR cyan
INK = (246, 238, 233)
DRK = (17, 11, 7)


def hud_box(im, x1, y1, x2, y2, label, conf, rng_m):
    """Aerospace HUD: corner brackets + crosshair + mono telemetry block."""
    h, w = im.shape[:2]
    t = max(2, int(w / 640))
    L = max(14, int((x2 - x1) * 0.28))
    for (cx, cy, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(im, (cx, cy), (cx + dx * L, cy), ACC, t, cv2.LINE_AA)
        cv2.line(im, (cx, cy), (cx, cy + dy * L), ACC, t, cv2.LINE_AA)
    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
    g = max(8, int((x2 - x1) * 0.12))
    for (a, b) in [((mx - 2 * g, my), (mx - g, my)), ((mx + g, my), (mx + 2 * g, my)),
                   ((mx, my - 2 * g), (mx, my - g)), ((mx, my + g), (mx, my + 2 * g))]:
        cv2.line(im, a, b, CYN, max(1, t - 1), cv2.LINE_AA)
    txt = f"{label.upper()}  P{conf:.2f}  RNG {rng_m:.1f}M"
    fs = max(0.45, w / 1600)
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    bx, by = x1, max(th + 14, y1 - 10)
    cv2.rectangle(im, (bx - 4, by - th - 8), (bx + tw + 10, by + 6), DRK, -1)
    cv2.rectangle(im, (bx - 4, by - th - 8), (bx + tw + 10, by + 6), ACC, 1, cv2.LINE_AA)
    cv2.putText(im, txt, (bx + 3, by), cv2.FONT_HERSHEY_SIMPLEX, fs, INK, 1, cv2.LINE_AA)
    return im


def frame_chrome(im, tagl="THIRD IRIS // LIVE INFERENCE", tagr="VISION ONLY"):
    h, w = im.shape[:2]
    fs = max(0.42, w / 1900)
    cv2.putText(im, tagl, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, fs, CYN, 1, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(tagr, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    cv2.putText(im, tagr, (w - tw - 16, 26), cv2.FONT_HERSHEY_SIMPLEX, fs, ACC, 1, cv2.LINE_AA)
    for (cx, cy, dx, dy) in [(8, 8, 1, 1), (w - 8, 8, -1, 1), (8, h - 8, 1, -1), (w - 8, h - 8, -1, -1)]:
        cv2.line(im, (cx, cy), (cx + dx * 26, cy), CYN, 1, cv2.LINE_AA)
        cv2.line(im, (cx, cy), (cx, cy + dy * 26), CYN, 1, cv2.LINE_AA)
    return im


def real_detections(n_keep=6):
    """Pick frames with a clearly visible aircraft: sane box size (0.2-6% area), decent resolution,
    target-patch sharpness + contrast, ranked by conf*sharpness. Moderate zoom only (<=3x)."""
    from ultralytics import YOLO
    model = YOLO(str(REPO / "runs/train/uav_real/weights/best.pt"))
    cands = []
    for ds, splits in (("mjolnir", ("valid", "train")),):   # shahed labels degenerate — excluded
      for split in splits:
        root = REPO / f"datasets/realworld/{ds}/{split}"
        lbl = root / "labels"
        if not lbl.exists():
            continue
        for lf in lbl.glob("*.txt"):
            best = 0.0
            for line in lf.read_text().splitlines():
                p = line.split()
                if not p or p[0] != "0":
                    continue
                if len(p) == 5:                          # det row: cx cy w h
                    a = float(p[3]) * float(p[4])
                elif len(p) >= 7:                        # seg polygon: x1 y1 x2 y2 ...
                    xs = [float(v) for v in p[1::2]]
                    ys = [float(v) for v in p[2::2]]
                    a = (max(xs) - min(xs)) * (max(ys) - min(ys))
                else:
                    continue
                if 0.01 <= a <= 0.55:                    # clearly visible, not degenerate
                    best = max(best, a)
            if best > 0:
                imf = root / "images" / (lf.stem + ".jpg")
                if imf.exists():
                    cands.append((best, ds, imf))
    cands.sort(reverse=True)
    print(f"[real] {len(cands)} sane-size candidates")
    scored = []
    for area, ds, imf in cands[:300]:
        im = cv2.imread(str(imf))
        if im is None or im.shape[1] < 600:
            continue
        r = model.predict(str(imf), imgsz=960, conf=0.45, verbose=False)[0]
        boxes = [b for b in r.boxes if int(b.cls) == 0]
        if not boxes:
            continue
        b = max(boxes, key=lambda bb: float(bb.conf))
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        h, w = im.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        if bw < 0.06 * w or bw > 0.85 * w or bh < 12:
            continue
        patch = cv2.cvtColor(im[max(0, y1):y2, max(0, x1):x2], cv2.COLOR_BGR2GRAY)
        if patch.size < 64:
            continue
        sharp = cv2.Laplacian(patch, cv2.CV_64F).var()
        contrast = float(patch.std())
        if sharp < 60 or contrast < 20:
            continue
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        mask = np.ones_like(gray, bool)
        mask[max(0, y1):y2, max(0, x1):x2] = False
        bgmean, bgstd = float(gray[mask].mean()), float(gray[mask].std())
        if bgmean > 185 and bgstd < 45:                  # white catalog/product shot — not in flight
            continue
        scored.append((float(b.conf) * min(1.0, sharp / 400), area, ds, imf,
                       (x1, y1, x2, y2), float(b.conf)))
    scored.sort(reverse=True)
    print(f"[real] {len(scored)} pass sharp/contrast gates")
    kept = 0
    used = set()
    for sc, area, ds, imf, (x1, y1, x2, y2), conf in scored:
        if kept >= n_keep:
            break
        key = imf.name[:14]
        if key in used:                                   # avoid near-duplicate frames
            continue
        used.add(key)
        im = cv2.imread(str(imf))
        h, w = im.shape[:2]
        bw = x2 - x1
        cxm, cym = (x1 + x2) // 2, (y1 + y2) // 2
        cw = min(w, max(int(bw * 3), int(w * 0.6)))       # light zoom only
        ch = int(cw * 0.625)
        if ch > h:
            ch = h; cw = min(w, int(ch / 0.625))
        ox = int(np.clip(cxm - cw // 2, 0, w - cw)); oy = int(np.clip(cym - ch // 2, 0, h - ch))
        crop = im[oy:oy + ch, ox:ox + cw].copy()
        scale = 1280.0 / crop.shape[1]
        crop = cv2.resize(crop, (1280, int(crop.shape[0] * scale)), interpolation=cv2.INTER_LANCZOS4)
        nx1, ny1 = int((x1 - ox) * scale), int((y1 - oy) * scale)
        nx2, ny2 = int((x2 - ox) * scale), int((y2 - oy) * scale)
        rng = max(6.0, 14.0 / math.sqrt(area / 0.01))
        hud_box(crop, nx1, ny1, nx2, ny2, "drone", conf, rng)
        frame_chrome(crop, f"THIRD IRIS // {ds.upper()} VALIDATION", "REAL FRAME // VISION ONLY")
        fn = OUT / f"real_det_{kept}.jpg"
        cv2.imwrite(str(fn), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"[real] kept {fn.name} ds={ds} score={sc:.2f} conf={conf:.2f} area={area:.4f}")
        kept += 1
    print(f"[real] saved {kept}")


def rl_trajectories():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, str(REPO / "policy"))
    from pursuit_env import PursuitEnv, TRAJ, VMAX, VZ_MAX, YAW_MAX
    from stable_baselines3 import PPO
    AS = np.array([VMAX, VZ_MAX, YAW_MAX], dtype=np.float32)
    model = PPO.load(str(REPO / "runs/policy/rl_policy.zip"), device="cpu")
    env = PursuitEnv(seed=77, dyn=True)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=120)
    fig.patch.set_facecolor("#070b11")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#070b11")
    cols = ["#ff5c1a", "#3fd2ff", "#3ee08c", "#ffc233"]
    trajs = ["zigzag", "orbit", "weave", "dive"]
    for i, tj in enumerate(trajs):
        obs = env.reset({"gap": 12.0, "tspeed": 6.0, "traj": tj})
        ego, tgt = [env.ego.copy()], [env.tgt.copy()]
        done = False
        while not done:
            a = model.predict(obs, deterministic=True)[0] * AS
            obs, _, done, _ = env.step(a)
            ego.append(env.ego.copy()); tgt.append(env.tgt.copy())
        ego = np.array(ego); tgt = np.array(tgt)
        off = i * 90.0
        ax.plot(tgt[:, 0] + off, tgt[:, 1], color="#46586f", lw=1.2, ls="--")
        ax.plot(ego[:, 0] + off, ego[:, 1], color=cols[i], lw=1.8)
        ax.scatter([tgt[-1, 0] + off], [tgt[-1, 1]], s=42, marker="x", color="#8295ad", zorder=5)
        ax.scatter([ego[-1, 0] + off], [ego[-1, 1]], s=30, color=cols[i], zorder=5)
        ax.annotate(tj.upper(), (ego[0, 0] + off, ego[0, 1] - 14), color=cols[i],
                    fontsize=10, family="monospace")
    ax.set_aspect("equal")
    ax.grid(color="#15202e", lw=0.6)
    for sp in ax.spines.values():
        sp.set_color("#1e2a3a")
    ax.tick_params(colors="#4d6076", labelsize=8)
    ax.set_title("LEARNED PILOT // PURSUIT GEOMETRY (TOP-DOWN, 40 S EPISODES)",
                 color="#8295ad", family="monospace", fontsize=11, loc="left", pad=12)
    leg = ax.legend(["target path", "intercept path"], loc="upper right", frameon=False,
                    labelcolor="#8295ad", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "rl_pursuit_paths.png", facecolor="#070b11")
    print("[rl] saved rl_pursuit_paths.png")

    fig2, ax2 = plt.subplots(figsize=(12.8, 7.2), dpi=120)
    fig2.patch.set_facecolor("#070b11"); ax2.set_facecolor("#070b11")
    obs = env.reset({"gap": 12.0, "tspeed": 7.0, "traj": "zigzag"})
    errs, rngs = [], []
    done = False
    while not done:
        a = model.predict(obs, deterministic=True)[0] * AS
        obs, _, done, info = env.step(a)
        errs.append(info["center_err"]); rngs.append(info["rng"])
    t = np.arange(len(errs)) * 0.1
    ax2.plot(t, rngs, color="#3fd2ff", lw=1.6, label="range to target (m)")
    ax2.axhline(12.0, color="#3ee08c", lw=1.0, ls="--", label="commanded stand-off 12 m")
    ax2.plot(t, np.array(errs) * 20, color="#ff5c1a", lw=1.4, label="centering error (×20)")
    ax2.grid(color="#15202e", lw=0.6)
    for sp in ax2.spines.values():
        sp.set_color("#1e2a3a")
    ax2.tick_params(colors="#4d6076", labelsize=9)
    ax2.set_xlabel("time (s)", color="#8295ad", family="monospace")
    ax2.set_title("LEARNED PILOT // STAND-OFF HOLD VS EVASIVE TARGET (ZIGZAG 7 M/S)",
                  color="#8295ad", family="monospace", fontsize=11, loc="left", pad=12)
    ax2.legend(loc="upper right", frameon=False, labelcolor="#8295ad", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(OUT / "rl_standoff_hold.png", facecolor="#070b11")
    print("[rl] saved rl_standoff_hold.png")


if __name__ == "__main__":
    import sys
    if "--real" in sys.argv or len(sys.argv) == 1:
        real_detections()
    if "--rl" in sys.argv or len(sys.argv) == 1:
        rl_trajectories()
