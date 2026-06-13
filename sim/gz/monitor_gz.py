#!/usr/bin/env python3
"""TEAM SKY MONITOR: one up-facing camera watches MULTIPLE enemy drones overhead, detects + tracks all
of them at once with persistent IDs (YOLO + ByteTrack) and reconstructs each one's WORLD position from
the chaser pose + bearing + depth. This is the "team to monitor live" capability — situational awareness
over a sky full of drones, the front end to assigning interceptors.

Run in WSL (software GL):
  bash sim/gz/launch_gui.sh sim/gz/uav_track_team.sdf     # or launch_gz.sh for headless
  LIBGL_ALWAYS_SOFTWARE=1 python3 sim/gz/monitor_gz.py --secs 40 --yolo <weights>
"""
import argparse, math, time
import numpy as np
from gz.transport13 import Node
from track_gz import Cam, DepthCam, set_pose as _set_pose, HFOV   # reuse plumbing

WORLD = "uav_track_team"
VFOV = 2 * math.atan(math.tan(HFOV / 2) * 9 / 16)

def set_pose(node, name, x, y, z):
    import track_gz; track_gz.WORLD = WORLD; _set_pose(node, name, x, y, z)

# each target flies its own (modest) pattern so the monitor must hold multiple distinct tracks, but
# they stay CLUSTERED overhead within the up-cam FOV so all three remain observable.
def tgt_pose(name, t, base):
    bx, by, bz = base
    if name == "target_red":    return (bx + 2.0*math.cos(0.4*t),  by + 2.0*math.sin(0.4*t),  bz)
    if name == "target_green":  return (bx + 1.8*math.sin(0.5*t),  by + 1.8*math.cos(0.3*t),  bz)
    if name == "target_blue":   return (bx + 1.5*math.sin(0.7*t),  by + 2.0*math.sin(0.35*t), bz)
    return base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=40.0)
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.12)
    ap.add_argument("--out", default="/mnt/c/Users/admin/uav-vio-track/runs/videos")
    args = ap.parse_args()
    import cv2
    from ultralytics import YOLO

    node = Node(); cam = Cam(node); depth = DepthCam(node)
    model = YOLO(args.yolo)
    chaser = np.array([0.0, 0.0, 6.0])                 # fixed monitoring perch, looking up
    # targets ~5 m overhead -> all 3 small drones stay resolvable to the detector (at 10 m only 1-2 were)
    bases = {"target_red": (3, 2, 11), "target_green": (-3, 2, 11), "target_blue": (2, -3, 11)}
    set_pose(node, "chaser", *chaser)
    for n, b in bases.items():
        set_pose(node, n, *b)
    time.sleep(1.5)
    t0w = time.time()
    while cam.get() is None and time.time() - t0w < 10:
        time.sleep(0.1)
    if cam.get() is None:
        print("[monitor] ERROR no frames"); return

    COLORS = [(0,0,255),(0,200,0),(255,80,0),(0,220,220),(220,0,220)]   # per-id BGR
    tracks = {}                                          # id -> list of (t, world_pos)
    last = time.time(); simt = 0.0; frames = 0
    seen_counts = []                                     # how many distinct drones seen per frame
    vw = None; H = W = None
    while simt < args.secs:
        now = time.time(); dt = min(0.2, max(0.02, now - last)); last = now; simt += dt
        for n, b in bases.items():
            set_pose(node, n, *tgt_pose(n, simt, b))
        img = cam.get()
        if img is None:
            continue
        H, W = img.shape[:2]
        res = model.track(img, persist=True, imgsz=args.imgsz, conf=args.conf,
                          tracker="bytetrack.yaml", verbose=False)[0]
        frames += 1
        fr = img[:, :, ::-1].copy()
        n_seen = 0
        if res.boxes is not None and res.boxes.id is not None:
            ids = res.boxes.id.cpu().numpy().astype(int)
            xys = res.boxes.xyxy.cpu().numpy()
            for k, tid in enumerate(ids):
                x1, y1, x2, y2 = xys[k]; cx, cy = (x1+x2)/2, (y1+y2)/2
                rng = depth.range_at(cx / W, cy / H)
                if rng is None:
                    continue
                ex = (cx - W/2)/(W/2); ey = (cy - H/2)/(H/2)
                ray = np.array([ey*math.tan(VFOV/2), -ex*math.tan(HFOV/2), 1.0]); ray /= np.linalg.norm(ray)
                wp = chaser + rng * ray
                tracks.setdefault(int(tid), []).append((simt, wp))
                n_seen += 1
                c = COLORS[int(tid) % len(COLORS)]
                cv2.rectangle(fr, (int(x1),int(y1)), (int(x2),int(y2)), c, 2)
                cv2.putText(fr, f"ID{int(tid)} {rng:.0f}m", (int(x1), int(y1)-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        seen_counts.append(n_seen)
        cv2.putText(fr, f"t={simt:4.1f}  drones tracked: {n_seen}/3  unique IDs: {len(tracks)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        if vw is None:
            import os; os.makedirs(args.out, exist_ok=True)
            vw = cv2.VideoWriter(f"{args.out}/gz_team_monitor.mp4",
                                 cv2.VideoWriter_fourcc(*"mp4v"), 12, (W, H))
        vw.write(fr)
    if vw is not None:
        vw.release()

    sc = np.array(seen_counts) if seen_counts else np.array([0])
    # persistent tracks = ids seen in >40% of frames (filter ByteTrack flicker ids)
    persistent = [i for i, pts in tracks.items() if len(pts) > 0.4 * frames]
    print("\n================ GZ TEAM SKY MONITOR ================")
    print(f"frames={frames}  mean drones-in-view={sc.mean():.2f}/3  "
          f"all-3-frames={100.0*np.mean(sc>=3):.0f}%  >=2={100.0*np.mean(sc>=2):.0f}%")
    print(f"unique track IDs={len(tracks)}  persistent (seen >40% frames)={len(persistent)} -> {persistent}")
    print(f"video -> {args.out}/gz_team_monitor.mp4")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 7))
        for i in persistent:
            P = np.array([p for _, p in tracks[i]])
            ax.plot(P[:,0], P[:,1], '-', lw=2, label=f"ID{i}")
        ax.scatter([chaser[0]],[chaser[1]], c='k', marker='^', s=80, label='chaser')
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.axis("equal"); ax.grid(alpha=.3)
        ax.legend(); ax.set_title("Team sky monitor — reconstructed world tracks of all drones")
        fig.tight_layout(); fig.savefig(f"{args.out}/gz_team_monitor_map.png", dpi=110)
        print(f"map -> {args.out}/gz_team_monitor_map.png")
    except Exception as e:
        print("plot skipped:", e)

if __name__ == "__main__":
    main()
