#!/usr/bin/env python3
"""Multi-object tracking metrics (MOTA / IDF1 / ID-switches) using motmetrics.

Compares predicted tracks (tracks.jsonl from detect_track.py, or MOT-format txt) against
ground-truth in MOT Challenge format:
    frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z

    python perception/eval/eval_tracking.py --gt gt/MOT/seq01/gt.txt --pred runs/track/exp/tracks.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_pred_jsonl(path):
    """tracks.jsonl -> dict[frame] -> list of (id, x, y, w, h)."""
    frames = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("track_id") is None:
                continue
            x1, y1, x2, y2 = r["xyxy"]
            frames.setdefault(r["frame"] + 1, []).append(  # 1-based to match MOT gt
                (int(r["track_id"]), x1, y1, x2 - x1, y2 - y1))
    return frames


def load_mot_txt(path):
    """MOT-format txt -> dict[frame] -> list of (id, x, y, w, h)."""
    frames = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            fr, tid, x, y, w, h = (float(parts[i]) for i in range(6))
            frames.setdefault(int(fr), []).append((int(tid), x, y, w, h))
    return frames


def main(argv=None):
    import motmetrics as mm
    import numpy as np

    p = argparse.ArgumentParser(description="MOTA/IDF1 tracking eval")
    p.add_argument("--gt", required=True, help="ground-truth MOT-format txt")
    p.add_argument("--pred", required=True, help="tracks.jsonl or MOT-format txt")
    p.add_argument("--iou", type=float, default=0.5, help="match IoU distance threshold")
    args = p.parse_args(argv)

    gt = load_mot_txt(args.gt)
    pred = (load_pred_jsonl(args.pred) if str(args.pred).endswith(".jsonl")
            else load_mot_txt(args.pred))

    acc = mm.MOTAccumulator(auto_id=True)
    for frame in sorted(set(gt) | set(pred)):
        g = gt.get(frame, [])
        d = pred.get(frame, [])
        gids = [x[0] for x in g]
        dids = [x[0] for x in d]
        gboxes = np.array([x[1:] for x in g]) if g else np.empty((0, 4))
        dboxes = np.array([x[1:] for x in d]) if d else np.empty((0, 4))
        dist = mm.distances.iou_matrix(gboxes, dboxes, max_iou=1 - args.iou)
        acc.update(gids, dids, dist)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc, metrics=["mota", "idf1", "num_switches", "num_false_positives",
                      "num_misses", "precision", "recall", "mostly_tracked", "mostly_lost"],
        name="seq")
    print(mm.io.render_summary(summary, namemap=mm.io.motchallenge_metric_names,
                               formatters=mh.formatters))


if __name__ == "__main__":
    main()
