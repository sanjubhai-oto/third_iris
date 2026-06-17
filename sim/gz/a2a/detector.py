#!/usr/bin/env python3
"""Hybrid YOLO+KLT air-to-air detector for the chaser's forward camera. Same idea as
perception/hybrid_track.py / live_track_stream.py: run YOLO every N frames to (re)acquire, track with
cheap KLT optical flow in between -> real-world-grade detection at high effective rate. Dark-blob
fallback (numpy) if YOLO weights are missing, so it always returns something.

  from detector import HybridDetector
  det = HybridDetector(weights, every=8, conf=0.20, imgsz=640)
  out = det(frame_bgr)          # -> dict(cx,cy,w,h,conf,method) or None
"""
import os
import numpy as np
import cv2

DEFAULT_WEIGHTS = "/mnt/c/Users/admin/uav-vio-track/runs/train/uav_diverse/weights/best.pt"


def detect_blob(frame_bgr):
    """Dark drone vs bright sky: darkest ~0.4% of pixels, largest compact blob. -> dict or None."""
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    thr = float(np.percentile(g, 0.4))
    if g.min() > 90 or (np.median(g) - thr) < 20:      # no clearly-dark object
        return None
    mask = (g <= thr + 12).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w * h < 9 or w > 0.5 * g.shape[1] or h > 0.5 * g.shape[0]:
        return None
    return dict(cx=x + w / 2.0, cy=y + h / 2.0, w=float(w), h=float(h), conf=0.5, method="BLOB")


class HybridDetector:
    def __init__(self, weights=DEFAULT_WEIGHTS, every=8, conf=0.20, imgsz=640):
        self.every, self.conf, self.imgsz = every, conf, imgsz
        self.model = None
        if weights and os.path.exists(weights):
            try:
                from ultralytics import YOLO
                self.model = YOLO(weights)
            except Exception as e:
                print(f"[detector] YOLO load failed ({e}); using dark-blob fallback")
        self.lk = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        self.box = None; self.pts = None; self.prev = None; self.n = 0

    def _yolo(self, frame_bgr):
        r = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        b = r.boxes
        if b is None or not len(b):
            return None
        i = int(np.argmax(b.conf.cpu().numpy()))
        x1, y1, x2, y2 = b.xyxy[i].cpu().numpy().tolist()
        return [x1, y1, x2, y2, float(b.conf[i])]

    def _out(self, method, conf):
        x1, y1, x2, y2 = self.box
        return dict(cx=(x1 + x2) / 2.0, cy=(y1 + y2) / 2.0, w=x2 - x1, h=y2 - y1,
                    conf=float(conf), method=method)

    def __call__(self, frame_bgr):
        if self.model is None:
            return detect_blob(frame_bgr)
        self.n += 1
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        method = None; conf = 0.0
        need_nn = self.box is None or self.n % self.every == 0 or self.pts is None or len(self.pts) < 6
        if need_nn:
            d = self._yolo(frame_bgr)
            if d is not None:
                self.box = d[:4]; conf = d[4]; method = "YOLO"
                roi = gray[int(self.box[1]):int(self.box[3]), int(self.box[0]):int(self.box[2])]
                p = cv2.goodFeaturesToTrack(roi, 80, 0.01, 3) if roi.size else None
                self.pts = (p + np.array([[self.box[0], self.box[1]]], np.float32)).reshape(-1, 1, 2) \
                    if p is not None else None
            else:
                self.box = self.pts = None
        elif self.prev is not None and self.pts is not None:
            nn, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.pts, None, **self.lk)
            gn = nn[st == 1]; go = self.pts[st == 1]
            if len(gn) >= 6:
                dxy = np.median(gn - go.reshape(-1, 2), axis=0)
                self.box = [self.box[0] + dxy[0], self.box[1] + dxy[1],
                            self.box[2] + dxy[0], self.box[3] + dxy[1]]
                self.pts = gn.reshape(-1, 1, 2); method = "KLT"; conf = 0.4
            else:
                self.pts = None
        self.prev = gray
        if self.box is not None and method:
            return self._out(method, conf)
        b = detect_blob(frame_bgr)              # YOLO/KLT missed -> dark-blob (excellent vs bright sky)
        if b is not None:
            self.box = [b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2,
                        b["cx"] + b["w"] / 2, b["cy"] + b["h"] / 2]    # seed KLT next frame
            self.pts = None
        return b
