#!/usr/bin/env python3
"""Pluggable video + telemetry sources for the Web UI.

In simulation everything comes from AirSim. On a real aircraft you instead have two independent
downlinks: a **video receiver** (RTSP / UDP / analog-capture card / file) and a **telemetry
receiver** (MAVLink over UDP or serial from the flight controller / radio). This module lets the
same UI/tracking code switch between those, so the system is real-world ready.

* ``build_video_url`` maps a (protocol, endpoint) choice to an OpenCV capture URL/index.
* ``VideoReceiver`` opens that source and yields frames (lazily (re)opens on change).
* ``MavlinkReceiver`` is a background thread that reads attitude / position / speed from a MAVLink
  link (pymavlink) and exposes the latest values for the telemetry panel.

External video has no depth channel, so vision range falls back to GPS/coast in sim; on a real drone
range comes from the detector (known target size) or a stereo/depth payload.
"""
from __future__ import annotations

import math
import threading

import cv2

VIDEO_PROTOS = ("airsim", "rtsp", "udp", "http", "device", "file")
TELEM_PROTOS = ("airsim", "mavlink_serial", "mavlink_udp", "mavlink_tcp")


def build_video_url(proto: str, endpoint: str):
    """Translate a UI (protocol, endpoint) choice into something cv2.VideoCapture accepts."""
    endpoint = (endpoint or "").strip()
    if proto == "device":
        try:
            return int(endpoint) if endpoint != "" else 0
        except ValueError:
            return 0
    if proto == "file":
        return endpoint
    if proto in ("rtsp", "udp", "http"):
        if "://" in endpoint:
            return endpoint
        return f"{proto}://{endpoint}"          # allow bare host:port
    return endpoint


class VideoReceiver:
    """Lazily (re)opens an OpenCV capture and returns BGR frames. Returns (None, None) on failure so
    the caller can fall back. ``depth`` is always None for external video."""

    def __init__(self):
        self.cap = None
        self.key = None

    def _ensure(self, proto, endpoint):
        key = (proto, endpoint)
        if key != self.key:
            self.release()
            url = build_video_url(proto, endpoint)
            self.cap = cv2.VideoCapture(url)
            self.key = key
        return self.cap

    def read(self, proto, endpoint):
        cap = self._ensure(proto, endpoint)
        if cap is None or not cap.isOpened():
            return None, None
        ok, frame = cap.read()
        return (frame if ok else None), None

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.key = None


class MavlinkReceiver(threading.Thread):
    """Read attitude / global position / speed from a MAVLink link in the background.

    ``conn_str`` is a pymavlink connection string, e.g.::
        mavlink_udp   -> "udpin:0.0.0.0:14550"  or  "udp:127.0.0.1:14550"
        mavlink_tcp   -> "tcp:127.0.0.1:5760"
        mavlink_serial-> "COM5" (uses 57600 baud) or "/dev/ttyUSB0,921600"
    """

    def __init__(self, proto: str, endpoint: str):
        super().__init__(daemon=True)
        self.proto = proto
        self.conn_str, self.baud = self._conn(proto, endpoint)
        self.latest = {}
        self.connected = False
        self.error = None
        self._stop = False

    @staticmethod
    def _conn(proto, endpoint):
        endpoint = (endpoint or "").strip()
        if proto == "mavlink_udp":
            ep = endpoint or "udpin:0.0.0.0:14550"
            if "://" not in ep and not ep.startswith(("udp:", "udpin:", "udpout:")):
                ep = "udp:" + ep
            return ep, None
        if proto == "mavlink_tcp":
            ep = endpoint or "tcp:127.0.0.1:5760"
            if not ep.startswith("tcp:"):
                ep = "tcp:" + ep
            return ep, None
        # serial: "PORT" or "PORT,BAUD"
        if "," in endpoint:
            port, baud = endpoint.split(",", 1)
            return port.strip(), int(baud)
        return (endpoint or "COM5"), 57600

    def run(self):
        try:
            from pymavlink import mavutil
            m = (mavutil.mavlink_connection(self.conn_str, baud=self.baud)
                 if self.baud else mavutil.mavlink_connection(self.conn_str))
            m.wait_heartbeat(timeout=6)
            self.connected = True
        except Exception as e:                       # pragma: no cover - needs real link
            self.error = str(e)
            return
        while not self._stop:
            try:
                msg = m.recv_match(blocking=True, timeout=1)
            except Exception:
                continue
            if msg is None:
                continue
            t = msg.get_type()
            if t == "ATTITUDE":
                self.latest.update(roll=math.degrees(msg.roll), pitch=math.degrees(msg.pitch),
                                   yaw=math.degrees(msg.yaw))
            elif t == "GLOBAL_POSITION_INT":
                self.latest.update(alt_m=msg.relative_alt / 1000.0, lat=msg.lat / 1e7, lon=msg.lon / 1e7)
            elif t == "VFR_HUD":
                self.latest.update(groundspeed=msg.groundspeed, heading=msg.heading)

    def snapshot(self):
        return {"connected": self.connected, "error": self.error, **self.latest}

    def stop(self):
        self._stop = True
