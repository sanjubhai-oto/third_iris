#!/usr/bin/env bash
# Single BASE jetray (has the down RGB-D cam_vio) on the uav_a2a world for the GPS-denied VIO drift
# test (sim/gz/vio_sitl.py). Reuses env_a2a. Spawns jetray_0 on udp 14540.
#   bash sim/gz/a2a/launch_vio.sh
set +e
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env_a2a.sh" || exit 1

echo "[vio] killing stale gz / px4 / python ..."
pkill -9 python3 2>/dev/null; pkill -9 -f "bin/px4" 2>/dev/null; pkill -9 -f "gz sim" 2>/dev/null; sleep 2

echo "[vio] gz server: $A2A_WORLD_SDF"
nohup gz sim -s -r -v1 "$A2A_WORLD_SDF" > /tmp/vio_gz.log 2>&1 &
for i in $(seq 1 90); do
  sleep 1
  gz service -i --service "/world/$A2A_WORLD/scene/info" 2>/dev/null | grep -q "Service providers" && { echo "[vio] scene ready (${i}s)"; break; }
  [ "$i" = 90 ] && { echo "[vio] TIMEOUT scene"; exit 1; }
done

echo "[vio] spawning base jetray (i=0, has cam_vio) ..."
( cd "$PX4_ROOTFS" && \
  PX4_GZ_STANDALONE=1 PX4_GZ_WORLD="$A2A_WORLD" PX4_SYS_AUTOSTART="$A2A_AUTOSTART" PX4_SIM_MODEL="gz_jetray" \
  PX4_GZ_MODEL_POSE="0,0,0.2,0,0,0" PX4_GZ_NO_FOLLOW=1 HEADLESS=1 GZ_IP=127.0.0.1 \
  nohup "$PX4_BIN" -i 0 -d > /tmp/vio_px4.log 2>&1 & )

for i in $(seq 1 45); do
  sleep 1
  if python3 - <<'PY' >/dev/null 2>&1
from pymavlink import mavutil
m = mavutil.mavlink_connection("udpin:0.0.0.0:14540")
import sys; sys.exit(0 if m.wait_heartbeat(timeout=2) else 1)
PY
  then echo "[vio] jetray heartbeat on 14540 (${i}s)"; break; fi
  [ "$i" = 45 ] && { echo "[vio] TIMEOUT heartbeat"; tail -8 /tmp/vio_px4.log; exit 1; }
done
echo "[vio] cam_vio topic:"; gz topic -l 2>/dev/null | grep -E "cam_vio" || echo "  (missing!)"
echo "[vio] UP — run: python3 sim/gz/vio_sitl.py --world $A2A_WORLD --model jetray_0"
