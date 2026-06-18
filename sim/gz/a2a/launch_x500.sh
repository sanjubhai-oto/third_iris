#!/usr/bin/env bash
# Phase: perfect the front-cam tracking with STANDARD x500_mono_cam drones (chaser + runner), before
# moving to jetray. Both are gz_x500_mono_cam (autostart 4010, forward mono camera). Distinct per-model
# camera topics (no explicit <topic> -> gz scopes by model name). Keeps the suv rover in the world.
#   bash sim/gz/a2a/launch_x500.sh
set +e
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env_a2a.sh" || exit 1
AS=4010                                    # gz_x500_mono_cam airframe

echo "[x500] killing stale gz / px4 / python ..."
pkill -9 python3 2>/dev/null; pkill -9 -f "bin/px4" 2>/dev/null; pkill -9 -f "gz sim" 2>/dev/null; sleep 2

echo "[x500] gz server: $A2A_WORLD_SDF"
nohup gz sim -s -r -v1 "$A2A_WORLD_SDF" > /tmp/x500_gz.log 2>&1 &
for i in $(seq 1 90); do
  sleep 1
  gz service -i --service "/world/$A2A_WORLD/scene/info" 2>/dev/null | grep -q "Service providers" && { echo "[x500] scene ready (${i}s)"; break; }
  [ "$i" = 90 ] && { echo "[x500] TIMEOUT scene"; exit 1; }
done

start() {  # $1=instance $2=pose $3=log
  ( cd "$PX4_ROOTFS" && \
    PX4_GZ_STANDALONE=1 PX4_GZ_WORLD="$A2A_WORLD" PX4_SYS_AUTOSTART="$AS" PX4_SIM_MODEL="gz_x500_mono_cam" \
    PX4_GZ_MODEL_POSE="$2" PX4_GZ_NO_FOLLOW=1 HEADLESS=1 GZ_IP=127.0.0.1 \
    nohup "$PX4_BIN" -i "$1" -d > "/tmp/x500_$3.log" 2>&1 & )
}
wait_hb() {  # $1=port $2=label
  for i in $(seq 1 45); do
    sleep 1
    if python3 - "$1" <<'PY' >/dev/null 2>&1
import sys
from pymavlink import mavutil
m = mavutil.mavlink_connection(f"udpin:0.0.0.0:{sys.argv[1]}")
sys.exit(0 if m.wait_heartbeat(timeout=2) else 1)
PY
    then echo "[x500] $2 heartbeat udp:$1 (${i}s)"; return 0; fi
  done
  echo "[x500] TIMEOUT $2"; return 1
}

echo "[x500] CHASER (i=0) ..."; start 0 "0,0,0.2,0,0,0" chaser; wait_hb 14540 CHASER || { tail -12 /tmp/x500_chaser.log; exit 1; }
echo "[x500] RUNNER (i=1) ..."; start 1 "12,0,0.2,0,0,0" runner; wait_hb 14541 RUNNER || { tail -12 /tmp/x500_runner.log; exit 1; }

echo "[x500] ===== UP ====="
echo "  models:"; gz topic -e -t "/world/$A2A_WORLD/pose/info" -n 1 2>/dev/null | grep -oE 'name: "x500_mono_cam_[0-9]"' | sort -u
echo "  camera topics:"; gz topic -l 2>/dev/null | grep -iE "image|camera" | grep -iE "x500|camera_link|image" | head
