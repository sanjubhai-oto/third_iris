#!/usr/bin/env bash
# Bring up the two-vehicle PX4-SITL air-to-air world in WSL:
#   1) standalone gz server with sim/gz/uav_a2a.sdf (software GL)
#   2) CHASER px4 instance 0  -> spawns jetray_chaser_0, MAVLink offboard on udp 14540
#   3) RUNNER px4 instance 1  -> spawns jetray_runner_1,  MAVLink offboard on udp 14541
# Leaves everything running in the background. Re-run to restart cleanly.
#
#   bash sim/gz/a2a/launch_a2a.sh
set +e
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env_a2a.sh" || exit 1

echo "[a2a] (re)generating jetray_chaser / jetray_runner models ..."
python3 "$HERE/make_models.py" --px4 "$PX4_DIR" || exit 1

echo "[a2a] killing stale gz / px4 ..."
pkill -9 -f "bin/px4" 2>/dev/null; pkill -9 -f "gz sim" 2>/dev/null; sleep 2

# ---- 1) standalone gz server ----
echo "[a2a] launching gz server: $A2A_WORLD_SDF"
nohup gz sim -s -r -v1 "$A2A_WORLD_SDF" > /tmp/a2a_gz.log 2>&1 &
# Gate on the scene/info SERVICE being ready (what PX4's check_scene_info needs), NOT just the clock.
# First cold software-GL render warms llvmpipe shaders and can take a while; give it up to 90s.
for i in $(seq 1 90); do
  sleep 1
  if gz service -i --service "/world/$A2A_WORLD/scene/info" 2>/dev/null | grep -q "Service providers"; then
    echo "[a2a] gz world '$A2A_WORLD' scene ready (${i}s)"; break; fi
  [ "$i" = 90 ] && { echo "[a2a] TIMEOUT gz scene/info"; tail -8 /tmp/a2a_gz.log; exit 1; }
done

wait_vehicle() {  # $1=model name  $2=offboard port  $3=label
  local model="$1" port="$2" label="$3"
  for i in $(seq 1 45); do
    sleep 1
    if gz topic -i -t "/world/$A2A_WORLD/pose/info" >/dev/null 2>&1 && \
       python3 - "$port" <<'PY' >/dev/null 2>&1
import sys
from pymavlink import mavutil
m = mavutil.mavlink_connection(f"udpin:0.0.0.0:{sys.argv[1]}")
sys.exit(0 if m.wait_heartbeat(timeout=2) else 1)
PY
    then echo "[a2a] $label heartbeat on udp:$port (${i}s)"; return 0; fi
  done
  echo "[a2a] TIMEOUT waiting for $label heartbeat on $port"; return 1
}

start_px4() {  # $1=instance  $2=model  $3=pose  $4=logname
  ( cd "$PX4_ROOTFS" && \
    PX4_GZ_STANDALONE=1 PX4_GZ_WORLD="$A2A_WORLD" PX4_SYS_AUTOSTART="$A2A_AUTOSTART" PX4_SIM_MODEL="gz_$2" \
    PX4_GZ_MODEL_POSE="$3" PX4_GZ_NO_FOLLOW=1 HEADLESS=1 GZ_IP=127.0.0.1 \
    nohup "$PX4_BIN" -i "$1" -d > "/tmp/a2a_$4.log" 2>&1 & )
}

# ---- 2) CHASER (instance 0) at origin, nose +X ----
echo "[a2a] starting CHASER (i=0, jetray_chaser) ..."
start_px4 0 jetray_chaser "0,0,0.2,0,0,0" chaser
wait_vehicle jetray_chaser_0 14540 CHASER || { tail -15 /tmp/a2a_chaser.log; exit 1; }

# ---- 3) RUNNER (instance 1) 12 m ahead along +X (in the chaser's forward FOV) ----
echo "[a2a] starting RUNNER (i=1, jetray_runner) ..."
start_px4 1 jetray_runner "12,0,0.2,0,0,0" runner
wait_vehicle jetray_runner_1 14541 RUNNER || { tail -15 /tmp/a2a_runner.log; exit 1; }

echo "[a2a] ===== UP ====="
echo "  models in /world/$A2A_WORLD/pose/info:"
gz topic -e -t "/world/$A2A_WORLD/pose/info" -n 1 2>/dev/null | grep -oE 'name: "jetray_[a-z]+_[0-9]"' | sort -u
echo "  camera topic:"; gz topic -l 2>/dev/null | grep -E "camera_front|cam_vio" || echo "   (none yet)"
echo "  CHASER offboard udp:14540   RUNNER offboard udp:14541"
echo "[a2a] logs: /tmp/a2a_gz.log /tmp/a2a_chaser.log /tmp/a2a_runner.log"
