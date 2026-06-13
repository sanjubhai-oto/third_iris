#!/usr/bin/env bash
# Canonical headless launch of the up-cam tracking world in WSL.
# KEY: LIBGL_ALWAYS_SOFTWARE=1 (llvmpipe). Hardware ogre2 over WSLg/D3D12 CORE-DUMPS on camera
# render; software GL renders fine. GZ_SIM_SERVER_CONFIG_PATH adds the Sensors system (default
# server.config lacks it -> no camera). Leaves gz running in the background.
set +e
REPO=/mnt/c/Users/admin/uav-vio-track
export GZ_SIM_RESOURCE_PATH=/root/PX4-Autopilot/Tools/simulation/gz/models
export GZ_SIM_SERVER_CONFIG_PATH=/root/PX4-Autopilot/src/modules/simulation/gz_bridge/server.config
export LIBGL_ALWAYS_SOFTWARE=1
WORLD="${1:-$REPO/sim/gz/uav_track_up.sdf}"
pkill -9 -f "gz sim" 2>/dev/null; sleep 2
nohup gz sim -s -r -v1 "$WORLD" > /tmp/gz_up.log 2>&1 &
echo "gz launched (software GL) pid $!; waiting for camera topic..."
for i in $(seq 1 40); do
  sleep 1
  gz topic -l 2>/dev/null | grep -q camera_up && { echo "camera_up LIVE after ${i}s"; exit 0; }
done
echo "TIMEOUT waiting for camera_up"; tail -5 /tmp/gz_up.log; exit 1
