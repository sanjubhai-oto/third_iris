#!/usr/bin/env bash
# Clean launch of the up-cam world + grab one frame (de-risk render). Run in WSL.
set +e
export GZ_SIM_RESOURCE_PATH=/root/PX4-Autopilot/Tools/simulation/gz/models
REPO=/mnt/c/Users/admin/uav-vio-track
pkill -9 -f "gz sim" 2>/dev/null; sleep 2
echo "=== launching gz headless (uav_track_up) ==="
nohup gz sim -s -r -v1 "$REPO/sim/gz/uav_track_up.sdf" > /tmp/gz_up.log 2>&1 &
GZPID=$!
echo "gz pid $GZPID"
sleep 12
echo "=== topics ==="
gz topic -l | grep -iE "camera|image"
echo "=== grab ==="
python3 "$REPO/sim/gz/grab_one.py" camera_up
