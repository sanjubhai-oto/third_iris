#!/usr/bin/env bash
# Diagnose the headless camera-render crash. Logs to a Windows-readable path.
set +e
REPO=/mnt/c/Users/admin/uav-vio-track
LOG=$REPO/runs/videos/gz_render_diag.log
export GZ_SIM_RESOURCE_PATH=/root/PX4-Autopilot/Tools/simulation/gz/models
export GZ_SIM_SERVER_CONFIG_PATH=/root/PX4-Autopilot/src/modules/simulation/gz_bridge/server.config
pkill -9 -f "gz sim" 2>/dev/null; sleep 2
echo "### attempt 1: ogre2 default" > "$LOG"
timeout 18 gz sim -s -r -v3 "$REPO/sim/gz/uav_track_up.sdf" >> "$LOG" 2>&1
echo "rc=$? (124=timeout=alive)" >> "$LOG"
pkill -9 -f "gz sim" 2>/dev/null; sleep 2
echo "" >> "$LOG"; echo "### attempt 2: ogre2 + LIBGL_ALWAYS_SOFTWARE=1 (llvmpipe)" >> "$LOG"
LIBGL_ALWAYS_SOFTWARE=1 timeout 25 gz sim -s -r -v3 "$REPO/sim/gz/uav_track_up.sdf" >> "$LOG" 2>&1 &
SWPID=$!
sleep 20
echo "--- topics during sw render ---" >> "$LOG"
gz topic -l 2>/dev/null | grep -i camera >> "$LOG" || echo "NONE" >> "$LOG"
kill -9 $SWPID 2>/dev/null
echo "done" >> "$LOG"
