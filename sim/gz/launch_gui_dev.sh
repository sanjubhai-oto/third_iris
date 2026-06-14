#!/usr/bin/env bash
# Launch the Gazebo GUI as the NON-ROOT user (dev) so WSLg actually paints the window on the Windows
# desktop (root-owned windows don't surface). Models copied to /tmp/gzshare (readable by dev).
# Invoke:  wsl -d Ubuntu-22.04 -u dev bash sim/gz/launch_gui_dev.sh [world]
set +e
REPO=/mnt/c/Users/admin/uav-vio-track
export DISPLAY=${DISPLAY:-:0}
export QT_QPA_PLATFORM=xcb
export LIBGL_ALWAYS_SOFTWARE=1
export GZ_SIM_RESOURCE_PATH=/tmp/gzshare/models
export GZ_SIM_SERVER_CONFIG_PATH=/tmp/gzshare/server.config
WORLD="${1:-$REPO/sim/gz/uav_track_team.sdf}"
pkill -9 -f "gz sim" 2>/dev/null; sleep 2
nohup gz sim -r "$WORLD" --gui-config "$REPO/sim/gz/gui_upcam.config" > /tmp/gz_gui_dev.log 2>&1 &
echo "gz GUI (dev) pid $!"
for i in $(seq 1 45); do sleep 1; gz topic -l 2>/dev/null | grep -q camera_up && { echo "camera live ${i}s"; break; }; done
sleep 6
WID=$(DISPLAY=:0 xwininfo -root -children 2>/dev/null | grep -i "Gazebo" | grep -oE "0x[0-9a-f]+" | head -1)
if [ -n "$WID" ]; then
  for a in "windowsize $WID 1280 800" "windowmove $WID 80 60" "windowmap $WID" "windowactivate $WID"; do
    DISPLAY=:0 xdotool $a 2>/dev/null; done
  echo "GUI window $WID mapped @1280x800"
  DISPLAY=:0 xwininfo -id "$WID" 2>/dev/null | grep -E "Width|Height|Map State"
else
  echo "GUI window not found; last log:"; tail -6 /tmp/gz_gui_dev.log
fi
