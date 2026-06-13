#!/usr/bin/env bash
# Launch the Gazebo GUI (server+gui) so the user can WATCH the up-cam tracking in the WSLg window.
# Software GL (ogre2 over WSLg/D3D12 hardware GL core-dumps). Leaves it running in the background.
set +e
REPO=/mnt/c/Users/admin/uav-vio-track
export DISPLAY=${DISPLAY:-:0}
export GZ_SIM_RESOURCE_PATH=/root/PX4-Autopilot/Tools/simulation/gz/models
export GZ_SIM_SERVER_CONFIG_PATH=/root/PX4-Autopilot/src/modules/simulation/gz_bridge/server.config
export LIBGL_ALWAYS_SOFTWARE=1
# WSLg window-mapping fix when running as ROOT: WSLg's runtime dir is owned by UID 1000, so give root
# its own XDG_RUNTIME_DIR and force X11 (XWayland) instead of the wayland socket root can't use.
export QT_QPA_PLATFORM=xcb
unset WAYLAND_DISPLAY
export XDG_RUNTIME_DIR=/tmp/xdg-root
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
# make sure root can reach the WSLg X socket
[ -e /tmp/.X11-unix/X0 ] || ln -sf /mnt/wslg/.X11-unix/X0 /tmp/.X11-unix/X0 2>/dev/null
WORLD="${1:-$REPO/sim/gz/uav_track_up.sdf}"
pkill -9 -f "gz sim" 2>/dev/null; sleep 2
echo "launching Gazebo GUI on DISPLAY=$DISPLAY ..."
nohup gz sim -r "$WORLD" --gui-config "$REPO/sim/gz/gui_upcam.config" > /tmp/gz_gui.log 2>&1 &
echo "gz GUI pid $!"
for i in $(seq 1 50); do
  sleep 1
  gz topic -l 2>/dev/null | grep -q camera_up && break
done
# WSLg + software-GL: the gz GUI Qt window opens 1x1 and UNMAPPED -> force a real size + map it so it
# actually shows on screen (needs wmctrl/xdotool: apt-get install -y wmctrl xdotool x11-utils).
sleep 6
WID=$(DISPLAY=:0 xwininfo -root -children 2>/dev/null | grep -i "Gazebo GUI" | grep -oE "0x[0-9a-f]+" | head -1)
if [ -n "$WID" ]; then
  DISPLAY=:0 xdotool windowsize "$WID" 1280 800 2>/dev/null
  DISPLAY=:0 xdotool windowmove "$WID" 80 60 2>/dev/null
  DISPLAY=:0 xdotool windowmap "$WID" 2>/dev/null
  DISPLAY=:0 xdotool windowactivate "$WID" 2>/dev/null
  echo "GUI window $WID mapped @1280x800"
else
  echo "GUI window not found (install wmctrl/xdotool); last log:"; tail -5 /tmp/gz_gui.log
fi
