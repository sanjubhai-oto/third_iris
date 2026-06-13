#!/usr/bin/env bash
# Recon the WSL Gazebo/PX4 environment for the up-facing-camera VIO tracking build.
set +e
echo "=== gz version ==="; gz sim --version 2>/dev/null || echo NO_GZ
echo "=== GPU / render (camera sensors need this) ==="
ls /dev/dri 2>/dev/null || echo NO_DRI
echo "DISPLAY=$DISPLAY  WAYLAND=$WAYLAND_DISPLAY"
glxinfo -B 2>/dev/null | grep -iE "renderer|opengl version" || echo "no glxinfo"
echo "=== gz python bindings ==="
for v in 13 12 11; do python3 -c "import gz.transport$v" 2>/dev/null && echo "gz.transport$v OK"; done
for v in 10 9 8; do python3 -c "import gz.msgs$v" 2>/dev/null && echo "gz.msgs$v OK"; done
echo "=== ros2 ==="; ls /opt/ros 2>/dev/null || echo NO_ROS
echo "=== python deps in WSL ==="
python3 -c "import ultralytics; print('ultralytics', ultralytics.__version__)" 2>&1 | head -1
python3 -c "import cv2; print('cv2', cv2.__version__)" 2>&1 | head -1
python3 -c "import numpy; print('numpy', numpy.__version__)" 2>&1 | head -1
echo "=== jetray world contents (cameras? sensors?) ==="
grep -iE "camera|sensor|<sensor|pose|jetray|target" /root/PX4-Autopilot/Tools/simulation/gz/worlds/jetray_garden.sdf 2>/dev/null | head -40
echo "=== does the jetray model define a camera link? ==="
grep -rliE "camera|<sensor" /root/PX4-Autopilot/Tools/simulation/gz/models/jetray* 2>/dev/null | head
