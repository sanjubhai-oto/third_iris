#!/usr/bin/env bash
# Test whether camera_up renders when it's the ONLY camera (two-camera headless render hypothesis).
set +e
REPO=/mnt/c/Users/admin/uav-vio-track
MODEL=/root/PX4-Autopilot/Tools/simulation/gz/models/jetray/model.sdf
export GZ_SIM_RESOURCE_PATH=/root/PX4-Autopilot/Tools/simulation/gz/models

pkill -9 -f "gz sim"; sleep 2

# back up once
[ -f "$MODEL.orig" ] || cp "$MODEL" "$MODEL.orig"

# produce a model variant with camera_front REMOVED (python, robust vs sed multiline)
python3 - "$MODEL" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
s2 = re.sub(r'\s*<!-- FPV camera.*?</sensor>\n', '\n', s, flags=re.S, count=1)
open(p, "w").write(s2)
print("camera_front removed:", "camera_front" not in s2, "| camera_up present:", "camera_up" in s2)
PY

echo "=== launch gz -v4 (only camera_up) ==="
nohup gz sim -s -r -v4 "$REPO/sim/gz/uav_track_up.sdf" > /tmp/gz_v4.log 2>&1 &
sleep 14
echo "=== camera topics ==="; gz topic -l | grep -i camera
echo "=== render/error log ==="; grep -iE "camera_up|render engine|ogre|opengl|Failed|error creating|sensor" /tmp/gz_v4.log | head -25
echo "=== grab ==="; python3 "$REPO/sim/gz/grab_one.py" camera_up
