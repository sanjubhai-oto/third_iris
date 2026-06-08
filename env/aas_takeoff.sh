#!/usr/bin/env bash
# Send takeoff to each AAS drone (run inside WSL where docker lives).
#   ALT=20 N=2 bash aas_takeoff.sh
ALT="${ALT:-20.0}"
N="${N:-2}"
for ID in $(seq 1 "$N"); do
  docker exec -d "aircraft-container-inst0_${ID}" bash -lc \
    "source /opt/ros/humble/setup.bash; source /aas/aircraft_ws/install/setup.bash 2>/dev/null; \
     export ROS_DOMAIN_ID=${ID}; \
     ros2 action send_goal /Drone${ID}/takeoff_action autopilot_interface_msgs/action/Takeoff '{takeoff_altitude: ${ALT}}' \
       > /tmp/takeoff_${ID}.log 2>&1"
  echo "takeoff (alt=${ALT}m) sent to Drone${ID}"
done
