#!/usr/bin/env bash
# Launch the AAS mission (conops) node per drone -> arms, takes off, orbits, lands.
#   N=2 CONOPS=yalla.yaml bash aas_mission.sh
N="${N:-2}"
CONOPS="${CONOPS:-yalla.yaml}"
for ID in $(seq 1 "$N"); do
  docker exec -d "aircraft-container-inst0_${ID}" bash -lc \
    "source /opt/ros/humble/setup.bash; source /aas/github_ws/install/setup.bash 2>/dev/null; \
     source /aas/aircraft_ws/install/setup.bash; export ROS_DOMAIN_ID=${ID}; \
     ros2 run mission mission --conops ${CONOPS} --ros-args -r __ns:=/Drone${ID} -p use_sim_time:=true \
       > /tmp/mission_${ID}.log 2>&1"
  echo "mission '${CONOPS}' started for Drone${ID}"
done
