#!/usr/bin/env bash
# Headless, DETACHED launch of aerial-autonomy-stack (2 quads) for non-interactive use.
# Replicates tools_and_docs/sim_run.sh but with `docker run -d` (no xterm/-it/GUI), so it can
# be driven from automation. GPS state (ODOM=none). Use sim_run.sh for the interactive GUI version.
#
#   wsl -d Ubuntu-22.04 -u root -- bash /mnt/c/Users/admin/uav-vio-track/env/aas_run_headless.sh
set -e

NUM_QUADS="${NUM_QUADS:-2}"
WORLD="${WORLD:-swiss_town}"
RTF="${RTF:-2.0}"
SIM_SUBNET=10.42
AIR_SUBNET=10.22
SIM_ID=100
SIM_NET=aas-sim-network-inst0
AIR_NET=aas-air-network-inst0

# -dit = detached but with a pseudo-TTY (tmuxinator/tmux entrypoint needs a TTY).
# No --rm so we can read logs if a container exits.
COMMON="-dit --gpus all --device /dev/dri --privileged \
  --env NVIDIA_DRIVER_CAPABILITIES=all --env LIBGL_ALWAYS_SOFTWARE=0 \
  --env MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA --env LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  --volume /usr/lib/wsl:/usr/lib/wsl"

echo "== networks =="
docker network inspect $SIM_NET >/dev/null 2>&1 || docker network create --subnet=${SIM_SUBNET}.0.0/16 $SIM_NET
docker network inspect $AIR_NET >/dev/null 2>&1 || docker network create --subnet=${AIR_SUBNET}.0.0/16 $AIR_NET

echo "== simulation container (Gazebo headless) =="
docker run $COMMON \
  --env AUTOPILOT=px4 --env HEADLESS=true --env CAMERA=true --env LIDAR=true \
  --env NUM_QUADS=$NUM_QUADS --env NUM_VTOLS=0 --env WORLD=$WORLD \
  --env SIMULATED_TIME=true --env RTF=$RTF --env START_AS_PAUSED=false \
  --env SIM_SUBNET=$SIM_SUBNET --env GROUND_ID=101 --env GND_CONTAINER=false \
  --env ROS_DOMAIN_ID=$SIM_ID \
  --net=$SIM_NET --ip=${SIM_SUBNET}.90.${SIM_ID} \
  --name simulation-container-inst0 simulation-image

echo "== aircraft containers (PX4 + autonomy) =="
for ID in $(seq 1 $NUM_QUADS); do
  sleep 1.0
  docker run $COMMON \
    --env AUTOPILOT=px4 --env HEADLESS=true --env CAMERA=true --env LIDAR=true --env ODOM=none \
    --env DRONE_TYPE=quad --env DRONE_ID=$ID \
    --env SIMULATED_TIME=true \
    --env SIM_SUBNET=$SIM_SUBNET --env AIR_SUBNET=$AIR_SUBNET --env SIM_ID=$SIM_ID --env GROUND_ID=101 \
    --env GND_CONTAINER=false \
    --env ROS_DOMAIN_ID=$ID \
    --net=$SIM_NET --ip=${SIM_SUBNET}.90.$ID \
    --name aircraft-container-inst0_$ID aircraft-image
done

echo "== connect aircraft to air network =="
for ID in $(seq 1 $NUM_QUADS); do
  docker network connect --ip=${AIR_SUBNET}.90.$ID $AIR_NET aircraft-container-inst0_$ID
done

echo "== running containers =="
docker ps --format '{{.Names}}  {{.Status}}'
echo "DONE. Trigger takeoff per drone with:"
echo "  docker exec aircraft-container-inst0_<ID> bash -c 'source /opt/ros/humble/setup.bash && source /aas/aircraft_ws/install/setup.bash && ros2 action send_goal /Drone<ID>/takeoff_action autopilot_interface_msgs/action/Takeoff \"{takeoff_altitude: 20.0}\"'"
