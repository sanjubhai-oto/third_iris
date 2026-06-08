#!/usr/bin/env bash
# AAS launch with the Gazebo GUI rendered to the Windows desktop via WSLg (X11).
# Detached (-dit) so it can be driven from automation, but HEADLESS=false on the sim container
# so a Gazebo window appears. Requires systemd DISABLED in wsl.conf (WSLg X11 works then).
#   NUM_QUADS=2 bash aas_run_gui.sh
set -e
NUM_QUADS="${NUM_QUADS:-2}"
WORLD="${WORLD:-swiss_town}"
RTF="${RTF:-1.0}"
SIM_SUBNET=10.42; AIR_SUBNET=10.22; SIM_ID=100
SIM_NET=aas-sim-network-inst0; AIR_NET=aas-air-network-inst0

xhost +local:docker 2>/dev/null || true

COMMON="-dit --gpus all --device /dev/dri --privileged \
  --env NVIDIA_DRIVER_CAPABILITIES=all --env DISPLAY=:0 --env QT_X11_NO_MITSHM=1 \
  --env WAYLAND_DISPLAY=wayland-0 --env XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir \
  --env PULSE_SERVER=/mnt/wslg/PulseServer \
  --env LIBGL_ALWAYS_SOFTWARE=0 --env MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA --env LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw --volume /mnt/wslg:/mnt/wslg --volume /usr/lib/wsl:/usr/lib/wsl"

docker network inspect $SIM_NET >/dev/null 2>&1 || docker network create --subnet=${SIM_SUBNET}.0.0/16 $SIM_NET
docker network inspect $AIR_NET >/dev/null 2>&1 || docker network create --subnet=${AIR_SUBNET}.0.0/16 $AIR_NET

echo "== simulation container (Gazebo GUI via WSLg) =="
docker run $COMMON \
  --env AUTOPILOT=px4 --env HEADLESS=false --env CAMERA=true --env LIDAR=true \
  --env NUM_QUADS=$NUM_QUADS --env NUM_VTOLS=0 --env WORLD=$WORLD \
  --env SIMULATED_TIME=true --env RTF=$RTF --env START_AS_PAUSED=false \
  --env SIM_SUBNET=$SIM_SUBNET --env GROUND_ID=101 --env GND_CONTAINER=false \
  --env ROS_DOMAIN_ID=$SIM_ID \
  --net=$SIM_NET --ip=${SIM_SUBNET}.90.${SIM_ID} \
  --name simulation-container-inst0 simulation-image

echo "== aircraft containers =="
for ID in $(seq 1 "$NUM_QUADS"); do
  sleep 1
  docker run $COMMON \
    --env AUTOPILOT=px4 --env HEADLESS=true --env CAMERA=true --env LIDAR=true --env ODOM=none \
    --env DRONE_TYPE=quad --env DRONE_ID=$ID --env SIMULATED_TIME=true \
    --env SIM_SUBNET=$SIM_SUBNET --env AIR_SUBNET=$AIR_SUBNET --env SIM_ID=$SIM_ID --env GROUND_ID=101 \
    --env GND_CONTAINER=false --env ROS_DOMAIN_ID=$ID \
    --net=$SIM_NET --ip=${SIM_SUBNET}.90.$ID \
    --name "aircraft-container-inst0_${ID}" aircraft-image
done
for ID in $(seq 1 "$NUM_QUADS"); do
  docker network connect --ip=${AIR_SUBNET}.90.$ID $AIR_NET "aircraft-container-inst0_${ID}"
done
docker ps --format '{{.Names}} {{.Status}}'
echo "Gazebo GUI should appear on the Windows desktop shortly."
