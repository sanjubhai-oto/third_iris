#!/usr/bin/env bash
# Report takeoff result + latest odometry altitude for each AAS drone.
N="${N:-2}"
for ID in $(seq 1 "$N"); do
  echo "=== Drone${ID} ==="
  docker exec "aircraft-container-inst0_${ID}" cat "/tmp/takeoff_${ID}.log" 2>/dev/null \
    | sed 's/\x1b\[[0-9;?]*[a-zA-Z]//g' | grep -iE 'accepted|succeed|result|aborted|goal|status' | tail -4
  echo "-- latest odometry pos (NED x y z; z negative = altitude) --"
  docker logs --tail 60 "aircraft-container-inst0_${ID}" 2>&1 \
    | sed 's/\x1b\[[0-9;?]*[a-zA-Z]//g' | tr -d '\r' \
    | grep -oE 'pos: [-0-9.]+ [-0-9.]+ [-0-9.]+' | tail -1
done
echo "=== containers ==="
docker ps --format '{{.Names}} {{.Status}}' | grep inst0
