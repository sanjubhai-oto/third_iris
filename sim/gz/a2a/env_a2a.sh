#!/usr/bin/env bash
# Shared env for the air-to-air PX4-SITL engagement. Source this (`. env_a2a.sh`) from the launch
# scripts. Detects PX4_DIR, exports the gz resource/plugin/server-config paths, and forces software GL
# (hardware ogre2 core-dumps on camera render under WSLg).
REPO=/mnt/c/Users/admin/uav-vio-track

# --- detect PX4 (must have the jetray model) ---
PX4_DIR=""
for d in /root/PX4-Autopilot /opt/PX4-Autopilot "$HOME/PX4-Autopilot"; do
  [ -d "$d/Tools/simulation/gz/models/jetray" ] && { PX4_DIR="$d"; break; }
done
[ -z "$PX4_DIR" ] && { echo "ERROR: PX4-Autopilot+jetray not found"; return 1 2>/dev/null || exit 1; }
export PX4_DIR

PX4_BUILD="$PX4_DIR/build/px4_sitl_default"
export PX4_BIN="$PX4_BUILD/bin/px4"
export PX4_ROOTFS="$PX4_BUILD/rootfs"

# --- gz paths (mirror gz_env.sh; needed because PX4 standalone mode does NOT source it) ---
export PX4_GZ_MODELS="$PX4_DIR/Tools/simulation/gz/models"
export PX4_GZ_WORLDS="$PX4_DIR/Tools/simulation/gz/worlds"
export PX4_GZ_PLUGINS="$PX4_BUILD/src/modules/simulation/gz_plugins"
export PX4_GZ_SERVER_CONFIG="$PX4_DIR/src/modules/simulation/gz_bridge/server.config"
export GZ_SIM_RESOURCE_PATH="$PX4_GZ_MODELS:$PX4_GZ_WORLDS:$REPO/sim/gz"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$PX4_GZ_PLUGINS"
export GZ_SIM_SERVER_CONFIG_PATH="$PX4_GZ_SERVER_CONFIG"

export LIBGL_ALWAYS_SOFTWARE=1
export GZ_IP=127.0.0.1

# Geographic origin (PX4 SITL default = Zurich). Makes PX4 call set_spherical_coordinates so the gz
# Magnetometer/NavSat systems produce a real field + heading reference (else arming is denied:
# "Strong magnetic interference" / "no heading reference").
export PX4_HOME_LAT=47.397742
export PX4_HOME_LON=8.545594
export PX4_HOME_ALT=488.0
export A2A_WORLD=uav_a2a
export A2A_WORLD_SDF="$REPO/sim/gz/uav_a2a.sdf"
export A2A_AUTOSTART=4071            # jetray airframe (rotor geometry)

echo "[env_a2a] PX4_DIR=$PX4_DIR  world=$A2A_WORLD  (software GL)"
