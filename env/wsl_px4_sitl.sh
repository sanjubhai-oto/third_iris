#!/usr/bin/env bash
# Minimal PX4 SITL + Gazebo (Harmonic) install for WSL2 Ubuntu 22.04 — fastest path to a
# launchable flying drone with a camera. PX4-Autopilot is also the backend for Isaac/Pegasus,
# so none of this is wasted if you later switch to the Isaac path.
# Run as root:  wsl -d Ubuntu-22.04 -u root -- bash /mnt/c/Users/admin/uav-vio-track/env/wsl_px4_sitl.sh
set -e
export DEBIAN_FRONTEND=noninteractive
PX4_DIR=/opt/PX4-Autopilot

echo "== 1. base packages =="
apt-get update -qq
apt-get install -y -qq git curl wget lsb-release gnupg ca-certificates

echo "== 2. Gazebo Harmonic (gz) repo + install =="
curl -sSL https://packages.osrfoundation.org/gazebo.gpg \
     -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
  > /etc/apt/sources.list.d/gazebo-stable.list
apt-get update -qq
apt-get install -y -qq gz-harmonic

echo "== 3. clone PX4-Autopilot (recursive submodules — the long pole) =="
if [ ! -d "$PX4_DIR" ]; then
  git clone https://github.com/PX4/PX4-Autopilot.git --recursive "$PX4_DIR"
else
  echo "   already cloned; updating submodules"
  git -C "$PX4_DIR" submodule update --init --recursive
fi

echo "== 4. PX4 SITL toolchain (no NuttX; SITL only) =="
bash "$PX4_DIR/Tools/setup/ubuntu.sh" --no-nuttx

echo "== 5. build PX4 SITL (gz_x500_mono_cam: quad + downward mono camera) =="
cd "$PX4_DIR"
# PX4_GZ_STANDALONE keeps the build from auto-spawning the GUI during the build step.
HEADLESS=1 make px4_sitl gz_x500_mono_cam_build 2>/dev/null || make px4_sitl

echo "== DONE =="
echo "Launch later with:  cd $PX4_DIR && make px4_sitl gz_x500_mono_cam"
