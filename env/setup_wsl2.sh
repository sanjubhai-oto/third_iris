#!/usr/bin/env bash
# Phase 2-3 environment: PX4 SITL + ROS 2 Humble + MAVROS + XRCE-DDS inside WSL2 Ubuntu 22.04.
# Isaac Sim 5.1 + Pegasus are installed separately (see sim/README.md) — recommended to run Isaac
# natively on Windows and PX4/ROS2 here in WSL2, bridged over localhost.
#
# Prereqs on Windows FIRST (PowerShell, admin):
#     wsl --install -d Ubuntu-22.04
#     wsl --update
# Then inside the Ubuntu shell:
#     bash setup_wsl2.sh
set -euo pipefail

echo "== 0. Sanity: GPU visible in WSL2? =="
if command -v nvidia-smi >/dev/null; then nvidia-smi || true; else
  echo "WARN: nvidia-smi not found. Install the NVIDIA Windows driver (>=550) and 'wsl --update'."
fi

echo "== 1. Base packages =="
sudo apt update
sudo apt install -y git wget curl build-essential cmake python3-pip python3-venv \
                    software-properties-common locales lsb-release gnupg2
sudo locale-gen en_US en_US.UTF-8

echo "== 2. ROS 2 Humble (Ubuntu 22.04) =="
sudo add-apt-repository -y universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
     -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
sudo apt update
sudo apt install -y ros-humble-desktop ros-dev-tools ros-humble-mavros ros-humble-mavros-extras
# MAVROS needs the GeographicLib datasets for the geoid model.
sudo /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh
grep -q 'source /opt/ros/humble/setup.bash' ~/.bashrc || \
  echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc

echo "== 3. Micro XRCE-DDS Agent (PX4 <-> ROS2 uORB bridge) =="
mkdir -p ~/px4_ws && cd ~/px4_ws
if [ ! -d Micro-XRCE-DDS-Agent ]; then
  git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
fi
cd Micro-XRCE-DDS-Agent && mkdir -p build && cd build
cmake .. && make -j"$(nproc)" && sudo make install && sudo ldconfig /usr/local/lib/

echo "== 4. PX4-Autopilot (SITL) =="
cd ~/px4_ws
if [ ! -d PX4-Autopilot ]; then
  git clone https://github.com/PX4/PX4-Autopilot.git --recursive
fi
cd PX4-Autopilot
bash ./Tools/setup/ubuntu.sh   # installs the PX4 toolchain (no nuttx needed for SITL)
echo ">> Build SITL with:  make px4_sitl   (or 'make px4_sitl gz_x500_mono_cam' for a camera frame)"

echo "== 5. px4_msgs / px4_ros_com (ROS2 messages) =="
mkdir -p ~/px4_ws/ros2_ws/src && cd ~/px4_ws/ros2_ws/src
[ -d px4_msgs ]    || git clone https://github.com/PX4/px4_msgs.git
[ -d px4_ros_com ] || git clone https://github.com/PX4/px4_ros_com.git
cd ~/px4_ws/ros2_ws
source /opt/ros/humble/setup.bash
colcon build || echo "colcon build had issues — re-run after sourcing ROS."

cat <<'EOF'

== DONE (PX4 + ROS2 layer) ==
Next:
  - Isaac Sim 5.1 + Pegasus: see sim/README.md (run on Windows host or here).
  - Start the DDS bridge:        MicroXRCEAgent udp4 -p 8888
  - Start PX4 SITL:              cd ~/px4_ws/PX4-Autopilot && make px4_sitl gz_x500_mono_cam
  - Source ROS workspaces:       source ~/px4_ws/ros2_ws/install/setup.bash
  - VIO bringup + EKF2 params:   see vio/README.md and px4_config/.
EOF
