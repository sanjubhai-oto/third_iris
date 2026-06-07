#!/usr/bin/env bash
# Install Docker Engine + NVIDIA Container Toolkit in WSL2 Ubuntu (for aerial-autonomy-stack).
# Run as root:  wsl -d Ubuntu-22.04 -u root -- bash /mnt/c/Users/admin/uav-vio-track/env/wsl_docker_setup.sh
set -e
export DEBIAN_FRONTEND=noninteractive

echo "== 1. Docker Engine repo =="
apt-get update -qq
apt-get install -y -qq ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  > /etc/apt/sources.list.d/docker.list

echo "== 2. Install Docker =="
apt-get update -qq
apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "== 3. NVIDIA Container Toolkit =="
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -qq
apt-get install -y -qq nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker

echo "== 4. Start docker daemon (WSL2, no systemd) =="
service docker start || (dockerd > /var/log/dockerd.log 2>&1 &)
sleep 4

echo "== 5. Verify =="
docker --version
docker info --format '{{.ServerVersion}}' 2>/dev/null || echo "docker daemon not ready yet"
echo "== GPU-in-container test =="
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | tail -3 || echo "GPU container test failed (will diagnose)"
echo "== DONE =="
