# vio/ — Ego Visual-Inertial Odometry → PX4 EKF2 (Phase 3)

Goal: the ego drone localizes itself with **no GPS**, using its camera + IMU. The estimate is
fused by PX4's EKF2 via MAVROS `/mavros/vision_pose/pose`.

## Two-step strategy

1. **Bootstrap with ground truth** (`gt_pose_bridge/gt_pose_bridge.py`)
   Relay the sim's true ego pose to MAVROS as external vision. This proves the EKF2 plumbing
   and parameters (`px4_config/ekf2_vision.params`) work, isolated from VIO-estimator quality.
   - Load `px4_config/ekf2_vision.params`, reboot PX4.
   - `ros2 run <pkg> gt_pose_bridge --ros-args -p gt_topic:=/pegasus/ego/pose -p rate:=50.0`
   - Arm in Position mode with GPS disabled → EKF2 should converge and hold position.

2. **Swap in real VIO** (`openvins_bringup/`)
   Run **OpenVINS** on the sim camera+IMU, publish its odometry to `/mavros/vision_pose/pose`
   (replacing the GT bridge), then re-tune `EKF2_EV_DELAY` to the real latency and check drift.

## OpenVINS pointers
- Repo: https://github.com/rpng/open_vins (ROS 2 supported).
- Needs camera intrinsics + cam↔IMU extrinsics + IMU noise params matching the Isaac sensors.
  Pull intrinsics from the Isaac camera prim; IMU noise from the Pegasus IMU sensor config.
- Feed OpenVINS `/camera/image_raw` + `/imu`; remap its pose output to `/mavros/vision_pose/pose`.
- Alternative: **VINS-Fusion** (same idea, different config).

## Watch-items
- **Rate**: keep vision pose at 30–50 Hz or EKF2 won't fuse it.
- **Frames**: MAVROS expects ENU; EKF2 handles the ENU→NED internally. Confirm axes by checking
  `ekf2` innovations are small (`ros2 topic echo /mavros/local_position/pose` tracks truth).
- **Time sync**: VIO timestamps must align with the IMU clock; `EKF2_EV_DELAY` absorbs constant lag.

## Verify
With GPS off, `/mavros/local_position/pose` tracks the Isaac ground-truth pose within tolerance,
and a closed-loop flight returns near the start with bounded drift.
