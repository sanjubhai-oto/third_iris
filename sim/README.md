# sim/ — Isaac Sim + Pegasus + PX4 (Phase 2)

Perception in-the-loop: a PX4-backed ego drone with an RGB camera + IMU flies in Isaac Sim;
its camera streams over ROS 2 into the YOLO26 + ByteTrack pipeline from Phase 1.

## Recommended topology (Windows + WSL2)

```
   Windows host                         WSL2 Ubuntu 22.04
 ┌─────────────────────┐   MAVLink   ┌────────────────────────────┐
 │ Isaac Sim 5.1        │◄──UDP──────►│ PX4 SITL                   │
 │  + Pegasus Simulator │             │ MicroXRCEAgent (uORB<->DDS)│
 │  (RGB cam + IMU)      │   ROS2/DDS  │ ROS2 Humble + MAVROS       │
 │  ROS2 camera publisher│◄──────────►│ uav_perception_node        │
 └─────────────────────┘             │ gt_pose_bridge / OpenVINS  │
                                      └────────────────────────────┘
```

Isaac renders best **natively on Windows** with the RTX 5070 Ti; PX4/ROS 2 must be Linux → WSL2.
Bridge them over localhost. Simpler (but heavier-rendering) fallback: run Isaac Sim inside WSL2 too.

> ROS 2 across the Windows↔WSL2 boundary needs matching `ROS_DOMAIN_ID` and a DDS config that
> works over the WSL virtual NIC (mirrored networking mode in recent WSL, or CycloneDDS with the
> WSL IP). Budget setup time — this is the #1 friction point of Phase 2.

## Install

1. **Isaac Sim 5.1.0** — NVIDIA driver ≥550. Install via the Omniverse launcher / pip
   (`isaacsim` package). Tested combo: Isaac Sim 5.1.0 + Ubuntu 22.04 + driver 550.163.
2. **Pegasus Simulator** — https://github.com/PegasusSimulator/PegasusSimulator
   Follow its install docs; it provides the PX4 backend and example drone scenes.
3. **PX4 + ROS 2** — `env/setup_wsl2.sh`.

## Files

- `pegasus_scenes/` — scenario scripts: ego drone (PX4) + N target UAVs on scripted paths.
- `isaac_synth.py` — render RGB + **ground-truth instance segmentation** + poses → synthetic
  training data (closes the loop back to Phase 1 training, with free perfect masks).
- `ros2_perception_node/uav_perception_node.py` — live YOLO26+ByteTrack node → `/tracks`.

## Camera publishing
In Isaac Sim: **Tools → Robotics → ROS2 OmniGraphs → Camera** to publish the ego RGB camera
as `sensor_msgs/Image` (default topic `/camera/image_raw`, matched by the perception node).

## Verify
- `ros2 topic hz /camera/image_raw` shows frames flowing.
- `ros2 topic echo /tracks` shows detections with stable `id` as a target UAV moves.
- `/tracks/overlay` (rqt_image_view) shows boxes/masks on the live feed.
