#!/usr/bin/env python3
"""Phase 3 bootstrap: publish Isaac/Pegasus ground-truth ego pose as PX4 external vision.

This validates the EKF2 external-vision plumbing BEFORE wiring in a real VIO estimator.
It relays a ground-truth pose (from the sim) to MAVROS so EKF2 fuses it as vision odometry.
Once EKF2 holds position on this with GPS disabled, swap this node for OpenVINS (see README).

Subscribes : geometry_msgs/PoseStamped  (sim GT pose, e.g. /pegasus/ego/pose)
Publishes  : geometry_msgs/PoseStamped  -> /mavros/vision_pose/pose   (EKF2 fuses this)

EKF2 must be configured for external vision (see px4_config/ekf2_vision.params):
    EKF2_EV_CTRL  = 11  (horiz pos + vert pos + yaw from EV)
    EKF2_HGT_REF  = 3   (Vision)
    EKF2_GPS_CTRL = 0   (disable GPS aiding for GPS-denied test)
    EKF2_EV_DELAY tuned to the pipeline latency.

Rate: keep 30-50 Hz; EKF2 will not fuse if the stream is too slow.

    ros2 run <pkg> gt_pose_bridge --ros-args -p gt_topic:=/pegasus/ego/pose -p rate:=50.0
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped


class GtPoseBridge(Node):
    def __init__(self):
        super().__init__("gt_pose_bridge")
        self.declare_parameter("gt_topic", "/pegasus/ego/pose")
        self.declare_parameter("rate", 50.0)

        gt_topic = self.get_parameter("gt_topic").value
        self.rate = float(self.get_parameter("rate").value)

        # MAVROS vision_pose wants a reliable, sensor-friendly QoS.
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self.latest: PoseStamped | None = None
        self.sub = self.create_subscription(PoseStamped, gt_topic, self._on_gt, qos)
        self.pub = self.create_publisher(PoseStamped, "/mavros/vision_pose/pose", qos)
        self.timer = self.create_timer(1.0 / self.rate, self._tick)
        self.get_logger().info(f"relaying {gt_topic} -> /mavros/vision_pose/pose @ {self.rate} Hz")

    def _on_gt(self, msg: PoseStamped):
        self.latest = msg

    def _tick(self):
        if self.latest is None:
            return
        out = PoseStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "odom"
        out.pose = self.latest.pose
        self.pub.publish(out)


def main():
    rclpy.init()
    node = GtPoseBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
