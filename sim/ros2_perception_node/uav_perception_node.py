#!/usr/bin/env python3
"""ROS 2 node: subscribe to the ego drone's camera, run YOLO26 + ByteTrack, publish tracks.

Phase 2 (perception in-the-loop). Reuses the same Ultralytics model as the offline
perception/detect_track.py so behaviour matches the benchmarked pipeline.

Subscribes : sensor_msgs/Image  (default /camera/image_raw — from Isaac/Pegasus camera publisher)
Publishes  : vision_msgs/Detection2DArray (/tracks)  with track ids in results[].id
             sensor_msgs/Image           (/tracks/overlay, optional annotated frame)

Run (in WSL2, after sourcing ROS 2 + the project venv with ultralytics):
    ros2 run <pkg> uav_perception_node --ros-args \
        -p model:=runs/train/uav/weights/best.pt -p image_topic:=/camera/image_raw
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge


class UavPerceptionNode(Node):
    def __init__(self):
        super().__init__("uav_perception")
        self.declare_parameter("model", "yolo26n.pt")
        self.declare_parameter("tracker", "perception/trackers/bytetrack_uav.yaml")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("imgsz", 1280)
        self.declare_parameter("conf", 0.15)
        self.declare_parameter("publish_overlay", True)

        from ultralytics import YOLO
        self.model = YOLO(self.get_parameter("model").value)
        self.tracker = self.get_parameter("tracker").value
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf = float(self.get_parameter("conf").value)
        self.publish_overlay = bool(self.get_parameter("publish_overlay").value)

        self.bridge = CvBridge()
        topic = self.get_parameter("image_topic").value
        self.sub = self.create_subscription(Image, topic, self.on_image, 10)
        self.pub = self.create_publisher(Detection2DArray, "/tracks", 10)
        self.overlay_pub = self.create_publisher(Image, "/tracks/overlay", 5)
        self.get_logger().info(f"listening on {topic}, model loaded.")

    def on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # persist=True keeps ByteTrack's Kalman state across the stream of frames.
        results = self.model.track(
            frame, tracker=self.tracker, persist=True,
            imgsz=self.imgsz, conf=self.conf, verbose=False)
        r = results[0]

        out = Detection2DArray()
        out.header = msg.header
        boxes = r.boxes
        if boxes is not None and len(boxes):
            ids = boxes.id
            for i in range(len(boxes)):
                d = Detection2D()
                d.header = msg.header
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                d.bbox.center.position.x = (x1 + x2) / 2
                d.bbox.center.position.y = (y1 + y2) / 2
                d.bbox.size_x = x2 - x1
                d.bbox.size_y = y2 - y1
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(int(boxes.cls[i]))
                hyp.hypothesis.score = float(boxes.conf[i])
                d.results.append(hyp)
                d.id = str(int(ids[i])) if ids is not None else ""
                out.detections.append(d)
        self.pub.publish(out)

        if self.publish_overlay:
            annotated = r.plot()
            ov = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            ov.header = msg.header
            self.overlay_pub.publish(ov)


def main():
    rclpy.init()
    node = UavPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
