#!/usr/bin/env python3
import ctypes
ctypes.CDLL("/usr/lib/aarch64-linux-gnu/libgomp.so.1", mode=ctypes.RTLD_GLOBAL)

import os
import time
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO


class PersonSnapshotPublisher(Node):
    def __init__(self):
        super().__init__('person_snapshot_publisher')

        self.bridge = CvBridge()

        self.declare_parameter('model_path', '/home/unitree/capstone/yolov8s.pt')
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_rect_raw')
        self.declare_parameter('text_topic', '/yolo/detection_text')
        self.declare_parameter('snapshot_topic', '/yolo/person_snapshot/compressed')
        self.declare_parameter('conf_threshold', 0.2)
        self.declare_parameter('snapshot_cooldown_sec', 3.0)
        self.declare_parameter('save_dir', '/home/unitree/capstone/snapshots')

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        color_topic = self.get_parameter('color_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        text_topic = self.get_parameter('text_topic').get_parameter_value().string_value
        snapshot_topic = self.get_parameter('snapshot_topic').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        self.snapshot_cooldown_sec = self.get_parameter('snapshot_cooldown_sec').get_parameter_value().double_value
        self.save_dir = self.get_parameter('save_dir').get_parameter_value().string_value

        os.makedirs(self.save_dir, exist_ok=True)

        self.model = YOLO(model_path)

        self.text_pub = self.create_publisher(String, text_topic, 10)
        self.snapshot_pub = self.create_publisher(CompressedImage, snapshot_topic, 10)

        self.latest_depth_msg = None
        self.last_snapshot_time = 0.0
        self.is_processing = False

        self.color_sub = self.create_subscription(
            Image,
            color_topic,
            self.color_callback,
            qos_profile_sensor_data
        )
        self.depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info('Person snapshot publisher started.')
        self.get_logger().info(f'model_path: {model_path}')
        self.get_logger().info(f'color_topic: {color_topic}')
        self.get_logger().info(f'depth_topic: {depth_topic}')
        self.get_logger().info(f'snapshot_topic: {snapshot_topic}')
        self.get_logger().info(f'snapshot_cooldown_sec: {self.snapshot_cooldown_sec}')

    def depth_callback(self, msg):
        self.latest_depth_msg = msg

    def color_callback(self, color_msg):
        if self.is_processing:
            return
        if self.latest_depth_msg is None:
            return

        self.is_processing = True
        try:
            self.process_frame(color_msg, self.latest_depth_msg)
        finally:
            self.is_processing = False

    def get_depth_median(self, depth_frame, x1, y1, x2, y2):
        h, w = depth_frame.shape[:2]

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            return None

        bw = x2 - x1
        bh = y2 - y1

        roi_x1 = int(x1 + bw * 0.30)
        roi_x2 = int(x1 + bw * 0.70)
        roi_y1 = int(y1 + bh * 0.20)
        roi_y2 = int(y1 + bh * 0.80)

        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            return None

        roi = depth_frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            return None

        if depth_frame.dtype == np.uint16:
            valid = roi[(roi > 0) & (roi < 10000)]
            if valid.size == 0:
                return None
            return float(np.median(valid)) / 1000.0

        if depth_frame.dtype in (np.float32, np.float64):
            valid = roi[np.isfinite(roi)]
            valid = valid[(valid > 0.0) & (valid < 10.0)]
            if valid.size == 0:
                return None
            return float(np.median(valid))

        return None

    def draw_label(self, image, x1, y1, label, color):
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        text_y = max(y1 - 10, th + 5)

        cv2.rectangle(
            image,
            (x1, text_y - th - 6),
            (x1 + tw + 6, text_y + baseline - 6),
            color,
            -1
        )
        cv2.putText(
            image,
            label,
            (x1 + 3, text_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    def publish_snapshot(self, frame, header):
        now = time.time()
        if now - self.last_snapshot_time < self.snapshot_cooldown_sec:
            return

        self.last_snapshot_time = now

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(self.save_dir, f'person_snapshot_{timestamp}.jpg')

        success, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success:
            self.get_logger().warn('JPEG encoding failed.')
            return

        with open(file_path, 'wb') as f:
            f.write(encoded.tobytes())

        msg = CompressedImage()
        msg.header = header
        msg.format = 'jpeg'
        msg.data = encoded.tobytes()
        self.snapshot_pub.publish(msg)

        self.get_logger().info(f'Snapshot published and saved: {file_path}')

    def process_frame(self, color_msg, depth_msg):
        try:
            color_frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as exc:
            self.get_logger().error(f'Image conversion failed: {exc}')
            return

        try:
            results = self.model(
                color_frame,
                classes=[0],   # person only
                conf=self.conf_threshold,
                verbose=False
            )
        except Exception as exc:
            self.get_logger().error(f'YOLO inference failed: {exc}')
            return

        output = color_frame.copy()
        detection_lines = []
        detected_person = False

        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                if cls_id != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                distance_m = self.get_depth_median(depth_frame, x1, y1, x2, y2)

                detected_person = True
                color = (255, 0, 0)
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

                if distance_m is not None:
                    label = f'person {conf:.2f} {distance_m:.2f}m'
                    detection_lines.append(
                        f'person, conf={conf:.2f}, distance={distance_m:.2f}m, box=({x1},{y1},{x2},{y2})'
                    )
                else:
                    label = f'person {conf:.2f} N/A'
                    detection_lines.append(
                        f'person, conf={conf:.2f}, distance=N/A, box=({x1},{y1},{x2},{y2})'
                    )

                self.draw_label(output, x1, y1, label, color)

        text_msg = String()
        text_msg.data = '\n'.join(detection_lines) if detection_lines else 'no detections'
        self.text_pub.publish(text_msg)

        if detected_person:
            self.publish_snapshot(output, color_msg.header)


def main(args=None):
    rclpy.init(args=args)
    node = PersonSnapshotPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()