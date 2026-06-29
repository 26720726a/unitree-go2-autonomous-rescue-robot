#!/usr/bin/env python3
import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO

from message_filters import Subscriber, ApproximateTimeSynchronizer


class Go2YoloDistanceNode(Node):
    def __init__(self):
        super().__init__('go2_yolo_distance_node')

        self.bridge = CvBridge()

        # -------------------------
        # parameters
        # -------------------------
        self.declare_parameter('model_path', 'yolov8s.pt')
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('debug_image_topic', '/yolo/debug_image')
        self.declare_parameter('detection_text_topic', '/yolo/detection_text')
        self.declare_parameter('conf_threshold', 0.4)
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('sync_slop', 0.1)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        color_topic = self.get_parameter('color_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        debug_image_topic = self.get_parameter('debug_image_topic').get_parameter_value().string_value
        detection_text_topic = self.get_parameter('detection_text_topic').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        sync_queue_size = self.get_parameter('sync_queue_size').get_parameter_value().integer_value
        sync_slop = self.get_parameter('sync_slop').get_parameter_value().double_value

        # -------------------------
        # model
        # -------------------------
        self.model = YOLO(model_path)

        # -------------------------
        # publishers
        # -------------------------
        self.debug_image_pub = self.create_publisher(Image, debug_image_topic, 10)
        self.text_pub = self.create_publisher(String, detection_text_topic, 10)

        # -------------------------
        # subscribers
        # -------------------------
        self.color_sub = Subscriber(self, Image, color_topic)
        self.depth_sub = Subscriber(self, Image, depth_topic)

        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=sync_queue_size,
            slop=sync_slop
        )
        self.sync.registerCallback(self.synced_callback)

        self.get_logger().info('Go2 YOLO distance node started.')
        self.get_logger().info(f'model_path: {model_path}')
        self.get_logger().info(f'color_topic: {color_topic}')
        self.get_logger().info(f'depth_topic: {depth_topic}')
        self.get_logger().info(f'debug_image_topic: {debug_image_topic}')
        self.get_logger().info(f'detection_text_topic: {detection_text_topic}')
        self.get_logger().info(f'conf_threshold: {self.conf_threshold}')

    def compute_depth_median(self, depth_frame, x1, y1, x2, y2):
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

    def paint_label(self, image, x1, y1, label, color):
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

    def synced_callback(self, color_msg, depth_msg):
        try:
            color_frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as exc:
            self.get_logger().error(f'Image conversion failed: {exc}')
            return

        try:
            results = self.model(
                color_frame,
                classes=[0, 39],     # person, bottle
                conf=self.conf_threshold,
                verbose=False
            )
        except Exception as exc:
            self.get_logger().error(f'YOLO inference failed: {exc}')
            return

        output = color_frame.copy()

        person_count = 0
        bottle_count = 0
        detection_lines = []

        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                distance_m = self.compute_depth_median(depth_frame, x1, y1, x2, y2)

                if cls_id == 0:
                    name = 'person'
                    color = (255, 0, 0)
                    person_count += 1
                elif cls_id == 39:
                    name = 'bottle'
                    color = (0, 165, 255)
                    bottle_count += 1
                else:
                    continue

                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

                if distance_m is not None:
                    label = f'{name} {conf:.2f} {distance_m:.2f}m'
                    detection_lines.append(
                        f'{name}, conf={conf:.2f}, distance={distance_m:.2f}m, box=({x1},{y1},{x2},{y2})'
                    )
                else:
                    label = f'{name} {conf:.2f} N/A'
                    detection_lines.append(
                        f'{name}, conf={conf:.2f}, distance=N/A, box=({x1},{y1},{x2},{y2})'
                    )

                self.paint_label(output, x1, y1, label, color)

        cv2.putText(
            output, f'Persons: {person_count}', (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
        )
        cv2.putText(
            output, f'Bottles: {bottle_count}', (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2
        )

        # debug image publish
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(output, encoding='bgr8')
            debug_msg.header = color_msg.header
            self.debug_image_pub.publish(debug_msg)
        except Exception as exc:
            self.get_logger().error(f'Debug image publish failed: {exc}')

        # text publish
        text_msg = String()
        if detection_lines:
            text_msg.data = '\n'.join(detection_lines)
        else:
            text_msg.data = 'no detections'
        self.text_pub.publish(text_msg)


def main(args=None):
    rclpy.init(args=args)
    node = Go2YoloDistanceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()