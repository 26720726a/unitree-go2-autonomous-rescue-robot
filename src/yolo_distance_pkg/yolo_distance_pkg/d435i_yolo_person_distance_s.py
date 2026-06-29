import math
import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

from message_filters import Subscriber, ApproximateTimeSynchronizer


class D435iObjectDistanceNode(Node):
    def __init__(self):
        super().__init__('d435i_object_distance_node')

        self.bridge = CvBridge()

        # 🔥 YOLOv8s 사용 (정확도 ↑)
        self.model = YOLO("yolov8s.pt")

        # 카메라 토픽
        self.color_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')

        self.ts = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info("YOLOv8s Person + Bottle distance node started.")

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

        # 중앙 영역 ROI
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

    def sync_callback(self, color_msg, depth_msg):
        try:
            color_frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Conversion failed: {e}')
            return

        # person(0), bottle(39)
        results = self.model(color_frame, classes=[0, 39], conf=0.4, verbose=False)

        display = color_frame.copy()

        person_count = 0
        bottle_count = 0

        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                distance_m = self.get_depth_median(depth_frame, x1, y1, x2, y2)

                # 클래스별 설정
                if cls_id == 0:
                    name = "person"
                    color = (255, 0, 0)
                    person_count += 1
                elif cls_id == 39:
                    name = "bottle"
                    color = (0, 165, 255)
                    bottle_count += 1
                else:
                    continue

                # 박스
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                # 라벨
                if distance_m is not None:
                    label = f"{name} {conf:.2f} {distance_m:.2f}m"
                else:
                    label = f"{name} {conf:.2f} N/A"

                self.draw_label(display, x1, y1, label, color)

        # 카운트 표시
        cv2.putText(display, f"Persons: {person_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.putText(display, f"Bottles: {bottle_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

        cv2.imshow("YOLOv8s Distance", display)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = D435iObjectDistanceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
