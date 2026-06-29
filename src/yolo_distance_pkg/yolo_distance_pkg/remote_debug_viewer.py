#!/usr/bin/env python3
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class RemoteDebugViewer(Node):
    def __init__(self):
        super().__init__('remote_debug_viewer')

        self.bridge = CvBridge()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.sub = self.create_subscription(
            Image,
            '/yolo/debug_image',
            self.image_callback,
            qos
        )

        self.get_logger().info('Subscribed to /yolo/debug_image')

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow('Go2 YOLO Debug View', frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = RemoteDebugViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()