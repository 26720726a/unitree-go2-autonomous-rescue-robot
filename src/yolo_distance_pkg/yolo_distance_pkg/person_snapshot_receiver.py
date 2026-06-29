#!/usr/bin/env python3
import os
from datetime import datetime

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import qos_profile_sensor_data


class PersonSnapshotReceiver(Node):
    def __init__(self):
        super().__init__('person_snapshot_receiver')

        self.declare_parameter('snapshot_topic', '/yolo/person_snapshot/compressed')
        self.declare_parameter('save_dir', os.path.expanduser('~/person_snapshots'))
        self.declare_parameter('show_image', False)

        snapshot_topic = self.get_parameter('snapshot_topic').get_parameter_value().string_value
        self.save_dir = self.get_parameter('save_dir').get_parameter_value().string_value
        self.show_image = self.get_parameter('show_image').get_parameter_value().bool_value

        os.makedirs(self.save_dir, exist_ok=True)

        self.sub = self.create_subscription(
            CompressedImage,
            snapshot_topic,
            self.snapshot_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(f'Subscribed to {snapshot_topic}')
        self.get_logger().info(f'Saving snapshots to {self.save_dir}')
        self.get_logger().info(f'show_image = {self.show_image}')

    def snapshot_callback(self, msg):
        self.get_logger().info('snapshot_callback entered')

        try:
            if not msg.data:
                self.get_logger().warn('Received empty compressed image data.')
                return

            self.get_logger().info(f'compressed image bytes = {len(msg.data)}')

            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                self.get_logger().warn('Failed to decode compressed image.')
                return

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            file_path = os.path.join(self.save_dir, f'received_snapshot_{timestamp}.jpg')

            ok = cv2.imwrite(file_path, frame)

            if ok:
                self.get_logger().info(f'Snapshot received and saved: {file_path}')
            else:
                self.get_logger().error(f'cv2.imwrite failed: {file_path}')
                return

            if self.show_image:
                cv2.imshow('Received Person Snapshot', frame)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Exception in snapshot_callback: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = PersonSnapshotReceiver()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()