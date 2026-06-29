#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, Imu


class LidarImuStampSyncNode(Node):
    def __init__(self):
        super().__init__('lidar_imu_stamp_sync_node')

        qos_reliable_10 = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        qos_reliable_100 = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100
        )

        self.lidar_count = 0
        self.imu_count = 0

        self.lidar_pub = self.create_publisher(
            PointCloud2,
            '/lidar_points_sync',
            qos_reliable_10
        )

        self.imu_pub = self.create_publisher(
            Imu,
            '/utlidar/imu_sync',
            qos_reliable_100
        )

        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/lidar_points',
            self.lidar_callback,
            qos_reliable_10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/utlidar/imu',
            self.imu_callback,
            qos_reliable_100
        )

        self.get_logger().info('lidar_imu_stamp_sync_node started')
        self.get_logger().info('Sub: /lidar_points, /utlidar/imu')
        self.get_logger().info('Pub: /lidar_points_sync, /utlidar/imu_sync')

    def lidar_callback(self, msg: PointCloud2):
        out_msg = msg
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = msg.header.frame_id

        self.lidar_pub.publish(out_msg)

        self.lidar_count += 1
        if self.lidar_count % 20 == 0:
            self.get_logger().info(
                f'LiDAR sync pub count={self.lidar_count}, '
                f'stamp={out_msg.header.stamp.sec}.{out_msg.header.stamp.nanosec}'
            )

    def imu_callback(self, msg: Imu):
        out_msg = msg
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = msg.header.frame_id

        self.imu_pub.publish(out_msg)

        self.imu_count += 1
        if self.imu_count % 500 == 0:
            self.get_logger().info(
                f'IMU sync pub count={self.imu_count}, '
                f'stamp={out_msg.header.stamp.sec}.{out_msg.header.stamp.nanosec}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = LidarImuStampSyncNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()