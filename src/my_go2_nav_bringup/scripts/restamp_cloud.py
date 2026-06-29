#!/usr/bin/env python3
"""
Republishes a PointCloud2 topic with the current ROS time as the header stamp.
Needed when the source (e.g. Go2 UTLidar) publishes clouds with stale timestamps
that fall outside the TF buffer window, causing pointcloud_to_laserscan to drop them.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2


class RestampCloud(Node):
    def __init__(self):
        super().__init__('restamp_cloud')
        self.declare_parameter('input_topic',  '/utlidar/cloud_deskewed')
        self.declare_parameter('output_topic', '/utlidar/cloud_deskewed_restamped')

        in_topic  = self.get_parameter('input_topic').value
        out_topic = self.get_parameter('output_topic').value

        self.pub = self.create_publisher(PointCloud2, out_topic, 10)
        self.sub = self.create_subscription(PointCloud2, in_topic, self.cb, 10)
        self.get_logger().info(f'{in_topic} -> {out_topic} (restamped)')

    def cb(self, msg: PointCloud2):
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = RestampCloud()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
