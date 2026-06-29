import math

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class OdomRelay(Node):
    def __init__(self):
        super().__init__('odom_relay')

        self.declare_parameter('input_odom', '/utlidar/robot_odom')
        self.declare_parameter('output_odom', '/odom')
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('child_frame_id', 'base_link')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('force_2d', True)
        self.declare_parameter('restamp_with_now', True)

        # 방향/위치 보정값
        # yaw_offset: 원본 yaw에 더해줄 값(rad)
        # base_offset_x/y: base_link 기준 위치 보정값(m)
        # x: 전방(+), y: 좌측(+)
        self.declare_parameter('yaw_offset', 0.0)
        self.declare_parameter('base_offset_x', 0.0)
        self.declare_parameter('base_offset_y', 0.0)

        self.input_odom = self.get_parameter('input_odom').value
        self.output_odom = self.get_parameter('output_odom').value
        self.frame_id = self.get_parameter('frame_id').value
        self.child_frame_id = self.get_parameter('child_frame_id').value
        self.publish_tf = self.get_parameter('publish_tf').value
        self.force_2d = self.get_parameter('force_2d').value
        self.restamp_with_now = self.get_parameter('restamp_with_now').value

        self.yaw_offset = float(self.get_parameter('yaw_offset').value)
        self.base_offset_x = float(self.get_parameter('base_offset_x').value)
        self.base_offset_y = float(self.get_parameter('base_offset_y').value)

        self.pub = self.create_publisher(Odometry, self.output_odom, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.sub = self.create_subscription(
            Odometry,
            self.input_odom,
            self.callback,
            10
        )

        self.get_logger().info(
            f'Relay odom: {self.input_odom} -> {self.output_odom}'
        )
        self.get_logger().info(
            f'TF: {self.frame_id} -> {self.child_frame_id}, '
            f'publish_tf={self.publish_tf}, '
            f'force_2d={self.force_2d}, '
            f'restamp_with_now={self.restamp_with_now}, '
            f'yaw_offset={self.yaw_offset:.4f}, '
            f'base_offset_x={self.base_offset_x:.3f}, '
            f'base_offset_y={self.base_offset_y:.3f}'
        )

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def yaw_to_quaternion(self, yaw):
        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)
        return qz, qw

    def normalize_angle(self, a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def callback(self, msg):
        out = Odometry()

        if self.restamp_with_now:
            out.header.stamp = self.get_clock().now().to_msg()
        else:
            out.header.stamp = msg.header.stamp

        out.header.frame_id = self.frame_id
        out.child_frame_id = self.child_frame_id

        out.pose = msg.pose
        out.twist = msg.twist

        # 원본 yaw
        yaw_raw = self.quaternion_to_yaw(msg.pose.pose.orientation)

        # 방향 보정
        yaw = self.normalize_angle(yaw_raw + self.yaw_offset)

        # base_link 기준 위치 보정을 odom/map 좌표계로 회전해서 더함
        # base_offset_x: 로봇 전방 방향 보정
        # base_offset_y: 로봇 좌측 방향 보정
        dx = math.cos(yaw) * self.base_offset_x - math.sin(yaw) * self.base_offset_y
        dy = math.sin(yaw) * self.base_offset_x + math.cos(yaw) * self.base_offset_y

        out.pose.pose.position.x = msg.pose.pose.position.x + dx
        out.pose.pose.position.y = msg.pose.pose.position.y + dy

        if self.force_2d:
            qz, qw = self.yaw_to_quaternion(yaw)

            out.pose.pose.position.z = 0.0
            out.pose.pose.orientation.x = 0.0
            out.pose.pose.orientation.y = 0.0
            out.pose.pose.orientation.z = qz
            out.pose.pose.orientation.w = qw

            out.twist.twist.linear.z = 0.0
            out.twist.twist.angular.x = 0.0
            out.twist.twist.angular.y = 0.0
        else:
            qz, qw = self.yaw_to_quaternion(yaw)
            out.pose.pose.orientation.z = qz
            out.pose.pose.orientation.w = qw

        self.pub.publish(out)

        if self.publish_tf:
            t = TransformStamped()
            t.header.stamp = out.header.stamp
            t.header.frame_id = self.frame_id
            t.child_frame_id = self.child_frame_id

            t.transform.translation.x = out.pose.pose.position.x
            t.transform.translation.y = out.pose.pose.position.y
            t.transform.translation.z = out.pose.pose.position.z
            t.transform.rotation = out.pose.pose.orientation

            self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdomRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
