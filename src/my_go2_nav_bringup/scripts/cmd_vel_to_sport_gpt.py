#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from unitree_api.msg import Request
from unitree_sdk2py.go2.sport.sport_client import SportClient


class CmdVelClassicWalk(Node):
    def __init__(self):
        super().__init__("cmd_vel_classic_walk")

        self.pub = self.create_publisher(Request, "/api/sport/request", 10)

        self.sub = self.create_subscription(
            Twist,
            "/cmd_vel",
            self.cmd_vel_callback,
            10
        )

        self.client = SportClient()
        self.client.SetTimeout(10.0)
        self.client.Init()

        self.nav2_enabled = False
        self.last_cmd_time = self.get_clock().now()
        self.has_cmd = False

        self.init_step = 0

        self.init_timer = self.create_timer(0.5, self.init_sequence)
        self.watchdog_timer = self.create_timer(0.1, self.watchdog)

        self.get_logger().info("cmd_vel_classic_walk started")

    def init_sequence(self):
        if self.init_step == 0:
            self.get_logger().info("StandUp")
            self.client.StandUp()
            self.init_step += 1
            return

        if self.init_step < 7:
            self.init_step += 1
            return

        if self.init_step == 7:
            self.get_logger().info("ClassicWalk")
            self.client.ClassicWalk()
            self.init_step += 1
            return

        if self.init_step < 10:
            self.init_step += 1
            return

        self.get_logger().info("ClassicWalk ready. Nav2 cmd_vel accepted.")
        self.nav2_enabled = True
        self.init_timer.cancel()

    def cmd_vel_callback(self, msg):
        self.last_cmd_time = self.get_clock().now()
        self.has_cmd = True

        vx = self.clamp(msg.linear.x, -0.25, 0.25)
        vy = self.clamp(msg.linear.y, -0.15, 0.15)
        wz = self.clamp(msg.angular.z, -0.50, 0.50)

        if not self.nav2_enabled:
            return

        self.client.Move(vx, vy, wz)

    def watchdog(self):
        if not self.nav2_enabled:
            return

        if not self.has_cmd:
            return

        dt = (self.get_clock().now() - self.last_cmd_time).nanoseconds / 1e9

        if dt > 0.5:
            self.client.StopMove()
            self.has_cmd = False

    def clamp(self, value, min_value, max_value):
        return max(min_value, min(value, max_value))


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelClassicWalk()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.client.StopMove()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

