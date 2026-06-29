import rclpy
from rclpy.node import Node
from unitree_go.msg import SportModeState

class StateReader(Node):
    def __init__(self):
        super().__init__('state_reader')
        self.sub = self.create_subscription(
            SportModeState,
            '/sportmodestate',
            self.callback,
            10
        )
        self.get_logger().info('State reader started')
        self.count = 0

    def callback(self, msg):
        self.count += 1
        # 50번에 한 번만 출력 (보통 500Hz 정도라 너무 빠름)
        if self.count % 50 != 0:
            return

        self.get_logger().info(
            f'pos=({msg.position[0]:+.3f}, {msg.position[1]:+.3f}, {msg.position[2]:+.3f}) | '
            f'height={msg.body_height:.3f} | '
            f'vel=({msg.velocity[0]:+.3f}, {msg.velocity[1]:+.3f}) | '
            f'yaw_rate={msg.yaw_speed:+.3f} | '
            f'mode={msg.mode}'
        )

def main():
    rclpy.init()
    node = StateReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()