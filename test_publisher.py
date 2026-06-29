import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32

class TestPublisher(Node):
    def __init__(self):
        super().__init__('test_publisher')
        self.pub_str = self.create_publisher(String, '/capstone/test_string', 10)
        self.pub_int = self.create_publisher(Int32, '/capstone/test_int', 10)
        self.count = 0
        self.create_timer(1.0, self.tick)

    def tick(self):
        s = String(); s.data = f'hello from internal jetson {self.count}'
        i = Int32(); i.data = self.count
        self.pub_str.publish(s)
        self.pub_int.publish(i)
        self.get_logger().info(f'published: {s.data} / {i.data}')
        self.count += 1

def main():
    rclpy.init()
    node = TestPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
