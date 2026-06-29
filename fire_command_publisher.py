import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ── 키 → 명령 매핑 ────────────────────────────
#  나중에 이 발행부를 Go2가 대체하게 됨.
#  지금은 키보드 입력으로 /fire_command 토픽에 명령을 발행한다.
KEY_MAP = {
    'f': 'fire',     # 발사 시퀀스
    'm': 'marker',   # 모터3 패턴
    's': 'stop',     # 긴급 정지
}


class FireCommandPublisher(Node):
    def __init__(self):
        super().__init__('fire_command_publisher')
        self.publisher = self.create_publisher(String, '/fire_command', 10)
        self.get_logger().info('발행 노드 시작 - 토픽: /fire_command')

    def send(self, command):
        msg = String()
        msg.data = command
        self.publisher.publish(msg)
        self.get_logger().info(f'명령 발행: {command}')


def print_menu():
    print("=" * 50)
    print("발사 명령 발행기 (키보드 모드)")
    print("토픽: /fire_command")
    print("-" * 50)
    print("  f : fire    (발사 시퀀스)")
    print("  m : marker  (모터3 패턴)")
    print("  s : stop    (긴급 정지)")
    print("  q : 종료")
    print("=" * 50)
    print("명령을 입력하고 Enter:")


def main():
    rclpy.init()
    node = FireCommandPublisher()
    print_menu()

    try:
        while rclpy.ok():
            try:
                key = input("> ").strip().lower()
            except EOFError:
                break

            if key == 'q':
                print("종료합니다.")
                break

            if key in KEY_MAP:
                node.send(KEY_MAP[key])
            elif key in KEY_MAP.values():
                # 'fire' / 'marker' / 'stop' 처럼 전체 단어를 직접 입력한 경우
                node.send(key)
            elif key == '':
                continue
            else:
                print(f"알 수 없는 입력: {key}  (f / m / s / q)")

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
