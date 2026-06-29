import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from unitree_api.msg import Request
import sys
import termios
import tty
import select
import threading

# BalanceStand만 직접 로봇에 보냄 (이동 명령이 아니라 모드 명령)
SPORT_API_BALANCESTAND = 1002

HELP = """
=============================================
       Go2 Keyboard Teleop (via Safety Gate)
=============================================
  Movement:
      w  : forward       x : backward
      a  : turn left     d : turn right
      q  : strafe left   e : strafe right
      s  : stop

  Speed:
      +/= : speed up
      -/_ : speed down

  Special:
      b   : BalanceStand
      h   : show this help

  Quit: Ctrl+C
=============================================
  Note: Cmd is published to /cmd_vel.
        safety_gate forwards it to the robot
        and blocks forward motion if obstacle detected.
=============================================
"""

class TeleopKeyboard(Node):
    def __init__(self):
        super().__init__('teleop_keyboard')

        # 이동 명령 publisher — 표준 Twist로 /cmd_vel에 발행
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # BalanceStand 등 모드 명령 publisher — 로봇에 직접
        self.api_pub = self.create_publisher(Request, '/api/sport/request', 10)

        # 현재 명령 속도
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0

        # 속도 스케일 (한 번 누를 때 갱신되는 명령 크기)
        self.linear_step  = 0.45   # m/s
        self.angular_step = 1.0   # rad/s

        # 키 입력 → 마지막 갱신 시각
        self.last_key_time = self.get_clock().now()

        # 키가 안 눌리면 자동 정지 (안전)
        self.idle_timeout = 0.3  # 마지막 키 후 0.3초 지나면 멈춤

        # 50Hz로 현재 속도를 계속 publish
        self.timer = self.create_timer(0.02, self.publish_cmd)

        print(HELP)
        self.print_status()

    def print_status(self):
        print(f'\r  vx={self.vx:+.2f} m/s  vy={self.vy:+.2f} m/s  '
              f'vyaw={self.vyaw:+.2f} rad/s  | '
              f'step lin={self.linear_step:.2f} ang={self.angular_step:.2f}      ',
              end='', flush=True)

    def send_api_request(self, api_id, parameter=""):
        """BalanceStand 같은 모드 명령용 — 게이트를 거치지 않고 직접 보냄"""
        req = Request()
        req.header.identity.api_id = api_id
        req.parameter = parameter
        self.api_pub.publish(req)

    def publish_cmd(self):
        # 키 입력 끊긴 지 오래되면 자동 정지
        elapsed = (self.get_clock().now() - self.last_key_time).nanoseconds / 1e9
        if elapsed > self.idle_timeout:
            self.vx = 0.0
            self.vy = 0.0
            self.vyaw = 0.0

        # /cmd_vel로 Twist 발행 (safety_gate가 받아서 처리)
        twist = Twist()
        twist.linear.x  = float(self.vx)
        twist.linear.y  = float(self.vy)
        twist.angular.z = float(self.vyaw)
        self.cmd_pub.publish(twist)

    def on_key(self, key):
        """키 1회 입력 처리"""
        self.last_key_time = self.get_clock().now()

        if key == 'w':
            self.vx = +self.linear_step; self.vy = 0.0; self.vyaw = 0.0
        elif key == 'x':
            self.vx = -self.linear_step; self.vy = 0.0; self.vyaw = 0.0
        elif key == 'a':
            self.vx = 0.0; self.vy = 0.0; self.vyaw = +self.angular_step
        elif key == 'd':
            self.vx = 0.0; self.vy = 0.0; self.vyaw = -self.angular_step
        elif key == 'q':
            self.vx = 0.0; self.vy = +self.linear_step; self.vyaw = 0.0
        elif key == 'e':
            self.vx = 0.0; self.vy = -self.linear_step; self.vyaw = 0.0
        elif key == 's':
            self.vx = 0.0; self.vy = 0.0; self.vyaw = 0.0
        elif key in ('+', '='):
            self.linear_step  = min(self.linear_step  + 0.1, 1.0)
            self.angular_step = min(self.angular_step + 0.1, 1.5)
        elif key in ('-', '_'):
            self.linear_step  = max(self.linear_step  - 0.1, 0.1)
            self.angular_step = max(self.angular_step - 0.1, 0.1)
        elif key == 'b':
            self.send_api_request(SPORT_API_BALANCESTAND)
            print('\n  [BalanceStand sent]')
        elif key == 'h':
            print(HELP)
        else:
            return  # 모르는 키는 무시
        self.print_status()


# ---------- 키 입력 스레드 ----------
def get_key(timeout=0.05):
    """non-blocking 단일 키 읽기"""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            ch = sys.stdin.read(1)
            return ch
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def key_thread(node):
    while rclpy.ok():
        k = get_key()
        if k is None:
            continue
        if ord(k) == 3:  # Ctrl+C
            rclpy.shutdown()
            break
        node.on_key(k)


def main():
    rclpy.init()
    node = TeleopKeyboard()

    # 키 입력 스레드 시작
    t = threading.Thread(target=key_thread, args=(node,), daemon=True)
    t.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    # 종료 시 정지 명령 (Twist 0,0,0 발행)
    stop = Twist()
    node.cmd_pub.publish(stop)

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    print('\n[Teleop stopped]')


if __name__ == '__main__':
    main()
