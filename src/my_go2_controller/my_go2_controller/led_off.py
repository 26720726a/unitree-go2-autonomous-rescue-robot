#!/usr/bin/env python3
# ~/capstone/src/my_go2_controller/my_go2_controller/led_off.py
"""
led_off.py
-------------------------------------------------------------------
Go2 전원 ON 시 기본으로 켜지는 전면 헤드 LED(초록색)를 끄는 노드.

LED 는 SportClient(모션)가 아니라 VUI(Visual UI) 서비스로 제어한다.
  - 밝기(brightness): 0 = 꺼짐 ~ 10 = 최대  -> api_id 1005
  - 색상(color)                            -> api_id 1007
밝기를 0으로 설정하면 LED 가 꺼진다.

move_test.py 가 /api/sport/request 로 Request 를 보내는 것과 동일하게,
VUI 는 /api/vui/request 로 Request 를 보낸다.
-------------------------------------------------------------------
"""

import rclpy
from rclpy.node import Node
from unitree_api.msg import Request
import json

# ---- VUI(Visual UI) API ID ----
VUI_API_SET_BRIGHTNESS = 1005   # parameter: {"brightness": 0~10}
VUI_API_SET_COLOR      = 1007   # parameter: {"color": ..., "time": ..., "flash_cycle": ...}


class LedOff(Node):
    def __init__(self):
        super().__init__('led_off')

        # ---------- 파라미터 ----------
        self.declare_parameter('brightness', 0)   # 0 = 끄기, 1~10 = 켜기
        self.declare_parameter('hold', False)     # True면 주기적으로 계속 재전송(상태 서비스가 다시 켜는 경우 대비)
        self.declare_parameter('period', 2.0)     # hold 재전송 주기(s)

        self.brightness = int(self.get_parameter('brightness').value)
        self.hold       = bool(self.get_parameter('hold').value)
        self.period     = float(self.get_parameter('period').value)

        self.pub = self.create_publisher(Request, '/api/vui/request', 10)

        # DDS 디스커버리 직후 1회만 보내면 누락될 수 있으므로
        # 처음엔 0.5s 간격으로 몇 번 보낸 뒤 종료(또는 hold)
        self.tick_count = 0
        self.timer = self.create_timer(0.5, self.tick)

        self.get_logger().info('=' * 60)
        self.get_logger().info('LED Off - 전면 헤드 LED 제어 (VUI SetBrightness)')
        self.get_logger().info(f'  target brightness = {self.brightness} (0=꺼짐)')
        self.get_logger().info(f'  topic = /api/vui/request, api_id = {VUI_API_SET_BRIGHTNESS}')
        self.get_logger().info(f'  hold  = {self.hold}')
        self.get_logger().info('=' * 60)

    def send_brightness(self, level):
        req = Request()
        req.header.identity.api_id = VUI_API_SET_BRIGHTNESS
        req.parameter = json.dumps({"brightness": int(level)})
        self.pub.publish(req)

    def tick(self):
        self.send_brightness(self.brightness)
        self.tick_count += 1

        # 초기 6회(약 3초) 전송으로 확실히 적용
        if self.tick_count <= 6:
            self.get_logger().info(f'  SetBrightness({self.brightness}) 전송 #{self.tick_count}')
            return

        if self.hold:
            # hold 모드: period 마다 재전송하도록 타이머 재설정
            if abs(self.timer.timer_period_ns / 1e9 - self.period) > 1e-3:
                self.timer.cancel()
                self.timer = self.create_timer(self.period, self.tick)
        else:
            self.get_logger().info('완료. LED 꺼짐 명령 전송 종료. (Ctrl+C)')
            self.timer.cancel()


def main():
    rclpy.init()
    node = LedOff()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
