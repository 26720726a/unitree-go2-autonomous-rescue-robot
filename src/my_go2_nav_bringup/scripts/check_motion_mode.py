#!/usr/bin/env python3
"""
check_motion_mode.py

motion_switcher 서비스에 CheckMode(api_id=1001) 요청을 보내고
/api/motion_switcher/response 응답을 출력한다.
현재 sport 모드 이름(name)과 form 을 확인해 AI 모드 지원 여부를 가늠한다.

사용:
  source ~/capstone/install/setup.bash   # unitree_api 메시지 필요
  python3 check_motion_mode.py
"""
import json
import time
import rclpy
from rclpy.node import Node
from unitree_api.msg import Request, Response

MOTION_SWITCHER_API_ID_CHECK_MODE = 1001
REQ_TOPIC = '/api/motion_switcher/request'
RES_TOPIC = '/api/motion_switcher/response'


class CheckMode(Node):
    def __init__(self):
        super().__init__('check_motion_mode')
        self.pub = self.create_publisher(Request, REQ_TOPIC, 10)
        self.sub = self.create_subscription(Response, RES_TOPIC, self.on_res, 10)
        self.got = False
        # 퍼블리셔 연결 대기 후 요청 송신
        self.create_timer(0.5, self.send_once)
        self._sent = False

    def send_once(self):
        if self._sent:
            return
        self._sent = True
        req = Request()
        req.header.identity.id = int(time.time_ns() & 0x7FFFFFFF)
        req.header.identity.api_id = MOTION_SWITCHER_API_ID_CHECK_MODE
        req.parameter = ''
        self.pub.publish(req)
        self.get_logger().info(f'Sent CheckMode (api_id={MOTION_SWITCHER_API_ID_CHECK_MODE})')

    def on_res(self, msg):
        # api_id 가 일치하는 응답만
        if msg.header.identity.api_id != MOTION_SWITCHER_API_ID_CHECK_MODE:
            return
        code = msg.header.status.code
        data = msg.data
        self.get_logger().info(f'Response status.code={code}')
        self.get_logger().info(f'Response data raw = {data!r}')
        try:
            js = json.loads(data) if data else {}
            self.get_logger().info(f'  name = {js.get("name")!r}   form = {js.get("form")!r}')
        except Exception as e:
            self.get_logger().warn(f'  JSON parse failed: {e}')
        self.got = True


def main():
    rclpy.init()
    node = CheckMode()
    t0 = time.time()
    while rclpy.ok() and not node.got and (time.time() - t0) < 5.0:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not node.got:
        node.get_logger().warn('No response within 5s. '
                               'motion_switcher service / domain id 확인 필요.')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
