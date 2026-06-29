#!/usr/bin/env python3
"""
Go2 gait 진단 스크립트 (펌웨어 1.9+ / advanced gait 진입 확인용)  -- 무(無)이동 버전

로봇을 걷게 하지 않고, 명령이 '수락'되는지만 응답 코드로 확인한다.
  - SelectMode 응답 code=0  -> 모드 전환 성공
  - ClassicWalk(2049) 응답 code=0 -> 그 모드가 ClassicWalk 를 받아들임 = 정답 모드
(실제 보행 모습은 나중에 nav2 로 움직일 때 확인)

사용법 (고투에서 직접):
    python3 test_gait.py                # 현재 모드만 조회 (CheckMode)
    python3 test_gait.py ai             # ai 모드 전환 후 ClassicWalk 시도
    python3 test_gait.py normal
    python3 test_gait.py advanced
※ 로봇은 움직이지 않음. 실행 후 최소 5초는 Ctrl-C 하지 말고 응답 대기.
"""
import sys
import json
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from unitree_api.msg import Request, Response

MOTION_REQ = '/api/motion_switcher/request'
MOTION_RES = '/api/motion_switcher/response'
SPORT_REQ  = '/api/sport/request'
SPORT_RES  = '/api/sport/response'

MS_CHECK_MODE     = 1001
MS_SELECT_MODE    = 1002
SPORT_CLASSICWALK = 2049

# 응답은 best-effort 로 구독 (publisher 가 reliable/best-effort 어느 쪽이든 수신)
RES_QOS = QoSProfile(depth=10,
                     reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST)


def make_req(api_id, param=''):
    r = Request()
    r.header.identity.id     = int(time.time_ns() & 0x7FFFFFFF)
    r.header.identity.api_id = api_id
    r.header.policy.priority = 0
    r.header.policy.noreply  = False
    r.parameter = param
    return r


class TestGait(Node):
    def __init__(self, mode):
        super().__init__('test_gait')
        self.mode = mode

        self.ms_pub = self.create_publisher(Request, MOTION_REQ, 10)
        self.sp_pub = self.create_publisher(Request, SPORT_REQ, 10)
        self.create_subscription(Response, MOTION_RES, self._ms_res, RES_QOS)
        self.create_subscription(Response, SPORT_RES, self._sp_res, RES_QOS)

        self.t0 = time.time()
        self.step = 0
        self.create_timer(0.1, self._tick)

    def _ms_res(self, m):
        self.get_logger().info(
            f'[MOTION RESP] api_id={m.header.identity.api_id} '
            f'code={m.header.status.code} data={m.data}')

    def _sp_res(self, m):
        self.get_logger().info(
            f'[SPORT  RESP] api_id={m.header.identity.api_id} '
            f'code={m.header.status.code}')

    def _tick(self):
        el = time.time() - self.t0

        if self.step == 0 and el >= 0.5:
            self.step = 1
            self.get_logger().info('==> CheckMode(1001)  (현재 모드 조회)')
            self.ms_pub.publish(make_req(MS_CHECK_MODE))

        elif self.step == 1 and el >= 2.0:
            self.step = 2
            if self.mode is None:
                self.get_logger().info('모드 인자 없음 -> 조회만. 위 [MOTION RESP] data= 확인. (Ctrl-C)')
            else:
                self.get_logger().info(f'==> SelectMode(1002) name="{self.mode}"')
                self.ms_pub.publish(make_req(MS_SELECT_MODE, json.dumps({'name': self.mode})))

        elif self.step == 2 and el >= 4.0 and self.mode is not None:
            self.step = 3
            self.get_logger().info('==> ClassicWalk(2049) data=True  (로봇 이동 안 함)')
            self.sp_pub.publish(make_req(SPORT_CLASSICWALK, json.dumps({'data': True})))
            self.get_logger().info('--- 응답 대기. [SPORT RESP] api_id=2049 code 확인. (Ctrl-C) ---')


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    rclpy.init()
    node = TestGait(mode)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
