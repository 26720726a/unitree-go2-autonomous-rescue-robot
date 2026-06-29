#!/usr/bin/env python3
"""
classic_walk_demo.py

Unitree Go2 를 "ClassicWalk(클래식 보행)" 모드로 걷게 하는 최소 예제.

동작 순서 (Unitree sport service 기준):
  1. BalanceStand  (api_id 1002)  : 균형 잡고 서기 (걷기 전 안정 상태)
  2. SpeedLevel    (api_id 1015)  : 속도 단계 설정 (선택)
  3. ClassicWalk   (api_id 2049)  : 보행 gait 을 '클래식'으로 1회 전환 {"data": true}
  4. Move          (api_id 1008)  : (vx, vy, vyaw) 속도로 주행 (반복 송신)
  5. StopMove      (api_id 1003)  : 정지

주의:
  - ClassicWalk 는 "걷는 명령"이 아니라 "보행 방식(gait)을 한 번 바꾸는" 명령이다.
    한 번 켜두고, 실제 이동은 Move 로 계속 보낸다.
  - Move 는 와치독 때문에 일정 주기(여기선 20Hz)로 계속 보내줘야 멈추지 않는다.
  - 안전: 로봇 주위 공간 확보, 비상정지(리모컨) 준비 후 실행할 것.

사용법:
  source ~/capstone/install/setup.bash       # unitree_api 메시지 필요
  python3 classic_walk_demo.py                                  # 기본: 0.3 m/s 로 3초 전진
  python3 classic_walk_demo.py --vx 0.3 --duration 5            # 5초 전진
  python3 classic_walk_demo.py --vx 0.0 --vyaw 0.5 --duration 4 # 제자리 회전
  python3 classic_walk_demo.py --vx 0.2 --vy 0.1               # 대각선
  python3 classic_walk_demo.py --speed fast                     # 속도 단계 fast
"""
import argparse
import json
import time

import rclpy
from rclpy.node import Node
from unitree_api.msg import Request

# --- Sport service API ID (unitree_sdk2_python/go2/sport/sport_api.py 기준) ---
SPORT_API_BALANCESTAND = 1002
SPORT_API_STOPMOVE     = 1003
SPORT_API_MOVE         = 1008
SPORT_API_SPEEDLEVEL   = 1015
SPORT_API_CLASSICWALK  = 2049   # ClassicWalk(flag) : {"data": true/false}

SPORT_REQUEST_TOPIC = '/api/sport/request'

# -1 = slow, 0 = normal, 1 = fast
SPEED_LEVEL_MAP = {'slow': -1, 'normal': 0, 'fast': 1}


class ClassicWalkDemo(Node):
    def __init__(self, vx, vy, vyaw, duration, speed_level, settle):
        super().__init__('classic_walk_demo')

        self.vx = vx
        self.vy = vy
        self.vyaw = vyaw
        self.duration = duration
        self.speed_level = speed_level
        self.settle = settle          # gait 전환 후 Move 전까지 안정화 대기(초)

        self.pub = self.create_publisher(Request, SPORT_REQUEST_TOPIC, 10)

        self.move_period = 0.05       # 20 Hz 로 Move 송신
        self.move_timer = None
        self.move_start = None

        self.get_logger().info(
            f'ClassicWalkDemo: vx={vx} vy={vy} vyaw={vyaw} '
            f'duration={duration}s speed={speed_level}')

        # 퍼블리셔가 /api/sport/request 와 연결될 시간을 살짝 준 뒤 시퀀스 시작
        self.create_timer(0.5, self._start_once)
        self._started = False

    # ------------------------------------------------------------------

    def _make_request(self, api_id, parameter=''):
        req = Request()
        req.header.identity.id = int(time.time_ns() & 0x7FFFFFFF)
        req.header.identity.api_id = api_id
        req.header.policy.priority = 0
        req.header.policy.noreply = True
        req.parameter = parameter
        return req

    def _send(self, api_id, parameter=''):
        self.pub.publish(self._make_request(api_id, parameter))

    # ------------------------------------------------------------------

    def _start_once(self):
        if self._started:
            return
        self._started = True

        # 1) 균형 잡고 서기
        self.get_logger().info('[1/4] BalanceStand')
        self._send(SPORT_API_BALANCESTAND)
        time.sleep(1.0)

        # 2) 속도 단계 설정 (선택)
        lvl = SPEED_LEVEL_MAP[self.speed_level]
        self.get_logger().info(f'[2/4] SpeedLevel = {self.speed_level} ({lvl})')
        self._send(SPORT_API_SPEEDLEVEL, json.dumps({'data': lvl}))
        time.sleep(0.3)

        # 3) ClassicWalk gait 켜기 (한 번만)
        self.get_logger().info('[3/4] ClassicWalk (gait ON)')
        self._send(SPORT_API_CLASSICWALK, json.dumps({'data': True}))
        time.sleep(self.settle)

        # 4) Move 를 주기적으로 송신 시작
        self.get_logger().info(
            f'[4/4] Move ({self.vx}, {self.vy}, {self.vyaw}) for {self.duration}s')
        self.move_start = time.time()
        self.move_timer = self.create_timer(self.move_period, self._move_tick)

    def _move_tick(self):
        elapsed = time.time() - self.move_start
        if elapsed >= self.duration:
            self.move_timer.cancel()
            self.get_logger().info('Done. StopMove.')
            self._send(SPORT_API_STOPMOVE)
            # 정지 명령이 확실히 전달되도록 잠시 후 종료
            self.create_timer(0.3, self._shutdown)
            return
        self._send(
            SPORT_API_MOVE,
            json.dumps({'x': float(self.vx),
                        'y': float(self.vy),
                        'z': float(self.vyaw)}))

    def _shutdown(self):
        rclpy.shutdown()


def parse_args():
    p = argparse.ArgumentParser(description='Go2 ClassicWalk 데모')
    p.add_argument('--vx', type=float, default=0.3, help='전진 속도 m/s (+앞)')
    p.add_argument('--vy', type=float, default=0.0, help='횡 이동 속도 m/s (+왼쪽)')
    p.add_argument('--vyaw', type=float, default=0.0, help='회전 속도 rad/s (+좌회전)')
    p.add_argument('--duration', type=float, default=3.0, help='이동 시간(초)')
    p.add_argument('--speed', choices=SPEED_LEVEL_MAP.keys(), default='normal',
                   help='속도 단계 slow/normal/fast')
    p.add_argument('--settle', type=float, default=0.5,
                   help='ClassicWalk 전환 후 안정화 대기(초)')
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = ClassicWalkDemo(
        vx=args.vx, vy=args.vy, vyaw=args.vyaw,
        duration=args.duration, speed_level=args.speed, settle=args.settle)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Ctrl-C 시 안전 정지
        try:
            node._send(SPORT_API_STOPMOVE)
            time.sleep(0.1)
        except Exception:
            pass
    finally:
        if rclpy.ok():
            try:
                node._send(SPORT_API_STOPMOVE)
                time.sleep(0.1)
            except Exception:
                pass
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
