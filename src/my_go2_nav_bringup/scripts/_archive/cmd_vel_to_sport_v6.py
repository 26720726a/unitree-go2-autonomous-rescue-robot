#!/usr/bin/env python3
"""
cmd_vel_to_sport (v4)

Nav2 /cmd_vel -> Go2 sport-mode Request 변환 노드.
요청(Request) 생성 규약은 unitree_ros2 공식 예제
  example/src/src/common/ros2_sport_client.cpp
를 그대로 따른다:

  - publish 토픽:  /api/sport/request
  - 각 명령은 req.header.identity.api_id 와 req.parameter(JSON 문자열)만 설정.
    (공식 C++ SportClient 는 policy/noreply 를 건드리지 않는다)
  - ClassicWalk(flag): api_id=2049, parameter={"data": flag}
  - SpeedLevel(level): api_id=1015, parameter={"data": level}
  - Move(vx,vy,vyaw): api_id=1008, parameter={"x":vx,"y":vy,"z":vyaw}
  - BalanceStand():   api_id=1002, parameter 없음
  - StopMove():       api_id=1003, parameter 없음

부팅 시퀀스: BalanceStand -> SpeedLevel -> ClassicWalk(True) -> /cmd_vel 수락.
즉 sport mode 의 ClassicWalk(클래식 보행)를 처음부터 켜고 시작한다.
"""
import json
import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from unitree_api.msg import Request

# --- unitree_ros2 ros2_sport_client.h 의 ROBOT_SPORT_API_ID_* 와 동일 ---
ROBOT_SPORT_API_ID_BALANCESTAND = 1002
ROBOT_SPORT_API_ID_STOPMOVE = 1003
ROBOT_SPORT_API_ID_MOVE = 1008
ROBOT_SPORT_API_ID_SPEEDLEVEL = 1015
ROBOT_SPORT_API_ID_CLASSICWALK = 2049

# -1=slow, 0=normal, 1=fast  (펌웨어 허용 범위는 실기로 확인 권장)
SPEED_LEVEL_MAP = {'slow': -1, 'normal': 0, 'fast': 1}


def clamp(value, limit):
    if limit <= 0.0:
        return 0.0
    return max(-limit, min(limit, value))


def smoothstep(edge0, edge1, x):
    """edge0 <= x <= edge1 구간에서 0->1 로 매끄럽게."""
    if edge1 == edge0:
        return 0.0 if x < edge0 else 1.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


class CmdVelToSport(Node):
    def __init__(self):
        super().__init__('cmd_vel_to_sport')

        # ===== Parameters =====
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('sport_request_topic', '/api/sport/request')
        self.declare_parameter('max_vx', 0.6)
        self.declare_parameter('max_vy', 0.3)
        self.declare_parameter('max_vyaw', 1.5)
        self.declare_parameter('cmd_timeout', 0.5)
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('zero_deadband', 0.02)
        self.declare_parameter('init_balance_stand', True)
        self.declare_parameter('init_wait_sec', 2.0)
        self.declare_parameter('speed_level', 'normal')   # slow / normal / fast
        self.declare_parameter('classic_walk_flag', True)  # ClassicWalk flag (True=enable)

        # ----- 회전-후-전진(rotate-to-face) -----
        self.declare_parameter('face_goal_direction', True)
        self.declare_parameter('align_yaw_gain', 1.5)             # heading 오차 -> 회전속도 게인
        self.declare_parameter('rotate_in_threshold', 0.6)         # rad. 이 이상 어긋나면 전진 게이트 닫힘
        self.declare_parameter('rotate_out_threshold', 0.35)       # rad. 이 이하로 정렬되면 전진 게이트 열림 (히스테리시스)
        self.declare_parameter('min_translation_speed', 0.05)      # m/s. 이 이하면 순수 회전 명령으로 통과

        # ----- 부드러운 동작(슬루/정지) -----
        self.declare_parameter('max_lin_accel', 1.0)               # m/s^2
        self.declare_parameter('max_yaw_accel', 3.0)               # rad/s^2
        self.declare_parameter('hold_with_move', True)             # 정지 시 Move(0,0,0) 홀드(True) vs StopMove(False)

        g = self.get_parameter
        self.cmd_vel_topic = g('cmd_vel_topic').value
        self.sport_request_topic = g('sport_request_topic').value
        self.max_vx = float(g('max_vx').value)
        self.max_vy = float(g('max_vy').value)
        self.max_vyaw = float(g('max_vyaw').value)
        self.cmd_timeout = float(g('cmd_timeout').value)
        self.publish_rate = float(g('publish_rate').value)
        self.zero_deadband = float(g('zero_deadband').value)
        self.init_balance_stand = bool(g('init_balance_stand').value)
        self.init_wait_sec = float(g('init_wait_sec').value)

        speed_level_str = g('speed_level').value.lower()
        if speed_level_str not in SPEED_LEVEL_MAP:
            self.get_logger().warn(
                f"Unknown speed_level '{speed_level_str}', defaulting to 'normal'"
            )
            speed_level_str = 'normal'
        self.speed_level = SPEED_LEVEL_MAP[speed_level_str]
        self.speed_level_str = speed_level_str
        self.classic_walk_flag = bool(g('classic_walk_flag').value)

        self.face_goal_direction = bool(g('face_goal_direction').value)
        self.align_yaw_gain = float(g('align_yaw_gain').value)
        self.rotate_in_threshold = float(g('rotate_in_threshold').value)
        self.rotate_out_threshold = float(g('rotate_out_threshold').value)
        self.min_translation_speed = float(g('min_translation_speed').value)

        self.max_lin_accel = float(g('max_lin_accel').value)
        self.max_yaw_accel = float(g('max_yaw_accel').value)
        self.hold_with_move = bool(g('hold_with_move').value)

        # ===== State =====
        # target_*  : 콜백이 설정하는 목표 속도(shape 적용 후)
        # cur_*     : 실제로 매 사이클 슬루 제한으로 따라가는 명령 속도
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_vyaw = 0.0
        self.cur_vx = 0.0
        self.cur_vy = 0.0
        self.cur_vyaw = 0.0

        self.last_cmd_time = None
        self.is_ready = False
        self.timed_out = False
        self.forward_gate_open = False   # 히스테리시스 상태

        self.init_start_time = self.get_clock().now()
        self.speed_level_sent = False
        self.classic_walk_sent = False

        # ===== Pub/Sub =====
        self.pub = self.create_publisher(Request, self.sport_request_topic, 10)
        self.sub = self.create_subscription(
            Twist, self.cmd_vel_topic, self.cmd_callback, 10
        )

        # ===== Timer =====
        self.dt = 1.0 / self.publish_rate
        self.send_timer = self.create_timer(self.dt, self.send_loop)

        # ===== Init =====
        if self.init_balance_stand:
            self.get_logger().info('[INIT] Sending BalanceStand immediately...')
            self.send_balance_stand()
        self.init_start_time = self.get_clock().now()
        self.init_timer = self.create_timer(0.1, self.do_init_sequence)

        L = self.get_logger()
        L.info('=' * 60)
        L.info('   cmd_vel_to_sport (v4 / ClassicWalk)  --  PARAMETER SUMMARY')
        L.info('=' * 60)
        L.info(f'  subscribe={self.cmd_vel_topic}  publish={self.sport_request_topic}')
        L.info(f'  rate={self.publish_rate}Hz  cmd_timeout={self.cmd_timeout}s')
        L.info(f'  max v: vx={self.max_vx} vy={self.max_vy} vyaw={self.max_vyaw}')
        L.info(f'  init: balance={self.init_balance_stand} wait={self.init_wait_sec}s '
               f'speed={self.speed_level_str}({self.speed_level}) '
               f'classic_walk_flag={self.classic_walk_flag}')
        L.info(f'  face_goal={self.face_goal_direction} '
               f'in={self.rotate_in_threshold} out={self.rotate_out_threshold} '
               f'yaw_gain={self.align_yaw_gain}')
        L.info(f'  smooth: lin_accel={self.max_lin_accel} yaw_accel={self.max_yaw_accel} '
               f'hold_with_move={self.hold_with_move}')
        L.info('=' * 60)

        self._topic_log_fired = False
        self.create_timer(1.0, self._log_active_topics)

    # ------------------------------------------------------------------

    def _log_active_topics(self):
        if self._topic_log_fired:
            return
        self._topic_log_fired = True
        all_topics = dict(self.get_topic_names_and_types())
        watch = [
            self.cmd_vel_topic, self.sport_request_topic,
            '/utlidar/cloud_deskewed', '/utlidar/cloud_deskewed_restamped',
            '/utlidar/robot_odom', '/scan', '/odom', '/map', '/tf', '/tf_static',
        ]
        L = self.get_logger()
        L.info('-' * 60)
        L.info('  [Active ROS2 Topics  (1s after start)]')
        for t in watch:
            if t in all_topics:
                L.info(f'    [O] {t}  ({", ".join(all_topics[t])})')
            else:
                L.info(f'    [X] {t}  -- NOT found')
        L.info('-' * 60)

    # ------------------------------------------------------------------

    def apply_deadband(self, value):
        return 0.0 if abs(value) < self.zero_deadband else value

    def shape_velocity(self, vx, vy, vyaw):
        """
        Nav2 의 (vx, vy) 를 진행 방향으로 해석해 유니사이클(전진+회전)로 변환.
        - 목표가 뒤/옆이면 후진/게걸음 대신 먼저 제자리 회전.
        - smoothstep + 히스테리시스로 전진 게이트를 부드럽게 여닫아 채터링 제거.
        """
        if not self.face_goal_direction:
            return vx, clamp(vy, self.max_vy), clamp(vyaw, self.max_vyaw)

        speed = math.hypot(vx, vy)

        # 병진 성분이 거의 없으면 -> 순수 회전 명령으로 통과 (최종 yaw 정렬 등)
        if speed < self.min_translation_speed:
            self.forward_gate_open = False
            return 0.0, 0.0, clamp(vyaw, self.max_vyaw)

        heading_err = math.atan2(vy, vx)       # [-pi, pi]
        abs_err = abs(heading_err)

        # 히스테리시스: 열려 있으면 in_threshold 넘어야 닫힘, 닫혀 있으면 out_threshold 밑으로 와야 열림
        if self.forward_gate_open:
            if abs_err > self.rotate_in_threshold:
                self.forward_gate_open = False
        else:
            if abs_err < self.rotate_out_threshold:
                self.forward_gate_open = True

        # 전진 게이트(0~1) : 정렬될수록 1. smoothstep 으로 경계 부드럽게.
        gate = 1.0 - smoothstep(self.rotate_out_threshold,
                                self.rotate_in_threshold, abs_err)
        if not self.forward_gate_open and abs_err > self.rotate_in_threshold:
            gate = 0.0

        alignment = max(0.0, math.cos(heading_err))   # 정면=1, 측면=0, 후방<0->0

        out_vyaw = clamp(self.align_yaw_gain * heading_err + alignment * vyaw,
                         self.max_vyaw)
        out_vx = clamp(max(0.0, speed * math.cos(heading_err)) * gate, self.max_vx)
        out_vy = 0.0
        return out_vx, out_vy, out_vyaw

    def _slew(self, cur, target, max_accel):
        max_delta = max_accel * self.dt
        if target > cur + max_delta:
            return cur + max_delta
        if target < cur - max_delta:
            return cur - max_delta
        return target

    # ------------------------------------------------------------------
    # unitree_ros2 SportClient 와 동일한 방식의 Request 빌더.
    # 공식 C++ 코드는 api_id 와 parameter(JSON) 만 설정한다.
    # identity.id 는 추적용으로만 채워 둔다(선택).
    # ------------------------------------------------------------------

    def _make_request(self, api_id, parameter=''):
        req = Request()
        req.header.identity.id = int(time.time_ns() & 0x7FFFFFFF)
        req.header.identity.api_id = api_id
        req.parameter = parameter
        return req

    def send_balance_stand(self):
        # SportClient::BalanceStand : api_id=1002, parameter 없음
        self.pub.publish(self._make_request(ROBOT_SPORT_API_ID_BALANCESTAND))

    def send_stopmove(self):
        # SportClient::StopMove : api_id=1003, parameter 없음
        self.pub.publish(self._make_request(ROBOT_SPORT_API_ID_STOPMOVE))

    def send_speed_level(self):
        # SportClient::SpeedLevel(level) : api_id=1015, {"data": level}
        self.pub.publish(self._make_request(
            ROBOT_SPORT_API_ID_SPEEDLEVEL, json.dumps({'data': self.speed_level})))
        self.get_logger().info(
            f'[INIT] SpeedLevel set to {self.speed_level_str} ({self.speed_level})')

    def send_classic_walk(self):
        # SportClient::ClassicWalk(flag) : api_id=2049, {"data": flag}
        self.pub.publish(self._make_request(
            ROBOT_SPORT_API_ID_CLASSICWALK,
            json.dumps({'data': self.classic_walk_flag})))
        self.get_logger().info(
            f'[INIT] ClassicWalk gait enabled (flag={self.classic_walk_flag})')

    def send_move(self, vx, vy, vyaw):
        # SportClient::Move(vx,vy,vyaw) : api_id=1008, {"x":vx,"y":vy,"z":vyaw}
        self.pub.publish(self._make_request(
            ROBOT_SPORT_API_ID_MOVE,
            json.dumps({'x': float(vx), 'y': float(vy), 'z': float(vyaw)})))

    # ------------------------------------------------------------------

    def do_init_sequence(self):
        elapsed = (self.get_clock().now() - self.init_start_time).nanoseconds * 1e-9
        if elapsed < self.init_wait_sec:
            return
        if not self.speed_level_sent:
            self.send_speed_level()
            self.speed_level_sent = True
            return
        if not self.classic_walk_sent:
            self.send_classic_walk()
            self.classic_walk_sent = True
            return
        self.get_logger().info('[INIT] Ready. Accepting /cmd_vel commands now.')
        self.is_ready = True
        self.init_timer.cancel()

    # ------------------------------------------------------------------

    def cmd_callback(self, msg):
        vx = self.apply_deadband(clamp(msg.linear.x, self.max_vx))
        vy = self.apply_deadband(clamp(msg.linear.y, self.max_vy))
        vyaw = self.apply_deadband(clamp(msg.angular.z, self.max_vyaw))
        self.target_vx, self.target_vy, self.target_vyaw = \
            self.shape_velocity(vx, vy, vyaw)
        self.last_cmd_time = self.get_clock().now()

    def send_loop(self):
        if not self.is_ready or self.last_cmd_time is None:
            return

        dt = (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9

        # --- 타임아웃: 안전정지(StopMove) 한 번, 내부 상태 0으로 리셋 ---
        if dt > self.cmd_timeout:
            if not self.timed_out:
                self.get_logger().warn('cmd_vel timeout. Send StopMove.')
                self.send_stopmove()
                self.timed_out = True
            self.cur_vx = self.cur_vy = self.cur_vyaw = 0.0
            self.target_vx = self.target_vy = self.target_vyaw = 0.0
            return
        self.timed_out = False

        # --- 슬루 제한으로 목표를 부드럽게 추종 (급변 제거) ---
        self.cur_vx = self._slew(self.cur_vx, self.target_vx, self.max_lin_accel)
        self.cur_vy = self._slew(self.cur_vy, self.target_vy, self.max_lin_accel)
        self.cur_vyaw = self._slew(self.cur_vyaw, self.target_vyaw, self.max_yaw_accel)

        # --- 단일 채널: 항상 Move 로 송신 (정지도 Move(0,0,0) 홀드) ---
        # StopMove 와 Move 를 섞지 않아 명령 충돌/끊김이 사라짐.
        near_zero = (abs(self.cur_vx) < 1e-3 and
                     abs(self.cur_vy) < 1e-3 and
                     abs(self.cur_vyaw) < 1e-3)

        if near_zero and not self.hold_with_move:
            # 옵션: 홀드 대신 명시적 StopMove
            self.send_stopmove()
            return

        self.send_move(self.cur_vx, self.cur_vy, self.cur_vyaw)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelToSport()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.send_stopmove()
            time.sleep(0.1)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
