#!/usr/bin/env python3
"""
cmd_vel_to_sport_demo2.py  --  Go2 ClassicWalk(AI sport) cmd_vel 브리지 (classic 유지판)

이 파일은 cmd_vel_to_sport_demo.py 를 기반으로 한 "ClassicWalk 지속 유지" 버전이다.
기존 demo 와의 차이는 단 하나, **ClassicWalk gait 가 비활성화될 때까지 계속 유지된다**는 점이다.

왜 만들었나 (기존 demo 의 문제):
  기존 cmd_vel_to_sport_demo.py 는 ClassicWalk(2049) 를 부팅 시 *딱 한 번* 켠다.
  그런데 첫 번째 방향 명령과 두 번째 방향 명령 사이에 /cmd_vel 이 잠깐 끊기면
  watchdog 이 StopMove(1003) 를 보낸다. Go2 의 AI sport 에서는 StopMove 를 받으면
  보행 gait 가 기본값(FreeWalk) 으로 되돌아간다.
    -> 그래서 첫 방향은 ClassicWalk 로 걷지만, 두 번째 방향부터는 FreeWalk 로 걷는
       (사용자가 보고한 정확히 그 증상) 일이 발생한다.

이 버전의 해결책:
  "정지(StopMove/timeout) 상태에서 다시 이동을 시작하는 매 순간 ClassicWalk 를
   재전송(re-assert)" 한다. 즉 gait 가 FreeWalk 로 풀렸을 가능성이 있는 시점마다
  ClassicWalk 를 다시 켜서, deactivate 전까지 classic 을 강제로 유지한다.
  추가로(옵션) 이동 중에도 일정 주기로 ClassicWalk 를 재전송하는 keepalive 도 둔다.

Request 규약 (unitree_ros2 공식 예제 ros2_sport_client.cpp 기준):
  - /api/motion_switcher/request
      SelectMode  : api_id=1002, parameter={"name": "<mode>"}   (기본 "ai")
      ReleaseMode : api_id=1003
  - /api/sport/request
      BalanceStand: api_id=1002, parameter 없음
      StopMove    : api_id=1003, parameter 없음
      Move        : api_id=1008, parameter={"x":vx,"y":vy,"z":vyaw}
      SpeedLevel  : api_id=1015, parameter={"data": level}
      ClassicWalk : api_id=2049, parameter={"data": flag}

부팅 시퀀스:
  SelectMode("ai") -> (대기) -> BalanceStand -> SpeedLevel
    -> ClassicWalk(True) -> /cmd_vel 수락
  이후: 정지에서 재출발할 때마다 ClassicWalk(True) 재전송 (classic 유지)

추가 기능 (기존 demo 와 동일):
  - rotate-to-face : Nav2 의 (vx,vy) 를 진행방향으로 해석해, 옆/뒤 목표는
    게걸음/후진 대신 먼저 제자리 회전 (히스테리시스 + smoothstep 으로 채터링 제거).
  - slew limit     : 가/감속을 제한해 급격한 명령 변화를 부드럽게.
  - watchdog       : cmd_timeout 동안 /cmd_vel 이 없으면 StopMove 로 안전정지.

사용법:
  source ~/capstone/install/setup.bash
  python3 cmd_vel_to_sport_demo2.py
  # 파라미터 예:
  python3 cmd_vel_to_sport_demo2.py --ros-args -p speed_level:=fast -p select_ai_mode:=true
  # classic keepalive 주기(초) 켜기 (0 이면 끔, 기본 0):
  python3 cmd_vel_to_sport_demo2.py --ros-args -p classic_keepalive_sec:=1.0
"""
import json
import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from unitree_api.msg import Request

# --- sport service (ros2_sport_client.h 의 ROBOT_SPORT_API_ID_*) ---
ROBOT_SPORT_API_ID_BALANCESTAND = 1002
ROBOT_SPORT_API_ID_STOPMOVE = 1003
ROBOT_SPORT_API_ID_MOVE = 1008
ROBOT_SPORT_API_ID_SPEEDLEVEL = 1015
ROBOT_SPORT_API_ID_CLASSICWALK = 2049   # AI sport 전용 gait

# --- motion_switcher service ---
MOTION_SWITCHER_API_ID_CHECK_MODE = 1001
MOTION_SWITCHER_API_ID_SELECT_MODE = 1002
MOTION_SWITCHER_API_ID_RELEASE_MODE = 1003

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


class CmdVelToSportDemo2(Node):
    def __init__(self):
        super().__init__('cmd_vel_to_sport_demo2')

        # ===== Parameters =====
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('sport_request_topic', '/api/sport/request')
        self.declare_parameter('motion_switcher_topic', '/api/motion_switcher/request')
        self.declare_parameter('max_vx', 0.6)
        self.declare_parameter('max_vy', 0.3)
        self.declare_parameter('max_vyaw', 1.5)
        self.declare_parameter('cmd_timeout', 0.5)
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('zero_deadband', 0.02)
        self.declare_parameter('init_wait_sec', 2.0)
        self.declare_parameter('speed_level', 'normal')   # slow / normal / fast

        # ----- 모드/보행 -----
        self.declare_parameter('select_ai_mode', True)     # 부팅 시 AI 모드로 전환
        self.declare_parameter('motion_mode', 'ai')        # SelectMode 에 보낼 이름
        self.declare_parameter('init_balance_stand', True)
        self.declare_parameter('set_classic_walk', True)   # ClassicWalk 전송 여부
        self.declare_parameter('classic_walk_flag', True)  # ClassicWalk flag (True=enable)
        self.declare_parameter('mode_switch_wait_sec', 1.5)  # SelectMode 후 안정화 대기

        # ----- ClassicWalk 유지(이 버전의 핵심) -----
        # 정지->재출발 시 ClassicWalk 를 다시 켜서 FreeWalk 로 풀리는 것을 막는다.
        self.declare_parameter('reassert_classic_on_resume', True)
        # 재전송 후 Move 까지 살짝 대기(초). gait 전환이 반영될 시간.
        self.declare_parameter('classic_reassert_settle_sec', 0.1)
        # 이동 중에도 주기적으로 ClassicWalk 재전송(초). 0 이면 끔(기본 끔).
        self.declare_parameter('classic_keepalive_sec', 0.0)

        # ----- 회전-후-전진(rotate-to-face) -----
        self.declare_parameter('face_goal_direction', True)
        self.declare_parameter('align_yaw_gain', 1.5)
        self.declare_parameter('rotate_in_threshold', 0.6)
        self.declare_parameter('rotate_out_threshold', 0.35)
        self.declare_parameter('min_translation_speed', 0.05)

        # ----- 부드러운 동작(슬루/정지) -----
        self.declare_parameter('max_lin_accel', 1.0)
        self.declare_parameter('max_yaw_accel', 3.0)
        self.declare_parameter('hold_with_move', True)

        g = self.get_parameter
        self.cmd_vel_topic = g('cmd_vel_topic').value
        self.sport_request_topic = g('sport_request_topic').value
        self.motion_switcher_topic = g('motion_switcher_topic').value
        self.max_vx = float(g('max_vx').value)
        self.max_vy = float(g('max_vy').value)
        self.max_vyaw = float(g('max_vyaw').value)
        self.cmd_timeout = float(g('cmd_timeout').value)
        self.publish_rate = float(g('publish_rate').value)
        self.zero_deadband = float(g('zero_deadband').value)
        self.init_wait_sec = float(g('init_wait_sec').value)

        speed_level_str = g('speed_level').value.lower()
        if speed_level_str not in SPEED_LEVEL_MAP:
            self.get_logger().warn(
                f"Unknown speed_level '{speed_level_str}', defaulting to 'normal'")
            speed_level_str = 'normal'
        self.speed_level = SPEED_LEVEL_MAP[speed_level_str]
        self.speed_level_str = speed_level_str

        self.select_ai_mode = bool(g('select_ai_mode').value)
        self.motion_mode = str(g('motion_mode').value)
        self.init_balance_stand = bool(g('init_balance_stand').value)
        self.set_classic_walk = bool(g('set_classic_walk').value)
        self.classic_walk_flag = bool(g('classic_walk_flag').value)
        self.mode_switch_wait_sec = float(g('mode_switch_wait_sec').value)

        self.reassert_classic_on_resume = bool(g('reassert_classic_on_resume').value)
        self.classic_reassert_settle_sec = float(g('classic_reassert_settle_sec').value)
        self.classic_keepalive_sec = float(g('classic_keepalive_sec').value)

        self.face_goal_direction = bool(g('face_goal_direction').value)
        self.align_yaw_gain = float(g('align_yaw_gain').value)
        self.rotate_in_threshold = float(g('rotate_in_threshold').value)
        self.rotate_out_threshold = float(g('rotate_out_threshold').value)
        self.min_translation_speed = float(g('min_translation_speed').value)

        self.max_lin_accel = float(g('max_lin_accel').value)
        self.max_yaw_accel = float(g('max_yaw_accel').value)
        self.hold_with_move = bool(g('hold_with_move').value)

        # ===== State =====
        self.target_vx = self.target_vy = self.target_vyaw = 0.0
        self.cur_vx = self.cur_vy = self.cur_vyaw = 0.0
        self.last_cmd_time = None
        self.is_ready = False
        self.timed_out = False
        self.forward_gate_open = False

        # ClassicWalk 유지 상태:
        #   classic_active = 현재 ClassicWalk gait 가 켜져 있다고 믿는 상태.
        #   StopMove 를 보내면 gait 가 FreeWalk 로 풀린다고 가정 -> False 로 내림.
        #   재출발 직전 ClassicWalk 재전송 -> True 로 올림.
        self.classic_active = False
        self.last_classic_send_time = None

        # init step flags
        self.mode_selected = False
        self.balance_sent = False
        self.speed_level_sent = False
        self.classic_walk_sent = False
        self.mode_select_time = None

        # ===== Pub/Sub =====
        self.pub = self.create_publisher(Request, self.sport_request_topic, 10)
        self.switcher_pub = self.create_publisher(
            Request, self.motion_switcher_topic, 10)
        self.sub = self.create_subscription(
            Twist, self.cmd_vel_topic, self.cmd_callback, 10)

        # ===== Timer =====
        self.dt = 1.0 / self.publish_rate
        self.send_timer = self.create_timer(self.dt, self.send_loop)

        # ===== Init =====
        self.init_start_time = self.get_clock().now()
        self.init_timer = self.create_timer(0.1, self.do_init_sequence)

        L = self.get_logger()
        L.info('=' * 60)
        L.info('   cmd_vel_to_sport_demo2 (ClassicWalk@AI, 유지판)  --  PARAMETER SUMMARY')
        L.info('=' * 60)
        L.info(f'  subscribe={self.cmd_vel_topic}  sport_pub={self.sport_request_topic}')
        L.info(f'  switcher_pub={self.motion_switcher_topic}')
        L.info(f'  rate={self.publish_rate}Hz  cmd_timeout={self.cmd_timeout}s')
        L.info(f'  max v: vx={self.max_vx} vy={self.max_vy} vyaw={self.max_vyaw}')
        L.info(f'  init: select_ai={self.select_ai_mode}(mode="{self.motion_mode}") '
               f'balance={self.init_balance_stand} '
               f'speed={self.speed_level_str}({self.speed_level})')
        L.info(f'        classic_walk={self.set_classic_walk}(flag={self.classic_walk_flag}) '
               f'switch_wait={self.mode_switch_wait_sec}s init_wait={self.init_wait_sec}s')
        L.info(f'  classic 유지: reassert_on_resume={self.reassert_classic_on_resume} '
               f'settle={self.classic_reassert_settle_sec}s '
               f'keepalive={self.classic_keepalive_sec}s')
        L.info(f'  face_goal={self.face_goal_direction} '
               f'in={self.rotate_in_threshold} out={self.rotate_out_threshold} '
               f'yaw_gain={self.align_yaw_gain}')
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
            self.cmd_vel_topic, self.sport_request_topic, self.motion_switcher_topic,
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
        if not self.face_goal_direction:
            return vx, clamp(vy, self.max_vy), clamp(vyaw, self.max_vyaw)

        speed = math.hypot(vx, vy)
        if speed < self.min_translation_speed:
            self.forward_gate_open = False
            return 0.0, 0.0, clamp(vyaw, self.max_vyaw)

        heading_err = math.atan2(vy, vx)
        abs_err = abs(heading_err)

        if self.forward_gate_open:
            if abs_err > self.rotate_in_threshold:
                self.forward_gate_open = False
        else:
            if abs_err < self.rotate_out_threshold:
                self.forward_gate_open = True

        gate = 1.0 - smoothstep(self.rotate_out_threshold,
                                self.rotate_in_threshold, abs_err)
        if not self.forward_gate_open and abs_err > self.rotate_in_threshold:
            gate = 0.0

        alignment = max(0.0, math.cos(heading_err))
        out_vyaw = clamp(self.align_yaw_gain * heading_err + alignment * vyaw,
                         self.max_vyaw)
        out_vx = clamp(max(0.0, speed * math.cos(heading_err)) * gate, self.max_vx)
        return out_vx, 0.0, out_vyaw

    def _slew(self, cur, target, max_accel):
        max_delta = max_accel * self.dt
        if target > cur + max_delta:
            return cur + max_delta
        if target < cur - max_delta:
            return cur - max_delta
        return target

    # ------------------------------------------------------------------
    # Request 빌더 (unitree_ros2 SportClient 방식: api_id + parameter)
    # ------------------------------------------------------------------

    def _make_request(self, api_id, parameter=''):
        req = Request()
        req.header.identity.id = int(time.time_ns() & 0x7FFFFFFF)
        req.header.identity.api_id = api_id
        req.parameter = parameter
        return req

    # --- motion_switcher service ---
    def send_select_mode(self):
        # MotionSwitcher::SelectMode(name) : api_id=1002, {"name": name}
        self.switcher_pub.publish(self._make_request(
            MOTION_SWITCHER_API_ID_SELECT_MODE,
            json.dumps({'name': self.motion_mode})))
        self.get_logger().info(
            f'[INIT] motion_switcher SelectMode -> "{self.motion_mode}"')

    # --- sport service ---
    def send_balance_stand(self):
        self.pub.publish(self._make_request(ROBOT_SPORT_API_ID_BALANCESTAND))

    def send_stopmove(self):
        self.pub.publish(self._make_request(ROBOT_SPORT_API_ID_STOPMOVE))
        # StopMove 후에는 gait 가 FreeWalk 로 풀린 것으로 간주한다.
        self.classic_active = False

    def send_speed_level(self):
        self.pub.publish(self._make_request(
            ROBOT_SPORT_API_ID_SPEEDLEVEL, json.dumps({'data': self.speed_level})))
        self.get_logger().info(
            f'[INIT] SpeedLevel set to {self.speed_level_str} ({self.speed_level})')

    def send_classic_walk(self, log=True):
        if not self.set_classic_walk:
            return
        self.pub.publish(self._make_request(
            ROBOT_SPORT_API_ID_CLASSICWALK,
            json.dumps({'data': self.classic_walk_flag})))
        self.classic_active = bool(self.classic_walk_flag)
        self.last_classic_send_time = self.get_clock().now()
        if log:
            self.get_logger().info(
                f'[CLASSIC] ClassicWalk gait re-asserted (flag={self.classic_walk_flag})')

    def send_move(self, vx, vy, vyaw):
        self.pub.publish(self._make_request(
            ROBOT_SPORT_API_ID_MOVE,
            json.dumps({'x': float(vx), 'y': float(vy), 'z': float(vyaw)})))

    # ------------------------------------------------------------------

    def do_init_sequence(self):
        now = self.get_clock().now()
        elapsed = (now - self.init_start_time).nanoseconds * 1e-9
        if elapsed < self.init_wait_sec:
            return

        # 1) AI 모드 선택 (ClassicWalk 가 동작하려면 필수)
        if self.select_ai_mode and not self.mode_selected:
            self.send_select_mode()
            self.mode_selected = True
            self.mode_select_time = now
            return
        # 모드 전환 후 안정화 대기
        if self.select_ai_mode and self.mode_select_time is not None:
            since = (now - self.mode_select_time).nanoseconds * 1e-9
            if since < self.mode_switch_wait_sec:
                return

        # 2) BalanceStand
        if self.init_balance_stand and not self.balance_sent:
            self.send_balance_stand()
            self.balance_sent = True
            return
        # 3) SpeedLevel
        if not self.speed_level_sent:
            self.send_speed_level()
            self.speed_level_sent = True
            return
        # 4) ClassicWalk
        if not self.classic_walk_sent:
            if self.set_classic_walk:
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

    def _maybe_keepalive_classic(self):
        """이동 중에도 주기적으로 ClassicWalk 를 재전송(옵션)."""
        if self.classic_keepalive_sec <= 0.0 or not self.set_classic_walk:
            return
        if self.last_classic_send_time is None:
            self.send_classic_walk(log=False)
            return
        since = (self.get_clock().now()
                 - self.last_classic_send_time).nanoseconds * 1e-9
        if since >= self.classic_keepalive_sec:
            self.send_classic_walk(log=False)

    def send_loop(self):
        if not self.is_ready or self.last_cmd_time is None:
            return

        dt = (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9

        if dt > self.cmd_timeout:
            if not self.timed_out:
                self.get_logger().warn('cmd_vel timeout. Send StopMove.')
                self.send_stopmove()   # 내부에서 classic_active=False
                self.timed_out = True
            self.cur_vx = self.cur_vy = self.cur_vyaw = 0.0
            self.target_vx = self.target_vy = self.target_vyaw = 0.0
            return
        self.timed_out = False

        self.cur_vx = self._slew(self.cur_vx, self.target_vx, self.max_lin_accel)
        self.cur_vy = self._slew(self.cur_vy, self.target_vy, self.max_lin_accel)
        self.cur_vyaw = self._slew(self.cur_vyaw, self.target_vyaw, self.max_yaw_accel)

        near_zero = (abs(self.cur_vx) < 1e-3 and
                     abs(self.cur_vy) < 1e-3 and
                     abs(self.cur_vyaw) < 1e-3)

        if near_zero and not self.hold_with_move:
            self.send_stopmove()   # 내부에서 classic_active=False
            return

        # === 이 버전의 핵심 ===
        # 정지(StopMove/timeout) 로 gait 가 FreeWalk 로 풀린 뒤 다시 움직이려는
        # 순간이면, Move 를 보내기 전에 ClassicWalk 를 다시 켠다.
        # 이렇게 해야 deactivate 전까지 classic 보행이 유지된다.
        if (self.reassert_classic_on_resume and self.set_classic_walk
                and not self.classic_active):
            self.send_classic_walk()
            # gait 전환이 반영될 짧은 시간 동안은 Move 를 보류한다.
            if self.classic_reassert_settle_sec > 0.0:
                return

        # 이동 중 주기적 keepalive (옵션)
        self._maybe_keepalive_classic()

        self.send_move(self.cur_vx, self.cur_vy, self.cur_vyaw)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelToSportDemo2()
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
