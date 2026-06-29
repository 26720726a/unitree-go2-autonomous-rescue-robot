#!/usr/bin/env python3
import json
import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from unitree_api.msg import Request

SPORT_API_BALANCESTAND = 1002
SPORT_API_STOPMOVE = 1003
SPORT_API_MOVE = 1008
SPORT_API_SPEEDLEVEL = 1015   # FIXED: 기존 1006 은 RecoveryStand 였음
SPORT_API_STATICWALK = 1061   # Normal(잔발 많이 짚는 안정 보행). 기존 ClassicWalk(2049) 대체

# -1=slow, 0=normal, 1=fast  (펌웨어에서 실제 허용 범위는 실기로 한 번 확인 권장)
SPEED_LEVEL_MAP = {'slow': -1, 'normal': 0, 'fast': 1}


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
        self.declare_parameter('set_static_walk', True)    # StaticWalk(Normal gait) on/off

        # ----- 회전-후-전진(rotate-to-face) 관련 파라미터 -----
        # True 면: 목표 방향을 향해 먼저 회전한 뒤 전진. (뒤쪽 목표일 때 후진하지 않음)
        self.declare_parameter('face_goal_direction', True)
        self.declare_parameter('align_yaw_gain', 1.5)            # heading 오차 -> 회전속도 비례게인
        self.declare_parameter('rotate_in_place_threshold', 0.5)  # rad. 이 이상 어긋나면 제자리 회전만
        self.declare_parameter('min_translation_speed', 0.05)     # m/s. 이 이하면 순수 회전 명령으로 간주(통과)

        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.sport_request_topic = self.get_parameter('sport_request_topic').value
        self.max_vx = float(self.get_parameter('max_vx').value)
        self.max_vy = float(self.get_parameter('max_vy').value)
        self.max_vyaw = float(self.get_parameter('max_vyaw').value)
        self.cmd_timeout = float(self.get_parameter('cmd_timeout').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.zero_deadband = float(self.get_parameter('zero_deadband').value)
        self.init_balance_stand = bool(self.get_parameter('init_balance_stand').value)
        self.init_wait_sec = float(self.get_parameter('init_wait_sec').value)

        speed_level_str = self.get_parameter('speed_level').value.lower()
        if speed_level_str not in SPEED_LEVEL_MAP:
            self.get_logger().warn(
                f"Unknown speed_level '{speed_level_str}', defaulting to 'normal'"
            )
            speed_level_str = 'normal'
        self.speed_level = SPEED_LEVEL_MAP[speed_level_str]
        self.speed_level_str = speed_level_str
        self.set_static_walk = bool(self.get_parameter('set_static_walk').value)

        self.face_goal_direction = bool(self.get_parameter('face_goal_direction').value)
        self.align_yaw_gain = float(self.get_parameter('align_yaw_gain').value)
        self.rotate_in_place_threshold = float(
            self.get_parameter('rotate_in_place_threshold').value
        )
        self.min_translation_speed = float(
            self.get_parameter('min_translation_speed').value
        )

        # ===== State =====
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_vyaw = 0.0
        self.last_cmd_time = None
        self.is_stopped = True
        self.is_ready = False
        self.init_start_time = self.get_clock().now()
        self.balance_sent = False
        self.speed_level_sent = False
        self.static_walk_sent = False

        # ===== Pub/Sub =====
        self.pub = self.create_publisher(Request, self.sport_request_topic, 10)
        self.sub = self.create_subscription(
            Twist, self.cmd_vel_topic, self.cmd_callback, 10
        )

        # ===== Timer: continuously resend last command =====
        period = 1.0 / self.publish_rate
        self.send_timer = self.create_timer(period, self.send_loop)

        # ===== Init: BalanceStand 즉시 전송 후 나머지 단계는 타이머로 =====
        if self.init_balance_stand:
            self.get_logger().info('[INIT] Sending BalanceStand immediately...')
            self.send_balance_stand()
        self.balance_sent = True
        self.init_start_time = self.get_clock().now()
        self.init_timer = self.create_timer(0.1, self.do_init_sequence)

        L = self.get_logger()
        L.info('=' * 60)
        L.info('   cmd_vel_to_sport  --  PARAMETER SUMMARY')
        L.info('=' * 60)
        L.info(f'  [Topics]')
        L.info(f'    subscribe : {self.cmd_vel_topic}')
        L.info(f'    publish   : {self.sport_request_topic}')
        L.info(f'  [Rate / Timeout]')
        L.info(f'    publish_rate  : {self.publish_rate} Hz')
        L.info(f'    cmd_timeout   : {self.cmd_timeout} s')
        L.info(f'  [Velocity Limits]')
        L.info(f'    max_vx   : {self.max_vx} m/s')
        L.info(f'    max_vy   : {self.max_vy} m/s')
        L.info(f'    max_vyaw : {self.max_vyaw} rad/s')
        L.info(f'    zero_deadband : {self.zero_deadband}')
        L.info(f'  [Init Sequence]')
        L.info(f'    balance_stand  : {self.init_balance_stand}  (sent immediately)')
        L.info(f'    init_wait_sec  : {self.init_wait_sec} s')
        L.info(f'    speed_level    : {self.speed_level_str} ({self.speed_level})')
        L.info(f'    set_static_walk: {self.set_static_walk}')
        L.info(f'  [Rotate-to-Face]')
        L.info(f'    face_goal_direction      : {self.face_goal_direction}')
        L.info(f'    align_yaw_gain           : {self.align_yaw_gain}')
        L.info(f'    rotate_in_place_threshold: {self.rotate_in_place_threshold} rad')
        L.info(f'    min_translation_speed    : {self.min_translation_speed} m/s')
        L.info('=' * 60)

        # 1초 후 ROS2 활성 토픽 조회 (DDS 연결 대기)
        self._topic_log_fired = False
        self.create_timer(1.0, self._log_active_topics)

    # ------------------------------------------------------------------

    def _log_active_topics(self):
        if self._topic_log_fired:
            return
        self._topic_log_fired = True

        all_topics = dict(self.get_topic_names_and_types())
        watch = [
            self.cmd_vel_topic,
            self.sport_request_topic,
            '/utlidar/cloud_deskewed',
            '/utlidar/cloud_deskewed_restamped',
            '/utlidar/robot_odom',
            '/scan',
            '/odom',
            '/map',
            '/tf',
            '/tf_static',
        ]

        L = self.get_logger()
        L.info('-' * 60)
        L.info('  [Active ROS2 Topics  (1s after start)]')
        for t in watch:
            if t in all_topics:
                types = ', '.join(all_topics[t])
                L.info(f'    [O] {t}  ({types})')
            else:
                L.info(f'    [X] {t}  -- NOT found')
        L.info('-' * 60)

    # ------------------------------------------------------------------

    def clamp(self, value, limit):
        if limit <= 0.0:
            return 0.0
        return max(-limit, min(limit, value))

    def apply_deadband(self, value):
        return 0.0 if abs(value) < self.zero_deadband else value

    # ------------------------------------------------------------------

    def shape_velocity(self, vx, vy, vyaw):
        """
        Nav2 가 내보낸 (vx, vy) 속도 벡터를 '로봇이 향해야 할 진행 방향'으로 해석해서
        유니사이클(전진+회전)처럼 동작하도록 변환한다.

        - 목표가 뒤/옆에 있으면 (heading 오차 큼) 후진/게걸음 대신 먼저 제자리 회전.
        - 정렬될수록 cos 램프로 전진 속도를 키운다 (항상 +vx 방향).
        - 거의 제자리 회전 명령(목표 도착 후 최종 방향 정렬 등)은 플래너 yaw 그대로 통과.
        """
        if not self.face_goal_direction:
            return vx, vy, vyaw

        speed = math.hypot(vx, vy)

        # 병진 성분이 거의 없으면 -> 순수 회전 명령으로 보고 통과
        if speed < self.min_translation_speed:
            return 0.0, 0.0, self.clamp(vyaw, self.max_vyaw)

        # 로봇 기준, 가고자 하는 방향의 각도 (이미 최단 부호각 [-pi, pi])
        heading_err = math.atan2(vy, vx)

        # 정렬도: 정면이면 1, 90도면 0, 뒤쪽이면 음수
        alignment = max(0.0, math.cos(heading_err))

        # 회전: heading 정렬 + (정렬됐을 때만) 플래너 곡률 yaw 피드포워드
        out_vyaw = self.clamp(
            self.align_yaw_gain * heading_err + alignment * vyaw,
            self.max_vyaw,
        )

        if abs(heading_err) > self.rotate_in_place_threshold:
            # 많이 어긋남(특히 뒤쪽) -> 제자리 회전부터
            out_vx = 0.0
        else:
            # 정렬될수록 전진 (cos 램프), 항상 전진(+) 방향
            out_vx = self.clamp(speed * math.cos(heading_err), self.max_vx)

        out_vy = 0.0  # 측면 이동은 회전+전진으로 대체
        return out_vx, out_vy, out_vyaw

    # ------------------------------------------------------------------

    def _make_request(self, api_id, parameter=''):
        req = Request()
        req.header.identity.id = int(time.time_ns() & 0x7FFFFFFF)
        req.header.identity.api_id = api_id
        req.header.policy.priority = 0
        req.header.policy.noreply = True
        req.parameter = parameter
        return req

    def send_balance_stand(self):
        self.pub.publish(self._make_request(SPORT_API_BALANCESTAND))

    def send_stopmove(self):
        self.pub.publish(self._make_request(SPORT_API_STOPMOVE))

    def send_speed_level(self):
        parameter = json.dumps({'data': self.speed_level})
        self.pub.publish(self._make_request(SPORT_API_SPEEDLEVEL, parameter))
        self.get_logger().info(
            f'[INIT] SpeedLevel set to {self.speed_level_str} ({self.speed_level})'
        )

    def send_static_walk(self):
        # StaticWalk(1061) 은 파라미터 없는 일회성 모드 전환 명령
        self.pub.publish(self._make_request(SPORT_API_STATICWALK))
        self.get_logger().info('[INIT] StaticWalk (normal/stable gait) enabled')

    def send_move(self, vx, vy, vyaw):
        parameter = json.dumps({
            'x': float(vx),
            'y': float(vy),
            'z': float(vyaw),
        })
        self.pub.publish(self._make_request(SPORT_API_MOVE, parameter))

    # ------------------------------------------------------------------

    def do_init_sequence(self):
        """
        단계 1: init_wait_sec 대기 (BalanceStand 는 __init__ 에서 즉시 전송 완료)
        단계 2: SpeedLevel 전송
        단계 3: StaticWalk 전송
        단계 4: 준비 완료
        """
        # 단계 1: 대기
        elapsed = (self.get_clock().now() - self.init_start_time).nanoseconds * 1e-9
        if elapsed < self.init_wait_sec:
            return

        # 단계 2: SpeedLevel
        if not self.speed_level_sent:
            self.send_speed_level()
            self.speed_level_sent = True
            return

        # 단계 3: StaticWalk
        if not self.static_walk_sent:
            if self.set_static_walk:
                self.send_static_walk()
            self.static_walk_sent = True
            return

        # 단계 4: 준비 완료
        self.get_logger().info('[INIT] Ready. Accepting /cmd_vel commands now.')
        self.is_ready = True
        self.init_timer.cancel()

    # ------------------------------------------------------------------

    def cmd_callback(self, msg):
        vx = self.apply_deadband(self.clamp(msg.linear.x, self.max_vx))
        vy = self.apply_deadband(self.clamp(msg.linear.y, self.max_vy))
        vyaw = self.apply_deadband(self.clamp(msg.angular.z, self.max_vyaw))

        # 회전-후-전진 변환
        vx, vy, vyaw = self.shape_velocity(vx, vy, vyaw)

        self.last_vx = vx
        self.last_vy = vy
        self.last_vyaw = vyaw
        self.last_cmd_time = self.get_clock().now()

    def send_loop(self):
        if not self.is_ready:
            return
        if self.last_cmd_time is None:
            return

        dt = (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9

        if dt > self.cmd_timeout:
            if not self.is_stopped:
                self.get_logger().warn('cmd_vel timeout. Send StopMove.')
                self.send_stopmove()
                self.is_stopped = True
            return

        if self.last_vx == 0.0 and self.last_vy == 0.0 and self.last_vyaw == 0.0:
            if not self.is_stopped:
                self.send_stopmove()
                self.is_stopped = True
            return

        self.send_move(self.last_vx, self.last_vy, self.last_vyaw)
        self.is_stopped = False


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
