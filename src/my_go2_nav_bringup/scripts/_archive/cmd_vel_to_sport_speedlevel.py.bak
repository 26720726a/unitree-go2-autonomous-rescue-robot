#!/usr/bin/env python3
import json
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from unitree_api.msg import Request

SPORT_API_BALANCESTAND = 1002
SPORT_API_STOPMOVE = 1003
SPORT_API_MOVE = 1008
SPORT_API_SPEEDLEVEL = 1006
SPORT_API_CLASSICWALK = 2049  # ClassicGait enable/disable

# -1=slow, 0=normal, 1=fast
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
        self.declare_parameter('classic_walk', True)       # ClassicGait on/off

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
        self.classic_walk = bool(self.get_parameter('classic_walk').value)

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
        self.classic_walk_sent = False

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
        L.info(f'    balance_stand : {self.init_balance_stand}  (sent immediately)')
        L.info(f'    init_wait_sec : {self.init_wait_sec} s')
        L.info(f'    speed_level   : {self.speed_level_str} ({self.speed_level})')
        L.info(f'    classic_walk  : {self.classic_walk}')
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

    def send_classic_walk(self, enable: bool):
        parameter = json.dumps({'data': enable})
        self.pub.publish(self._make_request(SPORT_API_CLASSICWALK, parameter))
        self.get_logger().info(f'[INIT] ClassicWalk set to {enable}')

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
        단계 1: init_wait_sec 대기 (BalanceStand는 __init__에서 즉시 전송 완료)
        단계 2: SpeedLevel 전송
        단계 3: ClassicWalk 전송
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

        # 단계 3: ClassicWalk
        if not self.classic_walk_sent:
            self.send_classic_walk(self.classic_walk)
            self.classic_walk_sent = True
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
