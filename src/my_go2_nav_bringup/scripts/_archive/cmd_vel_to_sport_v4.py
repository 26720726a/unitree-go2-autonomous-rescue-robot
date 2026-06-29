#!/usr/bin/env python3
import json
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from unitree_api.msg import Request, Response

SPORT_API_BALANCESTAND = 1002
SPORT_API_STOPMOVE     = 1003
SPORT_API_MOVE         = 1008
SPORT_API_SPEEDLEVEL   = 1015   # 1006 = RecoveryStand (오류였음)
SPORT_API_CLASSICWALK  = 2049   # ClassicWalk 보행 (parameter {"data":true} 필수)

SPEED_LEVEL_NORMAL = 0          # -1=slow  0=normal  1=fast


class CmdVelToSport(Node):
    def __init__(self):
        super().__init__('cmd_vel_to_sport')

        # ===== Parameters =====
        self.declare_parameter('cmd_vel_topic',        '/cmd_vel')
        self.declare_parameter('sport_request_topic',  '/api/sport/request')
        self.declare_parameter('sport_response_topic', '/api/sport/response')
        self.declare_parameter('max_vx',        0.42)
        self.declare_parameter('max_vy',        0.3)
        self.declare_parameter('max_vyaw',      1.5)
        self.declare_parameter('cmd_timeout',   0.5)
        self.declare_parameter('publish_rate',  20.0)
        self.declare_parameter('zero_deadband', 0.02)
        self.declare_parameter('init_wait_sec', 2.0)

        g = self.get_parameter
        self.cmd_vel_topic        = g('cmd_vel_topic').value
        self.sport_request_topic  = g('sport_request_topic').value
        self.sport_response_topic = g('sport_response_topic').value
        self.max_vx        = float(g('max_vx').value)
        self.max_vy        = float(g('max_vy').value)
        self.max_vyaw      = float(g('max_vyaw').value)
        self.cmd_timeout   = float(g('cmd_timeout').value)
        self.publish_rate  = float(g('publish_rate').value)
        self.zero_deadband = float(g('zero_deadband').value)
        self.init_wait_sec = float(g('init_wait_sec').value)

        # ===== State =====
        self.last_vx   = 0.0
        self.last_vy   = 0.0
        self.last_vyaw = 0.0
        self.last_cmd_time = None
        self.is_stopped    = True
        self.is_ready      = False
        self.timed_out     = False

        self.init_start_time  = self.get_clock().now()
        self.balance_sent     = False
        self.speed_level_sent = False
        self.classic_walk_sent = False

        # ===== Pub/Sub =====
        self.pub = self.create_publisher(Request, self.sport_request_topic, 10)
        self.sub = self.create_subscription(
            Twist, self.cmd_vel_topic, self.cmd_callback, 10)
        self.resp_sub = self.create_subscription(
            Response, self.sport_response_topic, self._resp_callback, 10)

        # ===== Timers =====
        self.dt = 1.0 / self.publish_rate
        self.send_timer = self.create_timer(self.dt, self.send_loop)
        self.init_timer = self.create_timer(0.1, self.do_init_sequence)

        self._topic_log_fired = False
        self.create_timer(1.0, self._log_active_topics)

        L = self.get_logger()
        L.info('=' * 60)
        L.info('  cmd_vel_to_sport  [passthrough mode]')
        L.info(f'  subscribe : {self.cmd_vel_topic}')
        L.info(f'  publish   : {self.sport_request_topic} @ {self.publish_rate} Hz')
        L.info(f'  limits    : vx={self.max_vx}  vy={self.max_vy}  vyaw={self.max_vyaw}')
        L.info(f'  init_wait : {self.init_wait_sec}s')
        L.info('  init seq  : BalanceStand(1002) -> SpeedLevel(1015/normal) -> ClassicWalk(2049)')
        L.info('=' * 60)

    # ------------------------------------------------------------------

    def _resp_callback(self, msg):
        """init 단계 API 응답 수신 로그 (noreply=False 로 보낸 명령들)."""
        api_id = msg.header.identity.api_id
        code   = msg.header.status.code
        self.get_logger().info(f'[RESP] api_id={api_id}  code={code}')

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

    def clamp(self, value, limit):
        if limit <= 0.0:
            return 0.0
        return max(-limit, min(limit, value))

    def apply_deadband(self, value):
        return 0.0 if abs(value) < self.zero_deadband else value

    # ------------------------------------------------------------------
    # API helpers
    # init 명령: noreply=False → 응답 코드 로그로 확인
    # Move 루프: noreply=True  → fire-and-forget (20 Hz)

    def _make_request(self, api_id, parameter='', noreply=True):
        req = Request()
        req.header.identity.id     = int(time.time_ns() & 0x7FFFFFFF)
        req.header.identity.api_id = api_id
        req.header.policy.priority = 0
        req.header.policy.noreply  = noreply
        req.parameter = parameter
        return req

    def send_balance_stand(self):
        self.pub.publish(self._make_request(SPORT_API_BALANCESTAND, noreply=False))
        self.get_logger().info('[INIT] Sending BalanceStand (1002)...')

    def send_speed_level(self):
        self.pub.publish(self._make_request(
            SPORT_API_SPEEDLEVEL,
            json.dumps({'data': SPEED_LEVEL_NORMAL}),
            noreply=False))
        self.get_logger().info(
            f'[INIT] Sending SpeedLevel (1015) = {SPEED_LEVEL_NORMAL} (normal)')

    def send_classic_walk(self):
        self.pub.publish(self._make_request(
            SPORT_API_CLASSICWALK,
            json.dumps({'data': True}),
            noreply=False))
        self.get_logger().info('[INIT] Sending ClassicWalk (2049)...')

    def send_stopmove(self):
        self.pub.publish(self._make_request(SPORT_API_STOPMOVE))

    def send_move(self, vx, vy, vyaw):
        self.pub.publish(self._make_request(
            SPORT_API_MOVE,
            json.dumps({'x': float(vx), 'y': float(vy), 'z': float(vyaw)})))

    # ------------------------------------------------------------------

    def do_init_sequence(self):
        """
        BalanceStand(1002) → init_wait_sec → SpeedLevel(1015) → ClassicWalk(2049) → ready
        """
        if not self.balance_sent:
            self.send_balance_stand()
            self.balance_sent    = True
            self.init_start_time = self.get_clock().now()
            return

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
        if not self.is_ready:
            return
        vx   = self.apply_deadband(self.clamp(msg.linear.x,  self.max_vx))
        vy   = self.apply_deadband(self.clamp(msg.linear.y,  self.max_vy))
        vyaw = self.apply_deadband(self.clamp(msg.angular.z, self.max_vyaw))
        self.last_vx       = vx
        self.last_vy       = vy
        self.last_vyaw     = vyaw
        self.last_cmd_time = self.get_clock().now()

    def send_loop(self):
        if not self.is_ready or self.last_cmd_time is None:
            return

        dt = (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9

        if dt > self.cmd_timeout:
            if not self.timed_out:
                self.get_logger().warn('cmd_vel timeout → StopMove')
                self.send_stopmove()
                self.timed_out  = True
                self.is_stopped = True
            return
        self.timed_out = False

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
