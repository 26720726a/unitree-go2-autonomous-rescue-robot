# ~/capstone/src/my_go2_controller/my_go2_controller/move_test.py
import rclpy
from rclpy.node import Node
from unitree_api.msg import Request
from unitree_go.msg import SportModeState
from nav_msgs.msg import Odometry
import json
import math

SPORT_API_BALANCESTAND = 1002
SPORT_API_STOPMOVE     = 1003
SPORT_API_MOVE         = 1008


class MoveTest(Node):
    def __init__(self):
        super().__init__('move_test')
        
        self.pub = self.create_publisher(Request, '/api/sport/request', 10)
        
        # 상태 구독 (sport mode)
        self.sub_state = self.create_subscription(
            SportModeState, '/sportmodestate', self.state_cb, 10
        )
        # odometry (utlidar 기반)
        self.sub_odom = self.create_subscription(
            Odometry, '/utlidar/robot_odom', self.odom_cb, 10
        )
        
        # 현재 상태
        self.pos = [0.0, 0.0, 0.0]
        self.vel = [0.0, 0.0]
        self.yaw = 0.0
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.start_odom = None
        
        # 시퀀스 변수
        self.phase = 0
        self.phase_start = self.get_clock().now()
        
        # 50Hz 제어 루프
        self.timer = self.create_timer(0.02, self.tick)
        
        # 파라미터 — 매우 보수적으로 시작
        self.fwd_speed = 0.2      # m/s (천천히)
        self.fwd_duration = 3.0   # 1초 → 약 20cm
        self.yaw_speed = 0.3      # rad/s (천천히)
        self.yaw_duration = 3.0   # 1초 → 약 17도
        
        self.get_logger().info('=' * 60)
        self.get_logger().info('Move Test - VERY SLOW for safety')
        self.get_logger().info(f'  Forward: {self.fwd_speed} m/s for {self.fwd_duration}s')
        self.get_logger().info(f'  Yaw:     {self.yaw_speed} rad/s for {self.yaw_duration}s')
        self.get_logger().info('!! Keep remote control ready (L2+B = Damp) !!')
        self.get_logger().info('=' * 60)

    # ---------- 콜백 ----------
    def state_cb(self, msg):
        self.pos = list(msg.position)
        self.vel = list(msg.velocity)
        self.yaw = msg.imu_state.rpy[2]

    def odom_cb(self, msg):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        if self.start_odom is None:
            self.start_odom = (self.odom_x, self.odom_y)

    # ---------- 명령 헬퍼 ----------
    def send_balance_stand(self):
        req = Request()
        req.header.identity.api_id = SPORT_API_BALANCESTAND
        self.pub.publish(req)

    def send_stop(self):
        req = Request()
        req.header.identity.api_id = SPORT_API_STOPMOVE
        self.pub.publish(req)

    def send_move(self, vx, vy, vyaw):
        req = Request()
        req.header.identity.api_id = SPORT_API_MOVE
        # ⚠️ JSON 키 이름은 cpp 확인 후 확정
        req.parameter = json.dumps({"x": vx, "y": vy, "z": vyaw})
        self.pub.publish(req)

    def elapsed(self):
        return (self.get_clock().now() - self.phase_start).nanoseconds / 1e9

    def next_phase(self):
        self.phase += 1
        self.phase_start = self.get_clock().now()

    # ---------- 메인 시퀀스 ----------
    # Phase 0: BalanceStand, 2초
    # Phase 1: 전진 (vx만), 1초
    # Phase 2: 정지 (StopMove), 2초
    # Phase 3: 제자리 회전 (vyaw만), 1초
    # Phase 4: 정지, 2초
    # Phase 5: 종료 (BalanceStand)
    def tick(self):
        t = self.elapsed()

        if self.phase == 0:
            if t < 0.02:
                self.get_logger().info('[0] BalanceStand')
                self.send_balance_stand()
            if t >= 2.0:
                self.next_phase()

        elif self.phase == 1:
            # 전진
            if t < 0.02:
                self.get_logger().info(f'[1] Move FORWARD vx={self.fwd_speed}')
            self.send_move(self.fwd_speed, 0.0, 0.0)
            if int(t * 50) % 25 == 0 and t > 0.1:
                self.get_logger().info(
                    f'    pos=({self.pos[0]:+.2f}, {self.pos[1]:+.2f}) | '
                    f'odom=({self.odom_x:+.2f}, {self.odom_y:+.2f})'
                )
            if t >= self.fwd_duration:
                self.next_phase()

        elif self.phase == 2:
            if t < 0.02:
                self.get_logger().info('[2] STOP')
                self.send_stop()
            if t >= 2.0:
                self.next_phase()

        elif self.phase == 3:
            # 회전
            if t < 0.02:
                self.get_logger().info(f'[3] Yaw rotate vyaw={self.yaw_speed}')
            self.send_move(0.0, 0.0, self.yaw_speed)
            if int(t * 50) % 25 == 0 and t > 0.1:
                self.get_logger().info(
                    f'    yaw={math.degrees(self.yaw):+.1f}deg'
                )
            if t >= self.yaw_duration:
                self.next_phase()

        elif self.phase == 4:
            if t < 0.02:
                self.get_logger().info('[4] STOP')
                self.send_stop()
            if t >= 2.0:
                self.next_phase()

        elif self.phase == 5:
            if t < 0.02:
                self.get_logger().info('[5] BalanceStand (done)')
                self.send_balance_stand()
                # 최종 결과 요약
                if self.start_odom:
                    dx = self.odom_x - self.start_odom[0]
                    dy = self.odom_y - self.start_odom[1]
                    dist = math.hypot(dx, dy)
                    self.get_logger().info('=' * 60)
                    self.get_logger().info(
                        f'Total displacement: dx={dx:+.3f}m, dy={dy:+.3f}m, dist={dist:.3f}m'
                    )
                    self.get_logger().info('=' * 60)
            if t >= 1.0:
                self.next_phase()

        else:
            self.get_logger().info('Sequence complete. Ctrl+C to exit.')
            self.timer.cancel()


def main():
    rclpy.init()
    node = MoveTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()