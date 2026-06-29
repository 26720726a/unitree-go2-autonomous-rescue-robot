import rclpy
from rclpy.node import Node
from unitree_api.msg import Request
from unitree_go.msg import SportModeState
import json
import math

# API IDs (ros2_sport_client.h 확인 완료)
SPORT_API_BALANCESTAND = 1002
SPORT_API_EULER        = 1007
SPORT_API_POSE         = 1028


class PitchTest(Node):
    def __init__(self):
        super().__init__('pitch_test')

        self.pub = self.create_publisher(Request, '/api/sport/request', 10)
        self.sub = self.create_subscription(
            SportModeState, '/sportmodestate', self.state_cb, 10
        )
        self.current_pitch_deg = 0.0

        # 목표 자세
        self.pitch_deg = -15.0
        self.pitch_rad = math.radians(self.pitch_deg)
        self.roll_rad  = 0.0
        self.yaw_rad   = 0.0

        # 단계 진행 변수
        self.phase = 0
        self.phase_start = self.get_clock().now()

        # C++ 원본과 동일하게 50Hz 타이머
        self.timer = self.create_timer(0.02, self.tick)

        self.get_logger().info('=' * 60)
        self.get_logger().info('Pitch Test (Python port of pitch_15deg.cpp)')
        self.get_logger().info(f'Target: pitch={self.pitch_deg}deg ({self.pitch_rad:+.4f} rad)')
        self.get_logger().info('!! Keep remote control ready (L2+B = Damp) !!')
        self.get_logger().info('=' * 60)

    # ---------- 헬퍼 ----------
    def state_cb(self, msg):
        # rpy[1] = pitch (rad)
        self.current_pitch_deg = math.degrees(msg.imu_state.rpy[1])

    def send_balance_stand(self):
        req = Request()
        req.header.identity.api_id = SPORT_API_BALANCESTAND
        self.pub.publish(req)

    def send_pose(self, flag: bool):
        req = Request()
        req.header.identity.api_id = SPORT_API_POSE
        req.parameter = json.dumps({"data": flag})  # ← cpp에서 js["data"] = flag
        self.pub.publish(req)

    def send_euler(self, roll, pitch, yaw):
        req = Request()
        req.header.identity.api_id = SPORT_API_EULER
        req.parameter = json.dumps({"x": roll, "y": pitch, "z": yaw})
        self.pub.publish(req)

    def elapsed_in_phase(self):
        return (self.get_clock().now() - self.phase_start).nanoseconds / 1e9

    def next_phase(self):
        self.phase += 1
        self.phase_start = self.get_clock().now()

    # ---------- 메인 시퀀스 ----------
    # C++ 원본 흐름:
    #  BalanceStand → sleep(2)
    #  Pose(true)  → usleep(0.5s)
    #  Euler(0, pitch, 0) × 50Hz × 3s
    #  Euler(0, 0, 0)     → sleep(1)
    #  Pose(false)
    #  BalanceStand
    def tick(self):
        t = self.elapsed_in_phase()

        # Phase 0: BalanceStand, 2초 대기
        if self.phase == 0:
            if t < 0.02:
                self.get_logger().info('[0] BalanceStand')
                self.send_balance_stand()
            if t >= 2.0:
                self.next_phase()

        # Phase 1: Pose(true), 0.5초 대기
        elif self.phase == 1:
            if t < 0.02:
                self.get_logger().info('[1] Pose(true) - enter pose mode')
                self.send_pose(True)
            if t >= 0.5:
                self.next_phase()

        # Phase 2: Euler 50Hz × 3초
        elif self.phase == 2:
            if t < 0.02:
                self.get_logger().info(f'[2] Euler pitch={self.pitch_deg}deg for 3s')
            self.send_euler(self.roll_rad, self.pitch_rad, self.yaw_rad)
            # 1초마다 현재 pitch 출력
            if int(t * 50) % 50 == 0 and t > 0.1:
                self.get_logger().info(
                    f'    current pitch = {self.current_pitch_deg:+.2f} deg'
                )
            if t >= 3.0:
                self.next_phase()

        # Phase 3: Euler 0으로 복귀, 1초 대기
        elif self.phase == 3:
            if t < 0.02:
                self.get_logger().info('[3] Reset Euler to 0')
            self.send_euler(0.0, 0.0, 0.0)
            if t >= 1.0:
                self.next_phase()

        # Phase 4: Pose(false), 짧게 대기
        elif self.phase == 4:
            if t < 0.02:
                self.get_logger().info('[4] Pose(false) - exit pose mode')
                self.send_pose(False)
            if t >= 0.5:
                self.next_phase()

        # Phase 5: BalanceStand, 종료
        elif self.phase == 5:
            if t < 0.02:
                self.get_logger().info('[5] BalanceStand (done)')
                self.send_balance_stand()
            if t >= 1.0:
                self.next_phase()

        else:
            self.get_logger().info('=' * 60)
            self.get_logger().info('Sequence complete. Ctrl+C to exit.')
            self.get_logger().info('=' * 60)
            self.timer.cancel()


def main():
    rclpy.init()
    node = PitchTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()