import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from unitree_api.msg import Request
import json
import math

SPORT_API_MOVE = 1008


class SafetyGate(Node):
    def __init__(self):
        super().__init__('safety_gate')

        # ─── 거리/시간 파라미터 (기존 유지) ───
        # 방향별 정지/해제 거리 (m)
        # 본체 끝까지 거리 + 안전 마진(약 20cm)
        self.stop_distance = {
            'front': 0.50,   # 본체 앞끝 30cm + 마진 20cm
            'back':  0.70,   # 본체 뒤끝 50cm + 마진 20cm  ← 후방 늘림
            'left':  0.40,   # 본체 좌끝 20cm + 마진 20cm
            'right': 0.40,   # 본체 우끝 20cm + 마진 20cm
        }
        self.clear_distance = {
            'front': 0.70,
            'back':  0.90,   # stop + 20cm
            'left':  0.60,
            'right': 0.60,
        }


        self.cmd_timeout       = 0.3    # /cmd_vel 끊김 시 자동 정지
        self.no_lidar_timeout  = 1.0    # /scan_360 끊김 시 안전 차단

        # ─── 방향별 검사 범위 (신규) ───
        # 각 방향 중심에서 ±몇 도까지 검사할지
        self.direction_half_width_deg = 20.0

        # ─── forward_axis (front_obstacle과 동일하게) ───
        self.forward_axis = '-y'

        # ─── 회전 차단 옵션 ───
        # True면 회전 방향 측면 막혔을 때 회전 차단
        # False면 회전은 항상 허용 (탈출 수단)
        self.block_rotation = False

        # ─── 상태: 기존 ───
        self.front_distance = -1.0   # /front_obstacle_distance (시간 필터 적용된 값)
        self.forward_blocked_legacy = False   # 기존 코드와 동일한 히스테리시스
        self.target_vx   = 0.0
        self.target_vy   = 0.0
        self.target_vyaw = 0.0
        self.last_cmd_time   = self.get_clock().now()
        self.last_lidar_time = self.get_clock().now()
        self.last_scan_time  = self.get_clock().now()

        # ─── 상태: 360도 ───
        self.dir_distance = {
            'front': float('nan'),
            'back':  float('nan'),
            'left':  float('nan'),
            'right': float('nan'),
        }
        self.blocked = {
            'front': False, 'back': False,
            'left':  False, 'right': False,
        }

        # ─── LaserScan 매핑 (첫 스캔 받으면 초기화) ───
        self.n_bins = 0
        self.angle_min = -math.pi
        self.angle_increment = 0.0
        self.bin_front = 0
        self.bin_left  = 0
        self.bin_back  = 0
        self.bin_right = 0
        self.bin_window_size = 0   # 방향당 검사할 빈 개수 (한쪽)

        # ─── 구독 ───
        self.create_subscription(Twist,     '/cmd_vel',
                                 self.cmd_cb, 10)
        # 기존 — 시간 필터 적용된 정면 검출 (보조 안전망)
        self.create_subscription(Float32,   '/front_obstacle_distance',
                                 self.front_legacy_cb, 10)
        # 신규 — 360도 거리 프로파일
        self.create_subscription(LaserScan, '/scan_360',
                                 self.scan_cb, 10)

        # ─── 송신 ───
        self.api_pub = self.create_publisher(Request, '/api/sport/request', 10)

        # 50Hz 송신 루프
        self.timer = self.create_timer(0.02, self.publish_to_robot)
        # 1Hz 상태 출력
        self.create_timer(1.0, self.print_status)

        self.get_logger().info(
            f'SafetyGate (360°) started: stop={self.stop_distance}m, '
            f'clear={self.clear_distance}m, '
            f'forward_axis={self.forward_axis}, '
            f'check=±{self.direction_half_width_deg}°, '
            f'block_rotation={self.block_rotation}'
        )

    # ──────────────────────────────────────────────
    # forward_axis에 따른 전방 빈 인덱스 계산
    # ──────────────────────────────────────────────
    def _compute_front_bin(self):
        if self.forward_axis == 'x':
            front_angle = 0.0
        elif self.forward_axis == 'y':
            front_angle = math.pi / 2
        elif self.forward_axis == '-y':
            front_angle = -math.pi / 2
        elif self.forward_axis == '-x':
            front_angle = math.pi
        else:
            front_angle = 0.0
        if self.angle_increment == 0:
            return 0
        return int((front_angle - self.angle_min) / self.angle_increment) % self.n_bins

    # ──────────────────────────────────────────────
    # 콜백들
    # ──────────────────────────────────────────────
    def cmd_cb(self, msg: Twist):
        self.target_vx   = msg.linear.x
        self.target_vy   = msg.linear.y
        self.target_vyaw = msg.angular.z
        self.last_cmd_time = self.get_clock().now()

    def front_legacy_cb(self, msg: Float32):
        """기존 /front_obstacle_distance (시간 필터 적용)"""
        self.front_distance = msg.data
        self.last_lidar_time = self.get_clock().now()

    def scan_cb(self, msg: LaserScan):
        """360도 LaserScan 처리"""
        self.last_scan_time = self.get_clock().now()

        # 첫 메시지 또는 빈 개수 변경 시 매핑 재계산
        if (self.n_bins != len(msg.ranges) or
            self.angle_increment != msg.angle_increment):
            self.n_bins = len(msg.ranges)
            self.angle_min = msg.angle_min
            self.angle_increment = msg.angle_increment

            self.bin_front = self._compute_front_bin()
            self.bin_left  = (self.bin_front +     self.n_bins // 4) % self.n_bins
            self.bin_back  = (self.bin_front +     self.n_bins // 2) % self.n_bins
            self.bin_right = (self.bin_front + 3 * self.n_bins // 4) % self.n_bins

            bin_deg = math.degrees(self.angle_increment)
            self.bin_window_size = max(1,
                int(self.direction_half_width_deg / bin_deg))

            self.get_logger().info(
                f'Scan mapping: n_bins={self.n_bins}, bin_size={bin_deg:.0f}°, '
                f'F={self.bin_front}, L={self.bin_left}, '
                f'B={self.bin_back}, R={self.bin_right}, '
                f'window=±{self.bin_window_size} bins (±{self.bin_window_size*bin_deg:.0f}°)'
            )

        # 방향별 최단 거리
        self.dir_distance['front'] = self._min_range_around(msg.ranges, self.bin_front)
        self.dir_distance['back']  = self._min_range_around(msg.ranges, self.bin_back)
        self.dir_distance['left']  = self._min_range_around(msg.ranges, self.bin_left)
        self.dir_distance['right'] = self._min_range_around(msg.ranges, self.bin_right)

        # 각 방향 히스테리시스 갱신
        for direction in ('front', 'back', 'left', 'right'):
            self._update_block_state_360(direction)

    def _min_range_around(self, ranges, center_bin):
        """center_bin ± bin_window_size 안의 최단 거리 (원형 배열)"""
        n = self.n_bins
        w = self.bin_window_size
        indices = [(center_bin + offset) % n for offset in range(-w, w + 1)]
        vals = [ranges[i] for i in indices
                if not math.isinf(ranges[i]) and ranges[i] > 0]
        if not vals:
            return float('nan')
        return min(vals)

    def _update_block_state_360(self, direction):
        d = self.dir_distance[direction]
        if math.isnan(d):
            return

        stop  = self.stop_distance[direction]
        clear = self.clear_distance[direction]

        if self.blocked[direction]:
            if d > clear:
                self.blocked[direction] = False
                self.get_logger().info(
                    f'✅ {direction.upper()} cleared ({d:.2f}m > {clear:.2f}m)'
                )
        else:
            if d <= stop:
                self.blocked[direction] = True
                self.get_logger().warn(
                    f'⚠️  {direction.upper()} blocked ({d:.2f}m ≤ {stop:.2f}m)'
                )

        
    def _update_block_state_legacy(self):
        d = self.front_distance
        if d < 0:
            self.forward_blocked_legacy = False
            return

        stop  = self.stop_distance['front']
        clear = self.clear_distance['front']

        if self.forward_blocked_legacy:
            if d > clear:
                self.forward_blocked_legacy = False
        else:
            if d <= stop:
                self.forward_blocked_legacy = True




    # ──────────────────────────────────────────────
    # 안전 게이트
    # ──────────────────────────────────────────────
    def apply_safety(self, vx, vy, vyaw):
        # 1) 전진 (vx > 0) — 전방 검사 + 기존 백업
        if vx > 0 and (self.blocked['front'] or self.forward_blocked_legacy):
            vx = 0.0
        # 2) 후진 (vx < 0)
        if vx < 0 and self.blocked['back']:
            vx = 0.0
        # 3) 좌측 옆걸음 (vy > 0)
        if vy > 0 and self.blocked['left']:
            vy = 0.0
        # 4) 우측 옆걸음 (vy < 0)
        if vy < 0 and self.blocked['right']:
            vy = 0.0
        # 5) 회전 (옵션)
        if self.block_rotation:
            if vyaw > 0 and self.blocked['left']:
                vyaw = 0.0
            if vyaw < 0 and self.blocked['right']:
                vyaw = 0.0
        return vx, vy, vyaw

    # ──────────────────────────────────────────────
    # 로봇 송신 (50Hz)
    # ──────────────────────────────────────────────
    def publish_to_robot(self):
        now = self.get_clock().now()

        # 1) cmd_vel 타임아웃 → 정지
        if (now - self.last_cmd_time).nanoseconds / 1e9 > self.cmd_timeout:
            vx, vy, vyaw = 0.0, 0.0, 0.0
        else:
            vx, vy, vyaw = self.target_vx, self.target_vy, self.target_vyaw

        # 2) 라이다 끊김 체크
        lidar_stale = (now - self.last_lidar_time).nanoseconds / 1e9 > self.no_lidar_timeout
        scan_stale  = (now - self.last_scan_time).nanoseconds  / 1e9 > self.no_lidar_timeout

        # 3) 기존 히스테리시스 갱신
        self._update_block_state_legacy()

        # 4) 라이다 끊기면 모든 직선 이동 차단 (회전은 허용)
        if lidar_stale or scan_stale:
            vx = 0.0
            vy = 0.0
        else:
            # 5) 360도 + 기존 검출로 안전 검사
            vx, vy, vyaw = self.apply_safety(vx, vy, vyaw)

        # 6) Move 메시지 송신
        req = Request()
        req.header.identity.api_id = SPORT_API_MOVE
        req.parameter = json.dumps({
            'x': float(vx), 'y': float(vy), 'z': float(vyaw)
        })
        self.api_pub.publish(req)

    # ──────────────────────────────────────────────
    # 상태 출력 (1Hz)
    # ──────────────────────────────────────────────
    def print_status(self):
        def fmt(d):
            return f'{d:.2f}m' if not math.isnan(d) else '--'

        # 360도 상태
        parts = []
        for direction in ('front', 'left', 'back', 'right'):
            mark = '🚫' if self.blocked[direction] else '✓ '
            parts.append(f'{mark}{direction[0].upper()}={fmt(self.dir_distance[direction])}')

        # 기존 시간 필터 정면 검출
        legacy = (f'{self.front_distance:.2f}m'
                  if self.front_distance >= 0 else 'clear')
        legacy_state = 'BLK' if self.forward_blocked_legacy else 'OK'

        self.get_logger().info(
            f'[{" | ".join(parts)}]  '
            f'legacy=[{legacy_state} {legacy}]  '
            f'cmd=({self.target_vx:+.2f},{self.target_vy:+.2f},{self.target_vyaw:+.2f})'
        )


def main():
    rclpy.init()
    node = SafetyGate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
