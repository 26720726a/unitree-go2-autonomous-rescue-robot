import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PointStamped
from std_msgs.msg import Float32, String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import math


class GapFinder(Node):
    def __init__(self):
        super().__init__('gap_finder')

        # ─── 파라미터 ───
        # 빈이 "안전"으로 판정되는 최소 거리
        self.safe_distance = 0.80   # m

        # 갭이 유효하려면 최소 몇 개 빈 연속이어야 하는지
        # (로봇 폭 ~40cm + 마진을 거리에 따라 환산하면 좋지만, 단순화)
        self.min_gap_bins = 3   # 3빈 = 30°

        # 점수 가중치
        self.w_width    = 1.0   # 갭 폭
        self.w_depth    = 0.5   # 갭 깊이 (멀수록 좋음)
        self.w_target   = 2.0   # 목표 방향과의 일치 (가장 중요)

        # forward_axis (front_obstacle, safety_gate와 동일)
        self.forward_axis = '-y'

        # ─── 상태 ───
        self.target_angle = 0.0   # 사용자 의도 방향 (rad, 로봇 좌표계)
        self.has_target = False    # 사용자가 직선 이동 명령 중인지

        # LaserScan 매핑
        self.n_bins = 0
        self.angle_min = -math.pi
        self.angle_increment = 0.0
        self.bin_front = 0

        # ─── ROS I/O ───
        self.create_subscription(LaserScan, '/scan_360', self.scan_cb, 10)
        self.create_subscription(Twist,     '/cmd_vel',  self.cmd_cb,  10)

        # 추천 방향 발행
        self.angle_pub  = self.create_publisher(Float32, '/recommended_heading', 10)
        self.text_pub   = self.create_publisher(String,  '/gap_hint', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/gap_markers', 10)

        # 1Hz 텍스트 힌트 로그
        self.create_timer(1.0, self.print_hint)

        self.latest_hint = ''

        self.get_logger().info(
            f'GapFinder started: safe={self.safe_distance}m, '
            f'min_gap={self.min_gap_bins} bins, forward_axis={self.forward_axis}'
        )

    # ──────────────────────────────────────────────
    # 매핑 헬퍼
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

    def bin_to_robot_angle(self, bin_idx):
        """빈 인덱스 → 로봇 좌표계 각도 (rad)
        로봇 정면 = 0, 좌측 = +π/2, 후방 = ±π, 우측 = -π/2"""
        lidar_angle = self.angle_min + bin_idx * self.angle_increment
        # 로봇 정면을 0으로 회전
        front_lidar_angle = self.angle_min + self.bin_front * self.angle_increment
        robot_angle = lidar_angle - front_lidar_angle
        # -π ~ +π 정규화
        while robot_angle > math.pi:
            robot_angle -= 2 * math.pi
        while robot_angle < -math.pi:
            robot_angle += 2 * math.pi
        return robot_angle

    def robot_angle_to_lidar_xy(self, angle, distance):
        """로봇 좌표계 각도 → 라이다 좌표계 (x, y)
        (마커 발행용 — 라이다 frame에 그림)"""
        lidar_angle = angle + (self.angle_min + self.bin_front * self.angle_increment)
        x = distance * math.cos(lidar_angle)
        y = distance * math.sin(lidar_angle)
        return x, y

    # ──────────────────────────────────────────────
    # 콜백
    # ──────────────────────────────────────────────
    def cmd_cb(self, msg: Twist):
        """사용자 명령에서 의도 방향 추출"""
        vx, vy = msg.linear.x, msg.linear.y

        # 직선 이동 명령이 있으면 그 방향이 목표
        if abs(vx) > 0.05 or abs(vy) > 0.05:
            self.target_angle = math.atan2(vy, vx)
            self.has_target = True
        else:
            self.has_target = False

    def scan_cb(self, msg: LaserScan):
        # 매핑 초기화
        if (self.n_bins != len(msg.ranges) or
            self.angle_increment != msg.angle_increment):
            self.n_bins = len(msg.ranges)
            self.angle_min = msg.angle_min
            self.angle_increment = msg.angle_increment
            self.bin_front = self._compute_front_bin()
            self.get_logger().info(
                f'Scan mapping: n_bins={self.n_bins}, front_bin={self.bin_front}'
            )

        if self.n_bins == 0:
            return

        ranges = list(msg.ranges)
        gaps = self.find_gaps(ranges)

        if not gaps:
            self.latest_hint = '⚠️  No safe gap found — surrounded'
            self.publish_recommendation(None)
            self.publish_markers(msg.header.frame_id, msg.header.stamp, [], None)
            return

        # 점수 매기기
        best_gap = self.score_gaps(gaps, ranges)

        # 추천 방향 발행
        self.publish_recommendation(best_gap)
        self.publish_markers(msg.header.frame_id, msg.header.stamp, gaps, best_gap)

        # 텍스트 힌트
        center_angle_deg = math.degrees(best_gap['center_robot_angle'])
        self.latest_hint = (
            f'Best gap: {center_angle_deg:+.0f}° '
            f'(width={best_gap["width"]} bins, '
            f'depth={best_gap["mean_depth"]:.2f}m, '
            f'score={best_gap["score"]:.2f})'
        )

    # ──────────────────────────────────────────────
    # 갭 찾기
    # ──────────────────────────────────────────────
    def find_gaps(self, ranges):
        """연속된 안전 빈을 갭으로 묶음 (원형 배열 고려)"""
        n = self.n_bins
        safe = [r > self.safe_distance for r in ranges]

        # 모두 안전 / 모두 위험 케이스
        if all(safe):
            return [{'start': 0, 'end': n - 1, 'width': n}]
        if not any(safe):
            return []

        # 원형이라 시작점을 위험한 곳으로 설정
        start_idx = 0
        for i in range(n):
            if not safe[i]:
                start_idx = i
                break

        # 시작점부터 한 바퀴 돌며 연속 구간 추출
        gaps = []
        i = 0
        while i < n:
            idx = (start_idx + i) % n
            if safe[idx]:
                gap_start = idx
                length = 0
                while i < n and safe[(start_idx + i) % n]:
                    length += 1
                    i += 1
                gap_end = (start_idx + i - 1) % n
                if length >= self.min_gap_bins:
                    gaps.append({
                        'start': gap_start,
                        'end':   gap_end,
                        'width': length,
                    })
            else:
                i += 1
        return gaps

    # ──────────────────────────────────────────────
    # 갭 점수
    # ──────────────────────────────────────────────
    def score_gaps(self, gaps, ranges):
        """각 갭에 점수 부여 후 최고점 반환"""
        best = None
        best_score = -float('inf')

        for gap in gaps:
            # 갭의 빈 인덱스들
            n = self.n_bins
            length = gap['width']
            indices = [(gap['start'] + k) % n for k in range(length)]

            # 갭 중심 빈
            center_idx = indices[length // 2]
            center_robot_angle = self.bin_to_robot_angle(center_idx)

            # 깊이 — 갭 내 평균 거리
            valid = [ranges[i] for i in indices
                     if not math.isinf(ranges[i]) and ranges[i] > 0]
            if not valid:
                continue
            mean_depth = sum(valid) / len(valid)
            min_depth  = min(valid)

            # 점수
            width_score = length / n   # 0~1
            depth_score = min(mean_depth / 5.0, 1.0)   # 5m 이상이면 만점

            # 목표 방향과의 각도 차이 (라디안 → 0~1 점수)
            if self.has_target:
                angle_diff = abs(self.normalize_angle(
                    center_robot_angle - self.target_angle))
                # 0° = 1.0, 180° = 0.0
                target_score = 1.0 - (angle_diff / math.pi)
            else:
                # 목표 없으면 정면 방향 선호
                angle_diff = abs(center_robot_angle)
                target_score = 1.0 - (angle_diff / math.pi)

            score = (self.w_width  * width_score +
                     self.w_depth  * depth_score +
                     self.w_target * target_score)

            gap['center_idx']         = center_idx
            gap['center_robot_angle'] = center_robot_angle
            gap['mean_depth']         = mean_depth
            gap['min_depth']          = min_depth
            gap['score']              = score

            if score > best_score:
                best_score = score
                best = gap

        return best

    def normalize_angle(self, a):
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

    # ──────────────────────────────────────────────
    # 발행
    # ──────────────────────────────────────────────
    def publish_recommendation(self, best_gap):
        msg = Float32()
        if best_gap is None:
            msg.data = float('nan')
        else:
            msg.data = float(best_gap['center_robot_angle'])
        self.angle_pub.publish(msg)

        text = String()
        text.data = self.latest_hint
        self.text_pub.publish(text)

    def publish_markers(self, frame_id, stamp, gaps, best_gap):
        """rviz 시각화 — 모든 갭은 회색 호, 최고점은 굵은 녹색 화살표"""
        ma = MarkerArray()

        # 1) 모든 갭 — 회색 선
        gaps_marker = Marker()
        gaps_marker.header.frame_id = frame_id
        gaps_marker.header.stamp = stamp
        gaps_marker.ns = 'gaps_all'
        gaps_marker.id = 0
        gaps_marker.type = Marker.LINE_LIST
        gaps_marker.action = Marker.ADD
        gaps_marker.scale.x = 0.05
        gaps_marker.color.r = 0.6
        gaps_marker.color.g = 0.6
        gaps_marker.color.b = 0.6
        gaps_marker.color.a = 0.6
        gaps_marker.pose.orientation.w = 1.0
        gaps_marker.lifetime.sec = 0
        gaps_marker.lifetime.nanosec = 300_000_000

        for gap in gaps:
            n = self.n_bins
            length = gap['width']
            indices = [(gap['start'] + k) % n for k in range(length)]
            # 갭 양 끝점만 그림 (원점→끝)
            for i_bin in [indices[0], indices[-1]]:
                angle_robot = self.bin_to_robot_angle(i_bin)
                x, y = self.robot_angle_to_lidar_xy(angle_robot, 2.0)
                p0 = Point()
                p0.x, p0.y, p0.z = 0.0, 0.0, 0.0
                p1 = Point()
                p1.x, p1.y, p1.z = x, y, 0.0
                gaps_marker.points.append(p0)
                gaps_marker.points.append(p1)
        ma.markers.append(gaps_marker)

        # 2) 최고 갭 — 굵은 녹색 화살표 (라이다에서 갭 중심 방향)
        best_marker = Marker()
        best_marker.header.frame_id = frame_id
        best_marker.header.stamp = stamp
        best_marker.ns = 'gap_best'
        best_marker.id = 1
        best_marker.type = Marker.ARROW
        best_marker.action = Marker.ADD if best_gap else Marker.DELETE
        if best_gap:
            angle_robot = best_gap['center_robot_angle']
            x, y = self.robot_angle_to_lidar_xy(angle_robot, best_gap['mean_depth'])
            p0 = Point()
            p0.x, p0.y, p0.z = 0.0, 0.0, 0.0
            p1 = Point()
            p1.x, p1.y, p1.z = x, y, 0.0
            best_marker.points = [p0, p1]
            best_marker.scale.x = 0.08    # shaft 두께
            best_marker.scale.y = 0.18    # head 두께
            best_marker.scale.z = 0.20    # head 길이
            best_marker.color.r = 0.0
            best_marker.color.g = 1.0
            best_marker.color.b = 0.0
            best_marker.color.a = 0.95
            best_marker.lifetime.sec = 0
            best_marker.lifetime.nanosec = 300_000_000
        ma.markers.append(best_marker)

        self.marker_pub.publish(ma)

    def print_hint(self):
        if self.latest_hint:
            self.get_logger().info(self.latest_hint)


def main():
    rclpy.init()
    node = GapFinder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
