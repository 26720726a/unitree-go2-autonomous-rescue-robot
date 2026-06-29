#!/usr/bin/env python3
"""
앞쪽 장애물 감지 + 360도 거리 프로파일 통합 노드
  - 기존: 전방 1구역 검출 (front_obstacle 토픽)
  - 추가: 360도를 36개 빈으로 나눠 omni 프로파일 발행 + rviz 시각화
  - 추가: 비대칭 self-filter (라이다가 머리 위에 위치)
  - 추가: rviz에 본체 박스 마커 표시
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, LaserScan
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
import collections
import math
from rclpy.qos import qos_profils_sensor_data

class FrontObstacleDetector(Node):
    def __init__(self):
        super().__init__('front_obstacle_detector')

        # ─── ROI 파라미터 (전방 1구역 검출) ───
        self.fwd_min   = 0.4
        self.fwd_max   = 5.0
        self.side_half = 0.5
        self.warn_distance = 0.5

        # 라이다 -y가 로봇 전방
        self.forward_axis = '-y'

        # ─── 높이 필터 (라이다 z=0 기준, 바닥은 z=-0.5) ───
        self.z_min = -0.40   # 바닥보다 살짝 위 (바닥 점 제외)
        self.z_max =  1.00   # 천장 제외

        # ─── 로봇 자체 몸체 필터 (self-filter, 비대칭) ───
        # 라이다가 머리 위에 위치 → 앞쪽은 짧고 뒤쪽은 김
        # 모두 로봇 좌표계 기준 (fwd=전방, left=좌측)
        self.body_fwd_front =  0.30   # 라이다 앞쪽 30cm까지가 본체
        self.body_fwd_back  = -0.50   # 라이다 뒤쪽 50cm까지가 본체
        self.body_half_side =  0.20   # 좌우 반폭 20cm

        # ─── 라이다 높이 (본체 박스 시각화용) ───
        self.lidar_height = 0.50      # 바닥에서 라이다까지

        # ─── 노이즈 필터 (전방 1구역용) ───
        self.min_points        = 8    # ROI 안 최소 점 개수
        self.n_closest         = 5    # 거리 계산용 가까운 점 개수
        self.history_size      = 5    # 시간 윈도우
        self.confirm_threshold = 3    # 확정에 필요한 프레임 수
        self.history = collections.deque(maxlen=self.history_size)

        # ─── 360도 프로파일 파라미터 ───
        self.n_bins             = 36           # 10도씩 36구간
        self.bin_size_rad       = 2 * math.pi / self.n_bins
        self.omni_min_points    = 5            # 빈당 최소 점 개수
        self.omni_max_range     = 5.0          # 너무 먼 점은 무시
        self.omni_show_max      = 3.0          # 이 거리 안쪽만 rviz 표시
        self.neighbor_max_dgap  = 0.5          # 이웃 연결 시 거리 차 임계

        # ─── 상태 변수 ───
        self.last_frame_id = None

        # ─── ROS I/O ───
        self.sub = self.create_subscription(
            PointCloud2, '/lidar_points', self.cb, qos_profile_sensor_data
        )
        # 전방 1구역 (기존)
        self.pub = self.create_publisher(
            Float32, '/front_obstacle_distance', 10
        )
        self.marker_pub = self.create_publisher(
            Marker, '/front_obstacle_marker', 10
        )
        # 360도 프로파일
        self.scan_pub = self.create_publisher(
            LaserScan, '/scan_360', 10
        )
        self.omni_marker_pub = self.create_publisher(
            MarkerArray, '/omni_obstacle_markers', 10
        )
        # 본체 박스 (디버깅용)
        self.body_box_pub = self.create_publisher(
            Marker, '/robot_body_box', 10
        )

        # 1초마다 본체 박스 마커 발행
        self.body_box_timer = self.create_timer(1.0, self.publish_body_box)

        self.get_logger().info(
            f'Detector started. forward_axis={self.forward_axis}, '
            f'omni_bins={self.n_bins} ({360//self.n_bins}° each), '
            f'body=[fwd:{self.body_fwd_back:+.2f}~{self.body_fwd_front:+.2f}, '
            f'side:±{self.body_half_side:.2f}]'
        )

    # ──────────────────────────────────────────────
    # Hesai 22-byte point 파싱
    # ──────────────────────────────────────────────
    def parse_points(self, msg: PointCloud2):
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        n_points = msg.width * msg.height
        raw = raw.reshape(n_points, msg.point_step)
        xyz = raw[:, 0:12].copy().view(np.float32).reshape(n_points, 3)
        return xyz

    # ──────────────────────────────────────────────
    # 좌표계 변환: 라이다 → 로봇
    #   fwd  = 로봇 전방
    #   left = 로봇 좌측 (양수)
    # ──────────────────────────────────────────────
    def to_robot_frame(self, x, y):
        if self.forward_axis == 'x':
            fwd, left = x, y
        elif self.forward_axis == 'y':
            fwd, left = y, -x
        elif self.forward_axis == '-y':
            fwd, left = -y, x
        elif self.forward_axis == '-x':
            fwd, left = -x, -y
        else:
            fwd, left = x, y
        return fwd, left

    # ──────────────────────────────────────────────
    # 좌표계 변환: 로봇 → 라이다 (역방향)
    # ──────────────────────────────────────────────
    def from_robot_frame(self, fwd, left):
        if self.forward_axis == 'x':
            return fwd, left
        elif self.forward_axis == 'y':
            return -left, fwd
        elif self.forward_axis == '-y':
            return left, -fwd
        elif self.forward_axis == '-x':
            return -fwd, -left
        return fwd, left

    # ──────────────────────────────────────────────
    # 전방 1구역 마커 발행
    # ──────────────────────────────────────────────
    def publish_front_marker(self, frame_id, stamp, position=None):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = 'front_obstacle'
        marker.id = 0
        if position is None:
            marker.action = Marker.DELETE
        else:
            x, y, z = position
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = float(z)
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 200_000_000
        self.marker_pub.publish(marker)

    # ──────────────────────────────────────────────
    # 본체 박스 마커 (rviz 시각화용)
    # ──────────────────────────────────────────────
    def publish_body_box(self):
        if self.last_frame_id is None:
            return  # 라이다 데이터 아직 없음

        marker = Marker()
        marker.header.frame_id = self.last_frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'robot_body'
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # 박스 중심 (로봇 좌표계)
        center_fwd  = (self.body_fwd_front + self.body_fwd_back) / 2
        center_left = 0.0
        size_fwd    = self.body_fwd_front - self.body_fwd_back
        size_side   = self.body_half_side * 2

        # 라이다 좌표계로 역변환
        cx, cy = self.from_robot_frame(center_fwd, center_left)

        # forward_axis가 'y' 계열이면 x/y 사이즈 swap
        if self.forward_axis in ('y', '-y'):
            sx, sy = size_side, size_fwd
        else:
            sx, sy = size_fwd, size_side

        marker.pose.position.x = float(cx)
        marker.pose.position.y = float(cy)
        marker.pose.position.z = -self.lidar_height / 2
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(sx)
        marker.scale.y = float(sy)
        marker.scale.z = self.lidar_height
        marker.color.r = 0.2
        marker.color.g = 0.5
        marker.color.b = 1.0   # 파란색
        marker.color.a = 0.3   # 반투명
        marker.lifetime.sec = 2
        self.body_box_pub.publish(marker)

    # ──────────────────────────────────────────────
    # 360도 omni 마커 (점 + 선)
    # ──────────────────────────────────────────────
    def publish_omni_markers(self, frame_id, stamp, bin_distances, bin_xy):
        ma = MarkerArray()

        # ─── 1) 점 마커 (각 빈의 대표점) ───
        points_marker = Marker()
        points_marker.header.frame_id = frame_id
        points_marker.header.stamp = stamp
        points_marker.ns = 'omni_points'
        points_marker.id = 0
        points_marker.type = Marker.SPHERE_LIST
        points_marker.action = Marker.ADD
        points_marker.scale.x = 0.15
        points_marker.scale.y = 0.15
        points_marker.scale.z = 0.15
        points_marker.color.r = 1.0
        points_marker.color.g = 1.0
        points_marker.color.b = 0.0   # 노란색
        points_marker.color.a = 0.9
        points_marker.pose.orientation.w = 1.0
        points_marker.lifetime.sec = 0
        points_marker.lifetime.nanosec = 200_000_000

        for i in range(self.n_bins):
            d = bin_distances[i]
            if np.isinf(d) or d > self.omni_show_max:
                continue
            p = Point()
            p.x = float(bin_xy[i, 0])
            p.y = float(bin_xy[i, 1])
            p.z = 0.0
            points_marker.points.append(p)
        ma.markers.append(points_marker)

        # ─── 2) 선 마커 (이웃 연결) ───
        line_marker = Marker()
        line_marker.header.frame_id = frame_id
        line_marker.header.stamp = stamp
        line_marker.ns = 'omni_lines'
        line_marker.id = 1
        line_marker.type = Marker.LINE_LIST
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.04
        line_marker.color.r = 0.2
        line_marker.color.g = 1.0
        line_marker.color.b = 0.2   # 녹색
        line_marker.color.a = 0.8
        line_marker.pose.orientation.w = 1.0
        line_marker.lifetime.sec = 0
        line_marker.lifetime.nanosec = 200_000_000

        for i in range(self.n_bins):
            j = (i + 1) % self.n_bins
            di, dj = bin_distances[i], bin_distances[j]
            if np.isinf(di) or np.isinf(dj):
                continue
            if di > self.omni_show_max or dj > self.omni_show_max:
                continue
            if abs(di - dj) > self.neighbor_max_dgap:
                continue   # 거리 차가 크면 다른 물체

            pi, pj = Point(), Point()
            pi.x, pi.y, pi.z = float(bin_xy[i, 0]), float(bin_xy[i, 1]), 0.0
            pj.x, pj.y, pj.z = float(bin_xy[j, 0]), float(bin_xy[j, 1]), 0.0
            line_marker.points.append(pi)
            line_marker.points.append(pj)
        ma.markers.append(line_marker)

        self.omni_marker_pub.publish(ma)

    # ──────────────────────────────────────────────
    # 메인 콜백
    # ──────────────────────────────────────────────
    def cb(self, msg: PointCloud2):
        # frame_id 저장 (본체 박스 마커용)
        self.last_frame_id = msg.header.frame_id

        # 점 파싱
        xyz = self.parse_points(msg)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        # 기본 마스크
        finite    = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        not_zero  = ~((x == 0) & (y == 0) & (z == 0))
        height_ok = (z >= self.z_min) & (z <= self.z_max)

        # 로봇 좌표계로 변환
        fwd, left = self.to_robot_frame(x, y)

        # self-filter: 로봇 본체 영역 안의 점 제외
        in_body = (
            (fwd >= self.body_fwd_back) & (fwd <= self.body_fwd_front) &
            (np.abs(left) <= self.body_half_side)
        )
        body_mask = ~in_body

        # 최종 유효 마스크
        valid = finite & not_zero & height_ok & body_mask

        # ====================================================================
        # [Part 1] 전방 1구역 검출 (시간 + 공간 노이즈 필터)
        # ====================================================================
        roi = (
            valid &
            (fwd >= self.fwd_min) & (fwd <= self.fwd_max) &
            (np.abs(left) <= self.side_half)
        )
        n_roi = int(roi.sum())

        if n_roi < self.min_points:
            current_distance = -1.0
            current_closest  = None
        else:
            d = np.sqrt(fwd[roi] ** 2 + left[roi] ** 2)
            sorted_idx = np.argsort(d)
            n_use = min(self.n_closest, len(d))
            top_idx = sorted_idx[:n_use]
            current_distance = float(np.median(d[top_idx]))
            x_roi, y_roi, z_roi = x[roi], y[roi], z[roi]
            current_closest = (
                float(np.mean(x_roi[top_idx])),
                float(np.mean(y_roi[top_idx])),
                float(np.mean(z_roi[top_idx])),
            )

        # 시간 필터
        self.history.append((current_distance, current_closest))
        threats = [(d, p) for (d, p) in self.history if 0 < d < self.warn_distance]
        n_threat = len(threats)

        if n_threat >= self.confirm_threshold:
            min_d, min_p = min(threats, key=lambda t: t[0])
            distance, closest = min_d, min_p
            self.get_logger().warn(
                f'⚠️  CONFIRMED OBSTACLE {distance:.2f}m '
                f'at (x={closest[0]:.2f}, y={closest[1]:.2f}, z={closest[2]:.2f})  '
                f'[{n_threat}/{len(self.history)}]',
                throttle_duration_sec=0.5
            )
            self.publish_front_marker(msg.header.frame_id, msg.header.stamp, closest)
        elif current_distance > 0 and current_closest is not None:
            distance, closest = current_distance, current_closest
            self.get_logger().info(
                f'Front: {distance:.2f}m [n_roi={n_roi}, threat={n_threat}/{len(self.history)}]',
                throttle_duration_sec=1.0
            )
            self.publish_front_marker(msg.header.frame_id, msg.header.stamp, closest)
        else:
            distance = -1.0
            self.get_logger().info(
                f'Front clear (n_roi={n_roi})',
                throttle_duration_sec=1.0
            )
            self.publish_front_marker(msg.header.frame_id, msg.header.stamp, None)

        out = Float32()
        out.data = distance
        self.pub.publish(out)

        # ====================================================================
        # [Part 2] 360도 거리 프로파일
        # ====================================================================
        # 유효한 점만 추출 (로봇 좌표계 사용)
        x_v    = x[valid]
        y_v    = y[valid]

        # 각 점의 각도와 거리
        ranges_v = np.sqrt(x_v ** 2 + y_v ** 2)
        angles_v = np.arctan2(y_v, x_v)   # 0 = 전방, + = 좌

        # 거리 범위 필터
        ok = (ranges_v > 0.1) & (ranges_v < self.omni_max_range)
        ranges_ok = ranges_v[ok]
        angles_ok = angles_v[ok]
        x_ok = x_v[ok]
        y_ok = y_v[ok]

        # 빈 인덱스 계산
        bin_idx = ((angles_ok + math.pi) / self.bin_size_rad).astype(int)
        bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)

        # 각 빈의 점 개수
        counts = np.bincount(bin_idx, minlength=self.n_bins)

        # 각 빈의 최단 거리 + 대표 좌표
        bin_distances = np.full(self.n_bins, np.inf)
        bin_xy        = np.zeros((self.n_bins, 2))

        for i in range(self.n_bins):
            if counts[i] < self.omni_min_points:
                continue
            sel = (bin_idx == i)
            d_sel = ranges_ok[sel]
            d_sorted_idx = np.argsort(d_sel)
            n_use = min(5, len(d_sel))
            top_idx = d_sorted_idx[:n_use]
            bin_distances[i] = float(np.median(d_sel[top_idx]))
            sel_indices = np.where(sel)[0]
            top_global  = sel_indices[top_idx]
            bin_xy[i, 0] = float(np.mean(x_ok[top_global]))
            bin_xy[i, 1] = float(np.mean(y_ok[top_global]))

        # LaserScan 발행
        scan = LaserScan()
        scan.header.frame_id = msg.header.frame_id
        scan.header.stamp = msg.header.stamp
        scan.angle_min = -math.pi
        scan.angle_max =  math.pi - self.bin_size_rad
        scan.angle_increment = self.bin_size_rad
        scan.range_min = 0.1
        scan.range_max = self.omni_max_range
        ranges_out = np.where(
            np.isinf(bin_distances),
            self.omni_max_range + 1.0,
            bin_distances
        )
        scan.ranges = ranges_out.astype(np.float32).tolist()
        self.scan_pub.publish(scan)

        # rviz 시각화 (점 + 선)
        self.publish_omni_markers(msg.header.frame_id, msg.header.stamp,
                                   bin_distances, bin_xy)


def main():
    rclpy.init()
    node = FrontObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
