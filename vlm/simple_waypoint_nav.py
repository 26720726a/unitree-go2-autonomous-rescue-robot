#!/usr/bin/env python3
"""
simple_waypoint_nav.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
vlfm_source_nav 와 동일한 인터페이스 사용:
  - 목표 발행 : /navigate_to_pose (nav2_msgs/NavigateToPose action)
  - 위치 읽기 : TF  map → base_link

동작:
  WP1(95% 도달 + yaw) → 5초 대기 → WP2 → 5초 대기 → WP3
  → [Enter 입력] → 시작 위치로 복귀

도달 판정 = 목표까지 남은거리가 ARRIVE_RADIUS(절대값, m) 이하가 되면 goal 취소.
            (이동거리가 1~2m로 짧아서 % 방식 대신 절대 반경 사용)
            ※ ARRIVE_RADIUS 가 Nav2 의 xy_goal_tolerance 보다 작거나 같으면
              Nav2 가 먼저 도착 처리하므로 조기취소는 안 걸리고 Nav2 기본 도착으로 동작.
              조기취소를 실제로 걸려면 ARRIVE_RADIUS > Nav2 xy_goal_tolerance 로.
yaw       = 95%에서 끊으면 Nav2의 마지막 yaw 정렬 단계에 못 가므로,
            도착 후 현재 위치에서 목표 yaw 로 제자리 회전을 따로 수행.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
import threading
import time
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import Quaternion
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⬇⬇⬇  직접 입력 칸  ⬇⬇⬇
#   토픽/RViz 에서 받은 pose 를 그대로 복붙: (position.x, position.y,
#   orientation.z, orientation.w). yaw 는 z/w 에서 자동 계산됨.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#          position.x       , position.y         , orientation.z       , orientation.w
# 시작 위치 (복귀 목표)
START = (  -0.563839852809906, 6.40927791595459,  0.9999952719813889, 0.0030750633925066564)   
# 웨이포인트 1
WP1   = (1.414407730102539, 6.038581371307373,-0.7449109108205111, 0.6671639490714082 )  
 # 웨이포인트 2
WP2   = (   2.588763475418091 ,2.1107046604156494 ,-0.594049615164282,0.8044284024841294)  
 # 웨이포인트 3
WP3   = ( 3.654364585876465 , 2.416710376739502  ,0.31248656053922613, 0.9499221807508047 )  


ARRIVE_RADIUS     = 0.20   # 목표 도달 판정 절대 반경(m). 거리에 무관. 짧은 이동(1~2m)용.
WAIT_SEC          = 12.0    # 웨이포인트 사이 대기 시간(초)
ALIGN_YAW         = True   # 도착 후 yaw 정렬 여부
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 세부 파라미터 (보통 안 건드려도 됨)
YAW_TOL_DEG       = 6.0    # yaw 정렬 허용 오차(도)
YAW_TIMEOUT       = 8.0    # yaw 정렬 최대 대기(초)

NAV_ACTION = "/navigate_to_pose"
MAP_FRAME  = "map"
BASE_FRAME = "base_link"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def quaternion_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2)
    q.w = math.cos(yaw / 2)
    return q

def yaw_from_quaternion(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y),
                      1 - 2 * (q.y * q.y + q.z * q.z))

def yaw_from_zw(qz: float, qw: float) -> float:
    """orientation.z, orientation.w → yaw(rad). (x=y=0 인 평면 회전 가정)"""
    return 2.0 * math.atan2(qz, qw)

def normalize_angle(a: float) -> float:
    while a > math.pi:  a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 노드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SimpleWaypointNav(Node):

    def __init__(self):
        super().__init__("simple_waypoint_nav")
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav_client  = ActionClient(self, NavigateToPose, NAV_ACTION)

    # ── 기본 헬퍼 ────────────────────────────────────────────
    def get_pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                MAP_FRAME, BASE_FRAME, rclpy.time.Time())
            return (tf.transform.translation.x,
                    tf.transform.translation.y,
                    yaw_from_quaternion(tf.transform.rotation))
        except TransformException:
            return None

    def _wait_future(self, fut, timeout: float) -> bool:
        t0 = time.time()
        while not fut.done():
            if time.time() - t0 > timeout:
                return False
            time.sleep(0.02)
        return True

    def _send_goal(self, x: float, y: float, yaw: float):
        """goal 전송 후 accept 된 goal_handle 반환 (실패 시 None)."""
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = MAP_FRAME
        goal.pose.header.stamp    = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation = quaternion_from_yaw(float(yaw))

        fut = self.nav_client.send_goal_async(goal)
        if not self._wait_future(fut, timeout=5.0):
            self.get_logger().error("goal 응답 타임아웃")
            return None
        gh = fut.result()
        if gh is None or not gh.accepted:
            self.get_logger().error("goal REJECTED")
            return None
        return gh

    def _cancel(self, gh, result_fut=None):
        """현재 goal 취소하고 잠깐 정리될 때까지 대기."""
        try:
            cf = gh.cancel_goal_async()
            self._wait_future(cf, timeout=2.0)
        except Exception:
            pass
        if result_fut is not None:
            t0 = time.time()
            while not result_fut.done() and time.time() - t0 < 1.5:
                time.sleep(0.05)

    # ── 핵심: 95% 도달 주행 ──────────────────────────────────
    def go_to_pose(self, x: float, y: float, qz: float, qw: float, label: str,
                   align_yaw: bool = ALIGN_YAW):
        """(x, y, orientation.z, orientation.w) 형식 웨이포인트로 이동."""
        return self.go_to(x, y, yaw_from_zw(qz, qw), label, align_yaw=align_yaw)

    def go_to(self, x: float, y: float, yaw: float, label: str,
              align_yaw: bool = ALIGN_YAW):
        pose = self._wait_pose()
        if pose is None:
            self.get_logger().error(f"[{label}] 로봇 위치(TF)를 못 읽음")
            return False
        rx, ry, _ = pose
        init_dist     = math.hypot(x - rx, y - ry)
        arrive_radius = ARRIVE_RADIUS

        self.get_logger().info(
            f"[{label}] → ({x:.2f}, {y:.2f}) yaw={yaw:.3f}rad | "
            f"초기거리={init_dist:.2f}m, 도달반경={arrive_radius:.2f}m")

        # 시작부터 도달반경 안이면 위치 이동 생략 (가까운 웨이포인트 보호)
        if init_dist <= arrive_radius:
            self.get_logger().warn(
                f"[{label}] 시작부터 도달반경 안 (init={init_dist:.2f}m) → 위치 이동 생략")
            if align_yaw:
                self._align_yaw(yaw, label)
            return True

        gh = self._send_goal(x, y, yaw)
        if gh is None:
            return False
        result_fut = gh.get_result_async()

        # ── 위치 단계: 도달 반경 감시 ──
        while True:
            if result_fut.done():
                self.get_logger().info(f"[{label}] Nav2가 먼저 종료(목표 도달).")
                break
            pose = self.get_pose()
            if pose is not None:
                d = math.hypot(x - pose[0], y - pose[1])
                if d <= arrive_radius:
                    self.get_logger().info(
                        f"[{label}] 도달반경 진입 "
                        f"(남은거리={d:.2f}m ≤ {arrive_radius:.2f}m) → goal 취소")
                    self._cancel(gh, result_fut)
                    break
            time.sleep(0.1)

        # ── yaw 단계: 제자리 회전으로 목표 각도 정렬 ──
        if align_yaw:
            self._align_yaw(yaw, label)
        return True

    def _align_yaw(self, yaw: float, label: str):
        pose = self.get_pose()
        if pose is None:
            return
        rx, ry, cur = pose
        if abs(normalize_angle(yaw - cur)) < math.radians(YAW_TOL_DEG):
            return  # 이미 충분히 맞음

        self.get_logger().info(f"[{label}] yaw 정렬 → {math.degrees(yaw):.0f}°")
        gh = self._send_goal(rx, ry, yaw)   # 현재 위치 그대로, yaw만 목표
        if gh is None:
            return
        result_fut = gh.get_result_async()

        t0 = time.time()
        while time.time() - t0 < YAW_TIMEOUT:
            if result_fut.done():
                break
            pose = self.get_pose()
            if pose is not None and \
               abs(normalize_angle(yaw - pose[2])) < math.radians(YAW_TOL_DEG):
                self._cancel(gh, result_fut)
                self.get_logger().info(f"[{label}] yaw 정렬 완료")
                return
            time.sleep(0.1)
        self._cancel(gh, result_fut)
        self.get_logger().warn(f"[{label}] yaw 정렬 타임아웃")

    # ── 준비 대기 ────────────────────────────────────────────
    def _wait_pose(self, timeout: float = 10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            pose = self.get_pose()
            if pose is not None:
                return pose
            time.sleep(0.1)
        return None

    def wait_ready(self):
        self.get_logger().info("Nav2 action server 대기 중...")
        while not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("  ...아직 안 뜸")
        self.get_logger().info("TF(map→base_link) 대기 중...")
        if self._wait_pose(timeout=15.0) is None:
            self.get_logger().warn("TF 못 받음 — 그래도 진행은 함")
        self.get_logger().info("준비 완료.")

    # ── 미션 (순차 실행) ─────────────────────────────────────
    def run_mission(self):
        self.wait_ready()
        self.get_logger().info("===== 미션 시작 =====")

        self.go_to_pose(*WP1, "WP1")
        self.get_logger().info(f"{WAIT_SEC:.0f}초 대기...")
        time.sleep(WAIT_SEC)

        self.go_to_pose(*WP2, "WP2")
        self.get_logger().info(f"{WAIT_SEC:.0f}초 대기...")
        time.sleep(WAIT_SEC)

        self.go_to_pose(*WP3, "WP3")


        try:
            input("\n[WP3 도착] Enter 를 누르면 시작 위치로 복귀합니다... ")
        except EOFError:
            pass

        self.go_to_pose(*START, "START(복귀)")
        self.get_logger().info("===== 미션 종료 =====")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    rclpy.init()
    node = SimpleWaypointNav()

    # 콜백/TF/액션 처리를 위해 executor 는 백그라운드에서 spin,
    # 미션(순차 + Enter 입력)은 메인 스레드에서 실행.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.run_mission()
    except KeyboardInterrupt:
        node.get_logger().info("중단됨.")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
