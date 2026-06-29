#!/usr/bin/env python3
import cv2
import numpy as np
import os
import time

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from ultralytics import YOLO

from message_filters import Subscriber, ApproximateTimeSynchronizer


# YOLOv8-pose keypoint index
KP_NOSE           = 0
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6

# skeleton 연결 쌍
SKELETON_PAIRS = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

# 스냅샷 저장할 방향
SNAPSHOT_DIRS = ['front', 'left', 'right']

# 방향 판정 안정화: 몇 프레임 연속으로 같은 방향이어야 저장
STABLE_FRAMES_REQUIRED = 15  # 15 = 약 0.5초

# 같은 방향 재저장 방지
SAVE_COOLDOWN_SEC = 3.0


class YoloPoseDistanceNode(Node):
    def __init__(self):
        super().__init__('yolo_pose_distance_node')

        self.bridge = CvBridge()

        # -------------------------
        # parameters
        # -------------------------
        self.declare_parameter('pose_model_path',      'yolov8n-pose.pt')
        self.declare_parameter('det_model_path',       'yolov8n.pt')
        self.declare_parameter('color_topic',          '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic',          '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('debug_image_topic',    '/yolo_pose/debug_image')
        self.declare_parameter('detection_text_topic', '/yolo_pose/detection_text')
        self.declare_parameter('conf_threshold',       0.4)
        self.declare_parameter('sync_queue_size',      10)
        self.declare_parameter('sync_slop',            0.1)
        self.declare_parameter('snapshot_dir',         '~/victim_pictures')

        pose_model_path      = self.get_parameter('pose_model_path').get_parameter_value().string_value
        det_model_path       = self.get_parameter('det_model_path').get_parameter_value().string_value
        color_topic          = self.get_parameter('color_topic').get_parameter_value().string_value
        depth_topic          = self.get_parameter('depth_topic').get_parameter_value().string_value
        debug_image_topic    = self.get_parameter('debug_image_topic').get_parameter_value().string_value
        detection_text_topic = self.get_parameter('detection_text_topic').get_parameter_value().string_value
        self.conf_threshold  = self.get_parameter('conf_threshold').get_parameter_value().double_value
        sync_queue_size      = self.get_parameter('sync_queue_size').get_parameter_value().integer_value
        sync_slop            = self.get_parameter('sync_slop').get_parameter_value().double_value
        self.snapshot_dir    = os.path.expanduser(
            self.get_parameter('snapshot_dir').get_parameter_value().string_value
        )

        # -------------------------
        # models
        # -------------------------
        self.pose_model = YOLO(pose_model_path)
        self.det_model  = YOLO(det_model_path)

        # -------------------------
        # publishers
        # -------------------------
        self.debug_image_pub  = self.create_publisher(Image,        debug_image_topic,           10)
        self.text_pub         = self.create_publisher(String,       detection_text_topic,         10)
        # 추가: 사람 3D 위치 (매 프레임 publish)
        self.position_pub     = self.create_publisher(PointStamped, '/yolo_pose/person_position', 10)
        # 추가: 처음 발견 방향 (한 번만 publish)
        self.first_facing_pub = self.create_publisher(String,       '/yolo_pose/first_facing',    10)

        # -------------------------
        # subscribers
        # -------------------------
        self.color_sub = Subscriber(self, Image, color_topic)
        self.depth_sub = Subscriber(self, Image, depth_topic)

        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=sync_queue_size,
            slop=sync_slop
        )
        self.sync.registerCallback(self.synced_callback)

        # -------------------------
        # 스냅샷 상태 관리
        # track_id → {
        #   'saved'        : {front, left, right} → bool
        #   'stable_count' : {front, left, right} → int
        #   'last_facing'  : str
        #   'first_facing' : str  ← 처음 발견 방향 (한 번만 기록)
        #   'position'     : (x, y, z) ← 처음 발견 3D 위치 (한 번만 기록)
        # }
        # -------------------------
        self.person_states = {}

        self.get_logger().info('YoloPoseDistanceNode started.')
        self.get_logger().info(f'  pose_model  : {pose_model_path}')
        self.get_logger().info(f'  det_model   : {det_model_path}')
        self.get_logger().info(f'  snapshot_dir: {self.snapshot_dir}')

    # ------------------------------------------------------------------
    # 상태 초기화
    # ------------------------------------------------------------------
    def init_person_state(self, track_id):
        self.person_states[track_id] = {
            'saved':         {d: False for d in SNAPSHOT_DIRS},
            'stable_count':  {d: 0     for d in SNAPSHOT_DIRS},
            'last_facing':   'unknown',
            'first_facing':  None,   # 처음 발견 방향 (아직 미결정)
            'position':      None,   # 처음 발견 3D 위치 (아직 미결정)
        }

    # ------------------------------------------------------------------
    # 3D 위치 계산
    # bbox 중심 픽셀 + depth → 카메라 기준 3D 좌표 (m)
    # RealSense D435i @ 1280x720 기본 내부 파라미터 사용
    # ------------------------------------------------------------------
    def compute_3d_position(self, depth_frame, x1, y1, x2, y2):
        dist = self.compute_depth_median(depth_frame, x1, y1, x2, y2)
        if dist is None:
            return None

        cx_px = (x1 + x2) / 2.0
        cy_px = (y1 + y2) / 2.0

        # RealSense D435i @ 1280x720 기본값
        fx  = 921.0
        fy  = 921.0
        ppx = 640.0
        ppy = 360.0

        x = (cx_px - ppx) / fx * dist
        y = (cy_px - ppy) / fy * dist
        z = dist  # 전방 거리

        return (x, y, z)

    # ------------------------------------------------------------------
    # 깊이 median 계산
    # ------------------------------------------------------------------
    def compute_depth_from_keypoints(self, depth_frame, keypoints, conf_thresh=0.3):
    #visible 키포인트 위치에서 depth 샘플링 → median 반환
        h, w = depth_frame.shape[:2]
        depths = []

        for kp in keypoints:
            x, y, conf = int(kp[0]), int(kp[1]), float(kp[2])
            if conf < conf_thresh:
                continue

            # 키포인트 주변 5x5 패치
            x1 = max(0, x - 5)
            x2 = min(w, x + 5)
            y1 = max(0, y - 5)
            y2 = min(h, y + 5)

            patch = depth_frame[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            if depth_frame.dtype == np.uint16:
                valid = patch[(patch > 0) & (patch < 10000)]
                if valid.size:
                    depths.append(float(np.median(valid)) / 1000.0)

            elif depth_frame.dtype in (np.float32, np.float64):
                valid = patch[np.isfinite(patch) & (patch > 0) & (patch < 10.0)]
                if valid.size:
                    depths.append(float(np.median(valid)))

        if not depths:
            return None

        # 가장 가까운 값들의 median (벽보다 사람이 더 가까움)
        depths.sort()
        closest = depths[:max(1, len(depths) // 2)]  # 가까운 절반만 사용
        return float(np.median(closest))
    # 중앙기준 거리 추정 
    def compute_depth_median(self, depth_frame, x1, y1, x2, y2):
        h, w = depth_frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
    
        if x2 <= x1 or y2 <= y1:
            return None
    
        bw, bh = x2 - x1, y2 - y1
        roi = depth_frame[
            int(y1 + bh * 0.20):int(y1 + bh * 0.80),
            int(x1 + bw * 0.30):int(x1 + bw * 0.70)
        ]
        if roi.size == 0:
            return None
    
        if depth_frame.dtype == np.uint16:
            valid = roi[(roi > 0) & (roi < 10000)]
            return float(np.median(valid)) / 1000.0 if valid.size else None
    
        if depth_frame.dtype in (np.float32, np.float64):
            valid = roi[np.isfinite(roi) & (roi > 0.0) & (roi < 10.0)]
            return float(np.median(valid)) if valid.size else None
    
        return None

    # ------------------------------------------------------------------
    # 방향 추정
    # ------------------------------------------------------------------
    def estimate_facing(self, keypoints):
        kp_conf_thresh = 0.3

        nose  = keypoints[0]
        l_eye = keypoints[1]
        r_eye = keypoints[2]
        l_ear = keypoints[3]
        r_ear = keypoints[4]
        l_sh  = keypoints[5]
        r_sh  = keypoints[6]
        l_hip = keypoints[11]   
        r_hip = keypoints[12]   
        l_kne = keypoints[13]   
        r_kne = keypoints[14]   
        l_ank = keypoints[15]   
        r_ank = keypoints[16]   

        def vis(kp):
            return kp[2] > kp_conf_thresh

        # 1. 어깨 둘 다 보임
        if vis(l_sh) and vis(r_sh):
            shoulder_mid_x = (l_sh[0] + r_sh[0]) / 2.0
            shoulder_width = abs(r_sh[0] - l_sh[0])
            front_margin   = shoulder_width * 0.20
            if vis(nose):
                offset = nose[0] - shoulder_mid_x
                if abs(offset) < front_margin:
                    return 'front'
                return 'left' if offset < 0 else 'right'
            return 'side' if shoulder_width < 40 else 'front'

        # 2. 어깨 한쪽만 보임
        if vis(l_sh) and not vis(r_sh):
            return 'right'
        if vis(r_sh) and not vis(l_sh):
            return 'left'

        # 3. 귀로 판단
        if vis(l_ear) and not vis(r_ear):
            return 'right'
        if vis(r_ear) and not vis(l_ear):
            return 'left'
        if vis(l_ear) and vis(r_ear):
            return 'front'

        # 4. 눈으로 판단
        if vis(l_eye) and not vis(r_eye):
            return 'right'
        if vis(r_eye) and not vis(l_eye):
            return 'left'
        if vis(l_eye) and vis(r_eye):
            return 'front'
        # 4. 엉덩이 ← 추가
        if vis(l_hip) and vis(r_hip):
            hip_mid_x  = (l_hip[0] + r_hip[0]) / 2.0
            hip_width  = abs(r_hip[0] - l_hip[0])
            if hip_width < 40:
                return 'side'
            return 'front'
        if vis(l_hip) and not vis(r_hip):
            return 'right'
        if vis(r_hip) and not vis(l_hip):
            return 'left'

        # 5. 무릎 ← 추가
        if vis(l_kne) and vis(r_kne):
            knee_width = abs(r_kne[0] - l_kne[0])
            return 'side' if knee_width < 40 else 'front'
        if vis(l_kne) and not vis(r_kne):
            return 'right'
        if vis(r_kne) and not vis(l_kne):
            return 'left'

        # 6. 발목 ← 추가
        if vis(l_ank) and vis(r_ank):
            ank_width = abs(r_ank[0] - l_ank[0])
            return 'side' if ank_width < 40 else 'front'
        if vis(l_ank) and not vis(r_ank):
            return 'right'
        if vis(r_ank) and not vis(l_ank):
            return 'left'

        return 'unknown'
        


    # ------------------------------------------------------------------
    # 스냅샷 트리거 판단
    # ------------------------------------------------------------------
    def try_snapshot(self, track_id, facing, frame, x1, y1, x2, y2, dist_str):
        if facing not in SNAPSHOT_DIRS:
            for d in SNAPSHOT_DIRS:
                self.person_states[track_id]['stable_count'][d] = 0
            return

        state = self.person_states[track_id]

        # 3장 완료됐으면 스킵
        if all(state['saved'][d] for d in SNAPSHOT_DIRS):
            return

        # 이 방향 이미 저장했으면 스킵
        if state['saved'][facing]:
            return
        # ── 거리 조건: 1.5m ± 0.2m 범위일 때만 저장 ──────────────────
        try:
            dist_val = float(dist_str.replace('m', ''))
        except ValueError:
            return  # N/A 이면 스킵

        if not (1.3 <= dist_val <= 1.7):
        # 거리 조건 안맞으면 카운트 리셋
            state['stable_count'][facing] = 0
            return
        # 방향 바뀌면 다른 방향 카운트 리셋
        if state['last_facing'] != facing:
            for d in SNAPSHOT_DIRS:
                if d != facing:
                    state['stable_count'][d] = 0
        state['last_facing'] = facing

        # 안정화 카운트 증가
        state['stable_count'][facing] += 1

        # N프레임 연속 같은 방향이면 저장
        if state['stable_count'][facing] >= STABLE_FRAMES_REQUIRED:
            self.save_snapshot(track_id, facing, frame, x1, y1, x2, y2, dist_str)
            state['saved'][facing]        = True
            state['stable_count'][facing] = 0

            if all(state['saved'][d] for d in SNAPSHOT_DIRS):
                self.get_logger().info(
                    f'[COMPLETE] ID:{track_id} 3장 수집 완료! '
                    f'폴더: {os.path.join(self.snapshot_dir, f"ID_{track_id}")}'
                )

    # ------------------------------------------------------------------
    # 스냅샷 저장
    # ------------------------------------------------------------------
    def save_snapshot(self, track_id, facing, frame, x1, y1, x2, y2, dist_str):
        folder = os.path.join(self.snapshot_dir, f'ID_{track_id}')
        os.makedirs(folder, exist_ok=True)

        timestamp = time.strftime('%H%M%S')
        filename  = f'{facing}_{timestamp}_{dist_str.replace(".", "_")}.jpg'
        filepath  = os.path.join(folder, filename)

        cv2.imwrite(filepath, frame)   # ← crop 제거, 전체 프레임 저장
        self.get_logger().info(f'[SNAPSHOT] ID:{track_id} {facing} → {filepath}')
        return filepath

    # ------------------------------------------------------------------
    # 텍스트 라벨
    # ------------------------------------------------------------------
    def paint_label(self, image, x1, y1, label, color):
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        text_y = max(y1 - 10, th + 5)
        cv2.rectangle(image,
                      (x1, text_y - th - 6),
                      (x1 + tw + 6, text_y + baseline - 6),
                      color, -1)
        cv2.putText(image, label, (x1 + 3, text_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # ------------------------------------------------------------------
    # skeleton
    # ------------------------------------------------------------------
    def draw_skeleton(self, image, keypoints, color=(0, 255, 255)):
        kp_conf_thresh = 0.3
        for kp in keypoints:
            x, y, conf = int(kp[0]), int(kp[1]), kp[2]
            if conf > kp_conf_thresh:
                cv2.circle(image, (x, y), 4, color, -1)
        for i, j in SKELETON_PAIRS:
            xi, yi, ci = keypoints[i]
            xj, yj, cj = keypoints[j]
            if ci > kp_conf_thresh and cj > kp_conf_thresh:
                cv2.line(image, (int(xi), int(yi)), (int(xj), int(yj)), color, 2)

    # ------------------------------------------------------------------
    # 저장 상태 HUD
    # ------------------------------------------------------------------
    def draw_snapshot_status(self, image, track_id, x1, y2):
        if track_id not in self.person_states:
            return
        saved      = self.person_states[track_id]['saved']
        icons      = {'front': 'F', 'left': 'L', 'right': 'R'}
        status_str = ''
        for d, icon in icons.items():
            status_str += f'[{icon}✓]' if saved[d] else f'[{icon} ]'
        cv2.putText(image, status_str, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)

    # ------------------------------------------------------------------
    # 메인 콜백
    # ------------------------------------------------------------------
    def synced_callback(self, color_msg, depth_msg):
        try:
            color_frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as exc:
            self.get_logger().error(f'Image conversion failed: {exc}')
            return

        output          = color_frame.copy()
        save_frame      = color_frame.copy()
        detection_lines = []
        person_count    = 0
        bottle_count    = 0

        # ── 1. 사람: pose + tracking ────────────────────────────────────
        try:
            pose_results = self.pose_model.track(
                color_frame,
                classes=[0],
                conf=self.conf_threshold,
                persist=True,
                verbose=False
            )
        except Exception as exc:
            self.get_logger().error(f'Pose inference failed: {exc}')
            pose_results = []

        if pose_results and pose_results[0].boxes is not None:
            boxes     = pose_results[0].boxes
            keypoints = pose_results[0].keypoints

            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf     = float(box.conf[0].item())
                track_id = int(box.id[0].item()) if box.id is not None else -1
                person_count += 1

                # 새 ID 초기화
                if track_id != -1 and track_id not in self.person_states:
                    self.init_person_state(track_id)

                #기존  거리 + 3D 위치
                dist     = self.compute_depth_median(depth_frame, x1, y1, x2, y2)

                
                position = self.compute_3d_position(depth_frame, x1, y1, x2, y2)

                # 방향
                kps    = keypoints.data[idx].cpu().numpy()
                facing = self.estimate_facing(kps)

                # 수정 - 키포인트 기반 우선, 실패시 bbox fallback
                dist = self.compute_depth_from_keypoints(depth_frame, kps)
                if dist is None:
                    dist = self.compute_depth_median(depth_frame, x1, y1, x2, y2)

                dist_str = f'{dist:.2f}m' if dist is not None else 'N/A'

                color = (255, 80, 80)                                          # ← color 먼저 정의
                cv2.rectangle(save_frame, (x1, y1), (x2, y2), color, 2)       # ← 박스 그리기

                if track_id != -1:
                    state = self.person_states[track_id]

                    # ── 처음 발견 3D 위치 기록 (한 번만) ─────────────
                    if state['position'] is None and position is not None:
                        state['position'] = position
                        self.get_logger().info(
                            f'[NEW] ID:{track_id} 처음 발견 위치 → '
                            f'x={position[0]:.2f} y={position[1]:.2f} z={position[2]:.2f}m'
                        )

                    # ── 처음 발견 방향 기록 + publish (한 번만) ───────
                    if state['first_facing'] is None and facing in SNAPSHOT_DIRS:
                        state['first_facing'] = facing
                        self.get_logger().info(
                            f'[NEW] ID:{track_id} 처음 발견 방향 → {facing}'
                        )
                        msg      = String()
                        msg.data = f'id={track_id},facing={facing}'
                        self.first_facing_pub.publish(msg)

                    # ── 3D 위치 매 프레임 publish ─────────────────────
                    if position is not None:
                        pt          = PointStamped()
                        pt.header   = color_msg.header
                        pt.point.x  = position[0]
                        pt.point.y  = position[1]
                        pt.point.z  = position[2]
                        self.position_pub.publish(pt)

                    # ── 스냅샷 트리거 ──────────────────────────────────
                    self.try_snapshot(
                        track_id, facing, save_frame,
                        x1, y1, x2, y2, dist_str
                    )

                # 시각화
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                self.draw_skeleton(output, kps)
                self.draw_snapshot_status(output, track_id, x1, y2)
                self.draw_snapshot_status(save_frame, track_id, x1, y2)

                first_facing = self.person_states[track_id]['first_facing'] \
                    if track_id in self.person_states else '?'

                label = f'ID:{track_id} {conf:.2f} {dist_str} [{facing}] 1st:{first_facing}'
                self.paint_label(output, x1, y1, label, color)
                self.paint_label(save_frame, x1, y1, label, color)

                detection_lines.append(
                    f'person, id={track_id}, conf={conf:.2f}, dist={dist_str}, '
                    f'facing={facing}, first_facing={first_facing}, '
                    f'box=({x1},{y1},{x2},{y2})'
                )

        # ── 2. 물병 ────────────────────────────────────────────────────
        try:
            det_results = self.det_model(
                color_frame,
                classes=[39],
                conf=self.conf_threshold,
                verbose=False
            )
        except Exception as exc:
            self.get_logger().error(f'Det inference failed: {exc}')
            det_results = []

        if det_results and det_results[0].boxes is not None:
            for box in det_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf     = float(box.conf[0].item())
                bottle_count += 1

                dist     = self.compute_depth_median(depth_frame, x1, y1, x2, y2)
                dist_str = f'{dist:.2f}m' if dist is not None else 'N/A'

                color = (0, 165, 255)
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                self.paint_label(output, x1, y1, f'bottle {conf:.2f} {dist_str}', color)
                detection_lines.append(
                    f'bottle, conf={conf:.2f}, dist={dist_str}, box=({x1},{y1},{x2},{y2})'
                )

        # ── 3. HUD ─────────────────────────────────────────────────────
        cv2.putText(output, f'Persons: {person_count}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(output, f'Bottles: {bottle_count}', (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

        # ── 4. imshow ──────────────────────────────────────────────────
        cv2.imshow('YOLO Pose', output)
        cv2.waitKey(1)

        # ── 5. publish ─────────────────────────────────────────────────
        try:
            debug_msg        = self.bridge.cv2_to_imgmsg(output, encoding='bgr8')
            debug_msg.header = color_msg.header
            self.debug_image_pub.publish(debug_msg)
        except Exception as exc:
            self.get_logger().error(f'Debug image publish failed: {exc}')

        text_msg      = String()
        text_msg.data = '\n'.join(detection_lines) if detection_lines else 'no detections'
        self.text_pub.publish(text_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloPoseDistanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()