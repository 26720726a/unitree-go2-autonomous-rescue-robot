#!/usr/bin/env python3
"""
vlfm_source_nav_v20.py  –  VLFM 빛 발원지 탐색 v20
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v19 기반. 변경사항: PASSED 모드 + LiDAR depth fallback.

[PASSED 모드 (지나침)]
  조건: 현재(Panel 2) 안 보임 + 과거(Panel 1 BRIGHTEST) 보임.
  처리: BRIGHTEST 시점에 저장해둔 (image+depth+scan+pose)로 목표 계산.
        VLM이 Panel 1 픽셀 (u,v) 반환 → 그 시점 카메라 각도 역산 →
        그 위치/방향 기준으로 목표 산출.
  → 절대좌표 불필요. 과거 로봇 기준 상대 계산만.
  → 그 방향으로 가면 다시 보여서 DIRECT로 정밀 접근.

[LiDAR depth fallback]
  발광체는 시뮬에서 depth가 NaN으로 깨짐 (실제 환경은 원통+LED라 정상).
  거리 결정: depth ROI median → 무효면 cam_angle 방향 LiDAR 거리(±2빔 median).
  DIRECT: 현재 scan / PASSED: 과거 scan.
  둘 다 무효:
    DIRECT → frontier fallback (탐색 계속)
    PASSED → 발견 지점(_bright_frame_pose)으로 복귀 (그 방향 yaw로)
  ※ 픽셀→각도는 depth 없이 intrinsics(cx,fx)로 항상 계산.

[pixel_to_world 일반화]
  (u,v,pose,image,depth,scan,depth_scale,tag) 공용 함수.
  DIRECT/PASSED가 입력만 바꿔 호출. 좌표 리스케일(버그 A 수정) 유지.

[offset]
  발원지에서 1.5m 앞을 목표로 (투척 거리 고려).

[데이터 수집 (v19 유지)]
  --collect-data: Claude 입력과 동일한 jpg + 정답 라벨(BFS 경로거리) JSON.
  --source-x/y: 정답 라벨 계산 전용 (주행 판단엔 미사용).
  NOTE: source 좌표는 SLAM map frame 기준이어야 함. 로봇을 발원지에 놓고
        `tf2_echo map base_link`로 측정한 값 사용 (sdf world 좌표 아님).

[카메라-LiDAR 정렬 가정]
  둘 다 base_link 기준, 정면=0 으로 가정.
  실제 Go2 장착 오프셋은 --cam-lidar-yaw-offset 로 보정 (현재 0, 확인 필요).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse, base64, json, math, os, queue, threading, time
from collections import deque
from typing import Dict, List, Optional, Tuple

import anthropic
import cv2, numpy as np, rclpy
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

CANDIDATE_LABELS = ["A", "B", "C", "D", "E"]
PANEL_HEIGHT = 400   # 합성 이미지 각 패널 높이 (VLM u,v 좌표 기준)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def quaternion_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2)
    q.w = math.cos(yaw / 2)
    return q

def yaw_from_quaternion(q) -> float:
    return math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

def normalize_angle(a: float) -> float:
    while a >  math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def enhance_image(img: np.ndarray, clip: float = 3.0, grid: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def circular_variance(bearings: List[float]) -> float:
    if len(bearings) < 2:
        return 1.0
    sins = float(np.mean(np.sin(bearings)))
    coss = float(np.mean(np.cos(bearings)))
    return 1.0 - math.sqrt(sins**2 + coss**2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Claude API 워커
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ClaudeWorker:
    MODEL = "claude-sonnet-4-5"

    def __init__(self, api_key: str, timeout: float = 60.0):
        self.client  = anthropic.Anthropic(api_key=api_key)
        self.timeout = timeout
        self._q: queue.Queue = queue.Queue(maxsize=2)
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"[Claude] started model={self.MODEL}")

    def submit(self, composite_img: np.ndarray,
               candidates: List[Dict], cb) -> bool:
        item = (composite_img, candidates, cb)
        try:
            self._q.put_nowait(item)
            return True
        except queue.Full:
            try:
                self._q.get_nowait()
                self._q.put_nowait(item)
                return True
            except Exception:
                return False

    def _enc(self, img: np.ndarray, quality: int = 88) -> str:
        _, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return base64.b64encode(jpg.tobytes()).decode()

    def _call(self, prompt: str, composite_b64: str) -> str:
        msg = self.client.messages.create(
            model=self.MODEL,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64",
                                   "media_type": "image/jpeg",
                                   "data": composite_b64},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return msg.content[0].text.strip().lower()

    def _parse(self, raw: str, labels: List[str]) -> Dict:
        res: Dict = {
            "mode":       "frontier",
            "scores":     {l: 0 for l in labels},
            "best":       labels[0] if labels else "",
            "confidence": "low",
            "u": None, "v": None,
            "raw": raw,
        }
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("mode:"):
                if "direct" in line:
                    res["mode"] = "direct"
                elif "passed" in line:
                    res["mode"] = "passed"
            elif line.startswith("pixel:"):
                try:
                    coords = line.split(":", 1)[1].strip().strip("()").split(",")
                    res["u"] = int(float(coords[0].strip()))
                    res["v"] = int(float(coords[1].strip()))
                except Exception:
                    pass
            elif line.startswith("scores:"):
                for part in line.split(":", 1)[1].split():
                    if "=" in part:
                        k, v = part.split("=", 1)
                        k = k.strip().upper()
                        try:
                            if k in res["scores"]:
                                res["scores"][k] = int(v.strip())
                        except Exception:
                            pass
            elif line.startswith("best:"):
                b = line.split(":", 1)[1].strip().upper()
                if b in res["scores"]:
                    res["best"] = b
            elif line.startswith("confidence:"):
                for c in ["high", "medium", "low"]:
                    if c in line:
                        res["confidence"] = c
                        break
            elif line.startswith("reason:"):
                res["reason"] = line.split(":", 1)[1].strip()
        if res["mode"] == "frontier" and res["scores"]:
            res["best"] = max(res["scores"], key=res["scores"].get)
        return res

    def _loop(self):
        while True:
            cb = None
            try:
                composite_img, candidates, cb = self._q.get(timeout=1.0)
                labels  = [c["label"] for c in candidates]
                prompt  = build_unified_prompt(candidates)
                raw     = self._call(prompt, self._enc(composite_img, quality=90))
                cb(self._parse(raw, labels))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Claude] Error: {e}")
                if cb is not None:
                    try:
                        cb({"mode": "frontier", "scores": {}, "best": "",
                            "confidence": "low", "raw": f"error:{e}"})
                    except Exception:
                        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 통합 프롬프트 (3패널)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_unified_prompt(candidates: List[Dict]) -> str:
    labels    = [c["label"] for c in candidates]
    label_str = "/".join(labels)

    return f"""You are a robot navigation system searching for a red LED light source in a dark room.

You receive ONE composite image with THREE panels side by side:

[Panel 1: BRIGHTEST FRAME] The frame with the most red light captured during the last move.
  Use it to check if the LED source was directly visible while the robot was moving.

[Panel 2: CURRENT CAMERA] The current camera view.

[Panel 3: OCCUPANCY MAP] Top-down map. The robot's heading direction always points UP.
  Legend (use ONLY what you can actually see in the image):
  - Light gray = free space | Near-black = walls | Medium gray = unexplored
  - Yellow/orange arrows = directions where red light was observed earlier.
    Longer + brighter yellow = stronger observation. Arrows pointing/converging the same way
    indicate the likely direction of the real source.
  - Green line/dots = path the robot has ALREADY traveled.
  - Blue filled circle + arrow = robot's current position and the direction it faces;
    its arrow points UP. UP in the map = straight ahead of the robot.
    So the LEFT/RIGHT sides of the map are the robot's own LEFT/RIGHT.
    If the red light appears on the RIGHT side of the CURRENT CAMERA (Panel 2),
    the source is to the robot's RIGHT → prefer a candidate on the RIGHT side of the map.
    Camera left/right and map left/right mean the SAME thing.
  - Colored dots = frontier candidates ({label_str}). Each dot is the candidate
    location; its label is the white letter inside a small black box (with a
    colored border matching the dot) connected to the dot by a short line.
    The box sits just above the dot, or just below it near the top edge.
    Labels are always drawn upright and readable, even after map rotation.
    Colors: A=red, B=magenta, C=cyan, D=orange, E=green.

─── ALWAYS REASON IN THIS SAME ORDER ─────────────────────

STEP 1 — Look at Panel 2 (CURRENT CAMERA): is the red LED SOURCE itself directly visible NOW?
  ✓ "Direct" = a distinct glowing source object/orb is clearly visible.
  ✗ NOT direct = only red glow or reflection spread on walls/floor/ceiling.
  ✗ NOT direct = the room is merely lit red.
  - If directly visible NOW (medium/high confidence):
        → mode: direct, report the pixel (u, v) of the source center
          IN PANEL 2 coordinates (u=0 is the LEFT edge of Panel 2, not the composite).
  - Otherwise → go to STEP 2.

STEP 2 — Look at Panel 1 (BRIGHTEST FRAME): was the source directly visible while moving,
  even though it is NOT visible in Panel 2 now? (i.e. the robot has PASSED or turned away from it)
  - If the source IS clearly visible in Panel 1 but NOT in Panel 2:
        → mode: passed, report the pixel (u, v) of the source center
          IN PANEL 1 coordinates (u=0 is the LEFT edge of Panel 1).
  - If the source is not clearly visible in Panel 1 either → go to STEP 3.

STEP 3 — Choose exactly ONE frontier candidate ({label_str}) by combining the
  following pieces of information. Each piece MAY be absent; if a piece is absent,
  simply leave it out and judge from the remaining ones. Do NOT apply special-case
  rules — just weigh whatever information is actually present:

    (a) Direction of the red light seen in the CURRENT camera (Panel 2), if any.
    (b) Direction the yellow/orange arrows point or converge toward (accumulated
        observations on the map), if any arrows exist.
    (c) The robot's facing direction — it points UP in the map.
    (d) The green traveled path — avoid candidates that lead back over it.

  Combine the information that is present and pick the single most reasonable
  candidate. For example: when light cues (a)(b) exist they dominate the choice;
  when they are absent the decision naturally rests on (c) the robot's heading and
  (d) staying off the already-traveled path. Always output exactly one candidate.

─── OUTPUT FORMAT ────────────────────────────────────────

If source directly visible NOW (Panel 2):
mode: direct
pixel: (u, v)
confidence: low/medium/high
reason: one sentence

If source visible in Panel 1 (BRIGHTEST) but NOT in Panel 2 now:
mode: passed
pixel: (u, v)
confidence: low/medium/high
reason: one sentence

If source not directly visible in either panel:
mode: frontier
scores: {" ".join(f"{l}=?" for l in labels)}
best: ?
confidence: low/medium/high
reason: one sentence"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 노드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VLFMSourceNavV20(Node):

    def __init__(self, args):
        super().__init__("vlfm_source_nav_v20")
        self.args   = args
        self.bridge = CvBridge()

        # ── 센서 ─────────────────────────────────────────────
        self.latest_image: Optional[np.ndarray]    = None
        self.latest_depth: Optional[np.ndarray]    = None
        self.latest_scan:  Optional[LaserScan]     = None
        self.latest_map:   Optional[OccupancyGrid] = None
        self._occ_cache:   Optional[np.ndarray]    = None

        # 카메라 내부 파라미터
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None
        self.depth_scale = 1.0

        # ── Bearing 관측 이력 ─────────────────────────────────
        self.bearing_obs: deque = deque(maxlen=args.max_obs)
        self._last_obs_t = 0.0

        # ── 이동 이력 / 맵 ───────────────────────────────────
        self.route_history: deque = deque(maxlen=300)
        self.start_pose: Optional[Tuple] = None
        self.coverage_map: Optional[np.ndarray] = None
        self.map_shape:    Optional[Tuple]      = None
        self._last_cov_t = 0.0

        # ── 빛 상태 ──────────────────────────────────────────
        self.last_red:  Dict = {}
        self.last_mode: str  = ""

        # ── 이동 중 밝기 프레임 버퍼 ─────────────────────────
        self._bright_frame:       Optional[np.ndarray] = None
        self._bright_frame_score: float                = -1.0
        self._bright_frame_pose:  Optional[Tuple]      = None
        self._bright_frame_depth: Optional[np.ndarray] = None
        self._bright_frame_scan:  Optional[LaserScan]  = None
        self._last_bright_t:      float                = 0.0

        # ── VLM 트리거 상태 (80% 도달) ───────────────────────
        self._goal_target_pos: Tuple[float, float] = (0.0, 0.0)
        self._goal_init_dist:  float               = 0.0
        self._vlm_triggered:   bool                = False
        self._vlm_redirect_pending: bool           = False

        # ── Claude API 워커 ───────────────────────────────────
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            self.get_logger().warn("ANTHROPIC_API_KEY 없음.")
        self.claude_worker = ClaudeWorker(api_key, args.vlm_timeout)
        self.score_result: Dict = {}
        self._pending      = False
        self._score_last_t = 0.0

        # ── 데이터 수집 ──────────────────────────────────────
        self.source_xy: Optional[Tuple[float, float]] = None
        if args.source_x is not None and args.source_y is not None:
            self.source_xy = (args.source_x, args.source_y)
        self._pending_sample: Optional[Dict] = None
        if args.collect_data:
            os.makedirs(args.dataset_dir, exist_ok=True)
            self.get_logger().info(
                f"[dataset] 수집 ON → {args.dataset_dir} | source={self.source_xy}")

        # ── Nav2 ─────────────────────────────────────────────
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav_client  = ActionClient(self, NavigateToPose, args.nav_action)
        self.nav_busy    = False
        self._pending_goal: Dict  = {}
        self._current_goal_handle = None
        self.visited_goals: List[Dict] = []
        self.failed_goals:  List[Dict] = []

        # ── 루프 ─────────────────────────────────────────────
        self.done       = False
        self.iter_count = 0

        # ── 구독 / 타이머 ────────────────────────────────────
        self.create_subscription(Image,         args.image_topic,       self.image_cb,       10)
        self.create_subscription(Image,         args.depth_topic,       self.depth_cb,       10)
        self.create_subscription(CameraInfo,    args.camera_info_topic, self.camera_info_cb, 10)
        self.create_subscription(OccupancyGrid, args.map_topic,         self.map_cb,         10)
        # LiDAR는 보통 BEST_EFFORT(센서 QoS)로 발행됨 → RELIABLE로 구독하면 수신 0.
        self.create_subscription(LaserScan,     args.scan_topic,        self.scan_cb,
                                 qos_profile_sensor_data)
        self.create_timer(args.period, self.step)
        self.create_timer(1.0, self._nav_monitor_cb)

        os.makedirs(args.debug_dir, exist_ok=True)
        self.get_logger().info("Nav2 활성화 대기 중...")
        while not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("Nav2 대기 중...")
        self.get_logger().info("VLFMSourceNavV20 ready.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 콜백
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def image_cb(self, msg: Image):
        try:
            raw = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image = enhance_image(raw)
        except Exception as e:
            self.get_logger().warn(f"image_cb: {e}")
            return

        if self.latest_map is None:
            return
        self._ensure_maps()

        pose = self.get_robot_pose()
        if pose is None:
            return

        rx, ry, ryaw = pose
        now = time.time()

        red = self.detect_red(self.latest_image)
        self.last_red = red

        if red.get("visible") and (now - self._last_obs_t >= self.args.obs_interval):
            abs_bearing = normalize_angle(ryaw + red["bearing_rad"])
            self.bearing_obs.append({
                "rx": rx, "ry": ry,
                "abs_bearing": abs_bearing,
                "confidence":  red["confidence"],
                "t": now,
            })
            self._last_obs_t = now

        if self.nav_busy and (now - self._last_bright_t >= self.args.bright_capture_interval):
            score = self._red_brightness_score(self.latest_image)
            if score > self._bright_frame_score:
                self._bright_frame       = self.latest_image.copy()
                self._bright_frame_score = score
                self._bright_frame_pose  = (rx, ry, ryaw)
                self._bright_frame_depth = (self.latest_depth.copy()
                                            if self.latest_depth is not None else None)
                self._bright_frame_scan  = self.latest_scan
            self._last_bright_t = now

        if now - self._last_cov_t >= 0.5:
            self.update_coverage(pose)
            self._last_cov_t = now

        if self.start_pose is None:
            self.start_pose = pose
        if (not self.route_history or
                math.hypot(rx - self.route_history[-1][0],
                           ry - self.route_history[-1][1]) > 0.1):
            self.route_history.append((rx, ry))

    def depth_cb(self, msg: Image):
        try:
            enc = msg.encoding.lower()   # ROS 인코딩은 대문자(16UC1/32FC1) → 대소문자 무시
            if enc == "32fc1":
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
                self.depth_scale  = 1.0
            elif enc in ("16uc1", "mono16"):
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "16UC1").astype(np.float32)
                self.depth_scale  = 0.001
            else:
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg).astype(np.float32)
                self.depth_scale  = 1.0
        except Exception as e:
            self.get_logger().warn(f"depth_cb: {e}")

    def scan_cb(self, msg: LaserScan):
        self.latest_scan = msg

    def _lidar_range_at(self, cam_angle: float,
                        scan: Optional[LaserScan]) -> Optional[float]:
        if scan is None:
            return None
        ang = cam_angle + self.args.cam_lidar_yaw_offset
        if ang < scan.angle_min or ang > scan.angle_max:
            return None
        idx = int(round((ang - scan.angle_min) / scan.angle_increment))
        n = len(scan.ranges)
        if not (0 <= idx < n):
            return None
        w = self.args.lidar_window
        vals = []
        for i in range(idx - w, idx + w + 1):
            if 0 <= i < n:
                r = scan.ranges[i]
                if r is not None and not math.isnan(r) and not math.isinf(r):
                    if scan.range_min <= r <= scan.range_max:
                        vals.append(float(r))
        if not vals:
            return None
        return float(np.median(vals))

    def camera_info_cb(self, msg: CameraInfo):
        if self.fx is not None:
            return
        self.fx, self.fy = msg.k[0], msg.k[4]
        self.cx, self.cy = msg.k[2], msg.k[5]
        self.get_logger().info(
            f"[CameraInfo] fx={self.fx:.1f} fy={self.fy:.1f} "
            f"cx={self.cx:.1f} cy={self.cy:.1f}")

    def map_cb(self, msg: OccupancyGrid):
        new_shape = (msg.info.height, msg.info.width)

        if self.map_shape is not None and new_shape != self.map_shape:
            self.coverage_map = np.zeros(new_shape, np.uint8)
            self.get_logger().info(f"Map resized: {self.map_shape} → {new_shape}")
        elif self.map_shape is None:
            self.coverage_map = np.zeros(new_shape, np.uint8)

        self.latest_map = msg
        self._occ_cache = np.array(msg.data, dtype=np.int16).reshape(new_shape)
        self.map_shape  = new_shape

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 좌표 변환
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_robot_pose(self) -> Optional[Tuple]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.args.map_frame, self.args.base_frame, rclpy.time.Time())
            return (
                tf.transform.translation.x,
                tf.transform.translation.y,
                yaw_from_quaternion(tf.transform.rotation),
            )
        except TransformException as e:
            self.get_logger().warn(f"TF: {e}")
            return None

    def _sample_depth_roi(self, depth_img: np.ndarray, u_d: int, v_d: int,
                          half: int = 4) -> Tuple[Optional[float], Dict]:
        dh, dw = depth_img.shape[:2]
        x0, x1 = max(0, u_d - half), min(dw, u_d + half + 1)
        y0, y1 = max(0, v_d - half), min(dh, v_d + half + 1)
        roi = depth_img[y0:y1, x0:x1].astype(np.float32)
        valid = roi[(roi > 0) & ~np.isnan(roi) & ~np.isinf(roi)]
        total = roi.size if roi.size > 0 else 1
        ratio = len(valid) / total
        stats = {
            "median": float(np.median(valid)) if len(valid) else 0.0,
            "valid_ratio": round(ratio, 2),
            "n_valid": int(len(valid)),
        }
        if ratio < self.args.depth_min_valid_ratio or len(valid) == 0:
            return None, stats
        return float(np.median(valid)), stats

    def pixel_to_world(self, u: int, v: int, pose: Tuple,
                       image: np.ndarray, depth: Optional[np.ndarray],
                       scan: Optional[LaserScan],
                       depth_scale: float,
                       tag: str = "DIRECT"
                       ) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
        if image is None or self.fx is None:
            return None, None, None, "none"

        ih, iw = image.shape[:2]
        s_panel = ih / float(PANEL_HEIGHT)
        u_img   = u * s_panel
        v_img   = v * s_panel

        u_rgb_for_angle = float(np.clip(u_img, 0, iw - 1))
        x_over_z   = (u_rgb_for_angle - self.cx) / self.fx
        cam_angle  = math.atan2(-x_over_z, 1.0)
        rx, ry, ryaw = pose
        world_yaw  = normalize_angle(ryaw + cam_angle)

        dist_m = None
        source = "none"
        if depth is not None:
            dh, dw = depth.shape[:2]
            u_d = int(np.clip(round(u_img * dw / iw), 0, dw - 1))
            v_d = int(np.clip(round(v_img * dh / ih), 0, dh - 1))
            draw, stats = self._sample_depth_roi(depth, u_d, v_d,
                                                 half=self.args.depth_roi_half)
            if draw is not None:
                dm = draw * depth_scale
                if 0.1 <= dm <= self.args.max_depth:
                    dist_m, source = dm, "depth"
            self.get_logger().info(
                f"[{tag}] panel_uv=({u},{v}) depth_uv=({u_d},{v_d}) "
                f"ROI {stats} → depth_dist={dist_m}")

        if dist_m is None:
            ld = self._lidar_range_at(cam_angle, scan)
            if ld is not None and 0.1 <= ld <= self.args.max_depth:
                dist_m, source = ld, "lidar"
                self.get_logger().info(
                    f"[{tag}] depth invalid → LiDAR dist={ld:.2f}m "
                    f"@cam_angle={math.degrees(cam_angle):.1f}°")

        if dist_m is None:
            self.get_logger().warn(f"[{tag}] depth & LiDAR both invalid")
            return None, None, None, "none"

        offset      = self.args.direct_goal_offset
        min_forward = self.args.direct_min_forward
        effective_dist = max(min_forward, dist_m - offset)
        wx = rx + effective_dist * math.cos(world_yaw)
        wy = ry + effective_dist * math.sin(world_yaw)
        self.get_logger().info(
            f"[{tag}] dist={dist_m:.2f}m({source}) → goal_dist={effective_dist:.2f}m "
            f"yaw={math.degrees(world_yaw):.1f}°")
        return wx, wy, dist_m, source

    def w2g(self, wx: float, wy: float) -> Tuple[int, int]:
        g = self.latest_map
        r = g.info.resolution
        return (int((wx - g.info.origin.position.x) / r),
                int((wy - g.info.origin.position.y) / r))

    def g2w(self, mx: int, my: int) -> Tuple[float, float]:
        g = self.latest_map
        r = g.info.resolution
        return (g.info.origin.position.x + (mx + 0.5) * r,
                g.info.origin.position.y + (my + 0.5) * r)

    def inside(self, mx: int, my: int) -> bool:
        g = self.latest_map
        return 0 <= mx < g.info.width and 0 <= my < g.info.height

    def get_occ(self) -> np.ndarray:
        if self._occ_cache is not None:
            return self._occ_cache
        g = self.latest_map
        return np.array(g.data, dtype=np.int16).reshape((g.info.height, g.info.width))

    def _ensure_maps(self):
        if self.coverage_map is None and self.latest_map is not None:
            g = self.latest_map
            shape = (g.info.height, g.info.width)
            self.coverage_map = np.zeros(shape, np.uint8)
            self.map_shape    = shape

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HSV 탐지
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def detect_red(self, img: np.ndarray) -> Dict:
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s, v = self.args.red_min_s, self.args.red_min_v
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0,   s, v]), np.array([15,  255, 255])),
            cv2.inRange(hsv, np.array([160, s, v]), np.array([180, 255, 255])))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        null = {"visible": False, "whole_frame": False, "confidence": 0.0}
        if not cnts:
            return null
        c    = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self.args.min_red_area:
            return null
        h, w = img.shape[:2]
        m    = cv2.moments(c)
        if m["m00"] == 0:
            return null
        cx_val  = int(m["m10"] / m["m00"])
        cy_val  = int(m["m01"] / m["m00"])
        area_r  = area / float(w * h)
        whole_frame = area_r > self.args.whole_frame_ratio
        conf   = float(np.clip(0.3 + 0.7 * min(area_r / 0.05, 1.0), 0.0, 1.0))
        norm_x = (cx_val - w / 2.0) / (w / 2.0)
        brad   = -norm_x * math.radians(self.args.camera_fov_deg / 2)
        return {
            "visible": True, "whole_frame": whole_frame,
            "cx": cx_val, "cy": cy_val,
            "confidence": conf,
            "bearing_rad": brad,
            "bearing_deg": math.degrees(brad),
        }

    def _red_brightness_score(self, img: np.ndarray) -> float:
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s, v = self.args.red_min_s, self.args.red_min_v
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0,   s, v]), np.array([15,  255, 255])),
            cv2.inRange(hsv, np.array([160, s, v]), np.array([180, 255, 255])))
        return float(mask.sum())

    def update_coverage(self, pose: Tuple):
        if self.coverage_map is None:
            return
        g    = self.latest_map
        res  = g.info.resolution
        x, y, yaw = pose
        fov  = math.radians(self.args.camera_fov_deg)
        rng  = self.args.coverage_range
        rpx  = int(np.clip((x - g.info.origin.position.x) / res, 0, g.info.width-1))
        rpy  = int(np.clip((y - g.info.origin.position.y) / res, 0, g.info.height-1))
        pts  = [[rpx, rpy]]
        for a in np.linspace(yaw - fov/2, yaw + fov/2, 12):
            ex  = x + rng * math.cos(a)
            ey  = y + rng * math.sin(a)
            epx = int(np.clip((ex - g.info.origin.position.x) / res, 0, g.info.width-1))
            epy = int(np.clip((ey - g.info.origin.position.y) / res, 0, g.info.height-1))
            pts.append([epx, epy])
        cv2.fillPoly(self.coverage_map, [np.array(pts, np.int32)], 255)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 합성 이미지 빌더 (3패널)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _build_composite_image(self, current_cam: np.ndarray,
                                map_img: np.ndarray) -> np.ndarray:
        H = PANEL_HEIGHT

        def resize_to_h(img: np.ndarray, h: int) -> np.ndarray:
            ih, iw = img.shape[:2]
            scale  = h / ih
            return cv2.resize(img, (max(1, int(iw * scale)), h),
                              interpolation=cv2.INTER_LINEAR)

        bright = self._bright_frame if self._bright_frame is not None \
                 else np.full((current_cam.shape[0], current_cam.shape[1], 3),
                              30, np.uint8)

        p1 = resize_to_h(bright,      H)
        p2 = resize_to_h(current_cam, H)
        p3 = resize_to_h(map_img,     H)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(p1, "BRIGHTEST FRAME", (4, 22), font, 0.55, (180, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(p2, "CURRENT CAMERA",  (4, 22), font, 0.55, (180, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(p3, "MAP (UP=ROBOT)",  (4, 22), font, 0.55, (180, 255, 180), 1, cv2.LINE_AA)

        sep = np.full((H, 4, 3), 60, np.uint8)
        return np.hstack([p1, sep, p2, sep, p3])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 맵 렌더링
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _render_nav_map(self, occ: np.ndarray, pose: Tuple,
                        candidates: List[Dict]) -> np.ndarray:
        g   = self.latest_map
        res = g.info.resolution
        ox  = g.info.origin.position.x
        oy  = g.info.origin.position.y
        h, w = occ.shape
        rx, ry, ryaw = pose

        img = np.zeros((h, w, 3), np.uint8)
        img[occ == -1] = (80,  80,  80)
        img[occ ==  0] = (210, 210, 210)
        img[occ >  50] = (25,  25,  25)
        img = cv2.flip(img, 0)

        def w2p(wx: float, wy: float) -> Tuple[int, int]:
            px = int(np.clip((wx - ox) / res, 0, w - 1))
            py = int(np.clip(h - 1 - (wy - oy) / res, 0, h - 1))
            return px, py

        MIN_ARROW_M, MAX_ARROW_M = 0.5, 5.0
        MIN_SEP_PX,  MAX_ARROWS  = 15, 60
        drawn: List[Tuple[int, int]] = []
        for obs in sorted(self.bearing_obs, key=lambda o: o["t"], reverse=True):
            if len(drawn) >= MAX_ARROWS:
                break
            opx, opy = w2p(obs["rx"], obs["ry"])
            if any(math.hypot(opx-dx, opy-dy) < MIN_SEP_PX for dx, dy in drawn):
                continue
            drawn.append((opx, opy))
            conf     = obs["confidence"]
            arrow_m  = MIN_ARROW_M + (MAX_ARROW_M - MIN_ARROW_M) * conf
            ewx      = obs["rx"] + arrow_m * math.cos(obs["abs_bearing"])
            ewy      = obs["ry"] + arrow_m * math.sin(obs["abs_bearing"])
            epx, epy = w2p(ewx, ewy)
            g_ch     = int(140 + 115 * conf)
            col      = (0, g_ch, 255)
            thick    = max(1, round(1 + conf * 2))
            cv2.arrowedLine(img, (opx, opy), (epx, epy),
                            col, thick, cv2.LINE_AA, 0, 0.25)
            cv2.circle(img, (opx, opy), 3, col, -1)

        if len(self.route_history) >= 2:
            pts    = [w2p(wx, wy) for wx, wy in self.route_history]
            stride = max(1, len(pts) // 20)
            for i in range(1, len(pts)):
                cv2.line(img, pts[i-1], pts[i], (0, 230, 0), 3, cv2.LINE_AA)
            for pt in pts[::stride]:
                cv2.circle(img, pt, 3, (0, 230, 0), -1)

        rpx, rpy = w2p(rx, ry)
        ax = int(rpx + 24 * math.cos(ryaw))
        ay = int(rpy - 24 * math.sin(ryaw))
        cv2.circle(img, (rpx, rpy), 9, (220, 50, 0), -1)
        cv2.arrowedLine(img, (rpx, rpy), (ax, ay),
                        (255, 80, 0), 2, cv2.LINE_AA, 0, 0.4)

        CAND_COL = {
            "A": (0, 0, 255),
            "B": (255, 0, 255),
            "C": (255, 200, 0),
            "D": (0, 165, 255),
            "E": (0, 220, 0),
        }
        for cand in candidates:
            cpx, cpy = w2p(cand["wx"], cand["wy"])
            col      = CAND_COL.get(cand["label"], (200, 200, 200))
            cv2.circle(img, (cpx, cpy), 4, col, -1)
            cand["_px"] = (cpx, cpy)

        mode_col = ((0, 255, 100) if self.last_mode in ("DIRECT", "PASSED", "PASSED_RET")
                    else (0, 180, 255))
        cv2.putText(img, self.last_mode, (4, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_col, 1, cv2.LINE_AA)

        img, scale, x_off, y_off = self._resize_map(img, size=500)
        img, M, pad_x, pad_y     = self._rotate_to_heading(img, ryaw)

        # 라벨: 점과 겹치지 않게 박스로 띄워서 표시.
        #   검은 배경 박스 + 흰 글자 + 후보색 테두리 → 맵 축소(다운스케일)
        #   후에도 가독성 유지. 점-박스 사이는 짧은 연결선으로 연결.
        H_img, W_img = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fscale, fthick = 1.5, 3
        GAP = 16   # 점과 라벨 박스 사이 간격(px)
        PAD = 7    # 박스 안쪽 여백(px)
        for cand in candidates:
            cpx, cpy = cand["_px"]
            fx, fy   = self._apply_transforms(cpx, cpy, scale, x_off, y_off,
                                              M, pad_x, pad_y)
            label    = cand["label"]
            col      = CAND_COL.get(label, (200, 200, 200))
            cv2.circle(img, (fx, fy), 5, col, -1)
            cv2.circle(img, (fx, fy), 5, (0, 0, 0), 1, cv2.LINE_AA)

            (tw, th), _ = cv2.getTextSize(label, font, fscale, fthick)
            box_w, box_h = tw + 2 * PAD, th + 2 * PAD

            # 기본은 점 위쪽, 위 공간이 부족하면 아래쪽에 배치
            box_cx = int(np.clip(fx, box_w // 2, W_img - box_w // 2))
            up_cy   = fy - 5 - GAP - box_h // 2
            down_cy = fy + 5 + GAP + box_h // 2
            box_cy  = up_cy if up_cy - box_h // 2 >= 0 else down_cy
            box_cy  = int(np.clip(box_cy, box_h // 2, H_img - box_h // 2))

            x0, y0 = box_cx - box_w // 2, box_cy - box_h // 2
            x1, y1 = box_cx + box_w // 2, box_cy + box_h // 2

            # 연결선 (점 → 박스 가장자리). 박스가 위/아래 어느 쪽이든 자연스럽게.
            edge_y = y1 if box_cy < fy else y0
            cv2.line(img, (fx, fy), (box_cx, edge_y), col, 1, cv2.LINE_AA)

            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), -1)
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 2, cv2.LINE_AA)
            tx = box_cx - tw // 2
            ty = box_cy + th // 2
            cv2.putText(img, label, (tx, ty), font, fscale,
                        (255, 255, 255), fthick, cv2.LINE_AA)
        return img

    def _rotate_to_heading(self, img: np.ndarray, ryaw: float):
        h, w  = img.shape[:2]
        diag  = int(math.ceil(math.sqrt(h * h + w * w)))
        pad_y = (diag - h) // 2
        pad_x = (diag - w) // 2

        padded = np.full((diag, diag, 3), 60, np.uint8)
        padded[pad_y:pad_y + h, pad_x:pad_x + w] = img

        angle_deg = math.degrees(math.pi / 2 - ryaw)
        center    = (diag // 2, diag // 2)
        M         = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        rotated   = cv2.warpAffine(padded, M, (diag, diag),
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(60, 60, 60))
        return rotated, M, pad_x, pad_y

    @staticmethod
    def _apply_transforms(px: float, py: float,
                          scale: float, x_off: int, y_off: int,
                          M: np.ndarray, pad_x: int, pad_y: int) -> Tuple[int, int]:
        rx_ = px * scale + x_off
        ry_ = py * scale + y_off
        px_ = rx_ + pad_x
        py_ = ry_ + pad_y
        fx = M[0, 0] * px_ + M[0, 1] * py_ + M[0, 2]
        fy = M[1, 0] * px_ + M[1, 1] * py_ + M[1, 2]
        return int(round(fx)), int(round(fy))

    def _resize_map(self, img: np.ndarray, size: int = 500):
        h, w  = img.shape[:2]
        scale = size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        canvas  = np.full((size, size, 3), 60, np.uint8)
        y_off   = (size - new_h) // 2
        x_off   = (size - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas, scale, x_off, y_off

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Frontier / 후보 선택
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def extract_frontiers(self) -> List[Tuple[float, float]]:
        occ  = self.get_occ()
        g    = self.latest_map
        res  = g.info.resolution
        free = (occ == 0).astype(np.uint8) * 255
        unk  = (occ == -1).astype(np.uint8) * 255
        wall = (occ > 50).astype(np.uint8) * 255
        k    = np.ones((3, 3), np.uint8)
        fm   = cv2.bitwise_and(cv2.dilate(free, k), unk)
        mp   = max(1, int(0.15 / res))
        fm   = cv2.bitwise_and(
            fm, cv2.bitwise_not(cv2.dilate(wall, k, iterations=mp)))

        # ── free 영역 면적 라벨 (후보 주변 영역 support 측정용) ──────
        #    맵/영역은 건드리지 않는다. 각 frontier 가 닿아 있는 free
        #    영역의 면적을 '측정'해서, 작은 조각에만 걸친 후보(회랑 밖
        #    노이즈 등)를 후보 단계에서 걸러내기만 한다.
        _, free_lbl, free_stats, _ = cv2.connectedComponentsWithStats(
            free, connectivity=8)
        free_area     = free_stats[:, cv2.CC_STAT_AREA]      # idx 0 = background
        min_free_cell = self.args.frontier_min_free_area / (res * res)
        pr            = self.args.frontier_free_probe_radius

        n, _, stats, centroids = cv2.connectedComponentsWithStats(fm, connectivity=8)
        H, W      = free.shape
        pose      = self.get_robot_pose()
        frontiers = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 5:
                continue
            gx, gy = int(centroids[i][0]), int(centroids[i][1])

            # ── 주변 free 영역 support 필터 ──────────────────────
            #    centroid 이웃에 닿는 free 영역 중 가장 큰 면적이
            #    임계 미만이면 '주변 영역 없는 후보'로 보고 제외.
            y0, y1 = max(0, gy - pr), min(H, gy + pr + 1)
            x0, x1 = max(0, gx - pr), min(W, gx + pr + 1)
            labs   = np.unique(free_lbl[y0:y1, x0:x1])
            labs   = labs[labs > 0]
            if labs.size == 0 or free_area[labs].max() < min_free_cell:
                continue

            wx, wy = self.g2w(gx, gy)
            if pose:
                rx, ry, _ = pose
                d = math.hypot(wx - rx, wy - ry)
                if not (self.args.frontier_min_dist <= d <= self.args.frontier_max_dist):
                    continue
            frontiers.append((wx, wy))
        return frontiers

    def select_candidates(self, frontiers: List[Tuple], robot_pose: Tuple,
                          max_n: int = 5) -> List[Dict]:
        if not frontiers:
            return []
        rx, ry, _ = robot_pose

        raw = []
        for fx, fy in frontiers:
            angle = math.atan2(fy - ry, fx - rx)
            dist  = math.hypot(fx - rx, fy - ry)
            raw.append({"wx": fx, "wy": fy, "dist": dist, "angle": angle})

        sector_size = 2 * math.pi / max_n
        buckets: List[List[Dict]] = [[] for _ in range(max_n)]
        for c in raw:
            idx = int((c["angle"] + math.pi) / sector_size) % max_n
            buckets[idx].append(c)

        selected: List[Dict] = []
        for bucket in buckets:
            if bucket:
                best = min(bucket, key=lambda x: x["dist"])
                selected.append(best)

        if len(selected) < max_n:
            chosen_set = {(c["wx"], c["wy"]) for c in selected}
            remaining  = sorted(
                [c for c in raw if (c["wx"], c["wy"]) not in chosen_set],
                key=lambda x: x["dist"])
            min_sep = self.args.candidate_min_separation
            for c in remaining:
                too_close = any(
                    math.hypot(c["wx"] - s["wx"], c["wy"] - s["wy"]) < min_sep
                    for s in selected)
                if not too_close:
                    selected.append(c)
                if len(selected) >= max_n:
                    break

        selected = selected[:max_n]
        for i, s in enumerate(selected):
            s["label"] = CANDIDATE_LABELS[i]
        return selected

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # VLM 결과 처리
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _vlm_cb(self, result: Dict):
        self.score_result = {**result, "timestamp": time.time()}
        self._pending     = False

        mode = result.get("mode", "frontier")
        self.get_logger().info(
            f"[Claude] mode={mode} conf={result.get('confidence','?')} "
            f"reason={result.get('reason','')}")

        if self.args.collect_data and self._pending_sample is not None:
            self._pending_sample["claude_response"] = {
                "mode":       result.get("mode"),
                "best":       result.get("best"),
                "scores":     result.get("scores"),
                "u":          result.get("u"),
                "v":          result.get("v"),
                "confidence": result.get("confidence"),
                "reason":     result.get("reason"),
            }
            sid = self._pending_sample["sample_id"]
            try:
                jpath = os.path.join(self.args.dataset_dir, f"{sid}.json")
                with open(jpath, "w") as f:
                    json.dump(self._pending_sample, f, ensure_ascii=False, indent=2)
                ans = self._pending_sample["answer_label"]
                self.get_logger().info(
                    f"[dataset] saved {sid}.json | answer={ans} "
                    f"claude_best={result.get('best')}")
            except Exception as e:
                self.get_logger().warn(f"[dataset] save fail: {e}")
            self._pending_sample = None

        if self.nav_busy and self._current_goal_handle is not None:
            self._vlm_redirect_pending = True
            self._current_goal_handle.cancel_goal_async()
            self.get_logger().info("[Claude] 결과 도착 → 현재 goal 취소 중")

    def _resolve_goal(self, result: Dict, pose: Tuple) -> Optional[Dict]:
        mode = result.get("mode", "frontier")

        if mode == "direct":
            u, v = result.get("u"), result.get("v")
            if u is not None and v is not None:
                wx, wy, dist, src = self.pixel_to_world(
                    u, v, pose,
                    image=self.latest_image, depth=self.latest_depth,
                    scan=self.latest_scan, depth_scale=self.depth_scale,
                    tag="DIRECT")
                if wx is not None:
                    mx, my = self.w2g(wx, wy)
                    occ_val = self.get_occ()[my, mx] if self.inside(mx, my) else 0
                    if occ_val <= 50:
                        self.last_mode = "DIRECT"
                        # 광원까지 투척 거리 이내로 접근 완료 → 최종 종료
                        arrival = (self.args.direct_goal_offset +
                                   self.args.direct_arrival_margin)
                        if dist is not None and dist <= arrival:
                            self.done = True
                            self.get_logger().info(
                                f"=== DONE: source_reached "
                                f"(direct, dist={dist:.2f}m ≤ {arrival:.2f}m) ===")
                            return None
                        return {"x": wx, "y": wy,
                                "label": f"direct(d={dist:.1f}m,{src})"}
                    self.get_logger().warn("[DIRECT] goal hits wall → frontier fallback")

        elif mode == "passed":
            u, v = result.get("u"), result.get("v")
            if (u is not None and v is not None and
                    self._bright_frame is not None and
                    self._bright_frame_pose is not None):
                wx, wy, dist, src = self.pixel_to_world(
                    u, v, self._bright_frame_pose,
                    image=self._bright_frame, depth=self._bright_frame_depth,
                    scan=self._bright_frame_scan, depth_scale=self.depth_scale,
                    tag="PASSED")
                if wx is not None:
                    mx, my = self.w2g(wx, wy)
                    occ_val = self.get_occ()[my, mx] if self.inside(mx, my) else 0
                    if occ_val <= 50:
                        self.last_mode = "PASSED"
                        return {"x": wx, "y": wy,
                                "label": f"passed(d={dist:.1f}m,{src})"}
                    self.get_logger().warn("[PASSED] goal hits wall → return-to-spot")
                bx, by, byaw = self._bright_frame_pose
                self.last_mode = "PASSED_RET"
                self.get_logger().info(
                    f"[PASSED] sensors invalid → return to bright pose "
                    f"({bx:.2f},{by:.2f}) yaw={math.degrees(byaw):.1f}°")
                return {"x": bx, "y": by, "yaw": byaw, "label": "passed_return"}
            else:
                self.get_logger().warn(
                    "[PASSED] no bright frame/pixel → frontier fallback")

        frontiers  = self.extract_frontiers()
        candidates = self.select_candidates(frontiers, pose, self.args.max_candidates)
        best_label = result.get("best", "")
        best_cand  = next((c for c in candidates if c["label"] == best_label), None)
        if best_cand:
            self.last_mode = "VLFM"
            return {"x": best_cand["wx"], "y": best_cand["wy"],
                    "label": f"frontier_{best_label}"}

        self.get_logger().warn(
            f"[resolve_goal] best_label='{best_label}' not in new candidates → retry")
        return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 종료 조건
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def check_termination(self, pose: Tuple) -> Tuple[bool, str]:
        # 종료는 오직 DIRECT(광원 직접 발견 + 투척 거리까지 접근 완료) 시에만.
        #   → _resolve_goal 의 direct 분기에서 self.done 을 직접 설정한다.
        #   기존 source_confirmed(밝기+bearing) 휴리스틱은 오탐이 많아 제거.
        # 여기서는 안전장치로 최대 반복 횟수만 확인.
        if self.iter_count >= self.args.max_iters:
            return True, "max_iters"
        return False, ""

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Nav2
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def send_goal(self, x: float, y: float, label: str, pose: Tuple,
                  yaw: Optional[float] = None):
        rx, ry, ryaw = pose
        goal_yaw = yaw if yaw is not None else math.atan2(y - ry, x - rx)

        self._goal_init_dist  = math.hypot(x - rx, y - ry)
        self._goal_target_pos = (x, y)
        self._vlm_triggered   = False
        self._bright_frame       = None
        self._bright_frame_score = -1.0
        self._bright_frame_pose  = None
        self._bright_frame_depth = None
        self._bright_frame_scan  = None
        self._last_bright_t      = time.time()

        if not self.args.execute:
            self.get_logger().info(
                f"[DRY RUN] → ({x:.2f},{y:.2f}) [{label}] [{self.last_mode}]")
            return
        if not self.nav_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Nav2 unavailable")
            return

        self.nav_busy      = True
        self._pending_goal = {"x": x, "y": y}
        goal               = PoseStamped()
        goal.header.frame_id  = self.args.map_frame
        goal.header.stamp     = self.get_clock().now().to_msg()
        goal.pose.position.x  = float(x)
        goal.pose.position.y  = float(y)
        goal.pose.orientation = quaternion_from_yaw(float(goal_yaw))
        ng      = NavigateToPose.Goal()
        ng.pose = goal
        f       = self.nav_client.send_goal_async(ng)
        f.add_done_callback(self._goal_resp_cb)
        self.get_logger().info(
            f"Nav2 → ({x:.2f},{y:.2f}) [{label}] [{self.last_mode}] "
            f"yaw={math.degrees(goal_yaw):.1f}°")

    def _goal_resp_cb(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error(
                f"[Nav2] Goal REJECTED: ({self._pending_goal.get('x',0):.2f}, "
                f"{self._pending_goal.get('y',0):.2f})")
            self.failed_goals.append(self._pending_goal)
            self.nav_busy = False
            self._current_goal_handle = None
            return
        self.get_logger().info(
            f"[Nav2] Goal ACCEPTED: ({self._pending_goal.get('x',0):.2f}, "
            f"{self._pending_goal.get('y',0):.2f})")
        self._current_goal_handle = gh
        gh.get_result_async().add_done_callback(self._result_cb)

    def _result_cb(self, future):
        try:
            status = future.result().status
        except Exception:
            status = None

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.visited_goals.append(self._pending_goal)
            self.visited_goals = self.visited_goals[-10:]
        elif status != GoalStatus.STATUS_CANCELED:
            self.failed_goals.append(self._pending_goal)
            self.failed_goals = self.failed_goals[-10:]

        self.nav_busy = False
        self._current_goal_handle = None

        if self._vlm_redirect_pending:
            self._vlm_redirect_pending = False
            score = self.score_result
            if score:
                self.score_result = {}
                pose = self.get_robot_pose()
                if pose is not None:
                    new_goal = self._resolve_goal(score, pose)
                    if self.done:
                        self.get_logger().info(
                            "=== DONE: source_reached (direct) ===")
                        return
                    if new_goal is not None:
                        self.get_logger().info(
                            f"[Redirect] → ({new_goal['x']:.2f},{new_goal['y']:.2f}) "
                            f"[{new_goal['label']}]")
                        self.send_goal(new_goal["x"], new_goal["y"],
                                       new_goal["label"], pose,
                                       yaw=new_goal.get("yaw"))
                        self.iter_count += 1
                        return

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 80% 도달 모니터 (1Hz)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _nav_monitor_cb(self):
        if not self.nav_busy or self._vlm_triggered or self._pending or self.done:
            return
        if self._goal_init_dist < 0.3:
            return

        pose = self.get_robot_pose()
        if pose is None:
            return
        rx, ry, _ = pose
        gx, gy    = self._goal_target_pos
        cur_dist  = math.hypot(gx - rx, gy - ry)

        # DIRECT(디텍트) 모드는 광원으로 정밀 접근 중 → 기존대로 진행.
        # 프론티어(VLFM)/PASSED 탐색 모드만 더 일찍 재호출해 동선 효율화.
        if self.last_mode == "DIRECT":
            pct = self.args.approach_pct
        else:
            pct = self.args.approach_pct_explore

        ratio = cur_dist / self._goal_init_dist
        if ratio < (1.0 - pct):
            self.get_logger().info(
                f"[approach {int(pct*100)}%] goal 도달 "
                f"(ratio={ratio:.2f}, mode={self.last_mode or 'frontier'}) → VLM 호출")
            self._trigger_vlm(pose)
            self._vlm_triggered = True

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 데이터 수집
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _bfs_distance_field(self, start_gx: int, start_gy: int) -> Optional[np.ndarray]:
        g   = self.latest_map
        occ = self.get_occ()
        h, w = occ.shape

        if not (0 <= start_gx < w and 0 <= start_gy < h):
            return None

        passable = occ <= 50
        dist = np.full((h, w), np.inf, np.float32)

        from collections import deque as _dq
        if not passable[start_gy, start_gx]:
            found = False
            for r in range(1, 6):
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        nx, ny = start_gx + dx, start_gy + dy
                        if 0 <= nx < w and 0 <= ny < h and passable[ny, nx]:
                            start_gx, start_gy = nx, ny
                            found = True
                            break
                    if found: break
                if found: break
            if not found:
                return None

        dist[start_gy, start_gx] = 0.0
        q = _dq([(start_gx, start_gy)])
        neighbors = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
                     (-1,-1,1.4142),(1,-1,1.4142),(-1,1,1.4142),(1,1,1.4142)]
        while q:
            cx, cy = q.popleft()
            base = dist[cy, cx]
            for dx, dy, cost in neighbors:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and passable[ny, nx]:
                    nd = base + cost
                    if nd < dist[ny, nx]:
                        dist[ny, nx] = nd
                        q.append((nx, ny))
        return dist

    def _compute_answer_label(self, candidates: List[Dict]
                              ) -> Tuple[Optional[str], List[Dict]]:
        out = [dict(c) for c in candidates]
        if self.source_xy is None or self.latest_map is None:
            for c in out:
                c["path_dist"] = None
            return None, out

        g   = self.latest_map
        res = g.info.resolution
        sgx = int((self.source_xy[0] - g.info.origin.position.x) / res)
        sgy = int((self.source_xy[1] - g.info.origin.position.y) / res)

        dist = self._bfs_distance_field(sgx, sgy)
        if dist is None:
            for c in out:
                c["path_dist"] = None
            return None, out

        best_label, best_d = None, np.inf
        for c in out:
            cgx = int((c["wx"] - g.info.origin.position.x) / res)
            cgy = int((c["wy"] - g.info.origin.position.y) / res)
            d = np.inf
            if 0 <= cgx < dist.shape[1] and 0 <= cgy < dist.shape[0]:
                y0, y1 = max(0, cgy-1), min(dist.shape[0], cgy+2)
                x0, x1 = max(0, cgx-1), min(dist.shape[1], cgx+2)
                local = dist[y0:y1, x0:x1]
                finite = local[np.isfinite(local)]
                if len(finite) > 0:
                    d = float(finite.min())
            c["path_dist"] = (round(d * res, 3) if np.isfinite(d) else None)
            if np.isfinite(d) and d < best_d:
                best_d, best_label = d, c["label"]
        return best_label, out

    def _save_sample_image(self, sample_id: str, composite: np.ndarray) -> str:
        path = os.path.join(self.args.dataset_dir, f"{sample_id}.jpg")
        cv2.imwrite(path, composite, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return path

    def _trigger_vlm(self, pose: Tuple):
        if self.latest_image is None or self.latest_map is None:
            return

        occ        = self.get_occ()
        frontiers  = self.extract_frontiers()
        candidates = self.select_candidates(frontiers, pose, self.args.max_candidates)
        if not candidates:
            self.get_logger().warn("[VLM trigger] no frontier candidates")
            return

        nav_map   = self._render_nav_map(occ, pose, candidates)
        composite = self._build_composite_image(self.latest_image, nav_map)

        if self.claude_worker.submit(composite, candidates, self._vlm_cb):
            self._pending      = True
            self._score_last_t = time.time()
            n_obs = len(self.bearing_obs)
            self.get_logger().info(
                f"[VLM] submitted n_obs={n_obs} "
                f"candidates={[c['label'] for c in candidates]}")

            if self.args.collect_data:
                sample_id = f"{int(time.time() * 1000)}"
                answer_label, cands_pd = self._compute_answer_label(candidates)
                img_path = self._save_sample_image(sample_id, composite)
                self._pending_sample = {
                    "sample_id":     sample_id,
                    "image":         os.path.basename(img_path),
                    "answer_label":  answer_label,
                    "candidates":    [
                        {"label": c["label"],
                         "wx": round(c["wx"], 3), "wy": round(c["wy"], 3),
                         "path_dist": c.get("path_dist")}
                        for c in cands_pd
                    ],
                    "source_xy":     list(self.source_xy) if self.source_xy else None,
                    "robot_pose":    [round(pose[0], 3), round(pose[1], 3),
                                      round(pose[2], 4)],
                    "n_observations": n_obs,
                    "trigger_time":  time.time(),
                    "claude_response": None,
                }
                self.get_logger().info(
                    f"[dataset] sample {sample_id} answer={answer_label} "
                    f"path_dists={[(c['label'], c.get('path_dist')) for c in cands_pd]}")

            if self.args.save_debug:
                ts = int(time.time())
                cv2.imwrite(
                    os.path.join(self.args.debug_dir, f"composite_{ts}.jpg"),
                    composite)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 메인 루프
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def step(self):
        if self.done or self.nav_busy:
            return
        if self.latest_image is None or self.latest_map is None:
            self.get_logger().info("Waiting for sensor data...")
            return
        self._ensure_maps()

        pose = self.get_robot_pose()
        if pose is None:
            return

        rx, ry, ryaw = pose

        done, reason = self.check_termination(pose)
        if done:
            self.get_logger().info(f"=== DONE: {reason} ===")
            self.done = True
            return

        score = self.score_result
        s_age = time.time() - score.get("timestamp", 0)
        if score and s_age < self.args.vlm_cache_ttl:
            self.score_result = {}
            new_goal = self._resolve_goal(score, pose)
            if self.done:
                self.get_logger().info("=== DONE: source_reached (direct) ===")
                return
            if new_goal is not None:
                self.send_goal(new_goal["x"], new_goal["y"],
                               new_goal["label"], pose,
                               yaw=new_goal.get("yaw"))
                self.iter_count += 1
                return

        if self._pending:
            waited = time.time() - self._score_last_t
            if waited < self.args.vlm_timeout + 5.0:
                self.get_logger().info(f"[VLM pending {waited:.0f}s]")
                return
            else:
                self._pending = False
                self.get_logger().warn("[VLM] timeout, skip")

        n_obs = len(self.bearing_obs)
        self.get_logger().info(f"[step] n_obs={n_obs} → VLM 호출")
        self._trigger_vlm(pose)
        self.iter_count += 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    p = argparse.ArgumentParser(description="VLFM 빛 발원지 탐색 v20")

    # ── ROS 토픽 (RealSense D435i 기준으로 수정) ─────────────
    p.add_argument("--image-topic",
                   default="/camera/color/image_raw")
    p.add_argument("--depth-topic",
                   default="/camera/depth/image_rect_raw")
    p.add_argument("--camera-info-topic",
                   default="/camera/color/camera_info")
    p.add_argument("--map-topic",         default="/map")
    p.add_argument("--scan-topic",        default="/scan")
    p.add_argument("--nav-action",        default="/navigate_to_pose")
    p.add_argument("--map-frame",         default="map")
    p.add_argument("--base-frame",        default="base_link")

    # Claude API
    p.add_argument("--api-key",       default="",
                   help="Anthropic API key (없으면 ANTHROPIC_API_KEY 환경변수)")
    p.add_argument("--vlm-timeout",   type=float, default=60.0)
    p.add_argument("--vlm-cache-ttl", type=float, default=30.0)

    # VLM 호출 타이밍
    p.add_argument("--approach-pct", type=float, default=0.8,
                   help="DIRECT(디텍트) 모드 재호출 비율. goal 초기거리의 이 비율 도달 시 (0.8=80%%)")
    p.add_argument("--approach-pct-explore", type=float, default=0.4,
                   help="프론티어/PASSED 탐색 모드 재호출 비율 (0.4=40%% 도달 시). DIRECT엔 미적용")

    # DIRECT / PASSED 모드
    p.add_argument("--direct-goal-offset", type=float, default=1.5)
    p.add_argument("--direct-min-forward", type=float, default=0.4)
    p.add_argument("--direct-arrival-margin", type=float, default=0.3,
                   help="광원까지 거리 ≤ (direct_goal_offset+이 값)이면 "
                        "접근 완료로 보고 종료")
    p.add_argument("--max-depth",          type=float, default=20.0)
    p.add_argument("--depth-roi-half",     type=int,   default=4)
    p.add_argument("--depth-min-valid-ratio", type=float, default=0.25)
    p.add_argument("--lidar-window",       type=int,   default=2)
    p.add_argument("--cam-lidar-yaw-offset", type=float, default=0.0)

    # Bearing 관측
    p.add_argument("--obs-interval", type=float, default=1.0)
    p.add_argument("--max-obs",      type=int,   default=200)

    # 루프
    p.add_argument("--period",    type=float, default=1.0)
    p.add_argument("--max-iters", type=int,   default=100)

    # 카메라
    p.add_argument("--camera-fov-deg",    type=float, default=70.0)
    p.add_argument("--coverage-range",    type=float, default=3.0)
    p.add_argument("--whole-frame-ratio", type=float, default=0.30)

    # HSV
    p.add_argument("--red-min-s",    type=int,   default=60)
    p.add_argument("--red-min-v",    type=int,   default=60)
    p.add_argument("--min-red-area", type=float, default=10.0)

    # Frontier / 후보
    p.add_argument("--frontier-min-dist",        type=float, default=0.5)
    p.add_argument("--frontier-max-dist",        type=float, default=8.0)
    p.add_argument("--max-candidates",           type=int,   default=5)
    p.add_argument("--candidate-min-separation", type=float, default=1.5)
    p.add_argument("--frontier-min-free-area",     type=float, default=0.5,
                   help="후보가 접한 free 영역의 최소 면적(m^2). 미만이면 후보에서 제외")
    p.add_argument("--frontier-free-probe-radius", type=int,   default=2,
                   help="centroid 주변 free 라벨 탐색 반경(px). 정상 후보가 누락되면 3으로")

    # 밝기 버퍼
    p.add_argument("--bright-capture-interval", type=float, default=3.0)

    # 데이터 수집
    p.add_argument("--collect-data", action="store_true")
    p.add_argument("--dataset-dir", default=os.path.expanduser("~/vlfm_dataset"))
    p.add_argument("--source-x", type=float, default=None)
    p.add_argument("--source-y", type=float, default=None)

    # 디버그
    p.add_argument("--debug-dir",  default=os.path.expanduser("~/vlfm_v20_debug"))
    p.add_argument("--save-debug", action="store_true")
    p.add_argument("--execute",    action="store_true",
                   help="실제 Nav2 goal 전송 (없으면 DRY RUN)")

    args = p.parse_args()
    rclpy.init()
    node = VLFMSourceNavV20(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
