#!/usr/bin/env python3
"""
vlfm_source_nav_v13.py - VLFM Light Source Navigation v13
Changes from v12:
  - Fix bearing sign bug in detect_red (norm_x sign was inverted vs pixel_to_world)
  - Direct mode bearing fallback when depth unavailable
  - VLM score EMA accumulation for consistent candidate selection
  - Add --step-dist argument (was missing, caused silent crash in direct mode)
Usage: python3 vlfm_source_nav_v13.py --execute
"""

# =============================================================
# CONFIG - Edit only this section
# =============================================================
#
# Check available topics:
#   ros2 topic list | grep -E "image|depth|camera|map"
#
# Go2 + RealSense camera:
#   /camera/color/image_raw        -> IMAGE_TOPIC       : VLM/HSV input (RGB)
#   /camera/depth/image_rect_raw   -> DEPTH_TOPIC       : distance (16UC1, mm)
#   /camera/color/camera_info      -> CAMERA_INFO_TOPIC : fx, fy, cx, cy
#   /map                           -> MAP_TOPIC          : SLAM map
#   /navigate_to_pose              -> NAV_ACTION         : Nav2 action
#   map                            -> MAP_FRAME          : global TF frame
#   base_link                      -> BASE_FRAME         : robot body TF frame
#
# Run: python3 vlfm_source_nav_v12.py --execute --save-debug
# API key: export ANTHROPIC_API_KEY=sk-ant-...

IMAGE_TOPIC       = "/camera/color/image_raw"        # RGB color -> VLM/HSV
DEPTH_TOPIC       = "/camera/depth/image_rect_raw"   # Depth -> distance
CAMERA_INFO_TOPIC = "/camera/color/camera_info"      # fx, fy, cx, cy
MAP_TOPIC         = "/map"
NAV_ACTION        = "/navigate_to_pose"
MAP_FRAME         = "map"
BASE_FRAME        = "base_link"

# =============================================================

import argparse, base64, math, os, queue, threading, time
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
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

CANDIDATE_LABELS = ["A", "B", "C", "D", "E"]


# --------------------------------------------------
# ?? (v10 ??)
# --------------------------------------------------

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
    """
    0.0 = ?? ?? ?? (??? ??? ??)
    1.0 = ?? ??   (??? or ??? ??? ??)
    """
    if len(bearings) < 2:
        return 1.0
    sins = float(np.mean(np.sin(bearings)))
    coss = float(np.mean(np.cos(bearings)))
    return 1.0 - math.sqrt(sins**2 + coss**2)


# --------------------------------------------------
# Claude API ?? (v10 VLMWorker ??)
# --------------------------------------------------

class ClaudeWorker:
    """
    VLFM ?? ??. ???(?) + ?(??) ? ?? ??.
    ????? ????? Claude API ??.
    """

    MODEL = "claude-sonnet-4-5"

    def __init__(self, api_key: str, timeout: float = 60.0):
        self.client  = anthropic.Anthropic(api_key=api_key)
        self.timeout = timeout
        self._q: queue.Queue = queue.Queue(maxsize=2)
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"[Claude] started model={self.MODEL}")

    def submit(self, cam_img: np.ndarray, map_img: np.ndarray,
               candidates: List[Dict], ctx: Dict, cb) -> bool:
        item = (cam_img, map_img, candidates, ctx, cb)
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

    def _call(self, prompt: str, cam_b64: str, map_b64: str) -> str:
        # ???? ASCII ?? ??
        prompt_safe = prompt.encode("utf-8", errors="replace").decode("utf-8")
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
                                   "data": cam_b64},
                    },
                    {
                        "type": "image",
                        "source": {"type": "base64",
                                   "media_type": "image/jpeg",
                                   "data": map_b64},
                    },
                    {"type": "text", "text": prompt_safe},
                ],
            }],
        )
        return msg.content[0].text.strip().lower()

    def _parse(self, raw: str, labels: List[str]) -> Dict:
        """mode:direct ?? mode:frontier ? ?? ??."""
        res: Dict = {
            "mode": "frontier",
            "scores": {l: 0 for l in labels},
            "best": labels[0] if labels else "",
            "confidence": "low",
            "u": None, "v": None,
            "raw": raw,
        }
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("mode:"):
                if "direct" in line:
                    res["mode"] = "direct"
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
                cam_img, map_img, candidates, ctx, cb = self._q.get(timeout=1.0)
                labels = [c["label"] for c in candidates]
                prompt = build_unified_prompt(candidates, ctx)
                # ASCII ?? ? ?? ?? (??? ?? ??)
                prompt = prompt.encode("utf-8").decode("utf-8")
                raw    = self._call(prompt,
                                    self._enc(cam_img, quality=90),
                                    self._enc(map_img, quality=88))
                cb(self._parse(raw, labels))
            except queue.Empty:
                continue
            except Exception as e:#!/usr/bin/env python3
"""
vlfm_source_nav_v13.py - VLFM Light Source Navigation v13
Changes from v12:
  - Fix bearing sign bug in detect_red (norm_x sign was inverted vs pixel_to_world)
  - Direct mode bearing fallback when depth unavailable
  - VLM score EMA accumulation for consistent candidate selection
  - Add --step-dist argument (was missing, caused silent crash in direct mode)
Usage: python3 vlfm_source_nav_v13.py --execute
"""

# =============================================================
# CONFIG - Edit only this section
# =============================================================
#
# Check available topics:
#   ros2 topic list | grep -E "image|depth|camera|map"
#
# Go2 + RealSense camera:
#   /camera/color/image_raw        -> IMAGE_TOPIC       : VLM/HSV input (RGB)
#   /camera/depth/image_rect_raw   -> DEPTH_TOPIC       : distance (16UC1, mm)
#   /camera/color/camera_info      -> CAMERA_INFO_TOPIC : fx, fy, cx, cy
#   /map                           -> MAP_TOPIC          : SLAM map
#   /navigate_to_pose              -> NAV_ACTION         : Nav2 action
#   map                            -> MAP_FRAME          : global TF frame
#   base_link                      -> BASE_FRAME         : robot body TF frame
#
# Run: python3 vlfm_source_nav_v12.py --execute --save-debug
# API key: export ANTHROPIC_API_KEY=sk-ant-...

IMAGE_TOPIC       = "/camera/color/image_raw"        # RGB color -> VLM/HSV
DEPTH_TOPIC       = "/camera/depth/image_rect_raw"   # Depth -> distance
CAMERA_INFO_TOPIC = "/camera/color/camera_info"      # fx, fy, cx, cy
MAP_TOPIC         = "/map"
NAV_ACTION        = "/navigate_to_pose"
MAP_FRAME         = "map"
BASE_FRAME        = "base_link"

# =============================================================

import argparse, base64, math, os, queue, threading, time
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
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

CANDIDATE_LABELS = ["A", "B", "C", "D", "E"]


# --------------------------------------------------
# ?? (v10 ??)
# --------------------------------------------------

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
    """
    0.0 = ?? ?? ?? (??? ??? ??)
    1.0 = ?? ??   (??? or ??? ??? ??)
    """
    if len(bearings) < 2:
        return 1.0
    sins = float(np.mean(np.sin(bearings)))
    coss = float(np.mean(np.cos(bearings)))
    return 1.0 - math.sqrt(sins**2 + coss**2)


# --------------------------------------------------
# Claude API ?? (v10 VLMWorker ??)
# --------------------------------------------------

class ClaudeWorker:
    """
    VLFM ?? ??. ???(?) + ?(??) ? ?? ??.
    ????? ????? Claude API ??.
    """

    MODEL = "claude-sonnet-4-5"

    def __init__(self, api_key: str, timeout: float = 60.0):
        self.client  = anthropic.Anthropic(api_key=api_key)
        self.timeout = timeout
        self._q: queue.Queue = queue.Queue(maxsize=2)
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"[Claude] started model={self.MODEL}")

    def submit(self, cam_img: np.ndarray, map_img: np.ndarray,
               candidates: List[Dict], ctx: Dict, cb) -> bool:
        item = (cam_img, map_img, candidates, ctx, cb)
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

    def _call(self, prompt: str, cam_b64: str, map_b64: str) -> str:
        # ???? ASCII ?? ??
        prompt_safe = prompt.encode("utf-8", errors="replace").decode("utf-8")
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
                                   "data": cam_b64},
                    },
                    {
                        "type": "image",
                        "source": {"type": "base64",
                                   "media_type": "image/jpeg",
                                   "data": map_b64},
                    },
                    {"type": "text", "text": prompt_safe},
                ],
            }],
        )
        return msg.content[0].text.strip().lower()

    def _parse(self, raw: str, labels: List[str]) -> Dict:
        """mode:direct ?? mode:frontier ? ?? ??."""
        res: Dict = {
            "mode": "frontier",
            "scores": {l: 0 for l in labels},
            "best": labels[0] if labels else "",
            "confidence": "low",
            "u": None, "v": None,
            "raw": raw,
        }
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("mode:"):
                if "direct" in line:
                    res["mode"] = "direct"
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
                cam_img, map_img, candidates, ctx, cb = self._q.get(timeout=1.0)
                labels = [c["label"] for c in candidates]
                prompt = build_unified_prompt(candidates, ctx)
                # ASCII ?? ? ?? ?? (??? ?? ??)
                prompt = prompt.encode("utf-8").decode("utf-8")
                raw    = self._call(prompt,
                                    self._enc(cam_img, quality=90),
                                    self._enc(map_img, quality=88))
                cb(self._parse(raw, labels))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Claude] Error: {e}")
                if cb is not None:
                    try:
                        cb({"mode":"frontier","scores":{},"best":"","confidence":"low","raw":f"error:{e}"})
                    except Exception:
                        pass


# --------------------------------------------------
# ?? ???? (Claude? DIRECT/FRONTIER ?? ??)
# --------------------------------------------------

def build_unified_prompt(candidates: List[Dict], ctx: Dict) -> str:
    labels    = [c["label"] for c in candidates]
    label_str = "/".join(labels)
    var       = ctx.get("bearing_variance", 1.0)
    n_obs     = ctx.get("n_observations", 0)

    return f"""You are a robot navigation system searching for a red LED light source in a dark room.
You are given TWO images:

[Image 1: Camera] current robot camera view.
[Image 2: Map] top-down occupancy map with recorded light directions.

MAP legend:
- Light gray=free | Near-black=walls | Medium gray=unknown
- Arrows: recorded light directions. LONGER+YELLOWER = stronger.
  Converging arrows = likely real source. Scattered = reflected light.
- Green=robot route | Blue dot+arrow=robot position/heading
- Circles ({label_str}): navigation candidates

ARROW CONSISTENCY: variance={var:.2f} (0=consistent, 1=scattered) | OBSERVATIONS: {n_obs}

STEP 1 - Look at Image 1 (camera) for horizontal direction ONLY:
  Is the LED directly visible as a distinct glowing point/circle? (not just wall glow)
  YES + confidence medium/high -> mode:direct
  NO (only reflected glow or not visible) -> mode:frontier

  CAMERA DIRECTION RULES (horizontal axis only):
  - Light at pixel u < 30% of width  -> "far left"
  - Light at pixel u < 45% of width  -> "slightly left"
  - Light at pixel u 45~55% of width -> "center"
  - Light at pixel u > 55% of width  -> "slightly right"
  - Light at pixel u > 70% of width  -> "far right"
  NOTE: IMPORTANT: vertical position (up/down) in camera image means NOTHING for navigation.
  NOTE: Do NOT interpret "light at top of image" as "light is to the north on the map".
  NOTE: Only LEFT and RIGHT in the camera image matter for direction.

STEP 2 - If mode:frontier, use Image 2 (map) with camera direction from STEP 1:
  The robot is the blue dot.And blue arrow show the robot's bearing which is same with camera's bearing. Arrows show recorded light directions.
  Match the camera horizontal direction to the map:
  - "far left" in camera    -> pick candidate to the robot's LEFT on map
  - "slightly left"         -> pick candidate slightly left of robot heading
  - "center"                -> pick candidate straight ahead
  - "slightly right"        -> pick candidate slightly right of robot heading
  - "far right"             -> pick candidate to the robot's RIGHT on map
  If no light visible, trust the arrow convergence pattern on the map.
  Avoid candidates in already-explored (green) areas.

Answer in EXACTLY one of these formats:

If LED directly visible:
mode: direct
pixel: (u, v)
confidence: low/medium/high
reason: one sentence

If not directly visible:
mode: frontier
scores: {" ".join(f"{l}=?" for l in labels)}
best: ?
confidence: low/medium/high
reason: one sentence (include camera direction like "slightly left" and which map candidate matches)"""

# --------------------------------------------------
# ?? ??
# --------------------------------------------------

class VLFMSourceNavV13(Node):

    def __init__(self, args):
        super().__init__("vlfm_source_nav_v13")
        self.args   = args
        self.bridge = CvBridge()

        # ?? ?? ?????????????????????????????????????????????
        self.latest_image: Optional[np.ndarray]    = None
        self.latest_depth: Optional[np.ndarray]    = None
        self.latest_map:   Optional[OccupancyGrid] = None
        self._occ_cache:   Optional[np.ndarray]    = None

        # ??? ?? ????
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None
        self.depth_scale = 1.0   # 32FC1=1.0(m), 16UC1=0.001(mm?m)

        # ?? Bearing ?? ?? (v10 ??) ?????????????????????
        self.bearing_obs: deque = deque(maxlen=args.max_obs)
        self._last_obs_t = 0.0

        # ?? ?? ?? / ? ???????????????????????????????????
        self.route_history: deque = deque(maxlen=300)
        self.start_pose: Optional[Tuple] = None
        self.coverage_map: Optional[np.ndarray] = None
        self.map_shape:    Optional[Tuple]      = None
        self._last_cov_t = 0.0

        # ?? ? ?? ??????????????????????????????????????????
        self.last_red: Dict = {}

        # ?? ?? ?? (???) ???????????????????????????????
        self.last_mode = ""   # "DIRECT" or "VLFM"

        # ?? Claude API ?? (VLFM ??) ??????????????????????
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            self.get_logger().warn(
                "ANTHROPIC_API_KEY ??. VLFM ???? Claude ?? ??.")
        self.claude_worker = ClaudeWorker(api_key, args.vlm_timeout)
        self.score_result: Dict = {}
        self.score_accum:  Dict[str, float] = {}  # EMA accumulated scores
        self._pending      = False
        self._score_last_t = 0.0

        # ?? Nav2 ?????????????????????????????????????????????
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav_client  = ActionClient(self, NavigateToPose, args.nav_action)
        self.nav_busy    = False
        self._pending_goal: Dict   = {}
        self._current_goal_handle  = None
        self.visited_goals: List[Dict] = []
        self.failed_goals:  List[Dict] = []

        # ?? ?? ?????????????????????????????????????????????
        self.done       = False
        self.iter_count = 0

        # ?? ?? / ??? ????????????????????????????????????
        self.create_subscription(Image,         args.image_topic,       self.image_cb,       10)
        self.create_subscription(Image,         args.depth_topic,       self.depth_cb,       10)
        self.create_subscription(CameraInfo,    args.camera_info_topic, self.camera_info_cb, 10)
        self.create_subscription(OccupancyGrid, args.map_topic,         self.map_cb,         10)
        self.create_timer(args.period, self.step)

        os.makedirs(args.debug_dir, exist_ok=True)
        # Nav2 ??? ?? (bt_navigator? active ??? ? ???)
        self.get_logger().info("Nav2 ??? ?? ?...")
        while not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("Nav2 ?? ?...")
        self.get_logger().info("VLFMSourceNavV13 ready.")

    # --------------------------------------------------
    # ??
    # --------------------------------------------------

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

        # bearing ?? ??
        if red.get("visible") and (now - self._last_obs_t >= self.args.obs_interval):
            abs_bearing = normalize_angle(ryaw + red["bearing_rad"])
            self.bearing_obs.append({
                "rx":          rx,
                "ry":          ry,
                "abs_bearing": abs_bearing,
                "confidence":  red["confidence"],
                "t":           now,
            })
            self._last_obs_t = now

        # coverage (2Hz)
        if now - self._last_cov_t >= 0.5:
            self.update_coverage(pose)
            self._last_cov_t = now

    def depth_cb(self, msg: Image):
        """Depth ???. 32FC1(m) ?? 16UC1(mm) ?? ??."""
        try:
            if msg.encoding == "32fc1":
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
                self.depth_scale  = 1.0
            elif msg.encoding in ("16uc1", "mono16"):
                self.latest_depth = self.bridge.imgmsg_to_cv2(
                    msg, "16UC1").astype(np.float32)
                self.depth_scale  = 0.001
            else:
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg).astype(np.float32)
                self.depth_scale  = 1.0
        except Exception as e:
            self.get_logger().warn(f"depth_cb: {e}")

    def camera_info_cb(self, msg: CameraInfo):
        """?? ???? 1? ??."""
        if self.fx is not None:
            return
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.get_logger().info(
            f"[CameraInfo] fx={self.fx:.1f} fy={self.fy:.1f} "
            f"cx={self.cx:.1f} cy={self.cy:.1f}")

    def map_cb(self, msg: OccupancyGrid):
        self.latest_map = msg
        self._occ_cache = np.array(
            msg.data, dtype=np.int16
        ).reshape((msg.info.height, msg.info.width))
        shape = (msg.info.height, msg.info.width)
        if self.map_shape != shape:
            self.map_shape    = shape
            self.coverage_map = np.zeros(shape, np.uint8)
            self.get_logger().info(f"Map updated: {shape}")

    # --------------------------------------------------
    # ?? ??
    # --------------------------------------------------

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

    def pixel_to_world(self, u: int, v: int,
                       pose: Tuple) -> Tuple[Optional[float], Optional[float], float]:
        """
        ?? (u,v) + depth ? ?? ?? (wx, wy).
        ??: (wx, wy, depth_m). depth ?? ? (None, None, 0).
        """
        if self.latest_depth is None or self.fx is None:
            return None, None, 0.0

        dh, dw = self.latest_depth.shape[:2]
        u = int(np.clip(u, 0, dw-1))
        v = int(np.clip(v, 0, dh-1))

        # depth ?? (m ??)
        depth_raw = float(self.latest_depth[v, u])
        if depth_raw <= 0 or np.isnan(depth_raw) or np.isinf(depth_raw):
            # 5?5 ?? ?? ??
            patch = self.latest_depth[
                max(0, v-2):min(dh, v+3),
                max(0, u-2):min(dw, u+3)]
            valid = patch[(patch > 0) & ~np.isnan(patch) & ~np.isinf(patch)]
            if len(valid) == 0:
                return None, None, 0.0
            depth_raw = float(np.mean(valid))

        depth_m = depth_raw * self.depth_scale

        # depth ?? ??
        if depth_m < 0.1 or depth_m > self.args.max_depth:
            return None, None, depth_m

        # ??? ?? ? ?? ??? ?? ? ??
        # (???: x=right, y=down, z=forward)
        x_cam = (u - self.cx) * depth_m / self.fx   # ?? ???
        z_cam = depth_m                               # ?? ??

        rx, ry, ryaw = pose
        # ??? ??? ? ?? ??
        cam_angle  = math.atan2(-x_cam, z_cam)
        world_yaw  = normalize_angle(ryaw + cam_angle)
        horiz_dist = math.sqrt(x_cam**2 + z_cam**2)

        wx = rx + horiz_dist * math.cos(world_yaw)
        wy = ry + horiz_dist * math.sin(world_yaw)
        return wx, wy, depth_m

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
        return np.array(g.data, dtype=np.int16).reshape(
            (g.info.height, g.info.width))

    def _ensure_maps(self):
        if self.coverage_map is None and self.latest_map is not None:
            g = self.latest_map
            shape = (g.info.height, g.info.width)
            self.coverage_map = np.zeros(shape, np.uint8)
            self.map_shape    = shape

    # --------------------------------------------------
    # ?? ??
    # --------------------------------------------------

    def should_use_direct(self, red: Dict, pose: Tuple) -> Tuple[bool, str, float, float]:
        """
        DIRECT ?? ?? ?? ??.
        ??: (use_direct, reason, goal_wx, goal_wy)

        DIRECT ?? (?? ????):
          1. ?? ??
          2. whole_frame ??
          3. ?? bearing variance < threshold (?? ??)
          4. depth ?? + ?? ??
        """
        if not red.get("visible"):
            return False, "light_not_visible", 0.0, 0.0

        if red.get("whole_frame"):
            return False, "whole_frame", 0.0, 0.0

        # bearing variance ?? (?? 10?)
        recent = list(self.bearing_obs)[-10:]
        if len(recent) >= 3:
            var = circular_variance([o["abs_bearing"] for o in recent])
            if var > self.args.direct_variance_threshold:
                return False, f"high_variance({var:.2f})", 0.0, 0.0

        # depth ??
        cx = red.get("cx")
        cy = red.get("cy")
        if cx is None or cy is None:
            return False, "no_pixel", 0.0, 0.0

        wx, wy, depth_m = self.pixel_to_world(cx, cy, pose)
        if wx is None:
            return False, f"depth_invalid(d={depth_m:.2f}m)", 0.0, 0.0

        # ? ??: ?? ??? occupied cell??
        mx, my = self.w2g(wx, wy)
        if self.inside(mx, my):
            occ = self.get_occ()
            if occ[my, mx] > 50:
                return False, "depth_hits_wall", 0.0, 0.0

        return True, "ok", wx, wy

    # --------------------------------------------------
    # HSV ?? (v10 ??)
    # --------------------------------------------------

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
        m = cv2.moments(c)
        if m["m00"] == 0:
            return null

        cx     = int(m["m10"] / m["m00"])
        cy_val = int(m["m01"] / m["m00"])
        area_r = area / float(w * h)
        whole_frame = area_r > self.args.whole_frame_ratio
        conf   = float(np.clip(0.3 + 0.7 * min(area_r / 0.05, 1.0), 0.0, 1.0))
        norm_x = (cx - w / 2.0) / (w / 2.0)
        # Negate: camera-left = robot-left = positive bearing (matches pixel_to_world atan2(-x_cam,z_cam))
        brad   = -norm_x * math.radians(self.args.camera_fov_deg / 2)

        return {
            "visible":     True,
            "whole_frame": whole_frame,
            "cx":          cx,
            "cy":          cy_val,
            "confidence":  conf,
            "bearing_rad": brad,
            "bearing_deg": math.degrees(brad),
        }

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

    # --------------------------------------------------
    # ? ??? (v10 ??)
    # --------------------------------------------------

    def _render_nav_map(self, occ: np.ndarray, pose: Tuple,
                        candidates: List[Dict]) -> np.ndarray:
        g   = self.latest_map
        res = g.info.resolution
        ox  = g.info.origin.position.x
        oy  = g.info.origin.position.y
        h, w = occ.shape

        img = np.zeros((h, w, 3), np.uint8)
        img[occ == -1] = (80,  80,  80)
        img[occ == 0 ] = (210, 210, 210)
        img[occ >  50] = (25,  25,  25)
        img = cv2.flip(img, 0)

        rx, ry, ryaw = pose

        def w2p(wx: float, wy: float) -> Tuple[int, int]:
            px = int(np.clip((wx - ox) / res, 0, w-1))
            py = int(np.clip(h-1-(wy - oy) / res, 0, h-1))
            return px, py

        # ?? ?? (??)
        if len(self.route_history) >= 2:
            pts    = [w2p(wx, wy) for wx, wy in self.route_history]
            stride = max(1, len(pts) // 20)
            for i in range(1, len(pts)):
                cv2.line(img, pts[i-1], pts[i], (0, 230, 0), 2, cv2.LINE_AA)
            for pt in pts[::stride]:
                cv2.circle(img, pt, 3, (0, 230, 0), -1)

        # Bearing ??? ???
        MIN_ARROW_M = 0.5
        MAX_ARROW_M = 5.0
        MIN_SEP_PX  = 15
        MAX_ARROWS  = 60
        drawn: List[Tuple[int, int]] = []
        for obs in sorted(self.bearing_obs, key=lambda o: o["t"], reverse=True):
            if len(drawn) >= MAX_ARROWS:
                break
            opx, opy = w2p(obs["rx"], obs["ry"])
            if any(math.hypot(opx-dx, opy-dy) < MIN_SEP_PX for dx, dy in drawn):
                continue
            drawn.append((opx, opy))
            conf    = obs["confidence"]
            arrow_m = MIN_ARROW_M + (MAX_ARROW_M - MIN_ARROW_M) * conf
            ewx     = obs["rx"] + arrow_m * math.cos(obs["abs_bearing"])
            ewy     = obs["ry"] + arrow_m * math.sin(obs["abs_bearing"])
            epx, epy = w2p(ewx, ewy)
            g_ch    = int(140 + 115 * conf)
            col     = (0, g_ch, 255)
            thick   = max(1, round(1 + conf * 2))
            cv2.arrowedLine(img, (opx, opy), (epx, epy),
                            col, thick, cv2.LINE_AA, 0, 0.25)
            cv2.circle(img, (opx, opy), 3, col, -1)

        # ?? ?? + ??
        rpx, rpy = w2p(rx, ry)
        ax = int(rpx + 24 * math.cos(ryaw))
        ay = int(rpy - 24 * math.sin(ryaw))
        cv2.circle(img, (rpx, rpy), 9, (220, 50, 0), -1)
        cv2.arrowedLine(img, (rpx, rpy), (ax, ay),
                        (255, 80, 0), 2, cv2.LINE_AA, 0, 0.4)

        # ?? A~E
        CAND_COL = {
            "A": (0, 0, 255), "B": (255, 0, 220), "C": (255, 220, 0),
            "D": (0, 140, 255), "E": (160, 0, 180),
        }
        for cand in candidates:
            label    = cand["label"]
            cpx, cpy = w2p(cand["wx"], cand["wy"])
            col      = CAND_COL.get(label, (80, 80, 80))
            cv2.circle(img, (cpx, cpy), 15, col, 2)
            cv2.circle(img, (cpx, cpy),  5, col, -1)
            cv2.putText(img, label, (cpx+18, cpy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

        # ?? ??
        mode_col = (0, 255, 100) if self.last_mode == "DIRECT" else (0, 180, 255)
        cv2.putText(img, self.last_mode, (4, h-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_col, 1, cv2.LINE_AA)

        return self._resize_map(img)

    def _resize_map(self, img: np.ndarray, size: int = 600) -> np.ndarray:
        """?? ?? size?size? ????. Claude? ?? ??? ? ? ??."""
        h, w = img.shape[:2]
        # ?? ????? ? ?? size? ??
        scale = size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # ???? ???? ?? ??
        canvas = np.full((size, size, 3), 60, np.uint8)
        y_off  = (size - new_h) // 2
        x_off  = (size - new_w) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        return canvas

    # --------------------------------------------------
    # Frontier / ?? (v10 ??)
    # --------------------------------------------------

    def extract_frontiers(self) -> List[Tuple[float, float]]:
        occ  = self.get_occ()
        g    = self.latest_map
        free = (occ == 0).astype(np.uint8) * 255
        unk  = (occ == -1).astype(np.uint8) * 255
        wall = (occ > 50).astype(np.uint8) * 255
        k    = np.ones((3, 3), np.uint8)
        fm   = cv2.bitwise_and(cv2.dilate(free, k), unk)
        mp   = max(1, int(0.15 / g.info.resolution))
        fm   = cv2.bitwise_and(
            fm, cv2.bitwise_not(cv2.dilate(wall, k, iterations=mp)))
        n, _, stats, centroids = cv2.connectedComponentsWithStats(fm, connectivity=8)
        pose      = self.get_robot_pose()
        frontiers = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 5:
                continue
            wx, wy = self.g2w(int(centroids[i][0]), int(centroids[i][1]))
            if pose:
                rx, ry, _ = pose
                d = math.hypot(wx-rx, wy-ry)
                if not (self.args.frontier_min_dist <= d <= self.args.frontier_max_dist):
                    continue
            frontiers.append((wx, wy))
        return frontiers

    def select_candidates(self, frontiers: List[Tuple], robot_pose: Tuple,
                          max_n: int = 5) -> List[Dict]:
        if not frontiers:
            return []
        rx, ry, _ = robot_pose

        recent = [o for o in self.bearing_obs if time.time() - o["t"] < 60.0]
        mean_bearing: Optional[float] = None
        if recent:
            sins = float(np.mean([math.sin(o["abs_bearing"]) for o in recent]))
            coss = float(np.mean([math.cos(o["abs_bearing"]) for o in recent]))
            mean_bearing = math.atan2(sins, coss)

        scored = []
        for fx, fy in frontiers:
            dist       = math.hypot(fx-rx, fy-ry)
            dist_score = 1.0 - min(dist / self.args.frontier_max_dist, 1.0)
            if mean_bearing is not None:
                angle_to = math.atan2(fy-ry, fx-rx)
                align    = (math.cos(normalize_angle(angle_to - mean_bearing)) + 1.0) / 2.0
            else:
                align = 0.5
            score = 0.5 * align + 0.5 * dist_score
            scored.append({"wx": fx, "wy": fy, "dist": dist, "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        selected, min_sep = [], self.args.candidate_min_separation
        for cand in scored:
            too_close = any(
                math.hypot(cand["wx"]-s["wx"], cand["wy"]-s["wy"]) < min_sep
                for s in selected)
            if not too_close:
                selected.append(cand)
            if len(selected) >= max_n:
                break
        for i, s in enumerate(selected):
            s["label"] = CANDIDATE_LABELS[i]
        return selected

    # --------------------------------------------------
    # VLM ?? / ????
    # --------------------------------------------------

    def _vlm_cb(self, result: Dict):
        self._pending = False
        mode = result.get("mode", "frontier")

        # EMA score accumulation — reduces per-call noise
        if mode == "frontier":
            new_scores = result.get("scores", {})
            if new_scores:
                alpha = 0.4
                if not self.score_accum or set(self.score_accum) != set(new_scores):
                    self.score_accum = dict(new_scores)
                else:
                    for k in new_scores:
                        self.score_accum[k] = (alpha * new_scores[k]
                                               + (1 - alpha) * self.score_accum.get(k, new_scores[k]))
                result = dict(result)
                result["scores"] = {k: round(v, 1) for k, v in self.score_accum.items()}
                result["best"]   = max(self.score_accum, key=self.score_accum.get)

        self.score_result = {**result, "timestamp": time.time()}

        if mode == "direct":
            self.get_logger().info(
                f"[Claude] mode=direct pixel=({result.get('u')},{result.get('v')}) "
                f"conf={result.get('confidence')} reason={result.get('reason','')}")
        else:
            self.get_logger().info(
                f"[Claude] mode=frontier scores={result.get('scores',{})} "
                f"best={result.get('best')} conf={result.get('confidence')} "
                f"reason={result.get('reason','')}")

    def _build_ctx(self, pose: Tuple) -> Dict:
        red = self.last_red
        obs = list(self.bearing_obs)
        var = circular_variance([o["abs_bearing"] for o in obs]) if len(obs) >= 2 else 1.0
        rx, ry, _ = pose
        moved = 0.0
        if self.start_pose:
            sx, sy, _ = self.start_pose
            moved = math.hypot(rx-sx, ry-sy)
        return {
            "light_visible":    red.get("visible", False),
            "whole_frame":      red.get("whole_frame", False),
            "bearing_variance": round(var, 3),
            "n_observations":   len(obs),
            "moved_dist":       round(moved, 2),
            
        }

    # --------------------------------------------------
    # ?? ?? (v10 ??)
    # --------------------------------------------------

    def check_termination(self, pose: Tuple) -> Tuple[bool, str]:
        if self.iter_count >= self.args.max_iters:
            return True, "max_iters"
        obs = list(self.bearing_obs)
        if len(obs) >= 5 and self.last_red.get("whole_frame"):
            var = circular_variance([o["abs_bearing"] for o in obs[-10:]])
            if var < 0.1:
                return True, "source_confirmed"
        return False, ""

    # --------------------------------------------------
    # Nav2
    # --------------------------------------------------

    def send_goal(self, x: float, y: float, label: str, pose: Tuple):
        rx, ry, _ = pose
        goal_yaw  = math.atan2(y-ry, x-rx)
        if not self.args.execute:
            self.get_logger().info(f"[DRY RUN] ? ({x:.2f},{y:.2f}) ({label})")
            return
        if not self.nav_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Nav2 unavailable")
            return
        self.nav_busy      = True
        self._pending_goal = {"x": x, "y": y}
        goal = PoseStamped()
        goal.header.frame_id  = self.args.map_frame
        goal.header.stamp     = self.get_clock().now().to_msg()
        goal.pose.position.x  = float(x)
        goal.pose.position.y  = float(y)
        goal.pose.orientation = quaternion_from_yaw(float(goal_yaw))
        ng      = NavigateToPose.Goal()
        ng.pose = goal
        f = self.nav_client.send_goal_async(ng)
        f.add_done_callback(self._goal_resp_cb)
        self.get_logger().info(f"Nav2 ? ({x:.2f},{y:.2f}) [{label}] [{self.last_mode}]")

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
        else:
            self.failed_goals.append(self._pending_goal)
            self.failed_goals = self.failed_goals[-10:]
        self.nav_busy = False
        self._current_goal_handle = None

    # --------------------------------------------------
    # ?? ??
    # --------------------------------------------------

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
        if self.start_pose is None:
            self.start_pose = pose

        if (not self.route_history or
                math.hypot(rx - self.route_history[-1][0],
                           ry - self.route_history[-1][1]) > 0.1):
            self.route_history.append((rx, ry))

        done, reason = self.check_termination(pose)
        if done:
            self.get_logger().info(f"=== DONE: {reason} ===")
            self.done = True
            return

        img = self.latest_image.copy()

        # -- Claude pending ?? ---------------------------
        if self._pending:
            waited = time.time() - self._score_last_t
            if waited > self.args.vlm_timeout + 5.0:
                self._pending = False
                self.get_logger().warn("[Claude] timeout, skip")
            else:
                self.get_logger().info(
                    f"[Claude pending {waited:.0f}s]")
                return

        # -- ???? ?? ?? -----------------------------
        frontiers  = self.extract_frontiers()
        candidates = self.select_candidates(frontiers, pose, self.args.max_candidates)
        if not candidates:
            self.get_logger().warn("No frontier candidates.")
            return

        occ     = self.get_occ()
        nav_map = self._render_nav_map(occ, pose, candidates)
        ctx     = self._build_ctx(pose)
        now     = time.time()

        # -- Claude ?? (?? 1? ???? ??) -----------
        n_obs = len(self.bearing_obs)
        if (now - self._score_last_t >= self.args.score_interval
                and n_obs >= 1):
            if self.claude_worker.submit(img, nav_map, candidates, ctx, self._vlm_cb):
                self._score_last_t = now
                self._pending      = True
                self.get_logger().info(
                    f"[Claude req] n_obs={n_obs} "
                    f"candidates={[c['label'] for c in candidates]}")
                if self.args.save_debug:
                    cv2.imwrite(
                        os.path.join(self.args.debug_dir,
                                     f"cam_{self.iter_count:03d}.jpg"), img)
                    cv2.imwrite(
                        os.path.join(self.args.debug_dir,
                                     f"map_{self.iter_count:03d}.jpg"), nav_map)
                return
        elif n_obs < 1:
            self.get_logger().info("[wait] n_obs=0, waiting for light observations")

        # -- Goal ?? (Claude ?? ??) -------------------
        goal_x = goal_y = None
        goal_label = ""

        score  = self.score_result
        s_age  = now - score.get("timestamp", 0)
        s_valid = (bool(score)
                   and s_age < self.args.vlm_cache_ttl
                   and score.get("confidence", "low") != "low")

        if s_valid:
            mode = score.get("mode", "frontier")

            if mode == "direct":
                u = score.get("u")
                v = score.get("v")
                if u is not None and v is not None:
                    wx, wy, depth_m = self.pixel_to_world(u, v, pose)
                    if wx is not None:
                        dist_to_led = math.hypot(wx - rx, wy - ry)
                        MAX_DIRECT_STEP = self.args.step_dist * 2.0
                        if dist_to_led > MAX_DIRECT_STEP:
                            ratio = MAX_DIRECT_STEP / dist_to_led
                            goal_x = rx + (wx - rx) * ratio
                            goal_y = ry + (wy - ry) * ratio
                        else:
                            goal_x, goal_y = wx, wy
                        goal_label = f"claude_direct(u={u},v={v},d={depth_m:.1f}m,step={min(dist_to_led,MAX_DIRECT_STEP):.1f}m)"
                        self.last_mode = "DIRECT"
                    elif self.last_red.get("visible"):
                        # Depth unavailable — move along current bearing observation
                        bearing = normalize_angle(ryaw + self.last_red["bearing_rad"])
                        goal_x  = rx + self.args.step_dist * math.cos(bearing)
                        goal_y  = ry + self.args.step_dist * math.sin(bearing)
                        goal_label = f"direct_bearing_fallback(u={u},v={v})"
                        self.last_mode = "DIRECT"

            if goal_x is None:
                # frontier ?? ?? direct ??
                best_label = score.get("best", "")
                best_cand  = next(
                    (c for c in candidates if c["label"] == best_label), None)
                if best_cand:
                    goal_x, goal_y = best_cand["wx"], best_cand["wy"]
                    sc = score.get("scores", {}).get(best_label, 0)
                    goal_label = f"claude_frontier_{best_label}(score={sc})"
                    self.last_mode = "VLFM"

        # fallback
        if goal_x is None and candidates:
            goal_x, goal_y = candidates[0]["wx"], candidates[0]["wy"]
            goal_label = f"fallback_{candidates[0]['label']}"
            self.last_mode = "VLFM"

        if goal_x is None:
            self.get_logger().warn("No valid goal.")
            return

        obs = list(self.bearing_obs)
        var = circular_variance([o["abs_bearing"] for o in obs]) if obs else 1.0
        self.get_logger().info("=" * 55)
        self.get_logger().info(
            f"[iter={self.iter_count}] mode={self.last_mode} "
            f"n_obs={len(obs)} var={var:.2f}")
        self.get_logger().info(
            f"[Claude] mode={score.get('mode','?')} "
            f"conf={score.get('confidence','?')} age={s_age:.0f}s valid={s_valid}")
        self.get_logger().info(
            f"[decision] {goal_label} ? ({goal_x:.2f},{goal_y:.2f})")
        self.get_logger().info("=" * 55)

        self.send_goal(goal_x, goal_y, goal_label, pose)
        self.iter_count += 1


# --------------------------------------------------
# main
# --------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="VLFM light source navigation v13")

    # ROS ??
    p.add_argument("--image-topic",       default=IMAGE_TOPIC)
    p.add_argument("--depth-topic",       default=DEPTH_TOPIC)
    p.add_argument("--camera-info-topic", default=CAMERA_INFO_TOPIC)
    p.add_argument("--map-topic",         default=MAP_TOPIC)
    p.add_argument("--nav-action",        default=NAV_ACTION)
    p.add_argument("--map-frame",         default=MAP_FRAME)
    p.add_argument("--base-frame",        default=BASE_FRAME)

    # Claude API
    p.add_argument("--api-key",     default="",
                   help="Anthropic API key (??? ANTHROPIC_API_KEY ???? ??)")
    p.add_argument("--vlm-timeout", type=float, default=60.0)
    p.add_argument("--vlm-cache-ttl", type=float, default=30.0)
    p.add_argument("--score-interval", type=float, default=12.0,
                   help="VLFM Claude ?? ??(?)")

    # DIRECT mode
    p.add_argument("--direct-variance-threshold", type=float, default=0.35,
                   help="bearing variance threshold for DIRECT mode")
    p.add_argument("--max-depth", type=float, default=20.0,
                   help="max valid depth (m)")
    p.add_argument("--step-dist", type=float, default=2.0,
                   help="step distance for direct mode (m)")

    # Bearing ??
    p.add_argument("--obs-interval", type=float, default=1.0)
    p.add_argument("--max-obs",      type=int,   default=200)

    # ??
    p.add_argument("--period",    type=float, default=4.0)
    p.add_argument("--max-iters", type=int,   default=300)

    # ???
    p.add_argument("--camera-fov-deg",    type=float, default=70.0)
    p.add_argument("--coverage-range",    type=float, default=3.0)
    p.add_argument("--whole-frame-ratio", type=float, default=0.30)

    # HSV
    p.add_argument("--red-min-s",    type=int,   default=60)
    p.add_argument("--red-min-v",    type=int,   default=60)
    p.add_argument("--min-red-area", type=float, default=10.0)

    # Frontier / ??
    p.add_argument("--frontier-min-dist",        type=float, default=0.5)
    p.add_argument("--frontier-max-dist",        type=float, default=8.0)
    p.add_argument("--max-candidates",           type=int,   default=5)
    p.add_argument("--candidate-min-separation", type=float, default=1.5)

    # ???
    p.add_argument("--debug-dir",  default=os.path.expanduser("~/vlfm_v13_debug"))
    p.add_argument("--save-debug", action="store_true")
    p.add_argument("--execute",    action="store_true", default=True,
                   help="?? Nav2 goal ?? (?? ON)")

    args = p.parse_args()
    rclpy.init()
    node = VLFMSourceNavV13(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()


# --------------------------------------------------
# ?? ???? (Claude? DIRECT/FRONTIER ?? ??)
# --------------------------------------------------

def build_unified_prompt(candidates: List[Dict], ctx: Dict) -> str:
    labels    = [c["label"] for c in candidates]
    label_str = "/".join(labels)
    var       = ctx.get("bearing_variance", 1.0)
    n_obs     = ctx.get("n_observations", 0)

    return f"""You are a robot navigation system searching for a red LED light source in a dark room.
You are given TWO images:

[Image 1: Camera] current robot camera view.
[Image 2: Map] top-down occupancy map with recorded light directions.

MAP legend:
- Light gray=free | Near-black=walls | Medium gray=unknown
- Arrows: recorded light directions. LONGER+YELLOWER = stronger.
  Converging arrows = likely real source. Scattered = reflected light.
- Green=robot route | Blue dot+arrow=robot position/heading
- Circles ({label_str}): navigation candidates

ARROW CONSISTENCY: variance={var:.2f} (0=consistent, 1=scattered) | OBSERVATIONS: {n_obs}

STEP 1 - Look at Image 1 (camera) for horizontal direction ONLY:
  Is the LED directly visible as a distinct glowing point/circle? (not just wall glow)
  YES + confidence medium/high -> mode:direct
  NO (only reflected glow or not visible) -> mode:frontier

  CAMERA DIRECTION RULES (horizontal axis only):
  - Light at pixel u < 30% of width  -> "far left"
  - Light at pixel u < 45% of width  -> "slightly left"
  - Light at pixel u 45~55% of width -> "center"
  - Light at pixel u > 55% of width  -> "slightly right"
  - Light at pixel u > 70% of width  -> "far right"
  NOTE: IMPORTANT: vertical position (up/down) in camera image means NOTHING for navigation.
  NOTE: Do NOT interpret "light at top of image" as "light is to the north on the map".
  NOTE: Only LEFT and RIGHT in the camera image matter for direction.

STEP 2 - If mode:frontier, use Image 2 (map) with camera direction from STEP 1:
  The robot is the blue dot.And blue arrow show the robot's bearing which is same with camera's bearing. Arrows show recorded light directions.
  Match the camera horizontal direction to the map:
  - "far left" in camera    -> pick candidate to the robot's LEFT on map
  - "slightly left"         -> pick candidate slightly left of robot heading
  - "center"                -> pick candidate straight ahead
  - "slightly right"        -> pick candidate slightly right of robot heading
  - "far right"             -> pick candidate to the robot's RIGHT on map
  If no light visible, trust the arrow convergence pattern on the map.
  Avoid candidates in already-explored (green) areas.

Answer in EXACTLY one of these formats:

If LED directly visible:
mode: direct
pixel: (u, v)
confidence: low/medium/high
reason: one sentence

If not directly visible:
mode: frontier
scores: {" ".join(f"{l}=?" for l in labels)}
best: ?
confidence: low/medium/high
reason: one sentence (include camera direction like "slightly left" and which map candidate matches)"""

# --------------------------------------------------
# ?? ??
# --------------------------------------------------

class VLFMSourceNavV13(Node):

    def __init__(self, args):
        super().__init__("vlfm_source_nav_v13")
        self.args   = args
        self.bridge = CvBridge()

        # ?? ?? ?????????????????????????????????????????????
        self.latest_image: Optional[np.ndarray]    = None
        self.latest_depth: Optional[np.ndarray]    = None
        self.latest_map:   Optional[OccupancyGrid] = None
        self._occ_cache:   Optional[np.ndarray]    = None

        # ??? ?? ????
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None
        self.depth_scale = 1.0   # 32FC1=1.0(m), 16UC1=0.001(mm?m)

        # ?? Bearing ?? ?? (v10 ??) ?????????????????????
        self.bearing_obs: deque = deque(maxlen=args.max_obs)
        self._last_obs_t = 0.0

        # ?? ?? ?? / ? ???????????????????????????????????
        self.route_history: deque = deque(maxlen=300)
        self.start_pose: Optional[Tuple] = None
        self.coverage_map: Optional[np.ndarray] = None
        self.map_shape:    Optional[Tuple]      = None
        self._last_cov_t = 0.0

        # ?? ? ?? ??????????????????????????????????????????
        self.last_red: Dict = {}

        # ?? ?? ?? (???) ???????????????????????????????
        self.last_mode = ""   # "DIRECT" or "VLFM"

        # ?? Claude API ?? (VLFM ??) ??????????????????????
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            self.get_logger().warn(
                "ANTHROPIC_API_KEY ??. VLFM ???? Claude ?? ??.")
        self.claude_worker = ClaudeWorker(api_key, args.vlm_timeout)
        self.score_result: Dict = {}
        self.score_accum:  Dict[str, float] = {}  # EMA accumulated scores
        self._pending      = False
        self._score_last_t = 0.0

        # ?? Nav2 ?????????????????????????????????????????????
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav_client  = ActionClient(self, NavigateToPose, args.nav_action)
        self.nav_busy    = False
        self._pending_goal: Dict   = {}
        self._current_goal_handle  = None
        self.visited_goals: List[Dict] = []
        self.failed_goals:  List[Dict] = []

        # ?? ?? ?????????????????????????????????????????????
        self.done       = False
        self.iter_count = 0

        # ?? ?? / ??? ????????????????????????????????????
        self.create_subscription(Image,         args.image_topic,       self.image_cb,       10)
        self.create_subscription(Image,         args.depth_topic,       self.depth_cb,       10)
        self.create_subscription(CameraInfo,    args.camera_info_topic, self.camera_info_cb, 10)
        self.create_subscription(OccupancyGrid, args.map_topic,         self.map_cb,         10)
        self.create_timer(args.period, self.step)

        os.makedirs(args.debug_dir, exist_ok=True)
        # Nav2 ??? ?? (bt_navigator? active ??? ? ???)
        self.get_logger().info("Nav2 ??? ?? ?...")
        while not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("Nav2 ?? ?...")
        self.get_logger().info("VLFMSourceNavV13 ready.")

    # --------------------------------------------------
    # ??
    # --------------------------------------------------

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

        # bearing ?? ??
        if red.get("visible") and (now - self._last_obs_t >= self.args.obs_interval):
            abs_bearing = normalize_angle(ryaw + red["bearing_rad"])
            self.bearing_obs.append({
                "rx":          rx,
                "ry":          ry,
                "abs_bearing": abs_bearing,
                "confidence":  red["confidence"],
                "t":           now,
            })
            self._last_obs_t = now

        # coverage (2Hz)
        if now - self._last_cov_t >= 0.5:
            self.update_coverage(pose)
            self._last_cov_t = now

    def depth_cb(self, msg: Image):
        """Depth ???. 32FC1(m) ?? 16UC1(mm) ?? ??."""
        try:
            if msg.encoding == "32fc1":
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
                self.depth_scale  = 1.0
            elif msg.encoding in ("16uc1", "mono16"):
                self.latest_depth = self.bridge.imgmsg_to_cv2(
                    msg, "16UC1").astype(np.float32)
                self.depth_scale  = 0.001
            else:
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg).astype(np.float32)
                self.depth_scale  = 1.0
        except Exception as e:
            self.get_logger().warn(f"depth_cb: {e}")

    def camera_info_cb(self, msg: CameraInfo):
        """?? ???? 1? ??."""
        if self.fx is not None:
            return
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.get_logger().info(
            f"[CameraInfo] fx={self.fx:.1f} fy={self.fy:.1f} "
            f"cx={self.cx:.1f} cy={self.cy:.1f}")

    def map_cb(self, msg: OccupancyGrid):
        self.latest_map = msg
        self._occ_cache = np.array(
            msg.data, dtype=np.int16
        ).reshape((msg.info.height, msg.info.width))
        shape = (msg.info.height, msg.info.width)
        if self.map_shape != shape:
            self.map_shape    = shape
            self.coverage_map = np.zeros(shape, np.uint8)
            self.get_logger().info(f"Map updated: {shape}")

    # --------------------------------------------------
    # ?? ??
    # --------------------------------------------------

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

    def pixel_to_world(self, u: int, v: int,
                       pose: Tuple) -> Tuple[Optional[float], Optional[float], float]:
        """
        ?? (u,v) + depth ? ?? ?? (wx, wy).
        ??: (wx, wy, depth_m). depth ?? ? (None, None, 0).
        """
        if self.latest_depth is None or self.fx is None:
            return None, None, 0.0

        dh, dw = self.latest_depth.shape[:2]
        u = int(np.clip(u, 0, dw-1))
        v = int(np.clip(v, 0, dh-1))

        # depth ?? (m ??)
        depth_raw = float(self.latest_depth[v, u])
        if depth_raw <= 0 or np.isnan(depth_raw) or np.isinf(depth_raw):
            # 5?5 ?? ?? ??
            patch = self.latest_depth[
                max(0, v-2):min(dh, v+3),
                max(0, u-2):min(dw, u+3)]
            valid = patch[(patch > 0) & ~np.isnan(patch) & ~np.isinf(patch)]
            if len(valid) == 0:
                return None, None, 0.0
            depth_raw = float(np.mean(valid))

        depth_m = depth_raw * self.depth_scale

        # depth ?? ??
        if depth_m < 0.1 or depth_m > self.args.max_depth:
            return None, None, depth_m

        # ??? ?? ? ?? ??? ?? ? ??
        # (???: x=right, y=down, z=forward)
        x_cam = (u - self.cx) * depth_m / self.fx   # ?? ???
        z_cam = depth_m                               # ?? ??

        rx, ry, ryaw = pose
        # ??? ??? ? ?? ??
        cam_angle  = math.atan2(-x_cam, z_cam)
        world_yaw  = normalize_angle(ryaw + cam_angle)
        horiz_dist = math.sqrt(x_cam**2 + z_cam**2)

        wx = rx + horiz_dist * math.cos(world_yaw)
        wy = ry + horiz_dist * math.sin(world_yaw)
        return wx, wy, depth_m

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
        return np.array(g.data, dtype=np.int16).reshape(
            (g.info.height, g.info.width))

    def _ensure_maps(self):
        if self.coverage_map is None and self.latest_map is not None:
            g = self.latest_map
            shape = (g.info.height, g.info.width)
            self.coverage_map = np.zeros(shape, np.uint8)
            self.map_shape    = shape

    # --------------------------------------------------
    # ?? ??
    # --------------------------------------------------

    def should_use_direct(self, red: Dict, pose: Tuple) -> Tuple[bool, str, float, float]:
        """
        DIRECT ?? ?? ?? ??.
        ??: (use_direct, reason, goal_wx, goal_wy)

        DIRECT ?? (?? ????):
          1. ?? ??
          2. whole_frame ??
          3. ?? bearing variance < threshold (?? ??)
          4. depth ?? + ?? ??
        """
        if not red.get("visible"):
            return False, "light_not_visible", 0.0, 0.0

        if red.get("whole_frame"):
            return False, "whole_frame", 0.0, 0.0

        # bearing variance ?? (?? 10?)
        recent = list(self.bearing_obs)[-10:]
        if len(recent) >= 3:
            var = circular_variance([o["abs_bearing"] for o in recent])
            if var > self.args.direct_variance_threshold:
                return False, f"high_variance({var:.2f})", 0.0, 0.0

        # depth ??
        cx = red.get("cx")
        cy = red.get("cy")
        if cx is None or cy is None:
            return False, "no_pixel", 0.0, 0.0

        wx, wy, depth_m = self.pixel_to_world(cx, cy, pose)
        if wx is None:
            return False, f"depth_invalid(d={depth_m:.2f}m)", 0.0, 0.0

        # ? ??: ?? ??? occupied cell??
        mx, my = self.w2g(wx, wy)
        if self.inside(mx, my):
            occ = self.get_occ()
            if occ[my, mx] > 50:
                return False, "depth_hits_wall", 0.0, 0.0

        return True, "ok", wx, wy

    # --------------------------------------------------
    # HSV ?? (v10 ??)
    # --------------------------------------------------

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
        m = cv2.moments(c)
        if m["m00"] == 0:
            return null

        cx     = int(m["m10"] / m["m00"])
        cy_val = int(m["m01"] / m["m00"])
        area_r = area / float(w * h)
        whole_frame = area_r > self.args.whole_frame_ratio
        conf   = float(np.clip(0.3 + 0.7 * min(area_r / 0.05, 1.0), 0.0, 1.0))
        norm_x = (cx - w / 2.0) / (w / 2.0)
        # Negate: camera-left = robot-left = positive bearing (matches pixel_to_world atan2(-x_cam,z_cam))
        brad   = -norm_x * math.radians(self.args.camera_fov_deg / 2)

        return {
            "visible":     True,
            "whole_frame": whole_frame,
            "cx":          cx,
            "cy":          cy_val,
            "confidence":  conf,
            "bearing_rad": brad,
            "bearing_deg": math.degrees(brad),
        }

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

    # --------------------------------------------------
    # ? ??? (v10 ??)
    # --------------------------------------------------

    def _render_nav_map(self, occ: np.ndarray, pose: Tuple,
                        candidates: List[Dict]) -> np.ndarray:
        g   = self.latest_map
        res = g.info.resolution
        ox  = g.info.origin.position.x
        oy  = g.info.origin.position.y
        h, w = occ.shape

        img = np.zeros((h, w, 3), np.uint8)
        img[occ == -1] = (80,  80,  80)
        img[occ == 0 ] = (210, 210, 210)
        img[occ >  50] = (25,  25,  25)
        img = cv2.flip(img, 0)

        rx, ry, ryaw = pose

        def w2p(wx: float, wy: float) -> Tuple[int, int]:
            px = int(np.clip((wx - ox) / res, 0, w-1))
            py = int(np.clip(h-1-(wy - oy) / res, 0, h-1))
            return px, py

        # ?? ?? (??)
        if len(self.route_history) >= 2:
            pts    = [w2p(wx, wy) for wx, wy in self.route_history]
            stride = max(1, len(pts) // 20)
            for i in range(1, len(pts)):
                cv2.line(img, pts[i-1], pts[i], (0, 230, 0), 2, cv2.LINE_AA)
            for pt in pts[::stride]:
                cv2.circle(img, pt, 3, (0, 230, 0), -1)

        # Bearing ??? ???
        MIN_ARROW_M = 0.5
        MAX_ARROW_M = 5.0
        MIN_SEP_PX  = 15
        MAX_ARROWS  = 60
        drawn: List[Tuple[int, int]] = []
        for obs in sorted(self.bearing_obs, key=lambda o: o["t"], reverse=True):
            if len(drawn) >= MAX_ARROWS:
                break
            opx, opy = w2p(obs["rx"], obs["ry"])
            if any(math.hypot(opx-dx, opy-dy) < MIN_SEP_PX for dx, dy in drawn):
                continue
            drawn.append((opx, opy))
            conf    = obs["confidence"]
            arrow_m = MIN_ARROW_M + (MAX_ARROW_M - MIN_ARROW_M) * conf
            ewx     = obs["rx"] + arrow_m * math.cos(obs["abs_bearing"])
            ewy     = obs["ry"] + arrow_m * math.sin(obs["abs_bearing"])
            epx, epy = w2p(ewx, ewy)
            g_ch    = int(140 + 115 * conf)
            col     = (0, g_ch, 255)
            thick   = max(1, round(1 + conf * 2))
            cv2.arrowedLine(img, (opx, opy), (epx, epy),
                            col, thick, cv2.LINE_AA, 0, 0.25)
            cv2.circle(img, (opx, opy), 3, col, -1)

        # ?? ?? + ??
        rpx, rpy = w2p(rx, ry)
        ax = int(rpx + 24 * math.cos(ryaw))
        ay = int(rpy - 24 * math.sin(ryaw))
        cv2.circle(img, (rpx, rpy), 9, (220, 50, 0), -1)
        cv2.arrowedLine(img, (rpx, rpy), (ax, ay),
                        (255, 80, 0), 2, cv2.LINE_AA, 0, 0.4)

        # ?? A~E
        CAND_COL = {
            "A": (0, 0, 255), "B": (255, 0, 220), "C": (255, 220, 0),
            "D": (0, 140, 255), "E": (160, 0, 180),
        }
        for cand in candidates:
            label    = cand["label"]
            cpx, cpy = w2p(cand["wx"], cand["wy"])
            col      = CAND_COL.get(label, (80, 80, 80))
            cv2.circle(img, (cpx, cpy), 15, col, 2)
            cv2.circle(img, (cpx, cpy),  5, col, -1)
            cv2.putText(img, label, (cpx+18, cpy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

        # ?? ??
        mode_col = (0, 255, 100) if self.last_mode == "DIRECT" else (0, 180, 255)
        cv2.putText(img, self.last_mode, (4, h-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_col, 1, cv2.LINE_AA)

        return self._resize_map(img)

    def _resize_map(self, img: np.ndarray, size: int = 600) -> np.ndarray:
        """?? ?? size?size? ????. Claude? ?? ??? ? ? ??."""
        h, w = img.shape[:2]
        # ?? ????? ? ?? size? ??
        scale = size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # ???? ???? ?? ??
        canvas = np.full((size, size, 3), 60, np.uint8)
        y_off  = (size - new_h) // 2
        x_off  = (size - new_w) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        return canvas

    # --------------------------------------------------
    # Frontier / ?? (v10 ??)
    # --------------------------------------------------

    def extract_frontiers(self) -> List[Tuple[float, float]]:
        occ  = self.get_occ()
        g    = self.latest_map
        free = (occ == 0).astype(np.uint8) * 255
        unk  = (occ == -1).astype(np.uint8) * 255
        wall = (occ > 50).astype(np.uint8) * 255
        k    = np.ones((3, 3), np.uint8)
        fm   = cv2.bitwise_and(cv2.dilate(free, k), unk)
        mp   = max(1, int(0.15 / g.info.resolution))
        fm   = cv2.bitwise_and(
            fm, cv2.bitwise_not(cv2.dilate(wall, k, iterations=mp)))
        n, _, stats, centroids = cv2.connectedComponentsWithStats(fm, connectivity=8)
        pose      = self.get_robot_pose()
        frontiers = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 5:
                continue
            wx, wy = self.g2w(int(centroids[i][0]), int(centroids[i][1]))
            if pose:
                rx, ry, _ = pose
                d = math.hypot(wx-rx, wy-ry)
                if not (self.args.frontier_min_dist <= d <= self.args.frontier_max_dist):
                    continue
            frontiers.append((wx, wy))
        return frontiers

    def select_candidates(self, frontiers: List[Tuple], robot_pose: Tuple,
                          max_n: int = 5) -> List[Dict]:
        if not frontiers:
            return []
        rx, ry, _ = robot_pose

        recent = [o for o in self.bearing_obs if time.time() - o["t"] < 60.0]
        mean_bearing: Optional[float] = None
        if recent:
            sins = float(np.mean([math.sin(o["abs_bearing"]) for o in recent]))
            coss = float(np.mean([math.cos(o["abs_bearing"]) for o in recent]))
            mean_bearing = math.atan2(sins, coss)

        scored = []
        for fx, fy in frontiers:
            dist       = math.hypot(fx-rx, fy-ry)
            dist_score = 1.0 - min(dist / self.args.frontier_max_dist, 1.0)
            if mean_bearing is not None:
                angle_to = math.atan2(fy-ry, fx-rx)
                align    = (math.cos(normalize_angle(angle_to - mean_bearing)) + 1.0) / 2.0
            else:
                align = 0.5
            score = 0.5 * align + 0.5 * dist_score
            scored.append({"wx": fx, "wy": fy, "dist": dist, "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        selected, min_sep = [], self.args.candidate_min_separation
        for cand in scored:
            too_close = any(
                math.hypot(cand["wx"]-s["wx"], cand["wy"]-s["wy"]) < min_sep
                for s in selected)
            if not too_close:
                selected.append(cand)
            if len(selected) >= max_n:
                break
        for i, s in enumerate(selected):
            s["label"] = CANDIDATE_LABELS[i]
        return selected

    # --------------------------------------------------
    # VLM ?? / ????
    # --------------------------------------------------

    def _vlm_cb(self, result: Dict):
        self._pending = False
        mode = result.get("mode", "frontier")

        # EMA score accumulation — reduces per-call noise
        if mode == "frontier":
            new_scores = result.get("scores", {})
            if new_scores:
                alpha = 0.4
                if not self.score_accum or set(self.score_accum) != set(new_scores):
                    self.score_accum = dict(new_scores)
                else:
                    for k in new_scores:
                        self.score_accum[k] = (alpha * new_scores[k]
                                               + (1 - alpha) * self.score_accum.get(k, new_scores[k]))
                result = dict(result)
                result["scores"] = {k: round(v, 1) for k, v in self.score_accum.items()}
                result["best"]   = max(self.score_accum, key=self.score_accum.get)

        self.score_result = {**result, "timestamp": time.time()}

        if mode == "direct":
            self.get_logger().info(
                f"[Claude] mode=direct pixel=({result.get('u')},{result.get('v')}) "
                f"conf={result.get('confidence')} reason={result.get('reason','')}")
        else:
            self.get_logger().info(
                f"[Claude] mode=frontier scores={result.get('scores',{})} "
                f"best={result.get('best')} conf={result.get('confidence')} "
                f"reason={result.get('reason','')}")

    def _build_ctx(self, pose: Tuple) -> Dict:
        red = self.last_red
        obs = list(self.bearing_obs)
        var = circular_variance([o["abs_bearing"] for o in obs]) if len(obs) >= 2 else 1.0
        rx, ry, _ = pose
        moved = 0.0
        if self.start_pose:
            sx, sy, _ = self.start_pose
            moved = math.hypot(rx-sx, ry-sy)
        return {
            "light_visible":    red.get("visible", False),
            "whole_frame":      red.get("whole_frame", False),
            "bearing_variance": round(var, 3),
            "n_observations":   len(obs),
            "moved_dist":       round(moved, 2),
            
        }

    # --------------------------------------------------
    # ?? ?? (v10 ??)
    # --------------------------------------------------

    def check_termination(self, pose: Tuple) -> Tuple[bool, str]:
        if self.iter_count >= self.args.max_iters:
            return True, "max_iters"
        obs = list(self.bearing_obs)
        if len(obs) >= 5 and self.last_red.get("whole_frame"):
            var = circular_variance([o["abs_bearing"] for o in obs[-10:]])
            if var < 0.1:
                return True, "source_confirmed"
        return False, ""

    # --------------------------------------------------
    # Nav2
    # --------------------------------------------------

    def send_goal(self, x: float, y: float, label: str, pose: Tuple):
        rx, ry, _ = pose
        goal_yaw  = math.atan2(y-ry, x-rx)
        if not self.args.execute:
            self.get_logger().info(f"[DRY RUN] ? ({x:.2f},{y:.2f}) ({label})")
            return
        if not self.nav_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Nav2 unavailable")
            return
        self.nav_busy      = True
        self._pending_goal = {"x": x, "y": y}
        goal = PoseStamped()
        goal.header.frame_id  = self.args.map_frame
        goal.header.stamp     = self.get_clock().now().to_msg()
        goal.pose.position.x  = float(x)
        goal.pose.position.y  = float(y)
        goal.pose.orientation = quaternion_from_yaw(float(goal_yaw))
        ng      = NavigateToPose.Goal()
        ng.pose = goal
        f = self.nav_client.send_goal_async(ng)
        f.add_done_callback(self._goal_resp_cb)
        self.get_logger().info(f"Nav2 ? ({x:.2f},{y:.2f}) [{label}] [{self.last_mode}]")

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
        else:
            self.failed_goals.append(self._pending_goal)
            self.failed_goals = self.failed_goals[-10:]
        self.nav_busy = False
        self._current_goal_handle = None

    # --------------------------------------------------
    # ?? ??
    # --------------------------------------------------

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
        if self.start_pose is None:
            self.start_pose = pose

        if (not self.route_history or
                math.hypot(rx - self.route_history[-1][0],
                           ry - self.route_history[-1][1]) > 0.1):
            self.route_history.append((rx, ry))

        done, reason = self.check_termination(pose)
        if done:
            self.get_logger().info(f"=== DONE: {reason} ===")
            self.done = True
            return

        img = self.latest_image.copy()

        # -- Claude pending ?? ---------------------------
        if self._pending:
            waited = time.time() - self._score_last_t
            if waited > self.args.vlm_timeout + 5.0:
                self._pending = False
                self.get_logger().warn("[Claude] timeout, skip")
            else:
                self.get_logger().info(
                    f"[Claude pending {waited:.0f}s]")
                return

        # -- ???? ?? ?? -----------------------------
        frontiers  = self.extract_frontiers()
        candidates = self.select_candidates(frontiers, pose, self.args.max_candidates)
        if not candidates:
            self.get_logger().warn("No frontier candidates.")
            return

        occ     = self.get_occ()
        nav_map = self._render_nav_map(occ, pose, candidates)
        ctx     = self._build_ctx(pose)
        now     = time.time()

        # -- Claude ?? (?? 1? ???? ??) -----------
        n_obs = len(self.bearing_obs)
        if (now - self._score_last_t >= self.args.score_interval
                and n_obs >= 1):
            if self.claude_worker.submit(img, nav_map, candidates, ctx, self._vlm_cb):
                self._score_last_t = now
                self._pending      = True
                self.get_logger().info(
                    f"[Claude req] n_obs={n_obs} "
                    f"candidates={[c['label'] for c in candidates]}")
                if self.args.save_debug:
                    cv2.imwrite(
                        os.path.join(self.args.debug_dir,
                                     f"cam_{self.iter_count:03d}.jpg"), img)
                    cv2.imwrite(
                        os.path.join(self.args.debug_dir,
                                     f"map_{self.iter_count:03d}.jpg"), nav_map)
                return
        elif n_obs < 1:
            self.get_logger().info("[wait] n_obs=0, waiting for light observations")

        # -- Goal ?? (Claude ?? ??) -------------------
        goal_x = goal_y = None
        goal_label = ""

        score  = self.score_result
        s_age  = now - score.get("timestamp", 0)
        s_valid = (bool(score)
                   and s_age < self.args.vlm_cache_ttl
                   and score.get("confidence", "low") != "low")

        if s_valid:
            mode = score.get("mode", "frontier")

            if mode == "direct":
                u = score.get("u")
                v = score.get("v")
                if u is not None and v is not None:
                    wx, wy, depth_m = self.pixel_to_world(u, v, pose)
                    if wx is not None:
                        dist_to_led = math.hypot(wx - rx, wy - ry)
                        MAX_DIRECT_STEP = self.args.step_dist * 2.0
                        if dist_to_led > MAX_DIRECT_STEP:
                            ratio = MAX_DIRECT_STEP / dist_to_led
                            goal_x = rx + (wx - rx) * ratio
                            goal_y = ry + (wy - ry) * ratio
                        else:
                            goal_x, goal_y = wx, wy
                        goal_label = f"claude_direct(u={u},v={v},d={depth_m:.1f}m,step={min(dist_to_led,MAX_DIRECT_STEP):.1f}m)"
                        self.last_mode = "DIRECT"
                    elif self.last_red.get("visible"):
                        # Depth unavailable — move along current bearing observation
                        bearing = normalize_angle(ryaw + self.last_red["bearing_rad"])
                        goal_x  = rx + self.args.step_dist * math.cos(bearing)
                        goal_y  = ry + self.args.step_dist * math.sin(bearing)
                        goal_label = f"direct_bearing_fallback(u={u},v={v})"
                        self.last_mode = "DIRECT"

            if goal_x is None:
                # frontier ?? ?? direct ??
                best_label = score.get("best", "")
                best_cand  = next(
                    (c for c in candidates if c["label"] == best_label), None)
                if best_cand:
                    goal_x, goal_y = best_cand["wx"], best_cand["wy"]
                    sc = score.get("scores", {}).get(best_label, 0)
                    goal_label = f"claude_frontier_{best_label}(score={sc})"
                    self.last_mode = "VLFM"

        # fallback
        if goal_x is None and candidates:
            goal_x, goal_y = candidates[0]["wx"], candidates[0]["wy"]
            goal_label = f"fallback_{candidates[0]['label']}"
            self.last_mode = "VLFM"

        if goal_x is None:
            self.get_logger().warn("No valid goal.")
            return

        obs = list(self.bearing_obs)
        var = circular_variance([o["abs_bearing"] for o in obs]) if obs else 1.0
        self.get_logger().info("=" * 55)
        self.get_logger().info(
            f"[iter={self.iter_count}] mode={self.last_mode} "
            f"n_obs={len(obs)} var={var:.2f}")
        self.get_logger().info(
            f"[Claude] mode={score.get('mode','?')} "
            f"conf={score.get('confidence','?')} age={s_age:.0f}s valid={s_valid}")
        self.get_logger().info(
            f"[decision] {goal_label} ? ({goal_x:.2f},{goal_y:.2f})")
        self.get_logger().info("=" * 55)

        self.send_goal(goal_x, goal_y, goal_label, pose)
        self.iter_count += 1


# --------------------------------------------------
# main
# --------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="VLFM light source navigation v13")

    # ROS ??
    p.add_argument("--image-topic",       default=IMAGE_TOPIC)
    p.add_argument("--depth-topic",       default=DEPTH_TOPIC)
    p.add_argument("--camera-info-topic", default=CAMERA_INFO_TOPIC)
    p.add_argument("--map-topic",         default=MAP_TOPIC)
    p.add_argument("--nav-action",        default=NAV_ACTION)
    p.add_argument("--map-frame",         default=MAP_FRAME)
    p.add_argument("--base-frame",        default=BASE_FRAME)

    # Claude API
    p.add_argument("--api-key",     default="",
                   help="Anthropic API key (??? ANTHROPIC_API_KEY ???? ??)")
    p.add_argument("--vlm-timeout", type=float, default=60.0)
    p.add_argument("--vlm-cache-ttl", type=float, default=30.0)
    p.add_argument("--score-interval", type=float, default=12.0,
                   help="VLFM Claude ?? ??(?)")

    # DIRECT mode
    p.add_argument("--direct-variance-threshold", type=float, default=0.35,
                   help="bearing variance threshold for DIRECT mode")
    p.add_argument("--max-depth", type=float, default=20.0,
                   help="max valid depth (m)")
    p.add_argument("--step-dist", type=float, default=2.0,
                   help="step distance for direct mode (m)")

    # Bearing ??
    p.add_argument("--obs-interval", type=float, default=1.0)
    p.add_argument("--max-obs",      type=int,   default=200)

    # ??
    p.add_argument("--period",    type=float, default=4.0)
    p.add_argument("--max-iters", type=int,   default=300)

    # ???
    p.add_argument("--camera-fov-deg",    type=float, default=70.0)
    p.add_argument("--coverage-range",    type=float, default=3.0)
    p.add_argument("--whole-frame-ratio", type=float, default=0.30)

    # HSV
    p.add_argument("--red-min-s",    type=int,   default=60)
    p.add_argument("--red-min-v",    type=int,   default=60)
    p.add_argument("--min-red-area", type=float, default=10.0)

    # Frontier / ??
    p.add_argument("--frontier-min-dist",        type=float, default=0.5)
    p.add_argument("--frontier-max-dist",        type=float, default=8.0)
    p.add_argument("--max-candidates",           type=int,   default=5)
    p.add_argument("--candidate-min-separation", type=float, default=1.5)

    # ???
    p.add_argument("--debug-dir",  default=os.path.expanduser("~/vlfm_v13_debug"))
    p.add_argument("--save-debug", action="store_true")
    p.add_argument("--execute",    action="store_true", default=True,
                   help="?? Nav2 goal ?? (?? ON)")

    args = p.parse_args()
    rclpy.init()
    node = VLFMSourceNavV13(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
