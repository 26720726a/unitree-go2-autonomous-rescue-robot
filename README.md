# unitree-go2-physical-ai-rescue-robot

ROS2-based Physical AI system for autonomous navigation, human detection, visual-language-guided exploration, and rescue-assist operation using the Unitree Go2 quadruped robot.

This project integrates LiDAR, RGB-D vision, YOLO-based object detection, Nav2, SLAM-related processing, and VLM-based frontier decision logic to build an autonomous rescue robot platform for unstructured and disaster-like indoor environments.

---

## Project Overview

The goal of this project is to develop an autonomous robotic system based on the Unitree Go2 that can perceive its surroundings, detect people, estimate distance, navigate through unknown environments, and support rescue-related actions.

The system is designed for scenarios such as fire or disaster environments where prior maps may not be available. The robot uses onboard sensors and ROS2 nodes to build environmental awareness, move through the space, detect humans, and publish rescue-related commands.

---

## Key Features

* Unitree Go2-based autonomous mobile robot system
* ROS2-based robot software architecture
* LiDAR-based mapping and obstacle perception
* RealSense D435i RGB-D camera integration
* YOLOv8-based person and object detection
* Depth-based distance estimation from detected objects
* Person snapshot publishing when a human is detected
* `/cmd_vel` to Unitree sport API bridge for Go2 motion control
* ClassicWalk mode support for stable locomotion
* Nav2-compatible navigation pipeline
* VLM/VLFM-based frontier selection and light-source navigation experiments
* Fire command publisher for testing rescue-actuation commands

---

## Hardware Configuration

| Component                 | Purpose                                         |
| ------------------------- | ----------------------------------------------- |
| Unitree Go2               | Quadruped robot platform                        |
| Intel RealSense D435i     | RGB-D image acquisition and distance estimation |
| Hesai LiDAR               | Point cloud acquisition and obstacle perception |
| Onboard Jetson / Linux PC | ROS2 node execution and sensor processing       |

---

## Software Stack

| Category         | Tools / Frameworks                        |
| ---------------- | ----------------------------------------- |
| Middleware       | ROS2                                      |
| Robot Platform   | Unitree Go2                               |
| Navigation       | Nav2, SLAM Toolbox-related configuration  |
| Perception       | YOLOv8, OpenCV, cv_bridge                 |
| Camera           | Intel RealSense ROS                       |
| LiDAR Processing | PointCloud2, LaserScan conversion         |
| VLM Navigation   | Claude API / VLFM-style source navigation |
| Communication    | CycloneDDS                                |
| Language         | Python                                    |

---

## Repository Structure

```bash
unitree-go2-autonomous-rescue-robot/
├── maps/
│   └── Saved map files
│
├── src/
│   ├── lidar_stamp_sync/
│   │   └── LiDAR timestamp synchronization package
│   │
│   ├── my_go2_controller/
│   │   └── Go2 control-related package
│   │
│   ├── my_go2_nav_bringup/
│   │   ├── config/
│   │   ├── launch/
│   │   └── scripts/
│   │       ├── cmd_vel_to_sport_demo.py
│   │       ├── cmd_vel_to_sport.py
│   │       ├── classic_walk_demo.py
│   │       ├── check_motion_mode.py
│   │       └── restamp_cloud.py
│   │
│   ├── my_go2_odom_relay/
│   │   └── Unitree odometry relay package
│   │
│   └── yolo_distance_pkg/
│       └── YOLOv8 + RealSense distance estimation package
│
├── vlm/
│   ├── vlfm_source_nav_v20.py
│   ├── vlfm_source_nav_v21.py
│   ├── vlfm_source_nav_v22.py
│   └── simple_waypoint_nav.py
│
├── run_vlm_stack.sh
├── fire_command_publisher.py
├── test_publisher.py
├── cyclonedds_eth1.xml
└── README.md
```

---

## System Architecture

```text
RealSense D435i
   ├── RGB Image
   └── Depth Image
        ↓
YOLOv8 Detection + Depth Estimation
        ↓
Human/Object Distance Information
        ↓
ROS2 Topics
        ↓
Navigation / Snapshot / Debug Viewer


Hesai LiDAR or Go2 UTLidar
        ↓
PointCloud2
        ↓
LaserScan Conversion
        ↓
SLAM / Nav2 Costmap / Obstacle Avoidance


Nav2 / VLM Decision
        ↓
/cmd_vel
        ↓
cmd_vel_to_sport bridge
        ↓
/api/sport/request
        ↓
Unitree Go2 Locomotion
```

---

## Main ROS2 Packages

### `my_go2_nav_bringup`

This package contains the main navigation bringup files for the Go2 robot.

Main functions:

* Static TF publishing for LiDAR and camera frames
* PointCloud2 to LaserScan conversion
* Unitree odometry relay integration
* SLAM Toolbox launch configuration
* `/cmd_vel` to `/api/sport/request` bridge execution
* RViz visualization support

Main launch files:

```bash
go2_mapping.launch.py
go2_mapping_utlidar.launch.py
```

---

### `cmd_vel_to_sport_demo.py`

This script converts standard ROS2 `/cmd_vel` velocity commands into Unitree Go2 sport API requests.

Main functions:

* Subscribe to `/cmd_vel`
* Publish Unitree sport commands to `/api/sport/request`
* Switch Go2 into AI sport mode
* Enable ClassicWalk gait
* Apply speed limits
* Apply acceleration smoothing
* Stop the robot when command timeout occurs
* Rotate-to-face logic for more stable directional movement

The initialization sequence is:

```text
SelectMode("ai")
→ BalanceStand
→ SpeedLevel
→ ClassicWalk(True)
→ Accept /cmd_vel commands
```

This is important because ClassicWalk must be enabled after switching the robot into AI sport mode.

---

### `yolo_distance_pkg`

This package performs object detection and distance estimation using YOLOv8 and RealSense depth images.

Main nodes:

```bash
yolo_distance
go2_yolo_distance_node
remote_debug_viewer
person_snapshot_publisher
person_snapshot_receiver
yolo_pose_distance
```

Main functions:

* Detect humans using YOLOv8
* Estimate object distance using depth image median values
* Publish detection text
* Publish annotated debug image
* Save and publish compressed snapshots when a person is detected
* Support remote visualization of YOLO detection output

Main topics:

```bash
/camera/color/image_raw
/camera/depth/image_rect_raw
/camera/aligned_depth_to_color/image_raw
/yolo/debug_image
/yolo/detection_text
/yolo/person_snapshot/compressed
```

---

### `vlm`

The `vlm` directory contains experimental VLM/VLFM-based navigation scripts.

The VLM navigation logic uses camera images, map information, frontier candidates, and visual cues to select a navigation direction. Some versions include logic for:

* Direct source detection
* Passed-source recovery
* Frontier candidate scoring
* LiDAR fallback for invalid depth values
* Navigation goal generation for Nav2
* Debug data collection for VLM evaluation

Example script:

```bash
vlfm_source_nav_v22.py
```

---

### `fire_command_publisher.py`

This node publishes simple rescue-related commands to the `/fire_command` topic.

Supported commands:

| Key | Command  | Meaning                             |
| --- | -------- | ----------------------------------- |
| `f` | `fire`   | Fire/extinguishing sequence command |
| `m` | `marker` | Marker or motor pattern command     |
| `s` | `stop`   | Emergency stop command              |
| `q` | quit     | Exit program                        |

Run:

```bash
python3 fire_command_publisher.py
```

---

## Installation

### 1. Clone repository

```bash
git clone https://github.com/26720726a/unitree-go2-autonomous-rescue-robot.git
cd unitree-go2-autonomous-rescue-robot
```

### 2. Source ROS2

For the Unitree Go2 onboard Jetson environment:

```bash
source /opt/ros/foxy/setup.bash
```

For a laptop ROS2 Humble environment:

```bash
source /opt/ros/humble/setup.bash
```

Use the ROS2 version that matches your robot and workspace configuration.

### 3. Install dependencies

Install common ROS2 dependencies:

```bash
sudo apt update
sudo apt install -y \
  ros-${ROS_DISTRO}-cv-bridge \
  ros-${ROS_DISTRO}-image-transport \
  ros-${ROS_DISTRO}-message-filters \
  ros-${ROS_DISTRO}-tf2-ros \
  ros-${ROS_DISTRO}-nav2-bringup \
  ros-${ROS_DISTRO}-slam-toolbox \
  ros-${ROS_DISTRO}-pointcloud-to-laserscan
```

Install Python dependencies:

```bash
pip3 install ultralytics opencv-python numpy anthropic
```

---

## Build

From the workspace root:

```bash
colcon build --symlink-install
source install/setup.bash
```

---

## Running the System

### 1. Start mapping and navigation stack

```bash
ros2 launch my_go2_nav_bringup go2_mapping.launch.py
```

For Go2 built-in UTLidar:

```bash
ros2 launch my_go2_nav_bringup go2_mapping_utlidar.launch.py
```

---

### 2. Run Go2 `/cmd_vel` bridge

```bash
source install/setup.bash
python3 src/my_go2_nav_bringup/scripts/cmd_vel_to_sport_demo.py
```

Example with parameters:

```bash
python3 src/my_go2_nav_bringup/scripts/cmd_vel_to_sport_demo.py \
  --ros-args \
  -p speed_level:=fast \
  -p set_classic_walk:=true
```

---

### 3. Run YOLO + distance estimation

```bash
ros2 run yolo_distance_pkg go2_yolo_distance_node
```

Run person snapshot publisher:

```bash
ros2 run yolo_distance_pkg person_snapshot_publisher
```

Run remote debug viewer:

```bash
ros2 run yolo_distance_pkg remote_debug_viewer
```

---

### 4. Run VLM navigation stack

Before running VLM-related scripts, set your API key as an environment variable.

```bash
export ANTHROPIC_API_KEY="YOUR_API_KEY_HERE"
```

Do not write your real API key directly into source code.

Run the stack script:

```bash
chmod +x run_vlm_stack.sh
./run_vlm_stack.sh
```

Then run the VLM navigation script in another terminal:

```bash
python3 vlm/vlfm_source_nav_v22.py --execute --save-debug
```

---

## Important ROS2 Topics

| Topic                              | Description                       |
| ---------------------------------- | --------------------------------- |
| `/cmd_vel`                         | Velocity command input            |
| `/api/sport/request`               | Unitree Go2 sport command request |
| `/api/motion_switcher/request`     | Go2 motion mode switching request |
| `/utlidar/robot_odom`              | Unitree odometry input            |
| `/odom`                            | Relayed odometry output           |
| `/lidar_points`                    | Hesai LiDAR point cloud           |
| `/utlidar/cloud_deskewed`          | Go2 UTLidar point cloud           |
| `/scan`                            | LaserScan for navigation          |
| `/map`                             | Occupancy grid map                |
| `/yolo/debug_image`                | YOLO annotated image              |
| `/yolo/detection_text`             | Detection result text             |
| `/yolo/person_snapshot/compressed` | Compressed person snapshot        |
| `/fire_command`                    | Rescue-actuation command topic    |

---

## Safety Notes

* Test all robot motion in an open area before running autonomous navigation.
* Always keep an emergency stop method available.
* Do not commit API keys, tokens, model weights, ROS bag files, or large checkpoint files.
* Keep `.env`, `*.pth`, `*.pt`, `*.onnx`, `build/`, `install/`, and `log/` out of Git.
* Verify the robot is in the correct motion mode before sending velocity commands.
* ClassicWalk requires AI sport mode before activation.

---

## Suggested `.gitignore`

```gitignore
# ROS2 build files
build/
install/
log/

# Python
__pycache__/
*.pyc
venv/
.env

# Secrets
.env
*.key

# Large files
*.bag
*.db3
*.pth
*.pt
*.onnx
*.ckpt
checkpoints/
models/
weights/

# System
.DS_Store
```

---

## Project Purpose

This project demonstrates a real-world Physical AI robot system that connects perception, mapping, navigation, and action on physical hardware. Rather than remaining at the simulation level, the system focuses on deploying AI-based perception and autonomous navigation on the Unitree Go2 platform.

The project was developed as a capstone-style robotic system for rescue and disaster-response scenarios, with emphasis on practical ROS2 integration, sensor fusion, real robot control, and autonomous decision-making.

---

## Author

Kim Ji Seong
Robotics / Physical AI / ROS2 / Autonomous Systems
