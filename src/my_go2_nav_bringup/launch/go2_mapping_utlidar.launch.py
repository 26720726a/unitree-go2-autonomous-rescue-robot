import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    bringup_dir = get_package_share_directory('my_go2_nav_bringup')
    slam_params = os.path.join(bringup_dir, 'config', 'slam_toolbox.yaml')

    cmd_bridge_script = os.path.join(
        bringup_dir,
        'scripts',
        'cmd_vel_to_sport.py'
    )

    restamp_script = os.path.join(
        bringup_dir,
        'scripts',
        'restamp_cloud.py'
    )

    use_rviz = LaunchConfiguration('rviz')
    use_cmd_bridge = LaunchConfiguration('cmd_bridge')

    declare_rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Whether to start RViz2'
    )

    declare_cmd_bridge_arg = DeclareLaunchArgument(
        'cmd_bridge',
        default_value='true',
        description='Whether to start /cmd_vel to /api/sport/request bridge'
    )

    return LaunchDescription([

        declare_rviz_arg,
        declare_cmd_bridge_arg,

        # ============================================================
        # 1. Static TF: base_link -> camera_link
        # ============================================================
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_camera_tf',
            output='screen',
            arguments=[
                '0.37', '0.0', '0.09',   # x, y, z
                '0.0',  '0.0', '0.0',    # yaw, pitch, roll
                'base_link',
                'camera_link'
            ]
        ),

        # ============================================================
        # 2-1. Restamp UTLidar cloud to fix stale-timestamp TF lookup failure
        # ============================================================
        ExecuteProcess(
            cmd=[
                'python3', restamp_script,
                '--ros-args',
                '-p', 'input_topic:=/utlidar/cloud_deskewed',
                '-p', 'output_topic:=/utlidar/cloud_deskewed_restamped',
            ],
            output='screen',
        ),

        # ============================================================
        # 2-2. PointCloud2 -> LaserScan  (Go2 built-in UTLidar)
        # ============================================================
        Node(
            package='pointcloud_to_laserscan',
            executable='pointcloud_to_laserscan_node',
            name='pointcloud_to_laserscan',
            output='screen',
            remappings=[
                ('cloud_in', '/utlidar/cloud_deskewed_restamped'),
                ('scan', '/scan'),
            ],
            parameters=[{
                'target_frame': 'base_link',
                'transform_tolerance': 0.3,
                'min_height': -0.20,
                'max_height': 0.35,
                'angle_min': -3.14159265,
                'angle_max': 3.14159265,
                'angle_increment': 0.0087,
                'scan_time': 0.1,
                'range_min': 0.20,
                'range_max': 10.0,
                'use_inf': True,
                'inf_epsilon': 1.0,
            }]
        ),

        # ============================================================
        # 3. Unitree odom relay
        # ============================================================
        Node(
            package='my_go2_odom_relay',
            executable='odom_relay',
            name='odom_relay',
            output='screen',
            parameters=[{
                'input_odom': '/utlidar/robot_odom',
                'output_odom': '/odom',
                'frame_id': 'odom',
                'child_frame_id': 'base_link',
                'publish_tf': True,
                'force_2d': True,
                'restamp_with_now': True,
                'yaw_offset': -1.5708,
                'base_offset_x': 0.0,
                'base_offset_y': 0.10,
            }]
        ),

        # ============================================================
        # 4. SLAM Toolbox
        # ============================================================
        Node(
            package='slam_toolbox',
            executable='sync_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[slam_params],
        ),

        # ============================================================
        # 5. /cmd_vel -> /api/sport/request bridge
        # ============================================================
        ExecuteProcess(
            cmd=[
                'python3',
                cmd_bridge_script,
                '--ros-args',
                '-p', 'max_vx:=0.5',
                '-p', 'max_vy:=0.2',
                '-p', 'max_vyaw:=1.0',
                '-p', 'cmd_timeout:=0.5',
                '-p', 'publish_rate:=20.0',
                '-p', 'zero_deadband:=0.02',
            ],
            output='screen',
            condition=IfCondition(use_cmd_bridge)
        ),

        # ============================================================
        # 6. RViz2
        # ============================================================
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            condition=IfCondition(use_rviz)
        ),
    ])
