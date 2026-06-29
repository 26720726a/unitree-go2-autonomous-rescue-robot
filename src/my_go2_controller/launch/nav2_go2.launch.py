from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """
    Go2 + Hesai XT-16 통합 launch
      - Hesai 드라이버 + rviz2 (start.py 그대로 include)
      - base_link → hesai_lidar 정적 TF
      - 전방 장애물 검출 노드
      - Safety Gate 노드
    """

    # 1) Hesai 드라이버 launch 그대로 가져오기 (rviz도 같이 떠 있음)
    hesai_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hesai_ros_driver'),
                'launch',
                'start.py'
            )
        )
    )

    # 2) base_link → hesai_lidar 정적 TF (Unitree 공식 행렬)
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_hesai',
        arguments=[
            '0.171', '0.0', '0.0908',     # x y z (meters)
            '0.0', '0.0', '1.0', '0.0',   # qx qy qz qw (yaw=180)
            'base_link', 'hesai_lidar'
        ]
    )

    # 3) 전방 장애물 검출
    front_obstacle = Node(
        package='my_go2_controller',
        executable='front_obstacle',
        name='front_obstacle',
        output='screen',
        emulate_tty=True,
    )

    # 4) Safety Gate
    safety_gate = Node(
        package='my_go2_controller',
        executable='safety_gate',
        name='safety_gate',
        output='screen',
        emulate_tty=True,
    )
    # 5) gap_finder
    gap_finder  = Node (
        package='my_go2_controller',
        executable='gap_finder',
        name='gap_finder',
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        hesai_launch,
        static_tf,
        front_obstacle,
        safety_gate,
        gap_finder,
    ])
