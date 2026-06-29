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
    """

    # 1. Hesai 드라이버 launch 그대로 가져오기 (rviz도 같이 떠 있음)
    hesai_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hesai_ros_driver'),
                'launch',
                'start.py'
            )
        )
    )

    # 2. base_link → hesai_lidar 정적 TF (Unitree 공식 행렬)
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_hesai',
        arguments=[
            '0.171', '0.0', '0.0908',   # x y z (meters)
            '0.0', '0.0', '1.0', '0.0',         # qx qy qz qw (yaw=180)
            'base_link', 'hesai_lidar'
        ]
    )

    return LaunchDescription([
        hesai_launch,
        static_tf,
    ])