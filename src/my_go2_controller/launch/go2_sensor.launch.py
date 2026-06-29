from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Hesai 드라이버 (rviz는 제외하는 게 좋음 - 아래 주의사항 참고)
    hesai_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hesai_ros_driver'),
                'launch',
                'norviz_start.py'
            )
        )
    )

    # base_link → hesai_lidar 정적 TF
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_hesai',
        arguments=[
            '0.171', '0.0', '0.0908',
            '0.0', '0.0', '1.0', '0.0',
            'base_link', 'hesai_lidar'
        ]
    )

    return LaunchDescription([
        hesai_launch,
        static_tf,
    ])
