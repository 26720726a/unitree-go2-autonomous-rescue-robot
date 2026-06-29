from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """
    Go2 TF tree setup:
      base_link ──> hesai_lidar
    
    공식 변환 (Unitree 문서):
      T(Go2 body IMU → XT-16) = translation (0.171, 0, 0.0908), no rotation
    """
    return LaunchDescription([
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_hesai',
            arguments=[
                '0.171', '0.0', '0.0908',   # x y z (meters)
                '0.0', '0.0', '0.0',         # yaw pitch roll (rad)
                'base_link', 'hesai_lidar'
            ]
        ),
    ])