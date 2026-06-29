from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        # 1) 전방 장애물 검출
        Node(
            package='my_go2_controller',
            executable='front_obstacle',
            name='front_obstacle',
            output='screen',
            emulate_tty=True,
        ),

        # 2) Safety Gate
        Node(
            package='my_go2_controller',
            executable='safety_gate',
            name='safety_gate',
            output='screen',
            emulate_tty=True,
        ),
    ])