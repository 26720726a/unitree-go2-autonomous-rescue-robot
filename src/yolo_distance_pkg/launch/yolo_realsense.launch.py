from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ),
        launch_arguments={
            'align_depth.enable': 'true'
        }.items()
    )

    yolo_node = Node(
        package='yolo_distance_pkg',
        executable='yolo_distance',
        name='yolo_distance_node',
        output='screen'
    )

    return LaunchDescription([
        realsense_launch,
        yolo_node
    ])