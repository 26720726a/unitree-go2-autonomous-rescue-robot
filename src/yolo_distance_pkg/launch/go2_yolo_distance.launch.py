from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolo_distance_pkg',
            executable='go2_yolo_distance_node',
            name='go2_yolo_distance_node',
            output='screen',
            parameters=[
                {
                    'model_path': 'yolov8s.pt',
                    'color_topic': '/camera/camera/color/image_raw',
                    'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
                    'debug_image_topic': '/yolo/debug_image',
                    'detection_text_topic': '/yolo/detection_text',
                    'conf_threshold': 0.4,
                    'sync_queue_size': 10,
                    'sync_slop': 0.1
                }
            ]
        )
    ])