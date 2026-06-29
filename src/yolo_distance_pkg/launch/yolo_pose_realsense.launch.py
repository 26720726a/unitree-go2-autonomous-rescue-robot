from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        Node(
            package='yolo_distance_pkg',
            executable='yolo_pose_distance',
            name='yolo_pose_distance_node',

            # Use torch bundled libgomp directly
            prefix='env LD_PRELOAD=/home/unitree/.local/lib/python3.8/site-packages/torch.libs/libgomp-804f19d4.so.1.0.0',

            parameters=[{
                'pose_model_path':       'yolov8n-pose.pt',
                'det_model_path':        'yolov8n.pt',
                'color_topic':           '/camera/color/image_raw',
                'depth_topic':           '/camera/aligned_depth_to_color/image_raw',
                'debug_image_topic':     '/yolo_pose/debug_image',
                'detection_text_topic':  '/yolo_pose/detection_text',
                'conf_threshold':        0.4,
                'sync_queue_size':       10,
                'sync_slop':             0.1,
            }],
            output='screen',
        ),
    ])