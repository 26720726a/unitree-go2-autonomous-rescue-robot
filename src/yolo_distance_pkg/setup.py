from setuptools import setup
from glob import glob

package_name = 'yolo_distance_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            'share/' + package_name + '/launch',
            glob('launch/*.launch.py')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kjs',
    maintainer_email='kjs@todo.todo',
    description='YOLOv8 + RealSense D435i distance estimation package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_distance = yolo_distance_pkg.d435i_yolo_person_distance_s:main',
            'go2_yolo_distance_node = yolo_distance_pkg.go2_yolo_distance_node:main',
            'remote_debug_viewer = yolo_distance_pkg.remote_debug_viewer:main',
            'person_snapshot_publisher = yolo_distance_pkg.person_snapshot_publisher:main',
            'person_snapshot_receiver = yolo_distance_pkg.person_snapshot_receiver:main',
            'yolo_pose_distance = yolo_distance_pkg.yolo_pose_distance:main',
        ],
    },
)