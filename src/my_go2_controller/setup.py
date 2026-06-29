import os
from glob import glob
from setuptools import setup

package_name = 'my_go2_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='unitree',
    maintainer_email='unitree@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'state_reader = my_go2_controller.state_reader:main',
            'pitch_test = my_go2_controller.pitch_test:main',
            'move_test = my_go2_controller.move_test:main',
            'teleop_keyboard = my_go2_controller.teleop_keyboard:main',
            'front_obstacle = my_go2_controller.front_obstacle:main',
            'safety_gate = my_go2_controller.safety_gate:main',
            'gap_finder = my_go2_controller.gap_finder:main',
            'led_off = my_go2_controller.led_off:main',
        ],
    },
)
