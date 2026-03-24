from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'lidar_people_filter'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml') + glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='LiDAR people detection and removal for 3D mapping',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'people_filter_node = lidar_people_filter.people_filter_node:main',
            'background_subtractor_node = lidar_people_filter.background_subtractor_node:main',
            'dbscan_filter_node = lidar_people_filter.dbscan_filter_node:main',
        ],
    },
)
