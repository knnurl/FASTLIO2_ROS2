"""
Two-stage people filter launch file.
  Stage 1: DBSCAN filter      — /cloud_registered → /cloud_filtered
  Stage 2: octomap_server    — /cloud_filtered + TF → clean 3D map

Both nodes are CPU-pinned away from FastLIO to prevent odometry jitter.
FastLIO must already be running before launching this file.

Usage:
  ros2 launch lidar_people_filter people_filter.launch.py

If FastLIO uses a different world frame:
  ros2 launch lidar_people_filter people_filter.launch.py world_frame:=map
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg = get_package_share_directory('lidar_people_filter')

    world_frame_arg = DeclareLaunchArgument(
        'world_frame',
        default_value='camera_init',
        description=(
            'Fixed frame FastLIO publishes TF in. '
            'Verify with: ros2 topic echo --once /cloud_registered | grep frame_id'
        )
    )

    dbscan_node = Node(
        package='lidar_people_filter',
        executable='dbscan_filter_node',
        name='dbscan_filter_node',
        output='screen',
        parameters=[os.path.join(pkg, 'config', 'dbscan_filter.yaml')],
        # Pin to cores 4-5 — keeps FastLIO latency stable on Jetson Orin.
        # Adjust if your platform has fewer than 6 cores.
        prefix='taskset -c 4,5',
    )

    octomap_node = Node(
        package='octomap_server',
        executable='octomap_server_node',
        name='octomap_server',
        output='screen',
        prefix='taskset -c 6,7',
        remappings=[('cloud_in', '/fastlio2/world_cloud_filtered')],
        parameters=[
            os.path.join(pkg, 'config', 'octomap.yaml'),
            {'frame_id': LaunchConfiguration('world_frame')},
        ],
    )

    return LaunchDescription([
        world_frame_arg,
        dbscan_node,
        octomap_node,
    ])
