"""
localization_spot.launch.py
────────────────────────────
KISS-ICP + Global Localizer on Boston Dynamics Spot + Velodyne VLP-16 (IMU-free).

Topic wiring:
  /spot/velodyne/points → KISS-ICP input
  /kiss/odometry        → localizer odom_topic
  /spot/velodyne/points → localizer cloud_topic  (raw scan for ICP)
  /kiss/local_map       → localizer world_cloud_topic  (map update accumulation)

TF chain produced:
  map → odom_lidar  (from localizer)
  odom_lidar → <velodyne_frame>  (from KISS-ICP)

Prerequisites — source BOTH workspaces:
  source /home/local/ISDADS/ses634/ros2_ws/install/setup.bash
  source /home/local/ISDADS/ses634/fastlio2_ws/install/setup.bash

Usage:
  ros2 launch fastlio2_bringup localization_spot.launch.py

Custom map:
  ros2 launch fastlio2_bringup localization_spot.launch.py \
    map_path:=/home/local/ISDADS/ses634/fastlio2_ws/maps/map_spot.pcd

Warm-start from last saved pose:
  ros2 launch fastlio2_bringup localization_spot.launch.py force_global_search:=false

Trigger manual re-localization:
  ros2 service call /localizer/global_relocalize interface/srv/GlobalRelocalize \
      "{pcd_path: '', force: true}"
"""
import os
import launch
import launch_ros.actions
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

VELODYNE_TOPIC  = '/spot/velodyne/points'
POSE_SAVE_PATH  = '/tmp/spot_last_pose.yaml'


def generate_launch_description():
    kiss_config_path = PathJoinSubstitution(
        [FindPackageShare('fastlio2_bringup'), 'config', 'kiss_icp_spot.yaml']
    )
    loc_config_path = PathJoinSubstitution(
        [FindPackageShare('localizer'), 'config', 'localizer_spot.yaml']
    )

    declared_args = [
        DeclareLaunchArgument(
            'map_path',
            default_value=os.path.expanduser('~/fastlio2_ws/maps/map_spot.pcd'),
            description='Absolute path to pre-saved PCD map file',
        ),
        DeclareLaunchArgument(
            'kiss_config',
            default_value=kiss_config_path,
            description='Path to KISS-ICP YAML config',
        ),
        DeclareLaunchArgument(
            'loc_config',
            default_value=loc_config_path,
            description='Path to localizer YAML config',
        ),
        DeclareLaunchArgument(
            'force_global_search',
            default_value='true',
            description='Delete saved pose on startup to force full global search '
                        '(set false to use warm-start from last saved pose)',
        ),
    ]

    clear_saved_pose = ExecuteProcess(
        cmd=['rm', '-f', POSE_SAVE_PATH],
        output='screen',
        condition=IfCondition(LaunchConfiguration('force_global_search')),
    )

    # ── KISS-ICP odometry node ────────────────────────────────────────────────
    kiss_icp_node = launch_ros.actions.Node(
        package='kiss_icp',
        executable='kiss_icp_node',
        name='kiss_icp_node',
        output='screen',
        remappings=[('pointcloud_topic', VELODYNE_TOPIC)],
        parameters=[
            LaunchConfiguration('kiss_config'),
            {
                'base_frame':           '',
                'lidar_odom_frame':     'odom_lidar',
                'publish_odom_tf':      True,
                'invert_odom_tf':       True,
                'publish_debug_clouds': True,   # /kiss/local_map for world_cloud_topic
                'use_sim_time':         False,
            },
        ],
    )

    # ── Localizer node ────────────────────────────────────────────────────────
    localizer_node = launch_ros.actions.Node(
        package='localizer',
        namespace='localizer',
        executable='localizer_node',
        name='localizer_node',
        output='screen',
        parameters=[
            {'config_path': LaunchConfiguration('loc_config')},
            {'map_path':    LaunchConfiguration('map_path')},
            # cloud_topic override uses raw velodyne scan
            {'cloud_topic': VELODYNE_TOPIC},
        ],
    )

    return launch.LaunchDescription(declared_args + [
        clear_saved_pose,
        kiss_icp_node,
        localizer_node,
    ])
