"""
mapping_spot.launch.py
───────────────────────
KISS-ICP mapping on Boston Dynamics Spot + Velodyne VLP-16 (IMU-free).

Nodes launched:
  • dbscan_filter_node — people removal from raw scans (optional, on by default)
  • kiss_icp_node      — LiDAR odometry + incremental map (reads filtered cloud)
  • kiss_map_saver     — subscribes to /kiss/local_map, saves PCD on service call

Because KISS-ICP builds its local_map from its input, filtering before KISS-ICP
means the saved PCD is already people-free — no post-processing needed.

Prerequisites — source BOTH workspaces:
  source ~/ros2_ws/install/setup.bash
  source ~/fastlio2_ws/install/setup.bash

Launch:
  ros2 launch fastlio2_bringup mapping_spot.launch.py

Skip people filter:
  ros2 launch fastlio2_bringup mapping_spot.launch.py use_people_filter:=false

Save map when done:
  ros2 service call /kiss/save_map interface/srv/SaveMaps \\
      "{file_path: '$HOME/fastlio2_ws/maps/map_spot.pcd', save_patches: false}"
"""
import launch
import launch_ros.actions
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration, PathJoinSubstitution, PythonExpression
)
from launch_ros.substitutions import FindPackageShare

VELODYNE_TOPIC = '/spot/velodyne/points'


def generate_launch_description():
    kiss_config_path = PathJoinSubstitution(
        [FindPackageShare('fastlio2_bringup'), 'config', 'kiss_icp_spot.yaml']
    )
    filter_config_path = PathJoinSubstitution(
        [FindPackageShare('lidar_people_filter'), 'config', 'dbscan_filter.yaml']
    )

    declared_args = [
        DeclareLaunchArgument(
            'kiss_config',
            default_value=kiss_config_path,
            description='Path to KISS-ICP YAML config',
        ),
        DeclareLaunchArgument(
            'use_people_filter',
            default_value='true',
            description='Launch the DBSCAN people filter node (recommended for clean maps)',
        ),
        DeclareLaunchArgument(
            'filter_input_topic',
            default_value=VELODYNE_TOPIC,
            description='Raw cloud topic the people filter reads from',
        ),
        DeclareLaunchArgument(
            'filter_output_topic',
            default_value='/spot/velodyne/points_filtered',
            description='Filtered cloud topic the people filter publishes to',
        ),
        # KISS-ICP reads filtered cloud when filter is on, raw otherwise
        DeclareLaunchArgument(
            'kiss_input_topic',
            default_value=PythonExpression([
                "'", LaunchConfiguration('filter_output_topic'), "'",
                " if '", LaunchConfiguration('use_people_filter'), "' == 'true' else ",
                "'", LaunchConfiguration('filter_input_topic'), "'",
            ]),
            description='Cloud topic KISS-ICP subscribes to',
        ),
    ]

    # ── DBSCAN people filter (optional) ──────────────────────────────────────
    people_filter_node = launch_ros.actions.Node(
        package='lidar_people_filter',
        executable='dbscan_filter_node',
        name='dbscan_filter_node',
        output='screen',
        parameters=[
            filter_config_path,
            {
                'input_topic':  LaunchConfiguration('filter_input_topic'),
                'output_topic': LaunchConfiguration('filter_output_topic'),
            },
        ],
        condition=IfCondition(LaunchConfiguration('use_people_filter')),
    )

    # ── KISS-ICP odometry node ────────────────────────────────────────────────
    kiss_icp_node = launch_ros.actions.Node(
        package='kiss_icp',
        executable='kiss_icp_node',
        name='kiss_icp_node',
        output='screen',
        remappings=[('pointcloud_topic', LaunchConfiguration('kiss_input_topic'))],
        parameters=[
            LaunchConfiguration('kiss_config'),
            {
                'base_frame':           '',
                'lidar_odom_frame':     'odom_lidar',
                'publish_odom_tf':      True,
                'invert_odom_tf':       True,
                'publish_debug_clouds': True,   # enables /kiss/local_map
                'use_sim_time':         False,
            },
        ],
    )

    # ── Map saver node ────────────────────────────────────────────────────────
    map_saver_node = launch_ros.actions.Node(
        package='fastlio2_bringup',
        executable='kiss_map_saver.py',
        name='kiss_map_saver',
        output='screen',
    )

    return launch.LaunchDescription(declared_args + [
        people_filter_node,
        kiss_icp_node,
        map_saver_node,
    ])
