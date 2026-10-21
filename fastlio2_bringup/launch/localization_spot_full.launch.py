"""
localization_spot_full.launch.py
─────────────────────────────────
Full KISS-ICP localization stack on Boston Dynamics Spot + Velodyne VLP-16 (IMU-free).

Nodes launched:
  • kiss_icp_node       — LiDAR odometry
  • dbscan_filter_node  — people removal (optional, toggle with use_people_filter)
  • localizer_node      — global NDT+ICP search, then continuous ICP tracking
  • rviz2               — visualization (toggle with launch_rviz:=false)

TF chain produced:
  map → odom_lidar  (localizer)
  odom_lidar → <velodyne_frame>  (KISS-ICP)

Prerequisites — source BOTH workspaces:
  source ~/ros2_ws/install/setup.bash
  source ~/fastlio2_ws/install/setup.bash

Launch:
  ros2 launch fastlio2_bringup localization_spot_full.launch.py

Custom map:
  ros2 launch fastlio2_bringup localization_spot_full.launch.py \
    map_path:=$HOME/fastlio2_ws/maps/map_spot.pcd

Skip people filter:
  ros2 launch fastlio2_bringup localization_spot_full.launch.py use_people_filter:=false

Warm-start from last saved pose:
  ros2 launch fastlio2_bringup localization_spot_full.launch.py force_global_search:=false

Trigger manual re-localization:
  ros2 service call /localizer/global_relocalize interface/srv/GlobalRelocalize \
      "{pcd_path: '', force: true}"
"""
import os
import launch
import launch_ros.actions
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration, PathJoinSubstitution, PythonExpression
)
from launch_ros.substitutions import FindPackageShare

VELODYNE_TOPIC = '/spot/velodyne/points'
POSE_SAVE_PATH = '/tmp/spot_last_pose.yaml'


def generate_launch_description():
    kiss_config_path = PathJoinSubstitution(
        [FindPackageShare('fastlio2_bringup'), 'config', 'kiss_icp_spot.yaml']
    )
    loc_config_path = PathJoinSubstitution(
        [FindPackageShare('localizer'), 'config', 'localizer_spot.yaml']
    )
    filter_config_path = PathJoinSubstitution(
        [FindPackageShare('lidar_people_filter'), 'config', 'dbscan_filter.yaml']
    )
    rviz_config_path = PathJoinSubstitution(
        [FindPackageShare('localizer'), 'rviz', 'localizer_spot.rviz']
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
            'rviz_config',
            default_value=rviz_config_path,
            description='Path to RViz config file',
        ),
        # ── people filter ────────────────────────────────────────────────────
        DeclareLaunchArgument(
            'use_people_filter',
            default_value='true',
            description='Launch the DBSCAN people filter node',
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
        DeclareLaunchArgument(
            'launch_rviz',
            default_value='true',
            description='Launch RViz2 (set false when GUI manages RViz separately)',
        ),
        DeclareLaunchArgument(
            'force_global_search',
            default_value='true',
            description='Delete saved pose on startup to force full global search '
                        '(set false to use warm-start from last saved pose)',
        ),
        # Localizer cloud topic: filtered when filter is on, raw otherwise
        DeclareLaunchArgument(
            'localizer_cloud_topic',
            default_value=PythonExpression([
                "'", LaunchConfiguration('filter_output_topic'), "'",
                " if '", LaunchConfiguration('use_people_filter'), "' == 'true' else ",
                "'", LaunchConfiguration('filter_input_topic'), "'",
            ]),
            description='Cloud topic the localizer subscribes to.',
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
                'publish_debug_clouds': True,
                'use_sim_time':         False,
            },
        ],
    )

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

    # ── Localizer node ────────────────────────────────────────────────────────
    localizer_node = launch_ros.actions.Node(
        package='localizer',
        namespace='localizer',
        executable='localizer_node',
        name='localizer_node',
        output='screen',
        parameters=[
            {'config_path':  LaunchConfiguration('loc_config')},
            {'map_path':     LaunchConfiguration('map_path')},
            {'cloud_topic':  LaunchConfiguration('localizer_cloud_topic')},
        ],
    )

    # ── RViz2 ─────────────────────────────────────────────────────────────────
    rviz_node = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        condition=IfCondition(LaunchConfiguration('launch_rviz')),
    )

    return launch.LaunchDescription(declared_args + [
        clear_saved_pose,
        kiss_icp_node,
        people_filter_node,
        localizer_node,
        rviz_node,
    ])
