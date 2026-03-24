"""
localization_full.launch.py
────────────────────────────
Full localization stack: Livox MID360 driver + FASTLIO2 LIO
+ Global Localizer + RViz.

The localizer performs an automatic global search (NDT grid + ICP)
to find the robot's pose in the pre-saved map — no initial pose hint needed.

Prerequisites — source ALL workspaces before launching:
  source /home/local/ISDADS/ses634/livox_ws/install/setup.bash
  source /home/local/ISDADS/ses634/fastlio2_ws/install/setup.bash

Launch:
  ros2 launch fastlio2_bringup localization_full.launch.py

Custom map:
  ros2 launch fastlio2_bringup localization_full.launch.py \
    map_path:=/home/local/ISDADS/ses634/fastlio2_ws/maps/my_map.pcd

To keep the saved pose for warm-start (skip global search):
  ros2 launch fastlio2_bringup localization_full.launch.py force_global_search:=false

Trigger manual re-localization at any time (e.g. after kidnapping):
  ros2 service call /localizer/global_relocalize interface/srv/GlobalRelocalize \
      "{pcd_path: '', force: true}"
"""
import os
import launch
import launch_ros.actions
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare

POSE_SAVE_PATH = "/tmp/fastlio2_last_pose.yaml"

LIVOX_WS = "/home/local/ISDADS/ses634/livox_ws/install"
DEFAULT_LIVOX_CONFIG = os.path.join(
    LIVOX_WS, "livox_ros_driver2", "share",
    "livox_ros_driver2", "config", "MID360_config.json"
)


def generate_launch_description():
    lio_config_path = PathJoinSubstitution(
        [FindPackageShare("fastlio2"), "config", "lio.yaml"]
    )
    loc_config_path = PathJoinSubstitution(
        [FindPackageShare("localizer"), "config", "localizer.yaml"]
    )
    filter_config_path = PathJoinSubstitution(
        [FindPackageShare("lidar_people_filter"), "config", "dbscan_filter.yaml"]
    )
    rviz_config_path = PathJoinSubstitution(
        [FindPackageShare("localizer"), "rviz", "localizer.rviz"]
    )

    declared_args = [
        DeclareLaunchArgument(
            "map_path",
            default_value="/home/local/ISDADS/ses634/fastlio2_ws/maps/map.pcd",
            description="Absolute path to pre-saved PCD map file",
        ),
        DeclareLaunchArgument(
            "livox_config",
            default_value=DEFAULT_LIVOX_CONFIG,
            description="Path to Livox driver JSON config (MID360_config.json)",
        ),
        DeclareLaunchArgument(
            "lio_config",
            default_value=lio_config_path,
            description="Path to FASTLIO2 YAML config",
        ),
        DeclareLaunchArgument(
            "loc_config",
            default_value=loc_config_path,
            description="Path to localizer YAML config",
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=rviz_config_path,
            description="Path to RViz config file",
        ),
        # ── people filter flags ───────────────────────────────────────────────
        DeclareLaunchArgument(
            "use_people_filter",
            default_value="true",
            description="Launch the DBSCAN people filter node",
        ),
        DeclareLaunchArgument(
            "filter_input_topic",
            default_value="/fastlio2/body_cloud",
            description="Raw cloud topic the people filter reads from",
        ),
        DeclareLaunchArgument(
            "filter_output_topic",
            default_value="/fastlio2/body_cloud_filtered",
            description="Filtered cloud topic the people filter publishes to",
        ),
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Launch RViz2 (set false when GUI manages RViz separately)",
        ),
        DeclareLaunchArgument(
            "force_global_search",
            default_value="true",
            description="Delete saved pose on startup to force full global search "
                        "(set false to use warm-start from last saved pose)",
        ),
        DeclareLaunchArgument(
            "localizer_cloud_topic",
            default_value=PythonExpression([
                "'", LaunchConfiguration("filter_output_topic"), "'",
                " if '", LaunchConfiguration("use_people_filter"), "' == 'true' else ",
                "'", LaunchConfiguration("filter_input_topic"), "'",
            ]),
            description="Cloud topic the localizer subscribes to. "
                        "Defaults to filter_output_topic when use_people_filter=true, "
                        "else filter_input_topic.",
        ),
    ]

    # Delete the saved pose file so the localizer skips warm-start and runs a
    # full global search.  Runs only when force_global_search:=true (default).
    clear_saved_pose = ExecuteProcess(
        cmd=["rm", "-f", POSE_SAVE_PATH],
        output="screen",
        condition=IfCondition(LaunchConfiguration("force_global_search")),
    )

    # ── Livox MID360 driver ───────────────────────────────────────────────────
    livox_node = launch_ros.actions.Node(
        package="livox_ros_driver2",
        executable="livox_ros_driver2_node",
        name="livox_ros_driver2_node",
        output="screen",
        parameters=[{
            "user_config_path": LaunchConfiguration("livox_config"),
            "xfer_format": 1,       # 1 = CustomMsg (required by fastlio2)
            "multi_topic": 0,
            "data_src": 0,
            "publish_freq": 10.0,
            "output_data_type": 0,
            "frame_id": "livox_frame",
            "lvx_file_path": "",
            "ros_data_type": 1,
        }],
    )

    # ── FASTLIO2 LIO node ─────────────────────────────────────────────────────
    lio_node = launch_ros.actions.Node(
        package="fastlio2",
        namespace="fastlio2",
        executable="lio_node",
        name="lio_node",
        output="screen",
        parameters=[{"config_path": LaunchConfiguration("lio_config")}],
    )

    # ── DBSCAN people filter (optional) ──────────────────────────────────────
    people_filter_node = launch_ros.actions.Node(
        package="lidar_people_filter",
        executable="dbscan_filter_node",
        name="dbscan_filter_node",
        output="screen",
        parameters=[
            filter_config_path,
            {
                "input_topic":  LaunchConfiguration("filter_input_topic"),
                "output_topic": LaunchConfiguration("filter_output_topic"),
            },
        ],
        condition=IfCondition(LaunchConfiguration("use_people_filter")),
    )

    # ── Localizer node (global mode: auto-searches on startup) ───────────────
    localizer_node = launch_ros.actions.Node(
        package="localizer",
        namespace="localizer",
        executable="localizer_node",
        name="localizer_node",
        output="screen",
        parameters=[
            {"config_path":   LaunchConfiguration("loc_config")},
            {"map_path":      LaunchConfiguration("map_path")},
            {"cloud_topic":   LaunchConfiguration("localizer_cloud_topic")},
        ],
    )

    # ── RViz2 ─────────────────────────────────────────────────────────────────
    rviz_node = launch_ros.actions.Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        condition=IfCondition(LaunchConfiguration("launch_rviz")),
    )

    return launch.LaunchDescription(declared_args + [
        clear_saved_pose,
        livox_node,
        lio_node,
        people_filter_node,
        localizer_node,
        rviz_node,
    ])
