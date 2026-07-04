"""
localization.launch.py
──────────────────────
Run FASTLIO2 in LOCALIZATION mode against a pre-saved PCD map.

The localizer automatically performs global search (NDT grid + ICP) to
find the robot's starting pose, then switches to continuous ICP tracking.
No initial pose hint is required.

Prerequisites — source ALL workspaces before launching:
  source /home/local/ISDADS/ses634/livox_ws/install/setup.bash
  source /home/local/ISDADS/ses634/fastlio2_ws/install/setup.bash

Usage:
  ros2 launch fastlio2_bringup localization.launch.py

Custom map:
  ros2 launch fastlio2_bringup localization.launch.py \
    map_path:=/home/local/ISDADS/ses634/fastlio2_ws/maps/my_map.pcd

To keep the saved pose for warm-start (skip global search):
  ros2 launch fastlio2_bringup localization.launch.py force_global_search:=false

To manually trigger global re-localization at any time:
  ros2 service call /localizer/global_relocalize interface/srv/GlobalRelocalize \
      "{pcd_path: '', force: true}"
"""
import launch
import launch_ros.actions
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

POSE_SAVE_PATH = "/tmp/fastlio2_last_pose.yaml"


def generate_launch_description():
    lio_config_path = PathJoinSubstitution(
        [FindPackageShare("fastlio2"), "config", "lio.yaml"]
    )
    loc_config_path = PathJoinSubstitution(
        [FindPackageShare("localizer"), "config", "localizer.yaml"]
    )

    declared_args = [
        DeclareLaunchArgument(
            "map_path",
            default_value="/home/local/ISDADS/ses634/fastlio2_ws/maps/map.pcd",
            description="Absolute path to pre-saved PCD map file",
        ),
        DeclareLaunchArgument(
            "lio_config",
            default_value=lio_config_path,
            description="Path to LIO YAML config",
        ),
        DeclareLaunchArgument(
            "loc_config",
            default_value=loc_config_path,
            description="Path to localizer YAML config",
        ),
        DeclareLaunchArgument(
            "force_global_search",
            default_value="true",
            description="Delete saved pose on startup to force full global search "
                        "(set false to use warm-start from last saved pose)",
        ),
    ]

    # Delete the saved pose file so the localizer skips warm-start and runs a
    # full global search.  Runs only when force_global_search:=true (default).
    clear_saved_pose = ExecuteProcess(
        cmd=["rm", "-f", POSE_SAVE_PATH],
        output="screen",
        condition=IfCondition(LaunchConfiguration("force_global_search")),
    )

    # ── LIO node: computes odometry and publishes body_cloud + odom ───────────
    lio_node = launch_ros.actions.Node(
        package="fastlio2",
        namespace="fastlio2",
        executable="lio_node",
        name="lio_node",
        output="screen",
        parameters=[{"config_path": LaunchConfiguration("lio_config")}],
    )

    # ── Localizer node: global search then ICP tracking ───────────────────────
    localizer_node = launch_ros.actions.Node(
        package="localizer",
        namespace="localizer",
        executable="localizer_node",
        name="localizer_node",
        output="screen",
        parameters=[
            {"config_path": LaunchConfiguration("loc_config")},
            {"map_path":    LaunchConfiguration("map_path")},
        ],
    )

    return launch.LaunchDescription(declared_args + [
        clear_saved_pose,
        lio_node,
        localizer_node,
    ])
