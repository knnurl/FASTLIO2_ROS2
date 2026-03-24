"""
mapping_full.launch.py
──────────────────────
Full mapping stack: Livox MID360 driver + FASTLIO2 LIO + RViz.

Prerequisites — source BOTH workspaces before launching:
  source /home/local/ISDADS/ses634/livox_ws/install/setup.bash
  source /home/local/ISDADS/ses634/fastlio2_ws/install/setup.bash

Launch:
  ros2 launch fastlio2_bringup mapping_full.launch.py

Optional overrides:
  ros2 launch fastlio2_bringup mapping_full.launch.py \
      livox_config:=/path/to/MID360_config.json \
      lio_config:=/path/to/lio.yaml

After mapping, save the map:
  ros2 service call /fastlio2/save_map interface/srv/SaveMaps \
      "{file_path: '/home/local/ISDADS/ses634/fastlio2_ws/maps/map.pcd', save_patches: false}"
"""
import os
import launch
import launch_ros.actions
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

LIVOX_WS = "/home/local/ISDADS/ses634/livox_ws/install"
DEFAULT_LIVOX_CONFIG = os.path.join(
    LIVOX_WS, "livox_ros_driver2", "share",
    "livox_ros_driver2", "config", "MID360_config.json"
)


def generate_launch_description():
    lio_config_path = PathJoinSubstitution(
        [FindPackageShare("fastlio2"), "config", "lio.yaml"]
    )
    rviz_config_path = PathJoinSubstitution(
        [FindPackageShare("fastlio2"), "rviz", "fastlio2.rviz"]
    )

    declared_args = [
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Launch RViz2 (set false when GUI manages RViz separately)",
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
            "rviz_config",
            default_value=rviz_config_path,
            description="Path to RViz config file",
        ),
    ]

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
        livox_node,
        lio_node,
        rviz_node,
    ])
