"""
full_system.launch.py
─────────────────────
Full closed-loop mapping system: FAST-LIO2 + PGO + HBA + Localiser +
People filter + Map Health Monitor + Validator + GUI.

Launch:
  ros2 launch fastlio2_bringup full_system.launch.py

Without health monitoring:
  ros2 launch fastlio2_bringup full_system.launch.py use_health_monitor:=false
"""
import os
import launch
import launch_ros.actions
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    GroupAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def _find_livox_config():
    """Locate the Livox MID360 config; fall back gracefully if driver not installed."""
    # Check environment variable override first
    env_ws = os.environ.get("LIVOX_WS", "")
    if env_ws:
        candidate = os.path.join(
            env_ws, "install", "livox_ros_driver2", "share",
            "livox_ros_driver2", "config", "MID360_config.json")
        if os.path.isfile(candidate):
            return candidate
    # Try ament resource index (works when livox_ros_driver2 is in the same workspace)
    try:
        from ament_index_python.packages import get_package_share_directory
        pkg_share = get_package_share_directory("livox_ros_driver2")
        return os.path.join(pkg_share, "config", "MID360_config.json")
    except Exception:
        pass
    return ""


DEFAULT_LIVOX_CONFIG = _find_livox_config()


def generate_launch_description():
    lio_config = PathJoinSubstitution(
        [FindPackageShare("fastlio2"), "config", "lio.yaml"]
    )
    pgo_config = PathJoinSubstitution(
        [FindPackageShare("pgo"), "config", "pgo.yaml"]
    )
    hba_config = PathJoinSubstitution(
        [FindPackageShare("hba"), "config", "hba.yaml"]
    )
    localizer_config = PathJoinSubstitution(
        [FindPackageShare("localizer"), "config", "localizer.yaml"]
    )
    people_filter_config = PathJoinSubstitution(
        [FindPackageShare("lidar_people_filter"), "config", "dbscan_filter.yaml"]
    )

    declared_args = [
        DeclareLaunchArgument("use_health_monitor", default_value="true",
                              description="Enable map health monitor and validation"),
        DeclareLaunchArgument("use_live_data", default_value="true",
                              description="Launch Livox MID360 driver"),
        DeclareLaunchArgument("use_people_filter", default_value="true",
                              description="Enable DBSCAN people filter"),
        DeclareLaunchArgument("use_pgo", default_value="true",
                              description="Enable pose graph optimization"),
        DeclareLaunchArgument("map_path", default_value="",
                              description="Path to pre-saved PCD map (empty for mapping mode)"),
        DeclareLaunchArgument("launch_rviz", default_value="false",
                              description="Launch RViz2"),
        DeclareLaunchArgument("launch_gui", default_value="false",
                              description="Launch PyQt5 GUI"),

        # Health monitor parameters
        DeclareLaunchArgument("degradation_threshold", default_value="0.4"),
        DeclareLaunchArgument("improvement_threshold", default_value="0.05"),
        DeclareLaunchArgument("staleness_decay_s", default_value="300.0"),
        DeclareLaunchArgument("staleness_hba_threshold", default_value="0.8"),
        DeclareLaunchArgument("confidence_threshold", default_value="0.7"),
        DeclareLaunchArgument("verify_scan_count", default_value="30"),
        DeclareLaunchArgument("pre_merge_baseline_scans", default_value="20"),
        DeclareLaunchArgument("partition_cell_size_m", default_value="50.0"),
        DeclareLaunchArgument("partition_radius_cells", default_value="1"),
        DeclareLaunchArgument("merge_policy", default_value="TIMESTAMP_PRIORITY"),

        # Validation parameters
        DeclareLaunchArgument("delta_m", default_value="1.0"),
        DeclareLaunchArgument("ate_regression_threshold", default_value="0.05"),
    ]

    # ── 1. FAST-LIO2 core ────────────────────────────────────────────────────
    lio_node = launch_ros.actions.Node(
        package="fastlio2",
        namespace="fastlio2",
        executable="lio_node",
        name="lio_node",
        output="screen",
        parameters=[{"config_path": lio_config}],
    )

    # ── 2. PGO / Loop closure ────────────────────────────────────────────────
    pgo_node = launch_ros.actions.Node(
        package="pgo",
        executable="pgo_node",
        name="pgo_node",
        output="screen",
        parameters=[{"config_path": pgo_config}],
        condition=IfCondition(LaunchConfiguration("use_pgo")),
    )

    # ── 3. HBA ───────────────────────────────────────────────────────────────
    hba_node = launch_ros.actions.Node(
        package="hba",
        executable="hba_node",
        name="hba_node",
        output="screen",
        parameters=[{"config_path": hba_config}],
    )

    # ── 4. Localiser ─────────────────────────────────────────────────────────
    localizer_node = launch_ros.actions.Node(
        package="localizer",
        executable="localizer_node",
        name="localizer_node",
        output="screen",
        parameters=[{"config_path": localizer_config}],
    )

    # ── 5. People filter ─────────────────────────────────────────────────────
    people_filter_node = launch_ros.actions.Node(
        package="lidar_people_filter",
        executable="dbscan_filter_node",
        name="dbscan_filter_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_people_filter")),
    )

    # ── 6. Map health monitor (health system) ────────────────────────────────
    health_monitor = launch_ros.actions.Node(
        package="fastlio2_health",
        executable="map_health_monitor",
        name="map_health_monitor",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_health_monitor")),
        parameters=[{
            "degradation_threshold": LaunchConfiguration("degradation_threshold"),
            "improvement_threshold": LaunchConfiguration("improvement_threshold"),
            "staleness_decay_s": LaunchConfiguration("staleness_decay_s"),
            "staleness_hba_threshold": LaunchConfiguration("staleness_hba_threshold"),
            "confidence_threshold": LaunchConfiguration("confidence_threshold"),
            "verify_scan_count": LaunchConfiguration("verify_scan_count"),
            "pre_merge_baseline_scans": LaunchConfiguration("pre_merge_baseline_scans"),
            "partition_cell_size_m": LaunchConfiguration("partition_cell_size_m"),
            "partition_radius_cells": LaunchConfiguration("partition_radius_cells"),
            "merge_policy": LaunchConfiguration("merge_policy"),
        }],
    )

    # ── 7. Voxel merge node ──────────────────────────────────────────────────
    merge_node = launch_ros.actions.Node(
        package="fastlio2_health",
        executable="voxel_merge_node",
        name="voxel_merge_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_health_monitor")),
        parameters=[{
            "merge_policy": LaunchConfiguration("merge_policy"),
        }],
    )

    # ── 8. Map validator (standalone, no GT) ─────────────────────────────────
    validator_node = launch_ros.actions.Node(
        package="fastlio2_validation",
        executable="map_validator",
        name="map_validator",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_health_monitor")),
        parameters=[{
            "delta_m": LaunchConfiguration("delta_m"),
            "ate_regression_threshold": LaunchConfiguration("ate_regression_threshold"),
        }],
    )

    return launch.LaunchDescription(declared_args + [
        lio_node,
        pgo_node,
        hba_node,
        localizer_node,
        people_filter_node,
        health_monitor,
        merge_node,
        validator_node,
    ])
