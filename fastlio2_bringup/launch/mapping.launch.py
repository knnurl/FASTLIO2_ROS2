"""
mapping.launch.py
─────────────────
Run FASTLIO2 in mapping mode: LIO odometry only.
After mapping, call the SaveMap service to dump the map to PCD:

  ros2 service call /fastlio2/save_map interface/srv/SaveMaps \
      "{file_path: '/home/local/ISDADS/ses634/fastlio2_ws/maps/map.pcd', save_patches: false}"
"""
import launch
import launch_ros.actions
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    lio_config_path = PathJoinSubstitution(
        [FindPackageShare("fastlio2"), "config", "lio.yaml"]
    )

    declared_args = [
        DeclareLaunchArgument(
            "lio_config",
            default_value=lio_config_path,
            description="Path to LIO YAML config",
        ),
    ]

    lio_node = launch_ros.actions.Node(
        package="fastlio2",
        namespace="fastlio2",
        executable="lio_node",
        name="lio_node",
        output="screen",
        parameters=[{"config_path": LaunchConfiguration("lio_config")}],
    )

    return launch.LaunchDescription(declared_args + [lio_node])
