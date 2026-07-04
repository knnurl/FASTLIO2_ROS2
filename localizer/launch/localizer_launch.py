import launch
import launch_ros.actions
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

POSE_SAVE_PATH = "/tmp/fastlio2_last_pose.yaml"


def generate_launch_description():
    rviz_cfg = PathJoinSubstitution(
        [FindPackageShare("localizer"), "rviz", "localizer.rviz"]
    )
    localizer_config_path = PathJoinSubstitution(
        [FindPackageShare("localizer"), "config", "localizer.yaml"]
    )
    lio_config_path = PathJoinSubstitution(
        [FindPackageShare("fastlio2"), "config", "lio.yaml"]
    )

    declared_args = [
        DeclareLaunchArgument(
            "force_global_search",
            default_value="true",
            description="Delete saved pose on startup to force full global search "
                        "(set false to use warm-start from last saved pose)",
        ),
    ]

    clear_saved_pose = ExecuteProcess(
        cmd=["rm", "-f", POSE_SAVE_PATH],
        output="screen",
        condition=IfCondition(LaunchConfiguration("force_global_search")),
    )

    return launch.LaunchDescription(
        declared_args + [
            clear_saved_pose,
            launch_ros.actions.Node(
                package="fastlio2",
                namespace="fastlio2",
                executable="lio_node",
                name="lio_node",
                output="screen",
                parameters=[
                    {"config_path": lio_config_path.perform(launch.LaunchContext())}
                ],
            ),
            launch_ros.actions.Node(
                package="localizer",
                namespace="localizer",
                executable="localizer_node",
                name="localizer_node",
                output="screen",
                parameters=[
                    {
                        "config_path": localizer_config_path.perform(
                            launch.LaunchContext()
                        )
                    }
                ],
            ),
            launch_ros.actions.Node(
                package="rviz2",
                namespace="localizer",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_cfg.perform(launch.LaunchContext())],
            ),
        ]
    )
