"""
semantic.launch.py
──────────────────
Launches the RandLA-Net semantic segmentation node standalone.

Can also be included in localization_full.launch.py:
    from launch.actions import IncludeLaunchDescription
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    from launch_ros.substitutions import FindPackageShare

Usage:
    ros2 launch lidar_semantic semantic.launch.py
    ros2 launch lidar_semantic semantic.launch.py model_path:=/path/to/weights.tar device:=cuda
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_path = PathJoinSubstitution(
        [FindPackageShare('lidar_semantic'), 'config', 'semantic.yaml']
    )

    declared_args = [
        DeclareLaunchArgument(
            'input_topic',
            default_value='/fastlio2/body_cloud_filtered', # _filtered
            description='PointCloud2 topic to segment',
        ),
        DeclareLaunchArgument(
            'model_path',
            default_value='/home/local/ISDADS/ses634/fastlio2_ws/src/lidar_semantic/weights/semantickitti_tsunghan_1d.tar',
            description='Path to pretrained RandLA-Net checkpoint (.pth/.tar)',
        ),
        DeclareLaunchArgument(
            'device',
            default_value='cuda',
            description='Inference device: cpu or cuda',
        ),
        DeclareLaunchArgument(
            'num_points',
            default_value='8192',
            description='Points per inference call (subsample/pad input cloud)',
        ),
        DeclareLaunchArgument(
            'decimation',
            default_value='1',
            description='Run inference every Nth incoming cloud',
        ),
    ]

    node = Node(
        package='lidar_semantic',
        executable='randlanet_node',
        name='randlanet_node',
        output='screen',
        parameters=[
            config_path,
            {
                'input_topic': LaunchConfiguration('input_topic'),
                'model_path':  LaunchConfiguration('model_path'),
                'device':      LaunchConfiguration('device'),
                'num_points':  LaunchConfiguration('num_points'),
                'decimation':  LaunchConfiguration('decimation'),
            },
        ],
    )

    return LaunchDescription(declared_args + [node])
