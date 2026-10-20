from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('degradation_threshold', default_value='0.4'),
        DeclareLaunchArgument('improvement_threshold', default_value='0.05'),
        DeclareLaunchArgument('staleness_decay_s', default_value='300.0'),
        DeclareLaunchArgument('staleness_hba_threshold', default_value='0.8'),
        DeclareLaunchArgument('confidence_threshold', default_value='0.7'),
        DeclareLaunchArgument('verify_scan_count', default_value='30'),
        DeclareLaunchArgument('pre_merge_baseline_scans', default_value='20'),
        DeclareLaunchArgument('partition_cell_size_m', default_value='50.0'),
        DeclareLaunchArgument('partition_radius_cells', default_value='1'),
        DeclareLaunchArgument('merge_policy', default_value='TIMESTAMP_PRIORITY'),
        DeclareLaunchArgument('voxel_resolution', default_value='0.3'),

        Node(
            package='fastlio2_health',
            executable='map_health_monitor',
            name='map_health_monitor',
            output='screen',
            parameters=[{
                'degradation_threshold': LaunchConfiguration('degradation_threshold'),
                'improvement_threshold': LaunchConfiguration('improvement_threshold'),
                'staleness_decay_s': LaunchConfiguration('staleness_decay_s'),
                'staleness_hba_threshold': LaunchConfiguration('staleness_hba_threshold'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'verify_scan_count': LaunchConfiguration('verify_scan_count'),
                'pre_merge_baseline_scans': LaunchConfiguration('pre_merge_baseline_scans'),
                'partition_cell_size_m': LaunchConfiguration('partition_cell_size_m'),
                'partition_radius_cells': LaunchConfiguration('partition_radius_cells'),
                'merge_policy': LaunchConfiguration('merge_policy'),
            }],
        ),

        Node(
            package='fastlio2_health',
            executable='voxel_merge_node',
            name='voxel_merge_node',
            output='screen',
            parameters=[{
                'merge_policy': LaunchConfiguration('merge_policy'),
                'voxel_resolution': LaunchConfiguration('voxel_resolution'),
            }],
        ),
    ])
