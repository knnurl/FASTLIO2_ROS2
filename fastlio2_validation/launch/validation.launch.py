from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('delta_m', default_value='1.0'),
        DeclareLaunchArgument('ate_regression_threshold', default_value='0.05'),

        Node(
            package='fastlio2_validation',
            executable='map_validator',
            name='map_validator',
            output='screen',
            parameters=[{
                'delta_m': LaunchConfiguration('delta_m'),
                'ate_regression_threshold': LaunchConfiguration('ate_regression_threshold'),
            }],
        ),
    ])
