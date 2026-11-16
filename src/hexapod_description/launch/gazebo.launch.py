import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


def generate_launch_description():
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_hexapod_description = get_package_share_directory('hexapod_description')

    # Parent directory of 'hexapod_description' in install/share
    # Required for Gazebo Sim to resolve 'model://hexapod_description/meshes/...'
    install_share_dir = os.path.dirname(pkg_hexapod_description)

    # Conda/RoboStack environments do not export the Gazebo system plugin and
    # QML module paths, so Gazebo cannot find ign_ros2_control nor the IgnGazebo
    # QML module (Entity Tree / Component Inspector panels) without this.
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    conda_lib = os.path.join(conda_prefix, 'lib')
    qml_import_path = os.path.join(conda_lib, 'ign-gazebo-6', 'plugins', 'gui')

    if conda_prefix and os.path.isdir(conda_lib):
        for var in ('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', 'GZ_SIM_SYSTEM_PLUGIN_PATH'):
            current = os.environ.get(var)
            os.environ[var] = conda_lib if not current else os.pathsep.join((conda_lib, current))

    gz_resource_path = AppendEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=install_share_dir
    )

    ign_resource_path = AppendEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=install_share_dir
    )

    extra_env_actions = []
    if conda_prefix and os.path.isdir(qml_import_path):
        extra_env_actions.append(
            AppendEnvironmentVariable(
                name='QML2_IMPORT_PATH',
                value=qml_import_path
            )
        )

    robot_description_file = os.path.join(pkg_hexapod_description, 'urdf', 'hexapod.xacro')
    ros_gz_bridge_config = os.path.join(pkg_hexapod_description, 'config', 'ros_gz_bridge_gazebo.yaml')

    robot_description_config = xacro.process_file(robot_description_file)
    robot_description = {'robot_description': robot_description_config.toxml()}

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description],
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-r -v 4 empty.sdf'}.items()
    )

    spawn_robot = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='ros_gz_sim',
                executable='create',
                arguments=[
                    '-topic', '/robot_description',
                    '-name', 'hexapod',
                    '-allow_renaming', 'false',
                    '-x', '0.0',
                    '-y', '0.0',
                    '-z', '0.25',
                    '-Y', '0.0'
                ],
                output='screen'
            )
        ]
    )

    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{'config_file': ros_gz_bridge_config}],
        output='screen'
    )

    gait = LaunchConfiguration('gait')
    declare_gait = DeclareLaunchArgument(
        'gait',
        default_value='true',
        description='Start the Cartesian gait controller'
    )

    control_gui = LaunchConfiguration('control_gui')
    declare_control_gui = DeclareLaunchArgument(
        'control_gui',
        default_value='false',
        description='Start the PyQt desktop control interface'
    )

    gait_node = TimerAction(
        period=10.0,
        condition=IfCondition(gait),
        actions=[
            Node(
                package='hexapod_controller',
                executable='gait',
                name='hexapod_gait',
                parameters=[{'use_sim_time': True}],
                output='screen'
            )
        ]
    )

    control_gui_node = TimerAction(
        period=12.0,
        condition=IfCondition(control_gui),
        actions=[
            Node(
                package='hexapod_controller',
                executable='control_gui',
                name='hexapod_control_gui',
                output='screen'
            )
        ]
    )

    load_controllers = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_state_broadcaster', '--controller-manager-timeout', '60'],
                output='screen'
            ),
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_group_position_controller', '--controller-manager-timeout', '60'],
                output='screen'
            ),
        ]
    )

    return LaunchDescription([
        declare_gait,
        declare_control_gui,
        gz_resource_path,
        ign_resource_path,
        *extra_env_actions,
        gazebo,
        spawn_robot,
        ros_gz_bridge,
        robot_state_publisher,
        load_controllers,
        gait_node,
        control_gui_node,
    ])
