import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.actions import ExecuteProcess, DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription, SetEnvironmentVariable
from launch_ros.descriptions import ParameterValue
from launch_ros.actions import Node
from launch.conditions import IfCondition
from math import pi

import xacro

def spawn_model_from_xacro(xacro_file, name, x, y, z, R, P, Y, **kwargs):
    return Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': xacro.process_file(xacro_file, mappings={"robot_name": name, **kwargs}).toxml(),
            'frame_prefix': name
        }],
        remappings=[
            ('/tf', f'/{name}/tf'),
            ('/tf_static', f'/{name}/tf_static'),
        ],
        namespace=name
    ), Node(
        package='ros_gz_sim', executable='create',
        arguments=[
            '-world', 'empty',
            '-name', name,
            '-topic', f"/{name}/robot_description",
            '-x', f"{x}",
            '-y', f"{y}",
            '-z', f"{z}",
            '-R', f"{R}",
            '-P', f"{P}",
            '-Y', f"{Y}"
        ],
        output='screen'
    )

def spawn_func(context, *args, **kwargs):
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_environments = get_package_share_directory('environments')

    description_pkg_path = os.path.join(get_package_share_directory('f1tenth_description'))
    xacro_file = os.path.join(description_pkg_path, 'urdf', 'robot.urdf.xacro')

    gz_sim = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': f'-r {pkg_environments}/worlds/evaluation.sdf',
        }.items()
    )

    world = LaunchConfiguration('world').perform(context)
    name = LaunchConfiguration('name').perform(context)
    camera_name = LaunchConfiguration('camera_name').perform(context)
    opponent_name = LaunchConfiguration('opponent_name').perform(context)

    x = LaunchConfiguration('x').perform(context)
    y = LaunchConfiguration('y').perform(context)
    z = LaunchConfiguration('z').perform(context)

    R = LaunchConfiguration('R').perform(context)
    P = LaunchConfiguration('P').perform(context)

    Y = LaunchConfiguration('Y').perform(context)

    # with open("test.txt", 'w') as f:
    #     f.write(xacro.process_file(xacro_file, mappings={"robot_name": name, "add_camera":"true", "add_aruco":"true"}).toxml())

    return [
        gz_sim,
        Node(
            package ='perception',
            executable='localize',
            name='localize',
            output='screen',
            parameters=[{
                'agent_name': name,
                'camera_name': camera_name,
                'opponent_name': opponent_name
            }],
            emulate_tty=True
        ),
        *spawn_model_from_xacro(xacro_file, name, x, y, z, R, P, Y, add_camera="true", camera_name="d435"),
        *spawn_model_from_xacro(xacro_file, opponent_name, 0, 0, 0, 0, 0, 0, add_aruco="true"),
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                f'/model/{name}/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
                f'/{name}/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
                f'/{name}/d435/color/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
                f'/{name}/zed2/left/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
                f'/{name}/zed2/right/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
                f'/model/{name}/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry',
                f'/model/{name}/odometry_with_covariance@nav_msgs/msg/Odometry@gz.msgs.OdometryWithCovariance',
                # f'/model/{name}/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V',
                f'/world/{world}/model/{name}/joint_state@sensor_msgs/msg/JointState@gz.msgs.Model',
                # f'/world/{world}/pose/info@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V',
                f'/model/{name}/pose@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V',
                f'/model/{opponent_name}/pose@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V',
            ],
            remappings=[
                (f'/model/{name}/cmd_vel', f'/{name}/cmd_vel'),
                (f'/model/{name}/pose', f'/{name}/pose'),
                (f'/model/{opponent_name}/pose', f'/{opponent_name}/pose'),
                (f'/model/{name}/odometry', f'/{name}/odometry'),
                (f'/model/{name}/odometry_with_covariance', f'/{name}/odometry_with_covariance'),
                (f'/world/{world}/model/{name}/joint_state', f'/{name}/joint_states'),
                # (f'/model/{name}/tf', '/tf'),
            ]
        ),
        Node(
             package='ros_gz_image',
             executable='image_bridge',
             arguments=['/f1tenth/d435/color/image_raw'],
             output='screen',
        ),
        Node(
            package='ros_gz_image',
            executable='image_bridge',
            arguments=['/f1tenth/d435/depth/image_rect_raw'],
            output='screen',
        )
        # Node(
        #     package='ros_gz_image',
        #     executable='image_bridge',
        #     arguments=['/f1tenth/zed2/left/image_raw'],
        #     output='screen',
        # ),
        # Node(
        #     package='ros_gz_image',
        #     executable='image_bridge',
        #     arguments=['/f1tenth/zed2/right/image_raw'],
        #     output='screen',
        # ),
    ]


def generate_launch_description():
    pkg_f1tenth_description = get_package_share_directory('f1tenth_description')

    world_arg = DeclareLaunchArgument(
        name='world',
        description='name of world'
    )

    name_arg = DeclareLaunchArgument(
        name='name',
        description='name of robot spawned'
    )

    camera_name_arg = DeclareLaunchArgument(
        name='camera_name',
        description='name of camera mounted on agent robot',
        default_value='d435'
    )

    opponent_name_arg = DeclareLaunchArgument(
        name='opponent_name',
        description='name of opponent robot spawned',
        default_value='opponent'
    )

    x = DeclareLaunchArgument(
        name='x',
        description='x position of robot',
        default_value='3.0'
    )

    y = DeclareLaunchArgument(
        name='y',
        description='y position of robot',
        default_value='3.0'
    )

    z = DeclareLaunchArgument(
        name='z',
        description='z position of robot',
        default_value='3.0'
    )

    R = DeclareLaunchArgument(
        name='R',
        description='roll of robot',
        default_value='0.0'
    )

    P = DeclareLaunchArgument(
        name='P',
        description='pitch of robot',
        default_value='0.0'
    )

    Y = DeclareLaunchArgument(
        name='Y',
        description='yaw of robot',
        default_value='0.0'
    )

    return LaunchDescription([
        SetEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', value=pkg_f1tenth_description[:-19]),
        world_arg,
        name_arg,
        camera_name_arg,
        opponent_name_arg,
        x,
        y,
        z,
        R,
        P,
        Y,
        OpaqueFunction(function=spawn_func)
    ])
