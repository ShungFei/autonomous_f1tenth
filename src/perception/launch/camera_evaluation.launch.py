import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, EnvironmentVariable
from launch.actions import (
    ExecuteProcess,
    RegisterEventHandler,
    DeclareLaunchArgument,
    OpaqueFunction,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    Shutdown,
    LogInfo,
    EmitEvent,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
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

def stereo_nodes(name, camera_name, opponent_name, debug=False):
    return [
        Node(
            package="perception",
            executable="stereo_localize",
            name="stereo_localize",
            output="screen",
            parameters=[
                {
                    "agent_name": name,
                    "camera_name": camera_name,
                    "opponent_name": opponent_name,
                    "debug": debug,
                }
            ],
            emulate_tty=True,
        ),
        Node(
            package="ros_gz_image",
            executable="image_bridge",
            arguments=[f"{name}/{camera_name}/left/image_raw"],
            output="screen",
        ),
        Node(
            package="ros_gz_image",
            executable="image_bridge",
            arguments=[f"{name}/{camera_name}/right/image_raw"],
            output="screen",
        ),
    ]

def monocular_nodes(name, camera_name, opponent_name, eval_time, debug=False):
    return [
        Node(
            package="perception",
            executable="localize",
            name="localize",
            output="screen",
            parameters=[
                {
                    "agent_name": name,
                    "camera_name": camera_name,
                    "opponent_name": opponent_name,
                    "debug": debug,
                }
            ],
            emulate_tty=True,
        ),
        
        Node(
            package="ros_gz_image",
            executable="image_bridge",
            arguments=[f"/{name}/{camera_name}/color/image_raw"],
            output="screen",
        ),
        Node(
            package="ros_gz_image",
            executable="image_bridge",
            arguments=[f"{name}/{camera_name}/depth/image_rect_raw"],
            output="screen",
        ),
    ]


def spawn_func(context, *args, **kwargs):
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")
    pkg_environments = get_package_share_directory("environments")

    description_pkg_path = os.path.join(
        get_package_share_directory("f1tenth_description")
    )
    xacro_file = os.path.join(description_pkg_path, "urdf", "robot.urdf.xacro")

    gz_sim = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": f"-r {pkg_environments}/worlds/evaluation.sdf",
            'on_exit_shutdown': 'true'
        }.items(),
    )

    world = LaunchConfiguration("world").perform(context)
    name = LaunchConfiguration("name").perform(context)
    camera_name = LaunchConfiguration("camera_name").perform(context)
    stereo_camera_name = LaunchConfiguration("stereo_camera_name").perform(context)
    opponent_name = LaunchConfiguration("opponent_name").perform(context)
    is_stereo = LaunchConfiguration("stereo").perform(context).lower()

    debug = LaunchConfiguration("debug").perform(context).lower() == "true"
    eval_time = float(LaunchConfiguration("eval_time").perform(context))

    x = LaunchConfiguration("x").perform(context)
    y = LaunchConfiguration("y").perform(context)
    z = LaunchConfiguration("z").perform(context)

    R = LaunchConfiguration("R").perform(context)
    P = LaunchConfiguration("P").perform(context)
    Y = LaunchConfiguration("Y").perform(context)

    evaluation_node = Node(
        package="perception",
        executable="evaluation",
        name="evaluation",
        output="screen",
        parameters=[
            {
                "agent_name": name,
                "camera_name": camera_name if is_stereo == "false" else stereo_camera_name,
                "opponent_name": opponent_name,
                "eval_time": eval_time,
                "is_stereo": is_stereo == "true",
                "debug": debug,
            }
        ],
        emulate_tty=True,
    )

    return [
        gz_sim,
        *spawn_model_from_xacro(
            xacro_file,
            name,
            x,
            y,
            z,
            R,
            P,
            Y,
            add_camera="true",
            camera_name="d435",
            use_stereo=is_stereo,
        ),
        *spawn_model_from_xacro(
            xacro_file, opponent_name, 0, 0, 0, 0, 0, 0, add_aruco="true"
        ),
        Node(
            package="perception",
            executable="trajectory",
            name="trajectory",
            output="screen",
            parameters=[
                {
                    "agent_name": name,
                    "camera_name": camera_name if is_stereo == "false" else stereo_camera_name,
                    "opponent_name": opponent_name,
                    "is_stereo": is_stereo == "true",
                    "eval_time": eval_time,
                }
            ],
            emulate_tty=True,
        ),
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=[
                f"/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
                f"/model/{name}/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
                f"/model/{opponent_name}/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
                f"/{name}/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan",
                f"/{name}/{camera_name}/color/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
                f"/{name}/{stereo_camera_name}/left/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
                f"/{name}/{stereo_camera_name}/right/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
                f"/model/{name}/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry",
                f"/model/{name}/odometry_with_covariance@nav_msgs/msg/Odometry@gz.msgs.OdometryWithCovariance",
                f"/world/{world}/model/{name}/joint_state@sensor_msgs/msg/JointState@gz.msgs.Model",
                f"/model/{name}/pose@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
                f"/model/{opponent_name}/pose@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            ],
            remappings=[
                (f"/model/{name}/cmd_vel", f"/{name}/cmd_vel"),
                (f"/model/{opponent_name}/cmd_vel", f"/{opponent_name}/cmd_vel"),
                (f"/model/{name}/pose", f"/{name}/pose"),
                (f"/model/{opponent_name}/pose", f"/{opponent_name}/pose"),
                (f"/model/{name}/odometry", f"/{name}/odometry"),
                (
                    f"/model/{name}/odometry_with_covariance",
                    f"/{name}/odometry_with_covariance",
                ),
                (f"/world/{world}/model/{name}/joint_state", f"/{name}/joint_states"),
            ],
        ),
        evaluation_node,
        RegisterEventHandler(
            OnProcessExit(
                target_action=evaluation_node,
                on_exit=[
                    LogInfo(msg=(EnvironmentVariable(name='USER'),
                            ' destroyed the evaluation node')),
                    EmitEvent(event=Shutdown(
                        reason='Evaluation compelte'))
                ]
            )
        ),
        *(monocular_nodes(name, camera_name, opponent_name, eval_time, debug=debug) if is_stereo == "false" else stereo_nodes(name, stereo_camera_name, opponent_name, debug=debug)),
    ]


def generate_launch_description():
    pkg_f1tenth_description = get_package_share_directory("f1tenth_description")

    world_arg = DeclareLaunchArgument(name="world", description="name of world", default_value="empty")

    name_arg = DeclareLaunchArgument(name="name", description="name of robot spawned", default_value="f1tenth")

    camera_name_arg = DeclareLaunchArgument(
        name="camera_name",
        description="name of camera mounted on agent robot",
        default_value="d435",
    )

    stereo_camera_name_arg = DeclareLaunchArgument(
        name="stereo_camera_name",
        description="name of stereo camera mounted on agent robot",
        default_value="zed2",
    )

    opponent_name_arg = DeclareLaunchArgument(
        name="opponent_name",
        description="name of opponent robot spawned",
        default_value="opponent",
    )

    stereo_arg = DeclareLaunchArgument(
        name="stereo", description="stereo camera or not", default_value="false"
    )

    debug_arg = DeclareLaunchArgument(
        name="debug", description="debug mode", default_value="false"
    )

    eval_time_arg = DeclareLaunchArgument(
        name="eval_time", description="evaluation time", default_value="10.0"
    )

    x = DeclareLaunchArgument(
        name="x", description="x position of robot", default_value="-3.0"
    )

    y = DeclareLaunchArgument(
        name="y", description="y position of robot", default_value="0.0"
    )

    z = DeclareLaunchArgument(
        name="z", description="z position of robot", default_value="0.0"
    )

    R = DeclareLaunchArgument(
        name="R", description="roll of robot", default_value="0.0"
    )

    P = DeclareLaunchArgument(
        name="P", description="pitch of robot", default_value="0.0"
    )

    Y = DeclareLaunchArgument(name="Y", description="yaw of robot", default_value="0.0")

    return LaunchDescription(
        [
            SetEnvironmentVariable(
                name="GZ_SIM_RESOURCE_PATH", value=pkg_f1tenth_description[:-19]
            ),
            world_arg,
            name_arg,
            stereo_arg,
            camera_name_arg,
            debug_arg,
            eval_time_arg,
            stereo_camera_name_arg,
            opponent_name_arg,
            x,
            y,
            z,
            R,
            P,
            Y,            
            OpaqueFunction(function=spawn_func),
        ]
    )
