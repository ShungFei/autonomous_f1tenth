from datetime import datetime
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

def spawn_func(context, *args, **kwargs):

    name = LaunchConfiguration("name").perform(context)
    camera_name = LaunchConfiguration("camera_name").perform(context)
    bev_camera_name = LaunchConfiguration("stereo_camera_name").perform(context)
    opponent_name = LaunchConfiguration("opponent_name").perform(context)
    fps = int(LaunchConfiguration("fps").perform(context))
    side_length = float(LaunchConfiguration("side_length").perform(context))

    debug = LaunchConfiguration("debug").perform(context).lower() == "true"
    eval_time = float(LaunchConfiguration("eval_time").perform(context))

    current_time = datetime.now().strftime('%y_%m_%d_%H:%M:%S')
    debug_dir = f"perception_debug/{current_time}"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    return [
        Node(
            package="perception",
            executable="trajectory",
            name="trajectory",
            output="screen",
            parameters=[
                {
                    "agent_name": name,
                    "camera_name": camera_name,
                    "opponent_name": opponent_name,
                    "is_stereo": False,
                    "eval_time": eval_time,
                }
            ],
            emulate_tty=True,
        ),
        Node(
            package="perception",
            executable="state_estimation",
            name="state_estimation",
            output="screen",
            parameters=[
                {
                    "opponent_name": opponent_name,
                    "debug_dir": debug_dir,
                }
            ],
            emulate_tty=True,
        ),
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
                    "debug_dir": debug_dir,
                }
            ],
            emulate_tty=True,
        ),
        Node(
            package="perception",
            executable="bev_track",
            name="bev_track",
            output="screen",
            parameters=[
                {
                    "camera_name": bev_camera_name,
                    "debug": debug,
                    "fps": fps,
                    "side_length": side_length,
                    "debug_dir": debug_dir,
                }
                
            ],
            
        )
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

    fps_arg = DeclareLaunchArgument(
        name="fps", description="frames per second", default_value="60"
    )

    side_length_arg = DeclareLaunchArgument(
        name="side_length", description="side length of the ArUco marker", default_value="0.15"
    )

    debug_arg = DeclareLaunchArgument(
        name="debug", description="debug mode", default_value="false"
    )
    
    eval_time_arg = DeclareLaunchArgument(
        name="eval_time", description="evaluation time", default_value="10.0"
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable(
                name="GZ_SIM_RESOURCE_PATH", value=pkg_f1tenth_description[:-19]
            ),
            world_arg,
            name_arg,
            stereo_arg,
            camera_name_arg,
            fps_arg,
            side_length_arg,
            debug_arg,
            eval_time_arg,
            stereo_camera_name_arg,
            opponent_name_arg,       
            OpaqueFunction(function=spawn_func),
        ]
    )
