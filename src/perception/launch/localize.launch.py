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
    ]

def monocular_nodes(name, camera_name, opponent_name, enable_auto_exposure, exposure_time, gain, debug=False):
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
                    "enable_auto_exposure": enable_auto_exposure,
                    "exposure_time": exposure_time,
                    "gain": gain,
                    "debug": debug,
                }
            ],
            emulate_tty=True,
        ),
	]
def spawn_func(context, *args, **kwargs):
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")
    pkg_environments = get_package_share_directory("environments")

    description_pkg_path = os.path.join(
        get_package_share_directory("f1tenth_description")
    )

    name = LaunchConfiguration("name").perform(context)
    camera_name = LaunchConfiguration("camera_name").perform(context)
    stereo_camera_name = LaunchConfiguration("stereo_camera_name").perform(context)
    opponent_name = LaunchConfiguration("opponent_name").perform(context)
    is_stereo = LaunchConfiguration("stereo").perform(context).lower()
    debug = LaunchConfiguration("debug").perform(context).lower() == "true"
    enable_auto_exposure = LaunchConfiguration("enable_auto_exposure").perform(context).lower() == "true"
    exposure_time = int(LaunchConfiguration("exposure_time").perform(context))
    gain = int(LaunchConfiguration("gain").perform(context))

    return [
        *(monocular_nodes(name, camera_name, opponent_name, enable_auto_exposure, exposure_time, gain, debug=debug) if is_stereo == "false" \
			else stereo_nodes(name, stereo_camera_name, opponent_name, debug=debug)),
    ]

def generate_launch_description():
    pkg_f1tenth_description = get_package_share_directory("f1tenth_description")

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

    auto_exposure_arg = DeclareLaunchArgument(
        name="enable_auto_exposure", description="enable auto exposure", default_value="false"
    )

    exposure_time_arg = DeclareLaunchArgument(
        name="exposure_time", description="exposure time", default_value="50"
    )

    gain_arg = DeclareLaunchArgument(
        name="gain", description="gain", default_value="128"
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable(
                name="GZ_SIM_RESOURCE_PATH", value=pkg_f1tenth_description[:-19]
            ),
            name_arg,
            stereo_arg,
            camera_name_arg,
            debug_arg,
            stereo_camera_name_arg,
            opponent_name_arg,
            auto_exposure_arg,
            exposure_time_arg,
            gain_arg,
            OpaqueFunction(function=spawn_func),
        ]
    )
