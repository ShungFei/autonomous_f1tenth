from datetime import datetime
import os
from timeit import Timer

from ament_index_python import get_package_share_directory
from launch import LaunchDescription, LaunchContext
from launch.events.process import ProcessStarted
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
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.events import Shutdown
from launch_ros.descriptions import ParameterValue
from launch_ros.actions import Node
from launch.conditions import IfCondition
from math import pi

def spawn_func(context, *args, **kwargs):

    agent_name = LaunchConfiguration("name").perform(context)
    camera_name = LaunchConfiguration("camera_name").perform(context)
    bev_camera_name = LaunchConfiguration("bev_camera_name").perform(context)
    opponent_name = LaunchConfiguration("opponent_name").perform(context)
    fps = int(LaunchConfiguration("fps").perform(context))
    side_length = float(LaunchConfiguration("side_length").perform(context))

    debug = LaunchConfiguration("debug").perform(context).lower() == "true"
    eval_time = float(LaunchConfiguration("eval_time").perform(context))

    current_time = datetime.now().strftime('%y_%m_%d_%H:%M:%S')
    debug_dir = f"perception_debug/{current_time}"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    trajectory_node = Node(
        package="perception",
        executable="trajectory",
        name="trajectory",
        output="screen",
        parameters=[
            {
                "agent_name": agent_name,
                "camera_name": camera_name,
                "opponent_name": opponent_name,
                "is_sim": False,
                "is_stereo": False,
                "eval_time": eval_time,
            }
        ],
        emulate_tty=True,
    )
    
    state_estimation_node = Node(
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
    )

    localize_node = Node(
        package="perception",
        executable="localize",
        name="localize",
        output="screen",
        parameters=[
            {
                "agent_name": agent_name,
                "camera_name": camera_name,
                "opponent_name": opponent_name,
                "debug": debug,
                "debug_dir": debug_dir,
            }
        ],
        emulate_tty=True,
    )

    bev_track_node = Node(
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
        emulate_tty=True,
    )

    return [
        bev_track_node,
        state_estimation_node,
        localize_node,
        TimerAction(period=3.0, actions=[trajectory_node]), # trajectory should start publishing after the cameras have initialized
        RegisterEventHandler(
            OnProcessExit(
                target_action=trajectory_node,
                on_exit=[
                    LogInfo(msg=(EnvironmentVariable(name='USER'),
                            ' destroyed the trajectory node')),
                    EmitEvent(event=Shutdown(
                        reason='Trajectory completed. Shutting down.')),
                ]
            )
        ),
    ]
def generate_launch_description():

    world_arg = DeclareLaunchArgument(name="world", description="name of world", default_value="empty")

    agent_name_arg = DeclareLaunchArgument(name="name", description="name of agent vehicle", default_value="f1tenth")

    camera_name_arg = DeclareLaunchArgument(
        name="camera_name",
        description="name of camera mounted on agent vehicle",
        default_value="d435",
    )

    bev_camera_name_arg = DeclareLaunchArgument(
        name="bev_camera_name",
        description="name of bev camera",
        default_value="zed",
    )

    opponent_name_arg = DeclareLaunchArgument(
        name="opponent_name",
        description="name of opponent robot spawned",
        default_value="opponent",
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
            world_arg,
            agent_name_arg,
            camera_name_arg,
            fps_arg,
            side_length_arg,
            debug_arg,
            eval_time_arg,
            bev_camera_name_arg,
            opponent_name_arg,       
            OpaqueFunction(function=spawn_func),
        ]
    )
