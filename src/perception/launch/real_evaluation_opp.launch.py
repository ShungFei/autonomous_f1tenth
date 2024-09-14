from datetime import datetime
import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, EnvironmentVariable
from launch.actions import (
  RegisterEventHandler,
  DeclareLaunchArgument,
  OpaqueFunction,
  IncludeLaunchDescription,
  Shutdown,
  LogInfo,
  EmitEvent,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node

def spawn_func(context, *args, **kwargs):

  agent_name = LaunchConfiguration("name").perform(context)
  opponent_name = LaunchConfiguration("opponent_name").perform(context)

  debug = LaunchConfiguration("debug").perform(context).lower() == "true"

  current_time = datetime.now().strftime('%y_%m_%d_%H:%M:%S')
  debug_dir = f"perception_debug/{current_time}"
  if not os.path.exists(debug_dir):
    os.makedirs(debug_dir)

  pkg_f1tenth_bringup = get_package_share_directory("f1tenth_bringup")

  hardware_bringup = IncludeLaunchDescription(
      launch_description_source=PythonLaunchDescriptionSource(
          os.path.join(pkg_f1tenth_bringup, "hardware_bringup.launch.py")
      ),
  )

  car_trajectory_node = Node(
    package="perception",
    executable="car_trajectory",
    name="car_trajectory",
    output="screen",
    parameters=[
      {
        "is_opponent": True,
        "agent_name": agent_name,
        "opponent_name": opponent_name,
        "debug": debug,
        "debug_dir": debug_dir,
      }
    ],
    emulate_tty=True,
  )

  return [
    hardware_bringup,
    car_trajectory_node,
    RegisterEventHandler(
      OnProcessExit(
        target_action=car_trajectory_node,
        on_exit=[
          LogInfo(msg=(EnvironmentVariable(name='USER'),
                  ' destroyed the car_trajectory node')),
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

  opponent_name_arg = DeclareLaunchArgument(
    name="opponent_name",
    description="name of opponent robot spawned",
    default_value="opponent",
  )

  debug_arg = DeclareLaunchArgument(
    name="debug", description="debug mode", default_value="false"
  )
  
  return LaunchDescription(
    [
      world_arg,
      agent_name_arg,
      camera_name_arg,
      debug_arg,
      opponent_name_arg,       
      OpaqueFunction(function=spawn_func),
    ]
  )
