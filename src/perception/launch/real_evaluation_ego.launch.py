from datetime import datetime
import os
from timeit import Timer
from turtle import width

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
  opponent_name = LaunchConfiguration("opponent_name").perform(context)
  width = int(LaunchConfiguration("width").perform(context))
  height = int(LaunchConfiguration("height").perform(context))
  fps = int(LaunchConfiguration("fps").perform(context))
  side_length = float(LaunchConfiguration("side_length").perform(context))

  debug = LaunchConfiguration("debug").perform(context).lower() == "true"
  eval_time = float(LaunchConfiguration("eval_time").perform(context))

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

  realsense_node = Node(
    package="realsense2_camera",
    executable="realsense2_camera_node",
    name=camera_name,
    namespace=agent_name,
    output="screen",
    parameters=[
      {
        "rgb_camera.color_profile": f"{width}x{height}x{fps}",
        "enable_color": True,
        "enable_depth": False,
        "enable_infra1": False,
        "enable_infra2": False,
        "rgb_camera.enable_auto_exposure": False,
        "rgb_camera.exposure": 166, # 166 is the default value
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

  return [
    # hardware_bringup,
    realsense_node,
    state_estimation_node,
    localize_node,
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

  width_arg = DeclareLaunchArgument(
    name="width", description="width of the camera image", default_value="1920"
  )

  height_arg = DeclareLaunchArgument(
    name="height", description="height of the camera image", default_value="1080"
  )

  fps_arg = DeclareLaunchArgument(
    name="fps", description="frames per second", default_value="30"
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
      width_arg,
      height_arg,
      fps_arg,
      side_length_arg,
      debug_arg,
      eval_time_arg,
      opponent_name_arg,       
      OpaqueFunction(function=spawn_func),
    ]
  )
