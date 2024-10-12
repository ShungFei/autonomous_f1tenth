from datetime import datetime
import os
from ament_index_python import get_package_share_directory
from launch_ros.actions import Node 
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable
from launch.actions import (
    ExecuteProcess,
    RegisterEventHandler,
    DeclareLaunchArgument,
    OpaqueFunction,
    IncludeLaunchDescription,
    LogInfo,
)
from launch.event_handlers import OnShutdown
from launch.substitutions import LaunchConfiguration

import xacro

def launch(context, *args, **kwargs):
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_environments = get_package_share_directory('environments')
    pkg_f1tenth_bringup = get_package_share_directory('f1tenth_bringup')
    pkg_f1tenth_description = get_package_share_directory('f1tenth_description')
    xacro_file = os.path.join(pkg_f1tenth_description, "urdf", "robot.urdf.xacro")

    track = LaunchConfiguration('track').perform(context)
    car_one = LaunchConfiguration('car_one').perform(context)
    car_two = LaunchConfiguration('car_two').perform(context)
    camera_name = LaunchConfiguration("camera_name").perform(context)
    stereo_camera_name = LaunchConfiguration("stereo_camera_name").perform(context)
    is_stereo = LaunchConfiguration("stereo").perform(context).lower()
    fps = int(LaunchConfiguration("fps").perform(context))

    debug = LaunchConfiguration("debug").perform(context).lower() == "true"

    x = LaunchConfiguration("x").perform(context)
    y = LaunchConfiguration("y").perform(context)
    z = LaunchConfiguration("z").perform(context)

    R = LaunchConfiguration("R").perform(context)
    P = LaunchConfiguration("P").perform(context)
    Y = LaunchConfiguration("Y").perform(context)

    current_time = datetime.now().strftime('%y_%m_%d_%H:%M:%S')
    debug_dir = f"perception_debug/{current_time}"
    if debug:
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

    gz_sim = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': f'-s -r {pkg_environments}/worlds/{track}.sdf',
        }.items()
    )

    f1tenth_one = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_f1tenth_bringup, 'simulation_bringup.launch.py')),
        launch_arguments={
            'name': car_one,
            'world': 'empty',
            'x': '0',
            'y': '0',
            'z': '5',
            'add_camera': 'true',
            'camera_name': camera_name,
            'use_stereo': 'false',
        }.items()
    )

    f1tenth_two = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_f1tenth_bringup, 'simulation_bringup.launch.py')),
        launch_arguments={
            'name': car_two,
            'world': 'empty',
            'x': '0',
            'y': '1',
            'z': '5',
            'add_aruco': 'true',
        }.items()
    )

    service_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        arguments=[
            f'/world/empty/control@ros_gz_interfaces/srv/ControlWorld',
            f'/world/empty/create@ros_gz_interfaces/srv/SpawnEntity',
            f'/world/empty/remove@ros_gz_interfaces/srv/DeleteEntity',
            f'/world/empty/set_pose@ros_gz_interfaces/srv/SetEntityPose',
            f'/world/empty/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock',
            f"/{car_one}/{camera_name}/color/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            f"/{car_one}/{stereo_camera_name}/left/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            f"/{car_one}/{stereo_camera_name}/right/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ],
        remappings=[
            (f'/world/empty/clock', f'/clock'),
        ],
    )

    controller = Node(
        package='controllers',
        executable='ftg_policy',
        output='screen',
        parameters=[
            {'car_name': car_two, 'track_name': track},
        ],
    )

    state_estimation_node = Node(
        package="perception",
        executable="state_estimation",
        name="state_estimation",
        output="screen",
        parameters=[
            {
                "debug": debug,
                "debug_dir": debug_dir,
                "opponent_name": car_two,
                "fps": fps,
            }
        ],
        emulate_tty=True,
    )

    return[gz_sim, service_bridge, controller, state_estimation_node,
           f1tenth_one, f1tenth_two,
        #    *spawn_model_from_xacro(
        #         xacro_file,
        #         car_one,
        #         x,
        #         y,
        #         z,
        #         R,
        #         P,
        #         Y,
        #         add_camera="true",
        #         camera_name="d435",
        #         use_stereo=is_stereo,
        #     ),
        #     *spawn_model_from_xacro(
        #         xacro_file, car_two, 0, 0, 0, 0, 0, 0, add_aruco="true"
        #     ),
            *(monocular_nodes(car_one, camera_name, car_two, debug_dir, debug=debug) if is_stereo == "false" else stereo_nodes(car_one, stereo_camera_name, car_two, debug_dir, debug=debug)),
            RegisterEventHandler(
                OnShutdown(
                    on_shutdown=[
                        LogInfo(msg=(EnvironmentVariable(name='USER'),
                                ' destroyed the controller node')),
                        ExecuteProcess(
                            cmd=[[
                                "pkill -f \"gz\"", # Simulation does not stop on its own when Shutdown event is emitted
                                "pkill -f \"ros-args\"",
                            ]],
                            shell=True
                        ),
                    ]
                )
            )]

def generate_launch_description():
    track_arg = DeclareLaunchArgument(
        'track',
        default_value='track_1'
    )

    car_one = DeclareLaunchArgument(
        'car_one',
        default_value='f1tenth'
    )

    car_two = DeclareLaunchArgument(
        'car_two',
        default_value='opponent'
    )

    reset = Node(
            package='environments',
            executable='CarTrackReset', # I'm very sorry if this trips you up in the future
            output='screen',
            emulate_tty=True,
    )

    stepping_service = Node(
            package='environments',
            executable='SteppingService',
            output='screen',
            emulate_tty=True,
    )

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

    stereo_arg = DeclareLaunchArgument(
        name="stereo", description="stereo camera or not", default_value="false"
    )

    fps_arg = DeclareLaunchArgument(
        name="fps", description="frames per second", default_value="30"
    )

    debug_arg = DeclareLaunchArgument(
        name="debug", description="debug mode", default_value="false"
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

    ld = LaunchDescription([
        track_arg,
        car_one,
        car_two,
        reset,
        stepping_service,
        camera_name_arg,
        stereo_camera_name_arg,
        stereo_arg,
        fps_arg,
        debug_arg,
        x,
        y,
        z,
        R,
        P,
        Y,
        OpaqueFunction(function=launch),
    ])
    
    return ld


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

def stereo_nodes(name, camera_name, opponent_name, debug_dir, debug=False):
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
                    "debug_dir": f"perception_debug/{name}",
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

def monocular_nodes(name, camera_name, opponent_name, debug_dir, debug=False):
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
                    "is_sim": True,
                    "debug_dir": debug_dir,
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
