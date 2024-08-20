import os
from ament_index_python import get_package_share_directory
import launch_ros
from launch_ros.actions import Node, SetParameter
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.substitutions import TextSubstitution
import yaml


alg_launch = {
    'astar': 'astar',
    'dstarlite': 'dstarlite',
}

def generate_launch_description():
    pkg_controllers = get_package_share_directory('controllers')

    config_path = os.path.join(
        pkg_controllers,
        'plan.yaml'
    )

    config = yaml.load(open(config_path), Loader=yaml.Loader)
    alg = config['plan']['ros__parameters']['algorithm']
    
    alg = Node(
            package='controllers',
            executable='planner',
            output='screen',
            parameters=[{'alg': TextSubstitution(text=str(alg))}]
        )


    


    return LaunchDescription([
        SetParameter(name='use_sim_time', value=True),
        alg,
])

