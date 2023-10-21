from environments.CarBlockEnvironment import CarBlockEnvironment
from environments.CarGoalEnvironment import CarGoalEnvironment
from environments.CarTrackEnvironment import CarTrackEnvironment
from environments.CarWallEnvironment import CarWallEnvironment
from environments.CarBeatEnvironment import CarBeatEnvironment

class EnvironmentFactory:
    def __init__(self):
        pass

    def create(self, name, config):
        print(config)
        if name == 'CarBlock':
            return CarBlockEnvironment()
        elif name == 'CarGoal':
            return CarGoalEnvironment()
        elif name == 'CarTrack':
            return CarTrackEnvironment(
                config['car_name'], 
                config['reward_range'], 
                config['max_steps'], 
                config['collision_range'], 
                config['step_length'], 
                config['track'], 
                config['observation_mode'], 
                config['max_goals']
                )
        elif name == 'CarWall':
            return CarWallEnvironment()
        elif name == 'CarBeat':
            return CarBeatEnvironment(
                config['car_name'],
                config['ftg_car_name'], 
                config['reward_range'], 
                config['max_steps'], 
                config['collision_range'], 
                config['step_length'], 
                config['track'], 
                config['observation_mode'], 
                config['max_goals'],
                config['num_lidar_points']
            )
        else:
            raise Exception('EnvironmentFactory: Environment not found')
