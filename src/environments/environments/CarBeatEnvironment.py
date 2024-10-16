import math
import numpy as np
from environments.CarTrackEnvironment import CarTrackEnvironment
from environments.util import get_euler_from_quarternion
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage
from perception_interfaces.msg import StateEstimateStamped

class CarBeatEnvironment(CarTrackEnvironment):

    """
    CarBeat Reinforcement Learning Environment:

        Task:
            Agent learns to drive a track and overtake a car that is driving at a constant speed.
            The second car is using the Follow The Gap algorithm.

        Observation:
            full:
                Car Position (x, y)
                Car Orientation (x, y, z, w)
                Car Velocity
                Car Angular Velocity
                Lidar Data
            no_position:
                Car Orientation (x, y, z, w)
                Car Velocity
                Car Angular Velocity
                Lidar Data
            lidar_only:
                Car Velocity
                Car Angular Velocity
                Lidar Data

            No. of lidar points is configurable

        Action:
            It's linear and angular velocity (Twist)
        
        Reward:
            +2 if it comes within REWARD_RANGE units of a goal
            +100 if it overtakes the Follow The Gap car
            -25 if it collides with a wall

        Termination Conditions:
            When the agent collides with a wall or the Follow The Gap car
        
        Truncation Condition:
            When the number of steps surpasses MAX_GOALS
    """
    def __init__(self,
                 rl_car_name,
                 ftg_car_name,
                 reward_range=1,
                 max_steps=50,
                 collision_range=0.2,
                 step_length=0.5,
                 track='multi_track',
                 observation_mode='lidar_only',
                 opponent_state_estimation_mode='kalman_filter_ca',
                 max_goals=500
                 ):
        
        super().__init__(rl_car_name, reward_range, max_steps, collision_range, step_length, track)
        # Perception state estimates -------------------------------------
        self.opponent_state_estimation_mode = opponent_state_estimation_mode
        match opponent_state_estimation_mode:
            case 'none':
                self.latest_opponent_state_estimate = None
                self.state_length = 0
            case 'raw_marker_pose':
                self.state_length = 3
                self.latest_opponent_state_estimate = None
                self.state_estimate_sub = self.create_subscription(
                    PoseStamped,
                    f'/opponent/pose_estimate',
                    lambda data: setattr(self, 'latest_opponent_state_estimate', data),
                    10
                )
            case 'kalman_filter_ca':
                self.state_length = 9
                self.latest_opponent_state_estimate = None
                self.state_estimate_sub = self.create_subscription(
                    StateEstimateStamped,
                    f'/opponent/state_estimate/{opponent_state_estimation_mode}',
                    lambda data: setattr(self, 'latest_opponent_state_estimate', data),
                    10
                )
            case 'ground_truth':
                # TODO: Add ground truth velocity
                self.state_length = 3
                self.latest_opponent_state_estimate = None
                self.pose = self.create_subscription(
                    StateEstimateStamped,
                    f'{ftg_car_name}/ground_truth/relative_pose',
                    lambda data: setattr(self, 'latest_opponent_state_estimate', data),
                    10
                )
        
        self.latest_opponent_pose = None
        self.opponent_ground_truth_pose_sub = self.create_subscription(
            TFMessage,
            f'{ftg_car_name}/pose',
            lambda data: setattr(self, 'latest_opponent_pose', data),
            10
        )
        self.OBSERVATION_SIZE += self.state_length

        # Environment Details ----------------------------------------
        self.OPPONENT_NAME = ftg_car_name
        # self.MAX_GOALS = max_goals # Unused

        
        # Opponent & Overtaking -----------------------------------------------
        self.ftg_car_name = ftg_car_name
        self.is_over_taken = False
        self.num_steps_over_taken = 0
        self.old_t_ego = None
        self.old_t_opponent = None
        self.t_ego_laps = 0
        self.t_opponent_laps = 0
        self.is_first_reward_since_reset = True
        self.ftg_goals_reached = 0
        self.ftg_start_waypoint_index = 0
        self.ftg_offset = 0


    def reset(self):
        state, info = super().reset()
        car_index = info['index'] # This isn't great practice, but the starting position of the opponent car is dependent on the ego car's starting position
        
        self.ftg_offset = np.random.randint(8, 12)
        self.ftg_goals_reached = 0

        # Starting point for the ftg car
        if self.is_evaluating: # This check exists for ego vehicle in CarTrackEnvironment so do for consistency
            ftg_x, ftg_y, ftg_yaw, ftg_index = self.track_waypoints[3 + self.ftg_offset] # Ego vehicle starts at index 3
        else:
            ftg_x, ftg_y, ftg_yaw, ftg_index = self.track_waypoints[(car_index + self.ftg_offset) % len(self.track_waypoints)]

        self.ftg_start_waypoint_index = ftg_index

        goal_x, goal_y, _, _ = self.track_waypoints[self.ftg_start_waypoint_index+1 if self.ftg_start_waypoint_index+1 < len(self.track_waypoints) else 0]
        self.ftg_goal_position  = [goal_x, goal_y]

        self.call_reset_service(
            car_x=ftg_x,
            car_y=ftg_y,
            car_Y=ftg_yaw,
            goal_x=goal_x,
            goal_y=goal_y,
            car_name=self.OPPONENT_NAME
        )

        self.is_over_taken = False # Needs to come after the opponent vehicle is reset, or there may be an accidental "overtaking"
        self.num_steps_over_taken = 0
        self.old_t_ego = None
        self.old_t_opponent = None
        self.t_ego_laps = 0
        self.t_opponent_laps = 0
        self.is_first_reward_since_reset = True

        return state, info

    def get_observation(self):
        state, full_state, raw_lidar_range = super().get_observation()

        state_observation = []
        if self.latest_opponent_state_estimate:
            opp_state = self.latest_opponent_state_estimate
            if self.opponent_state_estimation_mode == 'raw_marker_pose':
                # Convert quaternion to euler angle
                _, rotation_y, _ = get_euler_from_quarternion(opp_state.pose.orientation.w, opp_state.pose.orientation.x, opp_state.pose.orientation.y, opp_state.pose.orientation.z)
                state_observation += [opp_state.pose.position.x, opp_state.pose.position.z, rotation_y]
            elif self.opponent_state_estimation_mode == 'ground_truth':
                state_observation += [opp_state.position.x, opp_state.position.y, opp_state.orientation.z]
            else:
                state_observation += [opp_state.position.x, opp_state.position.z,
                                    opp_state.linear_velocity.x, opp_state.linear_velocity.z,
                                    opp_state.linear_acceleration.x, opp_state.linear_acceleration.z,
                                    opp_state.orientation.y, opp_state.angular_velocity.y, opp_state.angular_acceleration.y]
        else:
            state_observation += [0] * self.state_length

        state += state_observation

        return state, full_state, raw_lidar_range

    def compute_reward(self, state, next_state, raw_lidar_range):
        reward, reward_info = super().compute_reward(state, next_state, raw_lidar_range)

        if not self.latest_opponent_pose:
            return reward, reward_info
        
        latest_opponent_translation = [tf for tf in self.latest_opponent_pose.transforms if tf.child_frame_id == self.ftg_car_name]
        if latest_opponent_translation == []:
            return reward, reward_info
        
        latest_opponent_translation = latest_opponent_translation[0].transform.translation
        latest_opponent_xy = [latest_opponent_translation.x, latest_opponent_translation.y]

        ftg_current_distance = math.dist(self.ftg_goal_position, latest_opponent_xy)

        # # Keeping track of FTG car goal number
        if ftg_current_distance < self.REWARD_RANGE:
            self.ftg_goals_reached += 1

            # Updating Goal Position
            goal_x, goal_y, _, _ = self.track_waypoints[(self.ftg_start_waypoint_index + self.ftg_goals_reached) % len(self.track_waypoints)]
            self.ftg_goal_position = [goal_x, goal_y]

            # self.update_goal_service(goal_x, goal_y, self.OPPONENT_NAME)

        # If ego car has overtaken opponent car (making sure to account for cases where the track loops back on itself)
        t_ego = self.track_model.get_closest_point_on_spline(next_state[:2], t_only=True)
        t_opponent = self.track_model.get_closest_point_on_spline(latest_opponent_xy, t_only=True)

        if self.is_first_reward_since_reset:
            if t_ego > t_opponent:
                self.t_ego_laps -= 1
        elif abs(self.old_t_opponent - t_opponent) > 0.8:
            self.t_opponent_laps += 1 if self.old_t_opponent > t_opponent else -1
        elif abs(self.old_t_ego - t_ego) > 0.8:
            self.t_ego_laps += 1 if self.old_t_ego > t_ego else -1

        print(f'Ego: {self.t_ego_laps + t_ego}, Opp: {self.t_opponent_laps + t_opponent}, Steps Overtaken: {self.num_steps_over_taken}')
        if not self.is_over_taken and self.t_ego_laps + t_ego > self.t_opponent_laps + t_opponent:
            self.num_steps_over_taken += 1
        else:
            self.num_steps_over_taken = 0
        
        if not self.is_over_taken and self.num_steps_over_taken > 5:
            print(f'RL Car has overtaken FTG Car')
            reward += 100

            # Ensure overtaking won't happen again
            self.is_over_taken = True

        self.is_first_reward_since_reset = False
        self.old_t_ego = t_ego
        self.old_t_opponent = t_opponent
        
        return reward, reward_info
