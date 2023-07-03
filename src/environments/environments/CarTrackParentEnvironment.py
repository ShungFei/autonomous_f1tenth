import time
import math
import numpy as np
import random

import rclpy
from rclpy.node import Node
from rclpy import Future

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger
from sensor_msgs.msg import LaserScan
from environment_interfaces.srv import Reset
from message_filters import Subscriber, ApproximateTimeSynchronizer

from environments.ParentCarEnvironment import ParentCarEnvironment

class CarTrackParentEnvironment(ParentCarEnvironment):
    """
    CarTrack Reinforcement Learning Environment:

        Task:
            Here the agent learns to drive the f1tenth car to a goal position

        Observation:
            It's position (x, y), orientation (w, x, y, z), lidar points (approx. ~600 rays) and the goal's position (x, y)

        Action:
            It's linear and angular velocity
        
        Reward:
            It's progress toward the goal plus,
            50+ if it reaches the goal plus,
            -25 if it collides with the wall

        Termination Conditions:
            When the agent is within REWARD_RANGE units or,
            When the agent is within COLLISION_RANGE units
        
        Truncation Condition:
            When the number of steps surpasses MAX_STEPS
    """

    def __init__(self, car_name, reward_range=1, max_steps=50, collision_range=0.2, step_length=0.5):
        super().__init__('car_track', car_name, reward_range, max_steps, collision_range, step_length)

        # Environment Details ----------------------------------------
        self.MAX_STEPS_PER_GOAL = max_steps
        self.MIN_ACTIONS = np.asarray([0, -3.14])
        self.OBSERVATION_SIZE = 8 + 10  # Car position + Lidar rays

        # Reset Client -----------------------------------------------
        self.goal_number = 0
        self.all_goals = []

        self.car_reset_positions = {
            'x': 0.0,
            'y': 0.0,
            'yaw': 0.0
        }

        self.step_counter = 0

        self.get_logger().info('Environment Setup Complete')
    
    def reset(self):
        self.step_counter = 0

        self.set_velocity(0, 0)

        #TODO: Remove Hard coded-ness of 10x10
        self.goal_number = 0
        self.goal_position = self.generate_goal(self.goal_number)

        while not self.timer_future.done():
            rclpy.spin_once(self)
        
        self.timer_future = Future()

        self.call_reset_service()
        
        observation = self.get_observation()
        
        info = {}

        return observation, info

    def generate_goal(self, number):
        print("Goal", number, "spawned")
        return self.all_goals[number % len(self.all_goals)]

    def call_reset_service(self):
        x, y = self.goal_position

        request = Reset.Request()
        request.gx = x
        request.gy = y
        request.cx = self.car_reset_positions['x']
        request.cy = self.car_reset_positions['y']
        request.cyaw = self.car_reset_positions['yaw']
        request.flag = "car_and_goal"

        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()

    def update_goal_service(self, number):
        x, y = self.generate_goal(number)
        self.goal_position = [x, y]

        request = Reset.Request()
        request.gx = x
        request.gy = y
        request.flag = "goal_only"

        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()


    def get_observation(self):

        # Get Position and Orientation of F1tenth
        odom, lidar = self.get_data()
        odom = self.process_odom(odom)
        # ranges, _ = self.process_lidar(lidar)

        reduced_range = self.sample_reduce_lidar(lidar)

        # Get Goal Position
        return odom + reduced_range 

    def is_terminated(self, observation):
        """
        Observation (ranges all inclusive):
            0 to 8 => odom
            -1 to -2 => goal x, y
            9 to -3 => lidar
        """
        collided_wall = self.has_collided(observation)
        flipped_over = self.has_flipped_over(observation)

        if collided_wall:
            print("Collided with wall")
        if flipped_over:
            print("Flipped over")

        return collided_wall or flipped_over

    def compute_reward(self, state, next_state):

        # TESTING ONLY

        # if self.goal_number < len(self.all_goals) - 1:
        #     self.goal_number += 1
        # else:
        #     self.goal_number = 0

        # self.update_goal_service(self.goal_number)
        # ==============================================================

        reward = 0

        goal_position = self.goal_position

        current_distance = math.dist(goal_position, next_state[:2])

        if current_distance < self.REWARD_RANGE:
            reward += 50
            self.goal_number += 1
            self.step_counter = 0
            self.update_goal_service(self.goal_number)
            
        if self.has_collided(next_state) or self.has_flipped_over(next_state):
            reward -= 25 # TODO: find optimal value for this
        
        return reward

    def process_lidar(self, lidar: LaserScan):
        ranges = lidar.ranges
        ranges = np.nan_to_num(ranges, posinf=float(-1))
        ranges = np.clip(ranges, 0, 10)

        ranges = list(ranges)

        intensities = list(lidar.intensities)
        return ranges, intensities

    def sample_reduce_lidar(self, lidar: LaserScan):
        ranges = lidar.ranges
        ranges = np.nan_to_num(ranges, posinf=float(10))
        ranges = np.clip(ranges, 0, 10)
        ranges = list(ranges)
        
        reduced_range = []

        for i in range(10):
            sample = ranges[i*64] 
            reduced_range.append(sample)

        return reduced_range
