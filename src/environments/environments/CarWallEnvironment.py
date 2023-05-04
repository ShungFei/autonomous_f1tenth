import time
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy import Future

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger
from sensor_msgs.msg import LaserScan
from environment_interfaces.srv import Reset
from message_filters import Subscriber, ApproximateTimeSynchronizer

class CarWallEnvironment(Node):
    """
    CarWall Reinforcement Learning Environment:

        Task:
            Here the agent learns to drive the f1tenth car to a goal position

        Observation:
            It's position (x, y), orientation (w, x, y, z), lidar points (approx. ~600 rays) and the goal's position (x, y)

        Action:
            It's linear and angular velocity
        
        Reward:
            It's progress toward the goal plus,
            100+ if it reaches the goal plus,
            -50 if it collides with the wall

        Termination Conditions:
            When the agent is within REWARD_RANGE units or,
            When the agent is within COLLISION_RANGE units
        
        Truncation Condition:
            When the number of steps surpasses MAX_STEPS
    """

    def __init__(self, car_name, reward_range=1, max_steps=15, collision_range=0.5, step_length=0.5):
        super().__init__('car_goal_environment')
        
        # Environment Details ----------------------------------------
        self.NAME = car_name
        self.REWARD_RANGE = reward_range
        self.MAX_STEPS = max_steps
        self.COLLISION_RANGE = collision_range
        self.STEP_LENGTH = step_length

        self.step_counter = 0

        # Pub/Sub ----------------------------------------------------
        self.cmd_vel_pub = self.create_publisher(
                Twist,
                f'/model/{self.NAME}/cmd_vel',
                10
            )
        
        self.odom_sub = Subscriber(
            self,
            Odometry,
            f'/model/{self.NAME}/odometry',
        )
        # TODO: Map the lidar to a dynamic topic => of the form /model/<name>/lidar
        self.lidar_sub = Subscriber(
            self,
            LaserScan,
            f'/lidar',
        )

        self.message_filter = ApproximateTimeSynchronizer(
            [self.odom_sub, self.lidar_sub],
            10,
            0.1,
        )

        self.message_filter.registerCallback(self.message_filter_callback)

        self.observation_future = Future()

        # Reset Client -----------------------------------------------
        self.reset_client = self.create_client(
            Reset,
            'car_goal_reset'
        )

        # while not self.reset_client.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('reset service not available, waiting again...')

        time.sleep(2)

        # TODO: generate goal
        self.goal_position = [0, 0] # x and y

        time.sleep(5)
        
    def reset(self):
        self.step_counter = 0

        # Call reset Service

        time.sleep(self.STEP_LENGTH)
        
        observation = self.get_observation()
        
        info = {}

        return observation, info

    def step(self, action):
        self.step_counter += 1

        state = self.get_observation()

        lin_vel, ang_vel = action
        self.set_velocity(lin_vel, ang_vel)

        time.sleep(self.STEP_LENGTH)
        
        next_state = self.get_observation()
        reward = self.compute_reward(state, next_state)
        terminated = self.is_terminated(next_state)
        truncated = self.step_counter >= self.MAX_STEPS
        info = {}

        return next_state, reward, terminated, truncated, info

    def get_observation(self):

        # Get Position and Orientation of F1tenth
        odom, lidar = self.get_data()
        odom = self.process_odom(odom)
        ranges, _ = self.process_lidar(lidar)

        # Get Goal Position
        return odom + ranges + self.goal_position 

    def is_terminated(self, observation):
        """
        Observation (ranges all inclusive):
            0 to 8 => odom
            -1 to -2 => goal x, y
            9 to -3 => lidar
        """
        current_distance = math.dist(observation[-2:], observation[:2])

        reached_goal = current_distance <= self.REWARD_RANGE

        collided_wall = self.has_collided(observation[9:-2])
        return reached_goal or collided_wall
    
    def has_collided(self, lidar_ranges):
        return any(ray < self.COLLISION_RANGE for ray in lidar_ranges)
        
    
    def compute_reward(self, state, next_state):

        goal_position = state[-2:]

        old_distance = math.dist(goal_position, state[:2])
        current_distance = math.dist(goal_position, next_state[:2])

        delta_distance = old_distance - current_distance

        reward = 0

        if current_distance < self.REWARD_RANGE:
            reward += 100

        if self.has_collided(next_state[9:-2]):
            reward -= 50 # TODO: find optimal value for this
        
        reward += delta_distance * 10

        return reward

    def message_filter_callback(self, odom: Odometry, lidar: LaserScan):
        self.observation_future.set_result({'odom': odom, 'lidar': lidar})

    def get_data(self):
        rclpy.spin_until_future_complete(self, self.observation_future)
        future = self.observation_future
        self.observation_future = Future()
        data = future.result()
        return data['odom'], data['lidar']
    
    def process_odom(self, odom: Odometry):
        pose = odom.pose.pose
        position = pose.position
        orientation = pose.orientation

        twist = odom.twist.twist
        lin_vel = twist.linear
        ang_vel = twist.angular

        return [position.x, position.y, orientation.w, orientation.x, orientation.y, orientation.z, lin_vel.x, ang_vel.z]

    def process_lidar(self, lidar: LaserScan):
        ranges = lidar.ranges
        ranges = np.nan_to_num(ranges, posinf=float(10))
        ranges = list(ranges)

        intensities = list(lidar.intensities)
        return ranges, intensities

    def set_velocity(self, linear: float, angular: float):
        """
        Publish Twist messages to f1tenth cmd_vel topic
        """
        velocity_msg = Twist()
        velocity_msg.angular.z = float(angular)
        velocity_msg.linear.x = float(linear)

        self.cmd_vel_pub.publish(velocity_msg)

        