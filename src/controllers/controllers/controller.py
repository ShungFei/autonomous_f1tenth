import rclpy
import math
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy import Future
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from environments.util import process_lidar, process_odom, reduce_lidar, forward_reduce_lidar

class Controller(Node):
    def __init__(self, node_name, car_name, step_length):
        #TODO: make node name dynamic
        super().__init__(node_name + 'controller')

        # Environment Details ----------------------------------------
        self.NAME = car_name
        self.STEP_LENGTH = step_length

        # Pub/Sub ----------------------------------------------------
        # Ackermann pub only works for physical version
        self.ackerman_pub = self.create_publisher(
            AckermannDriveStamped,
            f'/f1tenth/drive',
            10
        )

        self.odom_sub = Subscriber(
            self,
            Odometry,
            f'/f1tenth/odometry',
        )

        self.lidar_sub = Subscriber(
            self,
            LaserScan,
            f'/f1tenth/scan',
        )

        self.message_filter = ApproximateTimeSynchronizer(
            [self.odom_sub, self.lidar_sub],
            10,
            0.1,
        )

        self.message_filter.registerCallback(self.message_filter_callback)

        self.observation_future = Future()

        self.timer = self.create_timer(step_length, self.timer_cb)
        self.timer_future = Future()

    def step(self, action, policy):

        lin_vel, ang_vel = action
        self.set_velocity(lin_vel, ang_vel)

        self.sleep()

        self.timer_future = Future()

        state = self.get_observation(policy)

        return state

    def message_filter_callback(self, odom: Odometry, lidar: LaserScan):
        self.observation_future.set_result({'odom': odom, 'lidar': lidar})

    def get_observation(self, policy):
        odom, lidar = self.get_data()
        odom = process_odom(odom)
        if policy == 'ftg':
            lidar = forward_reduce_lidar(lidar)
        else:
            lidar = reduce_lidar(lidar)
        print(lidar)
        state = odom+lidar
        return state
        

    def get_data(self):
        rclpy.spin_until_future_complete(self, self.observation_future)
        future = self.observation_future
        self.observation_future = Future()
        data = future.result()
        return data['odom'], data['lidar']

    def set_velocity(self, linear, angular):
        """
        Publish Twist messages to f1tenth cmd_vel topic
        """
        angle = self.convert(angular, linear, 0.16)
        velocity_msg = AckermannDriveStamped()
        velocity_msg.drive.steering_angle = -float(angle*0.5)
        velocity_msg.drive.speed = float(linear)

        self.ackerman_pub.publish(velocity_msg)

    def omega_to_ackerman(omega, linear_v, L):
        '''
        Convert CG angular velocity to Ackerman steering angle.

        Parameters:
        - omega: CG angular velocity in rad/s
        - v: Vehicle speed in m/s
        - L: Wheelbase of the vehicle in m

        Returns:
        - delta: Ackerman steering angle in radians

        Derivation:
        R = v / omega 
        R = L / tan(delta)  equation 10 from https://www.researchgate.net/publication/228464812_Electric_Vehicle_Stability_with_Rear_Electronic_Differential_Traction#pf3
        tan(delta) = L * omega / v
        delta = arctan(L * omega/ v)
        '''
        if linear_v == 0:
            return 0

        delta = math.atan((L * omega) / linear_v)

        return delta



    def sleep(self):
        while not self.timer_future.done():
            rclpy.spin_once(self)
    
    def timer_cb(self):
        self.timer_future.set_result(True)
