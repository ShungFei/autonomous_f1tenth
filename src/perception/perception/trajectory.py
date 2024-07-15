import rclpy
from rclpy.node import Node
import rclpy.time
import rclpy.timer
from tf2_msgs.msg import TFMessage
import numpy as np
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.publisher import Publisher
from rclpy.timer import Timer

import perception.util.ground_truth as GroundTruth
from geometry_msgs.msg import Twist
from controllers.controller import Controller
 
class Trajectory(Node):
  """
  Create an ImageSubscriber class, which is a subclass of the Node class.
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    super().__init__('trajectory')
    self.vels = {
      "agent": 
        [
          {"time": 1.0, "linear": 0.2, "angular": 0.0},
          {"time": 5.0, "linear": 0.0, "angular": 0.0}
        ],
      "opponent": 
        [
          {"time": 1.0, "linear": 0.5, "angular": 0.0},
          {"time": 5.0, "linear": -0.2, "angular": 0.0}
        ],
    }
    # Name of the cameras to use
    self.SELECTED_CAMERA = "color"

    self.agent_name = self.declare_parameter('agent_name', "f1tenth").get_parameter_value().string_value
    self.camera_name = self.declare_parameter('camera_name', "d435").get_parameter_value().string_value
    self.opponent_name = self.declare_parameter('opponent_name', "opponent").get_parameter_value().string_value

    self.start_agent_vel_pub = False
    self.start_opp_vel_pub = False

    self.prev_camera_pose = None
    self.camera_pose = None
    self.prev_aruco_pose = None
    self.aruco_pose = None

    self.agent_vel_publisher = self.create_publisher(
      Twist,
      f'{self.agent_name}/cmd_vel',
      10
    )

    self.opp_vel_publisher = self.create_publisher(
      Twist,
      f'{self.opponent_name}/cmd_vel',
      10
    )

    self.agent_pose_sub = self.create_subscription(
      TFMessage,
      f'{self.agent_name}/pose',
      self.agent_pose_callback,
      10
    )

    self.opp_pose_sub = self.create_subscription(
      TFMessage,
      f'{self.opponent_name}/pose',
      self.opp_pose_callback,
      10
    )

    timer_period = 0.1  # seconds

    # Allow the agent and opponent timers to publish velocities concurrently
    self.callback_group = ReentrantCallbackGroup()

    self.agent_timer = self.create_timer(timer_period, None, callback_group=self.callback_group)
    self.agent_timer.callback = lambda : self.update_velocity("agent", self.agent_vel_publisher, self.agent_timer)
    self.opp_timer = self.create_timer(timer_period, None, callback_group=self.callback_group)
    self.opp_timer.callback = lambda : self.update_velocity("opponent", self.opp_vel_publisher, self.opp_timer)

  def agent_pose_callback(self, data: TFMessage):
    """
    Callback function for the agent's pose

    Determines if the agent has settled before publishing velocity commands
    """
    camera_world_pose, header = GroundTruth.get_camera_world_pose(data, self.agent_name, self.camera_name, self.SELECTED_CAMERA)

    self.prev_camera_pose = self.camera_pose
    self.camera_pose = camera_world_pose
    if self.prev_camera_pose is not None and self.camera_pose is not None and (self.prev_camera_pose == self.camera_pose).all():
      self.start_agent_vel_pub = True
      self.destroy_subscription(self.agent_pose_sub)
      
  def opp_pose_callback(self, data: TFMessage):
    """
    Callback function for the opponent's pose

    Determines if the opponent has settled before publishing velocity commands
    """
    aruco_world_pose, _ = GroundTruth.get_aruco_world_pose(data, self.opponent_name)

    self.prev_aruco_pose = self.aruco_pose
    self.aruco_pose = aruco_world_pose
    if self.prev_aruco_pose is not None and self.aruco_pose is not None and (self.prev_aruco_pose == self.aruco_pose).all():
      self.start_opp_vel_pub = True
      self.destroy_subscription(self.opp_pose_sub)

  def update_velocity(self, vehicle_name, publisher: Publisher, timer: Timer):
    prev_time = 0
    if self.start_agent_vel_pub and self.start_opp_vel_pub:
      self.destroy_timer(timer)

      for step in self.vels[vehicle_name]:
        time_diff = step["time"] - prev_time
        
        # Wait until the time step has passed 
        if time_diff > 0:
          rate = self.create_rate(1.0 / time_diff, self.get_clock())
          rate.sleep()
          self.destroy_rate(rate)
        # print(vehicle_name, time_diff)

        msg = Twist()
        msg.linear.x = step["linear"]
        msg.angular.z = step["angular"]

        publisher.publish(msg)

        prev_time = step["time"]

  
def main(args=None):
  rclpy.init(args=args)
  
  AGENT_CAR_NAME = "f1tenth"
  OPPONENT_CAR_NAME = "opponent"
  # rclpy.spin(trajectory)

  # trajectory.destroy_node()

  try:
    trajectory = Trajectory()

    # Need MultiThreadedExecutor for rates
    # 
    # "Care should be taken when calling this from a callback. 
    #  This may block forever if called in a callback in a SingleThreadedExecutor."
    executor = MultiThreadedExecutor()
    executor.add_node(trajectory)

    try:
      executor.spin()
    finally:
      executor.shutdown()
      trajectory.destroy_node()
  finally:
    rclpy.shutdown()

  # rclpy.shutdown()
  
  # agent_controller = Controller('ftg_policy_', AGENT_CAR_NAME, 0.1)
  # opp_controller = Controller('ftg_policy_', OPPONENT_CAR_NAME, 0.1)

  # agent_controller.set_velocity(2, 0)
  # opp_controller.set_velocity(3, 0)