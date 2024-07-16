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
from rosgraph_msgs.msg import Clock

import perception.util.ground_truth as GroundTruth
from perception.util.conversion import get_time_from_header, get_time_from_clock
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
          {"time": 7.0, "linear": -0.2, "angular": 0.0}
        ],
    }
    # Name of the cameras to use
    self.SELECTED_CAMERA = "color"

    self.agent_name = self.declare_parameter('agent_name', "f1tenth").get_parameter_value().string_value
    self.camera_name = self.declare_parameter('camera_name', "d435").get_parameter_value().string_value
    self.opponent_name = self.declare_parameter('opponent_name', "opponent").get_parameter_value().string_value 
    self.eval_time = self.declare_parameter('eval_time', 10.0).get_parameter_value().double_value

    self.start_agent_vel_pub = False
    self.start_opp_vel_pub = False
    self.start_agent_vel_time = None
    self.start_opp_vel_time = None

    self.prev_camera_pose = None
    self.camera_pose = None
    self.prev_aruco_pose = None
    self.aruco_pose = None

    self.agent_checkpoint = 0
    self.opp_checkpoint = 0

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

    self.clock_sub = self.create_subscription(
      Clock,
      '/clock',
      self.clock_callback,
      10
    )

    self.opp_pose_sub = self.create_subscription(
      TFMessage,
      f'{self.opponent_name}/pose',
      self.opp_pose_callback,
      10
    )


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
      self.start_agent_vel_time = get_time_from_header(header)
      self.destroy_subscription(self.agent_pose_sub)
      
  def opp_pose_callback(self, data: TFMessage):
    """
    Callback function for the opponent's pose

    Determines if the opponent has settled before publishing velocity commands
    """
    aruco_world_pose, header = GroundTruth.get_aruco_world_pose(data, self.opponent_name)

    self.prev_aruco_pose = self.aruco_pose
    self.aruco_pose = aruco_world_pose
    if self.prev_aruco_pose is not None and self.aruco_pose is not None and (self.prev_aruco_pose == self.aruco_pose).all():
      self.start_opp_vel_pub = True
      self.start_opp_vel_time = get_time_from_header(header)
      self.destroy_subscription(self.opp_pose_sub)
  
  def clock_callback(self, data: Clock):
    if self.start_agent_vel_pub and self.start_opp_vel_pub and self.start_agent_vel_time is not None and self.start_opp_vel_time is not None:
      time = get_time_from_clock(data)

      if self.agent_checkpoint >= len(self.vels["agent"]) and self.opp_checkpoint >= len(self.vels["opponent"]):
        self.destroy_subscription(self.clock_sub)
        return
    
      if self.agent_checkpoint < len(self.vels["agent"]):
        next_agent_timestep = self.vels["agent"][self.agent_checkpoint]
        # The velocity is only published since the time the vehicle has settled (to prevent potential flipping)
        if time >= self.start_agent_vel_time + next_agent_timestep["time"]:
          self.agent_checkpoint += 1
          self.agent_vel_publisher.publish(self.create_velocity_msg(next_agent_timestep["linear"], next_agent_timestep["angular"]))
      if self.opp_checkpoint < len(self.vels["opponent"]):
        next_opp_timestep = self.vels["opponent"][self.opp_checkpoint]
        if time >= self.start_opp_vel_time + next_opp_timestep["time"]:
          self.opp_checkpoint += 1
          self.opp_vel_publisher.publish(self.create_velocity_msg(next_opp_timestep["linear"], next_opp_timestep["angular"]))
      
      # Stop the vehicles after the evaluation time has passed
      if time >= self.eval_time:
        self.agent_vel_publisher.publish(self.create_velocity_msg(0.0, 0.0))
        self.opp_vel_publisher.publish(self.create_velocity_msg(0.0, 0.0))
        self.destroy_subscription(self.clock_sub)
        return

  def create_velocity_msg(self, linear, angular):
    """
    Create a Twist message with the given linear and angular velocities
    """
    msg = Twist()
    msg.linear.x = linear
    msg.angular.z = angular

    return msg
  
def main(args=None):
  rclpy.init(args=args)
  
  AGENT_CAR_NAME = "f1tenth"
  OPPONENT_CAR_NAME = "opponent"

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
