import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo 
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Vector3Stamped, PoseStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer
import cv2
import os
import numpy as np
import pandas as pd
from math import sqrt
import seaborn as sns
from matplotlib import pyplot as plt
import time
from datetime import datetime

import perception.util.ground_truth as GroundTruth
from perception.util.conversion import get_time_from_header, get_quaternion_from_rotation_matrix
from perception_interfaces.msg import StateEstimateStamped

class Evaluation(Node):
  def __init__(self):
    """
    Class constructor to set up the node
    """
    super().__init__('car_localizer')

    # Name of the cameras to use
    self.SELECTED_CAMERA = "color"
    self.SELECTED_DEPTH_CAMERA = "depth"

    self.SELECTED_LEFT_CAMERA = "left"
    self.SELECTED_RIGHT_CAMERA = "right"
    
    plt.style.use("seaborn-v0_8")

    self.agent_name = self.declare_parameter('agent_name', "f1tenth").get_parameter_value().string_value
    self.camera_name = self.declare_parameter('camera_name', "d435").get_parameter_value().string_value
    self.opponent_name = self.declare_parameter('opponent_name', "opponent").get_parameter_value().string_value
    self.debug = self.declare_parameter('debug', False).get_parameter_value().bool_value
    self.is_stereo = self.declare_parameter('is_stereo', False).get_parameter_value().bool_value
    self.eval_time = self.declare_parameter('eval_time', 10.0).get_parameter_value().double_value

    self.get_logger().info(f"Subscribing to {self.agent_name}")

    self.camera_pose_pub = self.create_publisher(
      Vector3Stamped,
      f'{self.agent_name}/camera_pose',
      10
    )

    self.aruco_pose_pub = self.create_publisher(
      Vector3Stamped,
      f'{self.opponent_name}/aruco_pose',
      10
    )
    
    self.agent_pose_sub = self.create_subscription(
      TFMessage,
      f'{self.agent_name}/pose', 
      self.agent_pose_callback, 
      10
    )

    self.agent_odom_sub = self.create_subscription(
      Odometry,
      f'/{self.agent_name}/odometry',
      lambda msg: self.agent_odom_list.append(msg),
      10
    )

    self.opponent_pose_sub = self.create_subscription(
      TFMessage,
      f'{self.opponent_name}/pose',
      self.opponent_pose_callback,
      10
    )

    self.opponent_odom_sub = self.create_subscription(
      Odometry,
      f'/{self.opponent_name}/odometry',
      lambda msg: self.opponent_odom_list.append(msg),
      10
    )

    self.opp_measured_pose_sub = Subscriber(
      self,
      PoseStamped,
      f'{self.opponent_name}/pose_estimate',
    )

    self.camera_pose_sub = Subscriber(
      self,
      Vector3Stamped,
      f'{self.agent_name}/camera_pose',
    )

    self.aruco_pose_sub = Subscriber(
      self,
      Vector3Stamped,
      f'{self.opponent_name}/aruco_pose',
    )

    self.state_estimate_sub = self.create_subscription(
      StateEstimateStamped,
      f'{self.opponent_name}/state_estimate',
      lambda msg: self.state_list.append(msg),
      10
    )

    self.eval_sub = TimeSynchronizer(
        [self.opp_measured_pose_sub, self.camera_pose_sub, self.aruco_pose_sub],
        70
    )
    self.eval_sub.registerCallback(self.eval_callback)

    self.time_list = []
    self.ground_truth_list = []
    self.agent_odom_list = []
    self.opponent_odom_list = []
    self.estimated_distance_list = []
    self.state_list = []
    self.ground_truth_state_list = []

    # Used to convert between ROS and OpenCV images
    self.bridge = CvBridge()
  
  def eval_callback(self, measured_pose: PoseStamped, camera_pose: Vector3Stamped, aruco_pose: Vector3Stamped):
    """
    Synced callback function for the camera image, depth image, camera pose, and ArUco pose for evaluation
    """
    aruco_pose = np.array([aruco_pose.vector.x, aruco_pose.vector.y, aruco_pose.vector.z])
    camera_pose = np.array([camera_pose.vector.x, camera_pose.vector.y, camera_pose.vector.z])
    curr_clock_time = get_time_from_header(measured_pose.header)
    if aruco_pose is not None and camera_pose is not None and measured_pose is not None:
      if self.debug == True:
        print('aruco', aruco_pose)
        print('camera', camera_pose)
        print('detected', np.array([measured_pose.pose.position.x, measured_pose.pose.position.y, measured_pose.pose.position.z]))

      estimated_tvec = np.array([measured_pose.pose.position.x, measured_pose.pose.position.y, measured_pose.pose.position.z])

      # Append the data to the lists
      self.time_list.append(curr_clock_time)
      self.ground_truth_list.append(sqrt(np.sum((aruco_pose - camera_pose)**2)))
      self.estimated_distance_list.append(sqrt(np.sum((estimated_tvec)**2)))

  def write_eval_data(self, time_list, ground_truth_list, estimated_distance_list, state_list, curr_time):
    """
    Write the evaluation data to a file
    """
    data = np.column_stack((time_list,
                            ground_truth_list,
                            estimated_distance_list,
                            [sqrt(np.sum(np.array([state.position.x, state.position.y, state.position.z])**2)) if state is not np.nan else np.nan for state in state_list],
                            [sqrt(np.sum(np.array([state.linear_velocity.x, state.linear_velocity.y, state.linear_velocity.z])**2)) if state is not np.nan else np.nan for state in state_list],
                            [sqrt(np.sum(np.array([state.linear_acceleration.x, state.linear_acceleration.y, state.linear_acceleration.z])**2)) if state is not np.nan else np.nan for state in state_list]))
    np.savetxt(f'{self.FIGURES_DIR}/{curr_time}/data.csv', data, delimiter=',', header='time,ground_truth_position,measured_distance,ground_truth_velocity,estimated_distance_magnitude', comments='')

  def agent_pose_callback(self, data: TFMessage):
    """
    Callback function for the agent's pose

    This publishes the 3D world position (x,y,z) of the camera on the agent vehicle
    """
    camera_world_pose, header = GroundTruth.get_camera_world_pose(
      data, self.agent_name, self.camera_name, 
      self.SELECTED_CAMERA if not self.is_stereo else self.SELECTED_LEFT_CAMERA)
    
    if camera_world_pose is not None:
      msg = Vector3Stamped()
      msg.header = header
      msg.vector.x, msg.vector.y, msg.vector.z = camera_world_pose

      self.camera_pose_pub.publish(msg)
  
  def opponent_pose_callback(self, data: TFMessage):
    """
    Callback function for the opponent's pose

    This publishes the 3D world position (x,y,z) of the ArUco marker on the opponent vehicle
    """
    aruco_world_pose, header = GroundTruth.get_aruco_world_pose(data, self.opponent_name)
    if aruco_world_pose is not None and header is not None:
      msg = Vector3Stamped()
      msg.header = header
      msg.vector.x, msg.vector.y, msg.vector.z = aruco_world_pose

      self.aruco_pose_pub.publish(msg)

def main(args=None):
  rclpy.init(args=args)

  evaluation = Evaluation()

  try:
    rclpy.spin(evaluation)
  finally:
    evaluation.destroy_node()
    rclpy.shutdown()