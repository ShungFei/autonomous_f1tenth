import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo 
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Vector3Stamped, PoseStamped
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
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
from perception_interfaces.msg import OptionalPoseStamped

class Evaluation(Node):
  def __init__(self):
    """
    Class constructor to set up the node
    """
    super().__init__('car_localizer')

    self.FIGURES_DIR = "perception_eval"
    # Create a directory to save evaluation figures
    if not os.path.exists(self.FIGURES_DIR):
      os.makedirs(self.FIGURES_DIR)

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

    self.opponent_pose_sub = self.create_subscription(
      TFMessage,
      f'{self.opponent_name}/pose',
      self.opponent_pose_callback,
      10
    )

    self.opp_estimated_pose_sub = Subscriber(
      self,
      OptionalPoseStamped,
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

    self.eval_sub = ApproximateTimeSynchronizer(
        [self.opp_estimated_pose_sub, self.camera_pose_sub, self.aruco_pose_sub],
        10,
        0.1,
    )
    self.eval_sub.registerCallback(self.eval_callback)

    self.time_list = []
    self.ground_truth_list = []
    self.estimated_distance_list = []

    # Used to convert between ROS and OpenCV images
    self.bridge = CvBridge()

  def eval_callback(self, estimated_pose: OptionalPoseStamped, camera_pose: Vector3Stamped, aruco_pose: Vector3Stamped):
    """
    Synced callback function for the camera image, depth image, camera pose, and ArUco pose for evaluation
    """
    aruco_pose = np.array([aruco_pose.vector.x, aruco_pose.vector.y, aruco_pose.vector.z])
    camera_pose = np.array([camera_pose.vector.x, camera_pose.vector.y, camera_pose.vector.z])
    curr_clock_time = get_time_from_header(estimated_pose.header)
    print('time', curr_clock_time)
    if aruco_pose is not None and camera_pose is not None and estimated_pose is not None:
      if self.debug == True:
        print('aruco', aruco_pose)
        print('camera', camera_pose)
        if estimated_pose.is_set:
          print('detected', np.array([estimated_pose.pose.position.x, estimated_pose.pose.position.y, estimated_pose.pose.position.z]))
        else:
          print('detected', None)

      estimated_tvec = np.array([estimated_pose.pose.position.x, estimated_pose.pose.position.y, estimated_pose.pose.position.z])

      # Append the data to the lists
      self.time_list.append(curr_clock_time)
      self.ground_truth_list.append(sqrt(np.sum((aruco_pose - camera_pose)**2)))

      if estimated_pose.is_set:
        self.estimated_distance_list.append(sqrt(np.sum((estimated_tvec)**2)))
      else:
        self.estimated_distance_list.append(np.nan)

      # print('time', curr_clock_time)
      # print('ground truth distance:', sqrt(np.sum((aruco_pose - camera_pose)**2)))
      # print('estimated distance:', sqrt(np.sum((estimated_tvec)**2)))
    if curr_clock_time >= self.eval_time:
      curr_time = datetime.now().strftime('%y_%m_%d_%H:%M:%S')

      # Create a directory to save the evaluation data
      os.makedirs(f'{self.FIGURES_DIR}/{curr_time}', exist_ok=True)

      # Save the data to a file
      self.write_eval_data(self.time_list, self.ground_truth_list, self.estimated_distance_list, curr_time)

      self.plot_distance(self.time_list, self.ground_truth_list, self.estimated_distance_list, curr_time)
      self.plot_time_diff(self.time_list, self.ground_truth_list, self.estimated_distance_list, curr_time)
      self.plot_dist_diff(self.ground_truth_list, self.estimated_distance_list, curr_time)

      # Destroy all relevant subscriptions
      self.destroy_subscription(self.opp_estimated_pose_sub.sub)
      self.destroy_subscription(self.camera_pose_sub.sub)
      self.destroy_subscription(self.aruco_pose_sub.sub)

  def write_eval_data(self, time_list, ground_truth_list, estimated_distance_list, curr_time):
    """
    Write the evaluation data to a file
    """
    data = np.column_stack((time_list, ground_truth_list, estimated_distance_list))
    np.savetxt(f'{self.FIGURES_DIR}/{curr_time}/data.txt', data, delimiter=',', header='Time,Ground Truth,Estimated Distance', comments='')

  def read_eval_data(self, curr_time):
    """
    Read the evaluation data from a file
    """
    data = np.genfromtxt(f'{self.FIGURES_DIR}/{curr_time}/data.txt', delimiter=',', skip_header=1)

    # Extract the columns into separate lists
    time_list = data[:, 0]
    ground_truth_list = data[:, 1]
    estimated_distance_list = data[:, 2]

    return time_list, ground_truth_list, estimated_distance_list
  
  def plot_time_diff(self, time_list, ground_truth_list, estimated_distance_list, curr_time):
    """
    Plot the difference between the ground truth and estimated distance
    """
    plt.clf()
    sns.pointplot(x=time_list, y=np.array(ground_truth_list) - np.array(estimated_distance_list),
                  color='blue', native_scale=True, label='Distance Error',
                  ms=3, linewidth=1, marker='.')

    # Add labels and title to the plot
    plt.xlabel('Time (s)')
    plt.ylabel('Difference (m)')
    plt.title('Difference between Ground Truth and Estimated Distance')

    # Save the plot as an image
    plt.savefig(f'{self.FIGURES_DIR}/{curr_time}/time_diff_plot.png')
  
  def plot_dist_diff(self, ground_truth_list, estimated_distance_list, curr_time):
    """
    Plot the difference between the ground truth and estimated distance with respect to ground truth
    """
    plt.clf()
    sns.pointplot(x=ground_truth_list, y=np.array(ground_truth_list) - np.array(estimated_distance_list),
                  color='blue', native_scale=True, label='Distance Error',
                  ms=3, linewidth=1, marker='.')

    # Add labels and title to the plot
    plt.xlabel('Ground truth distance (m)')
    plt.ylabel('Difference (m)')
    plt.title('Difference between Ground Truth and Estimated Distance')

    # Save the plot as an image
    plt.savefig(f'{self.FIGURES_DIR}/{curr_time}/dist_diff_plot.png')

  def plot_distance(self, time_list, ground_truth_list, estimated_distance_list, curr_time):
    """
    Plot the ground truth and estimated distance
    """
    plt.clf()

    sns.pointplot(x=time_list, y=estimated_distance_list, color='blue',
                  native_scale=True, label='Estimated Distance', ms=3,
                  linewidth=1, marker='x')
    sns.pointplot(x=time_list, y=ground_truth_list, color='green',
                  native_scale=True, label='Ground Truth', ms=3,
                  linewidth=1, marker='.')

    # Add labels and title to the plot
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (m)')
    plt.title('Ground Truth vs Estimated Distance')

    # Save the plot as an image
    plt.savefig(f'{self.FIGURES_DIR}/{curr_time}/distance_plot.png')
  
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