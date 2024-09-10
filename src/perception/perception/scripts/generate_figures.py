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

class Plotter():
  def __init__(self):
    # Create a directory to save evaluation figures
    self.FIGURES_DIR = "figures"
    if not os.path.exists(self.FIGURES_DIR):
      os.makedirs(self.FIGURES_DIR)

  def plot_time_diff(self, time_list, ground_truth_list, estimated_distance_list, state_list):
    """
    Plot the difference between the ground truth and estimated distance
    """
    plt.clf()

    distance_list = [sqrt(np.sum(np.array([state.position.x, state.position.y, state.position.z])**2)) for state in state_list]

    sns.pointplot(x=time_list, y=np.array(ground_truth_list) - np.array(estimated_distance_list),
                  color='blue', native_scale=True, label='Measured Distance Error',
                  ms=3, linewidth=1, marker='.')
    sns.pointplot(x=time_list, y=np.array(ground_truth_list) - np.array(distance_list),
                  color='red', native_scale=True, label='Filtered Distance Error',
                  ms=3, linewidth=1, marker='.')

    # Add labels and title to the plot
    plt.xlabel('Time (s)')
    plt.ylabel('Difference (m)')
    plt.title('Difference between Ground Truth and Estimated Distance')

    # Save the plot as an image
    plt.savefig(f'{self.FIGURES_DIR}/time_diff_plot.png')

  def plot_dist_diff(self, ground_truth_list, estimated_distance_list):
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
    plt.savefig(f'{self.FIGURES_DIR}/dist_diff_plot.png')

  def plot_distance(self, time_list, ground_truth_list, estimated_distance_list):
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
    plt.savefig(f'{self.FIGURES_DIR}/distance_plot.png')

  def plot_state(self, ground_truth_list, estimated_distance_list, state_list):
    """
    Plot the ground truth and estimated state
    """
    plt.clf()

    time_list = [get_time_from_header(state.header) for state in state_list]
    position_list = [(state.position.x, state.position.y, state.position.z) for state in state_list]
    linear_velocity_list = [(state.linear_velocity.x, state.linear_velocity.y, state.linear_velocity.z) for state in state_list]
    linear_acceleration_list = [(state.linear_acceleration.x, state.linear_acceleration.y, state.linear_acceleration.z) for state in state_list]

    distance_list = [sqrt(np.sum(np.array(pos)**2)) for pos in position_list]
    speed_list = [sqrt(np.sum(np.array(vel)**2)) for vel in linear_velocity_list]
    acceleration_list = [sqrt(np.sum(np.array(acc)**2)) for acc in linear_acceleration_list]

    twist_time_list = [get_time_from_header(self.opponent_odom_list[i].header) for i in range(len(self.opponent_odom_list))]
    twist_linear_velocity_list = [sqrt(np.sum(np.array([self.opponent_odom_list[i].twist.twist.linear.x - self.agent_odom_list[i].twist.twist.linear.x,
                                                        self.opponent_odom_list[i].twist.twist.linear.y - self.agent_odom_list[i].twist.twist.linear.y,
                                                        self.opponent_odom_list[i].twist.twist.linear.z - self.agent_odom_list[i].twist.twist.linear.z])**2)) if i > 0 else np.nan for i in range(len(self.opponent_odom_list))]

    sns.pointplot(x=time_list, y=distance_list, color='red',
                  native_scale=True, label='Estimated Distance', ms=3,
                  linewidth=1, marker='.')
    sns.pointplot(x=time_list, y=speed_list, color='green',
                  native_scale=True, label='Estimated Speed', ms=3,
                  linewidth=1, marker='.')
    sns.pointplot(x=time_list, y=acceleration_list, color='blue',
                  native_scale=True, label='Estimated Linear Acceleration', ms=3,
                  linewidth=1, marker='.')
    sns.pointplot(x=time_list, y=ground_truth_list, color='black',
                  native_scale=True, label='Ground Truth Position', ms=3,
                  linewidth=1, marker='.')
    sns.pointplot(x=time_list, y=estimated_distance_list, color='magenta',
                  native_scale=True, label='Measured Distance', ms=3,
                  linewidth=1, marker='.')
    sns.pointplot(x=twist_time_list, y=twist_linear_velocity_list, color='cyan',
                  native_scale=True, label='Twist Linear Velocity', ms=3,
                  linewidth=1, marker='.')

    # Add labels and title to the plot
    plt.xlabel('Time (s)')
    plt.ylabel('Meters')
    plt.title('Ground Truth vs Estimated State')
    plt.ylim(0, 6)

    # Save the plot as an image
    plt.savefig(f'{self.FIGURES_DIR}/state_plot.png')
  

if __name__ == '__main__':
  # Get the directory of the latest run in the debug folder
  debug_directory = f"{os.getcwd()}/perception_debug"
  latest_directory = max([f.path for f in os.scandir(debug_directory) if f.is_dir()], key=os.path.getmtime)
  os.makedirs(f"{latest_directory}/figures", exist_ok=True)

  time_list = []
  ground_truth_list = []
  agent_odom_list = []
  opponent_odom_list = []
  estimated_distance_list = []
  state_list = []
  ground_truth_state_list = []

  plt.style.use("seaborn-v0_8")

  plotter = Plotter()
  plotter.plot_distance(time_list, ground_truth_list, estimated_distance_list)
  plotter.plot_time_diff(time_list, ground_truth_list, estimated_distance_list, state_list)
  plotter.plot_dist_diff(ground_truth_list, estimated_distance_list)
  plotter.plot_state(ground_truth_list, estimated_distance_list, state_list)

