
from datetime import datetime
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag

import argparse
import cv2
import os
import numpy as np
import pandas as pd
from math import sqrt
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import datetime

from perception.util.conversion import get_euler_from_quaternion

class StateEstimator():
  def __init__(self, input_file, interval=1/30):
    self.input_file = input_file
    self.interval = interval

    self.measured_poses = pd.read_csv(self.input_file)
    print(self.input_file)
    print(self.measured_poses)

    # Index is all potential time intervals between the first and last time in the measured poses
    self.expected_times = np.linspace(self.measured_poses.iloc[0].time, self.measured_poses.iloc[-1].time, len(self.measured_poses))
    self.state_estimates = pd.DataFrame(columns=["position_x", "position_y", "position_z", "linear_velocity_x", "linear_velocity_y", "linear_velocity_z", "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z", "orientation_x", "orientation_y", "orientation_z", "angular_velocity_x", "angular_velocity_y", "angular_velocity_z", "angular_acceleration_x", "angular_acceleration_y", "angular_acceleration_z"], index=self.expected_times)
  
  def process_kalman_filter(self):
    # Initialize the Kalman filter with the first pose
    self.initialize_kalman_filter(self.interval, pose=self.measured_poses.iloc[0])

    for expected_time in self.state_estimates.index:
      self.kf.predict()

      try:
        i = self.measured_poses.index.get_loc(expected_time, method="nearest", tolerance=self.interval / 2)
        orientation = get_euler_from_quaternion(self.measured_poses.iloc[i].qx, self.measured_poses.iloc[i].qy, self.measured_poses.iloc[i].qz, self.measured_poses.iloc[i].qw)
        pose = np.concatenate((self.measured_poses.iloc[i][["tx", "ty", "tz"]], orientation))
        self.kf.update(pose)
      except KeyError:
        # Skip the update step if there is no measured pose close to the expected time
        pass

      state_estimate = self.kf.x
      self.state_estimates.loc[expected_time] = state_estimate

    self.write_data()

  def initialize_kalman_filter(self, dt, pose = np.zeros(6)):
    self.kf = KalmanFilter(dim_x=18, dim_z=6)

    # State
    self.kf.x = np.zeros(18)
    self.kf.x[:3] = pose[:3]
    self.kf.x[9:12] = pose[3:6]
    # State covariance
    self.kf.P = np.diag([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    # Process noise
    q = Q_discrete_white_noise(dim=3, dt=dt, var=0.05)

    q_second_order = np.array([[q[0, 0], 0, 0, q[0, 1], 0, 0, q[0, 2], 0, 0],
                                [0, q[0, 0], 0, 0, q[0, 1], 0, 0, q[0, 2], 0],
                                [0, 0, q[0, 0], 0, 0, q[0, 1], 0, 0, q[0, 2]],
                                [q[1, 0], 0, 0, q[1, 1], 0, 0, q[1, 2], 0, 0],
                                [0, q[1, 0], 0, 0, q[1, 1], 0, 0, q[1, 2], 0],
                                [0, 0, q[1, 0], 0, 0, q[1, 1], 0, 0, q[1, 2]],
                                [q[2, 0], 0, 0, q[2, 1], 0, 0, q[2, 2], 0, 0],
                                [0, q[2, 0], 0, 0, q[2, 1], 0, 0, q[2, 2], 0],
                                [0, 0, q[2, 0], 0, 0, q[2, 1], 0, 0, q[2, 2]],
                                ])

    self.kf.Q = block_diag(q_second_order, q_second_order)
    
    # Measurement noise
    self.kf.R = np.eye(6) * 0.05

    # Transition matrix
    a_t = np.array([[1, 0, 0, dt, 0, 0, 0.5*dt**2, 0, 0],
                    [0, 1, 0, 0, dt, 0, 0, 0.5*dt**2, 0],
                    [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt**2],
                    [0, 0, 0, 1, 0, 0, dt, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, dt, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, dt],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1],
                    ])
  
    self.kf.F = np.zeros((18, 18), dtype=a_t.dtype)
    self.kf.F[:9, :9] = a_t
    self.kf.F[9:, 9:] = a_t

    # Measurement matrix
    self.kf.H = np.zeros((6, 18))
    self.kf.H[:3, :3] = np.eye(3)
    self.kf.H[3:, 9:12] = np.eye(3)

  def process_velocity_rolling_window_regression(data, window_size):
    """
    Estimate position and velocity of the opponent using rolling window regression on the given pose data
    """
    pass

  def write_data(self, output_file=None):
    if output_file is None:
      output_file = f"{self.input_file.strip('.csv')}_state_estimates.csv"
    self.state_estimates.to_csv(output_file)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("-i", "--input", type=str, help="Path to the input csv file containing poses over time")
  group.add_argument("--latest", action="store_true", help="Process the relative opponent poses of the latest run in the perception_debug directory")
  group.add_argument("--all", action="store_true", help="Process the relative opponent poses of all runs in the perception_debug directory")

  parser.add_argument("--frame_rate", type=float, default=30, help="Frequency of the poses in the input data")

  args = parser.parse_args()

  DEBUG_DIR = "perception_debug"
  if (args.all or args.latest) and not os.path.exists(DEBUG_DIR):
      print("To use --all or --latest, the perception_debug directory must exist in the current terminal's working directory")
      exit()
    
  if args.all:
    for run_dir in os.listdir(DEBUG_DIR):
      if os.path.isdir(run_dir):
        input_file = os.path.join(run_dir, "opp_rel_poses.csv")
        node = StateEstimator(input_file, interval=1/args.frame_rate)
        node.process_kalman_filter()
  elif args.latest:
    latest_dir = max([f.path for f in os.scandir(DEBUG_DIR) if f.is_dir()], key=os.path.getmtime)
    input_file = os.path.join(latest_dir, "opp_rel_poses.csv")
    node = StateEstimator(input_file, interval=1/args.frame_rate)
    node.process_kalman_filter()
  elif args.input:
    if not os.path.exists(args.input):
      print(f"The file {args.input} does not exist")
      exit()
    node = StateEstimator(args.input, interval=1/args.frame_rate)
    node.process_kalman_filter()
  