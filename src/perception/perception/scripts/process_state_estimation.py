import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag

import argparse
import os
import numpy as np
import pandas as pd
from math import ceil

from perception.util.conversion import get_euler_from_quaternion

class StateEstimator():
  def __init__(self, run_dir, method, frame_rate=30):
    self.run_dir = run_dir
    self.method = method
    self.frame_rate = frame_rate
      
  def process(self):
    for process_sub_dir, _, _ in os.walk(self.run_dir):
      print('Processing:', process_sub_dir)

      if not os.path.exists(f"{process_sub_dir}/opp_rel_poses.csv"):
        print(f"Skipping {process_sub_dir} as opp_rel_poses.csv is missing")
        continue

      measured_poses = pd.read_csv(f"{process_sub_dir}/opp_rel_poses.csv").set_index("time")
      
      if self.method == "kalman_ca":
        state_estimates = self.process_kalman_filter(measured_poses, is_constant_velocity_model=False)
      elif self.method == "kalman_cv":
        state_estimates = self.process_kalman_filter(measured_poses, is_constant_velocity_model=True)
      elif self.method == "rwr":
        state_estimates = self.process_velocity_rolling_window_regression(measured_poses, window_size=5)
      
      self.write_data(state_estimates, f"{process_sub_dir}/state_estimates_{self.method}.csv")

  def process_kalman_filter(self, measured_poses, is_constant_velocity_model):
    # Index is all potential time intervals between the first and last time in the measured poses
    expected_times_count = ceil((measured_poses.index[-1] - measured_poses.index[0]) / (1e9 / self.frame_rate))
    end_time = measured_poses.index[0] + expected_times_count * (1e9 / self.frame_rate)
    expected_times = np.linspace(measured_poses.index[0], end_time, expected_times_count + 1)

    if is_constant_velocity_model:
      state_estimates = pd.DataFrame(columns=["position_x", "position_y", "position_z",
                                                      "linear_velocity_x", "linear_velocity_y", "linear_velocity_z",
                                                      "orientation_x", "orientation_y", "orientation_z",
                                                      "angular_velocity_x", "angular_velocity_y", "angular_velocity_z"], index=expected_times)
    else:
      state_estimates = pd.DataFrame(columns=["position_x", "position_y", "position_z",
                                                      "linear_velocity_x", "linear_velocity_y", "linear_velocity_z",
                                                      "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
                                                      "orientation_x", "orientation_y", "orientation_z",
                                                      "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
                                                      "angular_acceleration_x","angular_acceleration_y", "angular_acceleration_z"], index=expected_times)
    state_estimates.index.name = "time"
    
    # Initialize the Kalman filter with the first pose
    if is_constant_velocity_model:
      kf = self.initialize_kalman_filter_cv(1 / self.frame_rate, pose=measured_poses.iloc[0])
    else:
      kf = self.initialize_kalman_filter_ca(1 / self.frame_rate, pose=measured_poses.iloc[0])

    for expected_time in state_estimates.index:
      kf.predict()

      # Update the Kalman filter with the closest measured pose before the expected time (within the interval of one frame)
      i = measured_poses.index.get_indexer([expected_time], method="ffill", tolerance=1e9 / self.frame_rate)[0]
      
      # Skip the update step if there is no measured pose before the expected time
      if i != -1 and not pd.isna(measured_poses.iloc[i][["qx", "qy", "qz", "qw", "tx", "ty", "tz"]]).any():
        orientation = get_euler_from_quaternion(*measured_poses.iloc[i][["qx", "qy", "qz", "qw"]].values)
        pose = np.concatenate((measured_poses.iloc[i][["tx", "ty", "tz"]].values, orientation))
        kf.update(pose)

      state_estimate = kf.x
      state_estimates.loc[expected_time] = state_estimate
    
    return state_estimates

  def initialize_kalman_filter_cv(self, dt, pose = np.zeros(6)):
    """Constant velocity model Kalman filter"""
    kf = KalmanFilter(dim_x=12, dim_z=6)

    # State
    kf.x = np.zeros(12)
    kf.x[:3] = pose[:3]
    kf.x[6:9] = pose[3:6]
    # State covariance
    kf.P = np.diag([1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2])
    # Process noise
    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.05)

    q_second_order = np.array([[q[0, 0], 0, 0, q[0, 1], 0, 0],
                                [0, q[0, 0], 0, 0, q[0, 1], 0],
                                [0, 0, q[0, 0], 0, 0, q[0, 1]],
                                [q[1, 0], 0, 0, q[1, 1], 0, 0],
                                [0, q[1, 0], 0, 0, q[1, 1], 0],
                                [0, 0, q[1, 0], 0, 0, q[1, 1]],
                                ])

    kf.Q = block_diag(q_second_order, q_second_order)
    
    # Measurement noise
    kf.R = np.eye(6) * 0.05

    # Transition matrix
    a_t = np.array([[1, 0, 0, dt, 0, 0],
                    [0, 1, 0, 0, dt, 0],
                    [0, 0, 1, 0, 0, dt],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                    ])
  
    kf.F = np.zeros((12, 12), dtype=a_t.dtype)
    kf.F[:6, :6] = a_t
    kf.F[6:, 6:] = a_t

    # Measurement matrix
    kf.H = np.zeros((6, 12))
    kf.H[:3, :3] = np.eye(3)
    kf.H[3:, 6:9] = np.eye(3)

    return kf

  def initialize_kalman_filter_ca(self, dt, pose = np.zeros(6)):
    """Constant acceleration model Kalman filter"""
    kf = KalmanFilter(dim_x=18, dim_z=6)

    # State
    kf.x = np.zeros(18)
    kf.x[:3] = pose[:3]
    kf.x[9:12] = pose[3:6]
    # State covariance
    kf.P = np.diag([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3])
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

    kf.Q = block_diag(q_second_order, q_second_order)
    
    # Measurement noise
    kf.R = np.eye(6) * 0.05

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
  
    kf.F = np.zeros((18, 18), dtype=a_t.dtype)
    kf.F[:9, :9] = a_t
    kf.F[9:, 9:] = a_t

    # Measurement matrix
    kf.H = np.zeros((6, 18))
    kf.H[:3, :3] = np.eye(3)
    kf.H[3:, 9:12] = np.eye(3)

    return kf

  def process_velocity_rolling_window_regression(data, window_size):
    """
    Estimate position and velocity of the opponent using rolling window regression on the given pose data
    """
    pass

  def write_data(self, state_estimates: pd.DataFrame, output_file: str):
    state_estimates.to_csv(output_file)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("-r", "--run_dir", type=str, help="Path to the input file containing poses over time")
  group.add_argument("--latest", action="store_true", help="Process the relative opponent poses of the latest run in the perception_debug directory")
  group.add_argument("--all", action="store_true", help="Process the relative opponent poses of all runs in the perception_debug directory")

  parser.add_argument("-m", "--method", type=str, required=True, choices=["kalman_ca", "kalman_cv", "rwr"], help="Method to use for state estimation.")
  parser.add_argument("--frame_rate", type=float, default=30, help="Frequency of the poses in the input data")

  args = parser.parse_args()

  DEBUG_DIR = "perception_debug"
  if (args.all or args.latest) and not os.path.exists(DEBUG_DIR):
      print("To use --all or --latest, the perception_debug directory must exist in the current terminal's working directory")
      exit()
    
  if args.all:
    for run_dir in os.listdir(DEBUG_DIR):
      if os.path.isdir(run_dir):
        node = StateEstimator(run_dir, method=args.method, frame_rate=args.frame_rate)
        node.process()
  elif args.latest:
    latest_dir = max([f.path for f in os.scandir(DEBUG_DIR) if f.is_dir()], key=os.path.getmtime)
    node = StateEstimator(latest_dir, method=args.method, frame_rate=args.frame_rate)
    node.process()
  elif args.run_dir:
    if not os.path.exists(args.run_dir):
      print(f"The directory {args.run_dir} does not exist")
      exit()
    node = StateEstimator(args.run_dir, method=args.method, frame_rate=args.frame_rate)
    node.process()
  