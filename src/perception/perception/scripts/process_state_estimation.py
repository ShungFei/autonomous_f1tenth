import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
from sklearn.linear_model import LinearRegression

import argparse
import os
import pandas as pd
from math import ceil

from perception.scripts.ground_truth_real import stabilise_euler_angles
from perception.util.conversion import get_euler_from_quaternion

class StateEstimator():
  def __init__(self, run_dir, method, frame_rate=30, degrees=False):
    self.run_dir = run_dir
    self.method = method
    self.frame_rate = frame_rate
    self.degrees = degrees
      
  def process(self):
    all_state_estimates = []
    for process_sub_dir, _, _ in os.walk(self.run_dir):
      print('Processing:', process_sub_dir)

      if not os.path.exists(f"{process_sub_dir}/opp_rel_poses.csv"):
        print(f"Skipping {process_sub_dir} as opp_rel_poses.csv is missing")
        continue

      measured_poses = pd.read_csv(f"{process_sub_dir}/opp_rel_poses.csv")
      
      # get euler angles from quaternions and skip zero norm quaternions
      measured_poses[["roll", "pitch", "yaw"]] = measured_poses[["qx", "qy", "qz", "qw"]].apply(
        lambda x: get_euler_from_quaternion(*x, degrees=self.degrees) if np.linalg.norm(x) > 0 else [np.nan] * 3, axis=1, result_type="expand")
      # make the euler angles continuous
      measured_poses = stabilise_euler_angles(measured_poses, ["roll", "pitch", "yaw"], degrees=self.degrees)
      measured_poses.set_index("time", inplace=True)
      
      if self.method == "kalman_ca":
        state_estimates = self.process_kalman_filter(measured_poses, is_constant_velocity_model=False)
      elif self.method == "kalman_cv":
        state_estimates = self.process_kalman_filter(measured_poses, is_constant_velocity_model=True)
      elif self.method == "rwr":
        state_estimates = self.process_velocity_rolling_window_regression(measured_poses, window_size=5)
      elif self.method == "kalman_ca_depth_fusion":
        state_estimates = self.process_kalman_filter(measured_poses, is_constant_velocity_model=False, is_depth_fusion=True)
      
      all_state_estimates.append(state_estimates)
      self.write_data(state_estimates, f"{process_sub_dir}/state_estimates_{self.method}.csv")
    
    return all_state_estimates

  def process_kalman_filter(self, measured_poses, is_constant_velocity_model, is_depth_fusion=False):
    # Index is all potential time intervals between the first and last time in the measured poses
    expected_times_count = ceil((measured_poses.index[-1] - measured_poses.index[0]) / (1e9 / self.frame_rate))
    end_time = measured_poses.index[0] + expected_times_count * (1e9 / self.frame_rate)
    expected_times = np.linspace(measured_poses.index[0], end_time, expected_times_count + 1)

    if is_constant_velocity_model:
      state_estimates = pd.DataFrame(columns=["tx", "ty", "tz",
                                              "vx", "vy", "vz",
                                              "roll", "pitch", "yaw",
                                              "vroll", "vpitch", "vyaw"], index=expected_times)
    else:
      state_estimates = pd.DataFrame(columns=["tx", "ty", "tz",
                                              "vx", "vy", "vz",
                                              "ax", "ay", "az",
                                              "roll", "pitch", "yaw",
                                              "vroll", "vpitch", "vyaw",
                                              "aroll","apitch", "ayaw"], index=expected_times)
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
      if i != -1 and not pd.isna(measured_poses.iloc[i][["qx", "qy", "qz", "qw", "tx", "ty", "tz", "roll", "pitch", "yaw"]]).any():
        if is_depth_fusion:
          # Update the measurement matrix to only update the position
          kf.H[3:, 6 if is_constant_velocity_model else 9:9 if is_constant_velocity_model else 12] = np.zeros((3, 3))
          
          depth_position = measured_poses.iloc[i][["depth_tx", "depth_ty", "depth_tz"]].values
          kf.update(np.concatenate((depth_position, np.zeros(3))))
          
          # Reset the measurement matrix to update position and orientation
          kf.H[3:, 6 if is_constant_velocity_model else 9:9 if is_constant_velocity_model else 12] = np.eye(3)

        pose = measured_poses.iloc[i][["tx", "ty", "tz", "roll", "pitch", "yaw"]].values
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
    q = Q_discrete_white_noise(dim=3, dt=dt, var=10)

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
    kf.R = np.diag([0.05, 0.05, 0.05, 4, 4, 4])

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

  def process_velocity_rolling_window_regression(self, measured_poses: pd.DataFrame, window_size: int):
    """
    Estimate velocity of the opponent using linear regression across a rolling window of poses
    The position is taken from the linear regression line at the last time in the window
    """

    state_estimates = pd.DataFrame(columns=["tx", "ty", "tz",
                                            "roll", "pitch", "yaw",
                                            "vx", "vy", "vz",
                                            "vroll", "vpitch", "vyaw"], index=measured_poses.index)
    state_estimates.index.name = "time"

    # Remove NaN values from the measured poses
    measured_poses = measured_poses.dropna(subset=["tx", "ty", "tz", "qx", "qy", "qz", "qw"])

    for i in range(len(measured_poses) - window_size):
      window = measured_poses.iloc[i:i+window_size]
      time = window.index[-1]
      
      X = window.index.values.reshape(-1, 1)
      y = window[["tx", "ty", "tz", "roll", "pitch", "yaw"]].values

      reg = LinearRegression().fit(X, y)
      pose = reg.predict(np.array([[time]]))[0]
      velocity = reg.coef_.flatten() * 1e9

      state_estimates.loc[time] = np.concatenate((pose, velocity))

    return state_estimates

  def write_data(self, state_estimates: pd.DataFrame, output_file: str):
    state_estimates.to_csv(output_file)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("-r", "--run_dir", type=str, help="Path to the input file containing poses over time")
  group.add_argument("--latest", action="store_true", help="Process the relative opponent poses of the latest run in the perception_debug directory")
  group.add_argument("--all", action="store_true", help="Process the relative opponent poses of all runs in the perception_debug directory")

  parser.add_argument("-m", "--method", type=str, required=True, choices=["kalman_ca", "kalman_cv", "rwr", "kalman_ca_depth_fusion"], help="Method to use for state estimation.")
  parser.add_argument("--frame_rate", type=float, default=30, help="Frequency of the poses in the input data")
  parser.add_argument("--degrees", action="store_true", help="Output euler angles in degrees")

  args = parser.parse_args()

  DEBUG_DIR = "perception_debug"
  if (args.all or args.latest) and not os.path.exists(DEBUG_DIR):
      print("To use --all or --latest, the perception_debug directory must exist in the current terminal's working directory")
      exit()
    
  if args.all:
    for run_dir in os.listdir(DEBUG_DIR):
      if os.path.isdir(run_dir):
        node = StateEstimator(run_dir, method=args.method, frame_rate=args.frame_rate, degrees=args.degrees)
        node.process()
  elif args.latest:
    latest_dir = max([f.path for f in os.scandir(DEBUG_DIR) if f.is_dir()], key=os.path.getmtime)
    node = StateEstimator(latest_dir, method=args.method, frame_rate=args.frame_rate, degrees=args.degrees)
    node.process()
  elif args.run_dir:
    if not os.path.exists(args.run_dir):
      print(f"The directory {args.run_dir} does not exist")
      exit()
    node = StateEstimator(args.run_dir, method=args.method, frame_rate=args.frame_rate, degrees=args.degrees)
    node.process()
  