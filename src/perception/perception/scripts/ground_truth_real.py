from typing import Literal
import cv2
import pickle
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from perception import state_estimation
import perception.util.conversion as conv
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

BEV_TO_BACK_OPP_X = -100.0
BEV_TO_BACK_OPP_Y = 0.0
BEV_TO_BACK_OPP_Z = -120.0

BEV_TO_CAM_EGO_X = -184.8
BEV_TO_CAM_EGO_Y = -39.5
BEV_TO_CAM_EGO_Z = 12.5

# Make assumption that the top marker and the camera on the egovehicle have fixed relative orientation
# These constants are used for the Run 1 and Run 2 datasets
EGO_TOP_MARKER_TO_EGO_CAM_ROT = conv.get_rotation_matrix_from_euler(-90, 0, 90, degrees=True)
OPP_TOP_MARKER_TO_OPP_BACK_MARKER_ROT = conv.get_rotation_matrix_from_euler(-90, 90, 90, degrees=True)

class GroundTruthMeasurements:
  def __init__(self, bev_to_cam_ego_x = BEV_TO_CAM_EGO_X, bev_to_cam_ego_y = BEV_TO_CAM_EGO_Y, bev_to_cam_ego_z = BEV_TO_CAM_EGO_Z,
                bev_to_back_opp_x = BEV_TO_BACK_OPP_X, bev_to_back_opp_y = BEV_TO_BACK_OPP_Y, bev_to_back_opp_z = BEV_TO_BACK_OPP_Z,
                ego_top_marker_to_ego_cam_rot = EGO_TOP_MARKER_TO_EGO_CAM_ROT, opp_top_marker_to_opp_back_marker_rot = OPP_TOP_MARKER_TO_OPP_BACK_MARKER_ROT):
    self.bev_to_cam_ego_x = bev_to_cam_ego_x
    self.bev_to_cam_ego_y = bev_to_cam_ego_y
    self.bev_to_cam_ego_z = bev_to_cam_ego_z
    self.bev_to_back_opp_x = bev_to_back_opp_x
    self.bev_to_back_opp_y = bev_to_back_opp_y
    self.bev_to_back_opp_z = bev_to_back_opp_z
    self.ego_top_marker_to_ego_cam_rot = ego_top_marker_to_ego_cam_rot
    self.opp_top_marker_to_opp_back_marker_rot = opp_top_marker_to_opp_back_marker_rot

def generate_df_from_monte_carlo(monte_carlo_results):
  df = pd.DataFrame([{
    "time": results["time"],
    **{f"{t}_mean": results["mean"]["tvec"][i] for i, t in enumerate(["tx", "ty", "tz"])},
    **{f"{e}_mean": results["mean"]["euler"][i] for i, e in enumerate(["roll", "pitch", "yaw"])},
    **{f"{t}_std": results["std"]["tvec"][i] for i, t in enumerate(["tx", "ty", "tz"])},
    **{f"{e}_std": results["std"]["euler"][i] for i, e in enumerate(["roll", "pitch", "yaw"])},
    **{f"{t}_2.5": results["95"]["tvec"][0, i] for i, t in enumerate(["tx", "ty", "tz"])},
    **{f"{t}_97.5": results["95"]["tvec"][1, i] for i, t in enumerate(["tx", "ty", "tz"])},
    **{f"{e}_2.5": results["95"]["euler"][0, i] for i, e in enumerate(["roll", "pitch", "yaw"])},
    **{f"{e}_97.5": results["95"]["euler"][1, i] for i, e in enumerate(["roll", "pitch", "yaw"])},
    "cov": results["cov"]
  } for results in monte_carlo_results])

  return df
  
def read_single_run(bev_path, ego_path, window_before_start_time=1, selected_bev_cam="right"):
  start_time = int(np.loadtxt(f'{bev_path}/start_time.txt')) / 1e9
 
  ego_pose_df = remove_unused_frames(
    pd.read_csv(f'{bev_path}/bev/{selected_bev_cam}/ego_poses.csv'), start_time, window_before_start_time)

  opp_pose_df = remove_unused_frames(
    pd.read_csv(f'{bev_path}/bev/{selected_bev_cam}/opp_poses.csv'), start_time, window_before_start_time)
    
  tracking_df = remove_unused_frames(
    pd.read_csv(f'{ego_path}/opp_rel_poses.csv'), start_time, window_before_start_time)
  
  state_estimation_dfs = {
    f"{method}_df": remove_unused_frames(
      pd.read_csv(f'{ego_path}/state_estimates_{method}.csv'), start_time, window_before_start_time) \
      for method in ["kalman_ca", "kalman_cv", "rwr", "kalman_ca_depth_fusion"]
  }
  
  monte_carlo_results = pickle.load(open(f'{bev_path}/bev/{selected_bev_cam}/monte_carlo_results.pkl', 'rb'))
  monte_carlo_df = remove_unused_frames(generate_df_from_monte_carlo(monte_carlo_results), start_time, window_before_start_time)

  return ego_pose_df, opp_pose_df, tracking_df, state_estimation_dfs, monte_carlo_df

def read_all_runs(bev_paths, ego_paths, window_before_start_time=1):
  run_data = []
  if len(bev_paths) != len(ego_paths):
    raise ValueError("The number of BEV and ego paths should be the same")

  for bev_path, ego_path in zip(bev_paths, ego_paths):
    start_time = int(np.loadtxt(f'{bev_path}/start_time.txt')) / 1e9
    ego_bev_right_df, opp_bev_right_df, tracking_df, state_estimation_dfs, monte_carlo_right_df = read_single_run(bev_path, ego_path, window_before_start_time, selected_bev_cam="right")
    ego_bev_left_df, opp_bev_left_df, _, _, monte_carlo_left_df = read_single_run(bev_path, ego_path, window_before_start_time, selected_bev_cam="left")
    
    run_data.append({
      "start_time": start_time,
      "bev_path": bev_path,
      "ego_path": ego_path,
      "raw": {
        "ego_bev_right_df": ego_bev_right_df,
        "opp_bev_right_df": opp_bev_right_df,
        "ego_bev_left_df": ego_bev_left_df,
        "opp_bev_left_df": opp_bev_left_df,
      },
      "tracking_df": tracking_df,
      "monte_carlo_right_df": monte_carlo_right_df,
      "monte_carlo_left_df": monte_carlo_left_df,
      **state_estimation_dfs
    })
  return run_data

def remove_unused_frames(poses_df: pd.DataFrame, start_time, window_before_start_time=1):
  poses_df["time (sec)"] = poses_df["time"] / 1e9
  
  # Include a few frames before the trajectory start time
  poses_df_start_time: pd.DataFrame = poses_df[poses_df["time (sec)"] > start_time]
  first_index = poses_df_start_time.index.min()

  if window_before_start_time >= 1:
    poses_df_start_time = pd.concat([poses_df.iloc[first_index - window_before_start_time:first_index], poses_df_start_time])

  poses_df_start_time = poses_df_start_time.reset_index(drop=True)

  # normalise the time from df to start at 0
  poses_df_start_time['time_norm (sec)'] = (poses_df_start_time['time (sec)'] - start_time)
  
  return poses_df_start_time

def find_closest_equiv_angle(alpha, beta, degrees=True):
  """
  Find the equivalent angle to alpha that is closest to beta
  """
  threshold_diff = 180 if degrees else np.pi
  diff = beta - alpha
  
  if diff > threshold_diff:
    return alpha + 2 * threshold_diff
  elif diff < -threshold_diff:
    return alpha - 2 * threshold_diff
  return alpha

def stabilise_euler_angles(df: pd.DataFrame, cols: list[str], degrees=True) -> pd.DataFrame:
  """
  Stabilise the Euler angles by ensuring that the difference 
  between consecutive angles is less than 180 degrees
  """
  diff = 180 if degrees else np.pi
  for i, row in df.iterrows():
    if i == 0:
      continue

    for col in cols:
      prev = df.loc[i - 1, col]
      curr = row[col]

      if abs(prev - curr) > diff:
        df.loc[i, col] = find_closest_equiv_angle(curr, prev, degrees) 

  for col in cols:
    df.loc[0, col] = find_closest_equiv_angle(df.loc[0, col], df.loc[1, col], degrees)  
    
  return df

def generate_euler_angles(poses_df: pd.DataFrame, degrees=True) -> pd.DataFrame:
  poses_df['roll'], poses_df['pitch'], poses_df['yaw'] = \
    zip(*poses_df.apply(lambda row: conv.get_euler_from_quaternion(*row[['qx','qy','qz','qw']], degrees=degrees), axis=1))

  stabilised_df = stabilise_euler_angles(poses_df, ['roll', 'pitch', 'yaw'], degrees=degrees)
  return stabilised_df

def get_ego_cam_from_top_marker(bev_to_ego_top_marker_rvec, tvec, measurements: GroundTruthMeasurements):
  """
  Get the position of the ego camera in the BEV camera frame
  """
  ego_cam_in_ego_top_marker_frame = np.array([measurements.bev_to_cam_ego_x, measurements.bev_to_cam_ego_y, measurements.bev_to_cam_ego_z]) / 1000
  
  # Transforms points from the ego top marker frame to the BEV frame
  # Equivalently, aligns the BEV frame to the ego top marker frame
  bev_to_ego_top_marker_rot = cv2.Rodrigues(bev_to_ego_top_marker_rvec)[0]

  ego_cam_in_ego_top_marker_frame_rot_to_bev = bev_to_ego_top_marker_rot @ ego_cam_in_ego_top_marker_frame
  ego_top_marker_in_bev_frame = tvec

  ego_cam_in_bev_frame = ego_top_marker_in_bev_frame + ego_cam_in_ego_top_marker_frame_rot_to_bev
  return ego_cam_in_bev_frame

def get_opp_back_from_top_marker(bev_to_opp_top_marker_rvec, tvec, measurements: GroundTruthMeasurements):
  """
  Get the position of the opponent back marker in the BEV camera frame
  """
  opp_back_in_opp_top_marker_frame = np.array([measurements.bev_to_back_opp_x, measurements.bev_to_back_opp_y, measurements.bev_to_back_opp_z]) / 1000
  bev_to_opp_top_marker_rot = cv2.Rodrigues(bev_to_opp_top_marker_rvec)[0]

  opp_back_in_opp_top_marker_frame_rot_to_bev = bev_to_opp_top_marker_rot @ opp_back_in_opp_top_marker_frame
  opp_top_marker_in_bev_frame = tvec

  opp_back_in_bev_frame = opp_top_marker_in_bev_frame + opp_back_in_opp_top_marker_frame_rot_to_bev
  return opp_back_in_bev_frame

def rotate_bev_frame_to_ego_cam_frame(bev_to_ego_top_marker_rvec, tvec, measurements: GroundTruthMeasurements):
  """
  Rotate the pose vector in the BEV frame to the ego camera frame
  """
  roll, pitch, yaw = conv.get_euler_from_quaternion(*conv.get_quaternion_from_rodrigues(bev_to_ego_top_marker_rvec), degrees=True)

  # Sometimes the marker may be detected as tilted in the pitch axis. 
  # Here we assume that the camera is fixed with respect to the BEV camera 
  # (might have to tune this to get better matches since camera is sensitive to movement
  # TODO: make bev_to_ego_cam_rot a fixed global variable
  ego_top_marker_to_bev_rot = cv2.Rodrigues(conv.get_rodrigues_from_euler(roll, pitch, yaw, degrees=True))[0].T

  # Alternatively, this assumes that the camera and the top marker are aligned by 90 degree rotations
  # ego_top_marker_to_bev_rot = cv2.Rodrigues(bev_to_ego_top_marker_rvec)[0].T
  
  return measurements.ego_top_marker_to_ego_cam_rot.T @ ego_top_marker_to_bev_rot @ tvec

def get_ground_truth_rel_pose_ego_cam_frame(bev_to_ego_top_marker_rvec, ego_top_marker_in_bev_frame, bev_to_opp_top_marker_rvec, opp_top_marker_in_bev_frame,
                                             measurements: GroundTruthMeasurements, degrees=True):
  ego_cam_in_bev_frame = get_ego_cam_from_top_marker(bev_to_ego_top_marker_rvec, ego_top_marker_in_bev_frame, measurements)
  # Get position of opponent back marker in BEV frame
  opp_back_in_bev_frame = get_opp_back_from_top_marker(bev_to_opp_top_marker_rvec, opp_top_marker_in_bev_frame, measurements)

  # Get the relative position  of the opponent back marker in the BEV camera frame
  rel_pos_bev_frame = opp_back_in_bev_frame - ego_cam_in_bev_frame
  # Convert the relative position to the ego camera frame
  rel_pos_ego_cam_frame = rotate_bev_frame_to_ego_cam_frame(bev_to_ego_top_marker_rvec, rel_pos_bev_frame, measurements)

  # Get the relative orientation of the opponent back marker in the ego camera frame
  # TODO: make bev_to_ego_cam_rot a fixed global variable
  bev_to_ego_cam_rot = cv2.Rodrigues(bev_to_ego_top_marker_rvec)[0] @ measurements.ego_top_marker_to_ego_cam_rot
  bev_to_opp_back_rot = cv2.Rodrigues(bev_to_opp_top_marker_rvec)[0] @ measurements.opp_top_marker_to_opp_back_marker_rot
  ego_cam_to_opp_back_euler = conv.get_euler_from_rotation_matrix(bev_to_ego_cam_rot.T @ bev_to_opp_back_rot, degrees=degrees)
  
  return rel_pos_ego_cam_frame, ego_cam_to_opp_back_euler

def remove_nan_rows(run: list[dict]):
  combined_right_df = pd.concat([run["raw"]["ego_bev_right_df"], run["raw"]["opp_bev_right_df"]], axis=1, keys=['ego', 'opp'])
  combined_left_df = pd.concat([run["raw"]["ego_bev_left_df"], run["raw"]["opp_bev_left_df"]], axis=1, keys=['ego', 'opp'])

  # Drop rows with NaN values in any of the columns
  combined_right_cleaned = combined_right_df.dropna().reset_index(drop=True)
  combined_left_cleaned = combined_left_df.dropna().reset_index(drop=True)

  run["raw"]["ego_bev_right_df"] = combined_right_cleaned['ego']
  run["raw"]["opp_bev_right_df"] = combined_right_cleaned['opp']

  run["raw"]["ego_bev_left_df"] = combined_left_cleaned['ego']
  run["raw"]["opp_bev_left_df"] = combined_left_cleaned['opp']

  last_bev_pose_time = max(
    min(run["raw"]["ego_bev_right_df"]["time (sec)"].max(), 
        run["raw"]["opp_bev_right_df"]["time (sec)"].max()),
    min(run["raw"]["ego_bev_left_df"]["time (sec)"].max(),
        run["raw"]["opp_bev_left_df"]["time (sec)"].max())
  )
  
  # Trim the tracking data to the last BEV pose time
  run["tracking_df"] = run["tracking_df"][run["tracking_df"]["time (sec)"] <= last_bev_pose_time]
  run["tracking_df"] = run["tracking_df"].dropna().reset_index(drop=True)

  for state_est_method in ["kalman_ca", "kalman_cv", "rwr", "kalman_ca_depth_fusion"]:
    run[f"{state_est_method}_df"] = run[f"{state_est_method}_df"][run[f"{state_est_method}_df"]["time (sec)"] <= last_bev_pose_time]
    run[f"{state_est_method}_df"] = run[f"{state_est_method}_df"].dropna().reset_index(drop=True)

def generate_avg_bev_df(left_df: pd.DataFrame, right_df: pd.DataFrame, cols: list[str]):
  avg_df = left_df.copy()
  for col in cols:
    avg_df[col] = (left_df[col] + right_df[col]) / 2
  return avg_df[["time", "time (sec)", "time_norm (sec)", *cols]].dropna().reset_index(drop=True)

def generate_avg_monte_carlo_df(left_df: pd.DataFrame, right_df: pd.DataFrame):
  """
  Computes the average of the results from the left and right cameras' Monte Carlo runs

  Assumes a Gaussian distribution for the Monte Carlo results
  """
  avg_df = left_df.copy()
  for col in left_df.columns:
    if col in ["time", "time (sec)", "time_norm (sec)"]:
      continue
    if col.endswith("_std"):
      avg_df[col] = np.sqrt((left_df[col] ** 2 + right_df[col] ** 2)) / 2
    elif col.endswith("_2.5"):
      # Assume Gaussian distribution
      stats.norm.ppf(0.025) * np.sqrt((left_df[col] ** 2 + right_df[col] ** 2)) / 2 + (left_df[col] + right_df[col]) / 2
    elif col.endswith("_97.5"):
      # Assume Gaussian distribution
      stats.norm.ppf(0.975) * np.sqrt((left_df[col] ** 2 + right_df[col] ** 2)) / 2 + (left_df[col] + right_df[col]) / 2
    elif col.endswith("_mean"):
      avg_df[col] = (left_df[col] + right_df[col]) / 2
  return avg_df.dropna().reset_index(drop=True)

def correct_euler_offset(src_df: pd.DataFrame, compare_df: pd.DataFrame,
                         src_cols: list[str], compare_cols: list[str] | None = None, degrees=True, inplace=False):
  """
  Check if the Euler angle is larger or smaller than compare_df 
  by 180 degrees and correct
  """
  if compare_cols is None:
    compare_cols = src_cols
  if not inplace:
    src_df = src_df.copy()
  diff = 180 if degrees else np.pi

  for src_col, compare_col in zip(src_cols, compare_cols):
    if src_df.iloc[0][src_col] - compare_df.iloc[0][compare_col] > diff:
      src_df[src_col] = src_df[src_col] - 2 * diff
    elif src_df.iloc[0][src_col] - compare_df.iloc[0][compare_col] < -diff:
      src_df[src_col] = src_df[src_col] + 2 * diff
  return src_df

def process_run_data(run_data: list[dict], measurements: GroundTruthMeasurements, smoothing_types: list[Literal["savgol", "rolling", "lowess"]] = [], degrees=True):
  for run in run_data:
    remove_nan_rows(run)

    run["tracking_df"] = remove_unused_frames(run["tracking_df"], run["start_time"])
    run["tracking_df"] = generate_euler_angles(run["tracking_df"], degrees=degrees)

    monte_carlo_euler_cols = []
    monte_carlo_compare_cols = []
    for n in ["mean", "2.5", "97.5"]:
      for e in ["roll", "pitch", "yaw"]:
        monte_carlo_euler_cols.append(f"{e}_{n}")
        monte_carlo_compare_cols.append(f"{e}")
    
    # Correct the euler offsets for the monte carlo results
    for df_name in ["monte_carlo_right_df", "monte_carlo_left_df"]:
      run[df_name] = stabilise_euler_angles(run[df_name], monte_carlo_euler_cols, degrees=True)
      run[df_name] = correct_euler_offset(run[df_name], run["tracking_df"], monte_carlo_euler_cols, monte_carlo_compare_cols, degrees=True)
    
    run["monte_carlo_avg_df"] = generate_avg_monte_carlo_df(run["monte_carlo_left_df"], run["monte_carlo_right_df"])
    
    for df_name in run["raw"].keys():
      run["raw"][df_name] = remove_unused_frames(run["raw"][df_name], run["start_time"])
      run["raw"][df_name] = generate_euler_angles(run["raw"][df_name], degrees=degrees)
    
    # Generate the average BEV poses
    run["raw"]["ego_avg_bev_df"] = generate_avg_bev_df(
      run["raw"]["ego_bev_left_df"],
      run["raw"]["ego_bev_right_df"],
      ["tx", "ty", "tz", "roll", "pitch", "yaw"])

    for smoothing_type in smoothing_types:
      run[smoothing_type] = {
        df_name: generate_smoothed_data(run["raw"][df_name], smoothing_type) \
          for df_name in run["raw"]
      }
  
    for smoothing_type in ["raw"] + smoothing_types:
      run[smoothing_type]["rel_poses_left_df"] = compute_relative_pose(run[smoothing_type]["ego_bev_left_df"], run[smoothing_type]["opp_bev_left_df"], measurements)
      run[smoothing_type]["rel_poses_right_df"] = compute_relative_pose(run[smoothing_type]["ego_bev_right_df"], run[smoothing_type]["opp_bev_right_df"], measurements)
      
      # Make the angle difference between consecutive frames less than 180 degrees
      run[smoothing_type]["rel_poses_left_df"] = stabilise_euler_angles(run[smoothing_type]["rel_poses_left_df"], ["roll", "pitch", "yaw"], degrees=degrees)
      run[smoothing_type]["rel_poses_right_df"] = stabilise_euler_angles(run[smoothing_type]["rel_poses_right_df"], ["roll", "pitch", "yaw"], degrees=degrees)

      # Check if the Euler angle is larger or smaller than tracking_df by 180 degrees and correct
      for df_name in ["rel_poses_left_df", "rel_poses_right_df"]:
        run[smoothing_type][df_name] = correct_euler_offset(
          run[smoothing_type][df_name],
          run["tracking_df"],
          ["roll", "pitch", "yaw"],
          degrees=degrees
        )

      # generate the average relative poses
      run[smoothing_type]["rel_poses_avg_df"] = generate_avg_bev_df(
        run[smoothing_type]["rel_poses_left_df"], run[smoothing_type]["rel_poses_right_df"],
        ["tx", "ty", "tz", "roll", "pitch", "yaw"])

  return run_data

def apply_rolling_mean(df: pd.DataFrame, src: str, dest: str | None = None, window=20, min_periods=1, center=True):
  if dest is None:
    dest = src
  df.loc[:, dest] = df[src].rolling(window=window, min_periods=min_periods, center=center).mean()

def apply_savgol_filter(df: pd.DataFrame, src: str, dest: str | None = None, window=20, polyorder=5):
  if dest is None:
    dest = src
  df.loc[:, dest] = savgol_filter(df[src], window, polyorder)

def apply_lowess_smoothing(df: pd.DataFrame, src: str, dest: str | None = None, frac=0.1):
  if dest is None:
    dest = src
  df.loc[:, dest] = lowess(df[src], df["time_norm (sec)"], frac=frac)[:, 1]

def generate_smoothed_data(poses_df: pd.DataFrame, func: Literal["savgol", "rolling", "lowess"],
                            window=20, polyorder=5, frac=0.1, inplace=False):
  """
  Generate smoothed poses data using filters
  """
  if not inplace:
    poses_df = poses_df.copy()

  pos_columns = ['tx', 'ty', 'tz']
  rot_columns = ['roll', 'pitch', 'yaw']
  for col in pos_columns + rot_columns:
    if func == "savgol":
      apply_savgol_filter(poses_df, col, window=window, polyorder=polyorder)
    elif func == "rolling":
      apply_rolling_mean(poses_df, col, window=window)
    elif func == "lowess":
      apply_lowess_smoothing(poses_df, col, frac=frac)

  poses_df = stabilise_euler_angles(poses_df, rot_columns)

  poses_df["ax"], poses_df["ay"], poses_df["az"] = \
    zip(*poses_df.apply(lambda row: conv.get_rodrigues_from_euler(
      *row[["roll", "pitch", "yaw"]], degrees=True), axis=1))
  poses_df["qx"], poses_df["qy"], poses_df["qz"], poses_df["qw"] = \
    zip(*poses_df.apply(lambda row: conv.get_quaternion_from_euler(
      *row[["roll", "pitch", "yaw"]], degrees=True), axis=1))

  return poses_df

def compute_relative_pose(ego_bev_df: pd.DataFrame, opp_bev_df: pd.DataFrame, measurements: GroundTruthMeasurements, degrees=True):
  rel_poses = pd.DataFrame(columns=["time (sec)", "time_norm (sec)"])

  for i in range(len(ego_bev_df)):
    ego_top_marker_to_bev_rvec = ego_bev_df.iloc[i][["ax", "ay", "az"]].to_numpy()
    ego_top_marker_in_bev_frame = ego_bev_df.iloc[i][["tx", "ty", "tz"]].to_numpy()

    opp_top_marker_to_bev_rvec = opp_bev_df.iloc[i][["ax", "ay", "az"]].to_numpy()
    opp_top_marker_in_bev_frame = opp_bev_df.iloc[i][["tx", "ty", "tz"]].to_numpy()

    ground_truth_rel_pose_ego_cam_frame, ground_truth_ego_cam_to_opp_back_euler = get_ground_truth_rel_pose_ego_cam_frame(
      ego_top_marker_to_bev_rvec, ego_top_marker_in_bev_frame,
      opp_top_marker_to_bev_rvec, opp_top_marker_in_bev_frame,
      measurements, degrees)
    
    rel_poses = pd.concat([rel_poses, pd.DataFrame([{
      "time": ego_bev_df.loc[i, "time"],
      "time (sec)": ego_bev_df.loc[i, "time (sec)"],
      "time_norm (sec)": ego_bev_df.loc[i, "time_norm (sec)"],
      "tx": ground_truth_rel_pose_ego_cam_frame[0],
      "ty": ground_truth_rel_pose_ego_cam_frame[1],
      "tz": ground_truth_rel_pose_ego_cam_frame[2],
      "roll": ground_truth_ego_cam_to_opp_back_euler[0],
      "pitch": ground_truth_ego_cam_to_opp_back_euler[1],
      "yaw": ground_truth_ego_cam_to_opp_back_euler[2]
    }])])

  return rel_poses.reset_index(drop=True)

def fit_polynomial(x: np.ndarray, y: np.ndarray, degree=1, regularise=False, **kwargs):
  # https://ostwalprasad.github.io/machine-learning/Polynomial-Regression-using-statsmodel.html
  polynomial_features= PolynomialFeatures(degree=degree)
  x = x.reshape(-1, 1)
  xp = polynomial_features.fit_transform(x)

  if regularise:
    model = sm.OLS(y, xp).fit_regularized(**kwargs)
  else:
    model = sm.OLS(y, xp).fit()
  model = np.poly1d(model.params[::-1])
  model_derivative = np.polyder(model)
  model_2nd_derivative = np.polyder(model_derivative)

  return model, model_derivative, model_2nd_derivative