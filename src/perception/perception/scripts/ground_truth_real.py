import cv2
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import perception.util.conversion as conv

BEV_TO_BACK_OPP_X = -100.0
BEV_TO_BACK_OPP_Y = 0.0
BEV_TO_BACK_OPP_Z = -120.0

BEV_TO_CAM_EGO_X = -184.8
BEV_TO_CAM_EGO_Y = -39.5
BEV_TO_CAM_EGO_Z = 12.5

# Make assumption that the top marker and the camera on the egovehicle have fixed relative orientation
EGO_TOP_MARKER_TO_EGO_CAM_QUAT = np.array([-0.5, -0.5, 0.5, 0.5])

def process_single_run(bev_path, ego_path, window_before_start_time=1, selected_bev_cam="right"):
  start_time = int(np.loadtxt(f'{bev_path}/start_time.txt')) / 1e9
 
  ego_pose_df = process_poses(
    pd.read_csv(f'{bev_path}/bev/{selected_bev_cam}/ego_poses.csv'), start_time, window_before_start_time)

  opp_pose_df = process_poses(
    pd.read_csv(f'{bev_path}/bev/{selected_bev_cam}/opp_poses.csv'), start_time, window_before_start_time)

  tracking_df = process_poses(
    pd.read_csv(f'{ego_path}/opp_rel_poses.csv'), start_time, window_before_start_time)
  
  last_bev_pose_time = max(ego_pose_df["time (sec)"].max(), opp_pose_df["time (sec)"].max())
  # Trim the tracking data to the last BEV pose time
  tracking_df = tracking_df[tracking_df["time (sec)"] <= last_bev_pose_time]

  return ego_pose_df, opp_pose_df, tracking_df

def process_all_runs(bev_paths, ego_paths, window_before_start_time=1):
  bev_run_data = []
  if len(bev_paths) != len(ego_paths):
    raise ValueError("The number of BEV and ego paths should be the same")

  for bev_path, ego_path in zip(bev_paths, ego_paths):
    start_time = int(np.loadtxt(f'{bev_path}/start_time.txt')) / 1e9
    ego_bev_right_df, opp_bev_right_df, tracking_df = process_single_run(bev_path, ego_path, window_before_start_time, selected_bev_cam="right")
    ego_bev_left_df, opp_bev_left_df, _ = process_single_run(bev_path, ego_path, window_before_start_time, selected_bev_cam="left")
    
    bev_run_data.append({
      "start_time": start_time,
      "bev_path": bev_path,
      "ego_path": ego_path,
      "ego_bev_right_df": ego_bev_right_df,
      "opp_bev_right_df": opp_bev_right_df,
      "ego_bev_left_df": ego_bev_left_df,
      "opp_bev_left_df": opp_bev_left_df,
      "tracking_df": tracking_df
    })
  return bev_run_data

def process_poses(poses_df: pd.DataFrame, start_time, window_before_start_time=1):
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

def stabilise_euler_angles(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
  """
  Stabilise the Euler angles by ensuring that the difference 
  between consecutive angles is less than 180 degrees
  """
  def find_closest_equiv_angle(alpha, beta):
    """
    Find the equivalent angle to alpha that is closest to beta
    """
    diff = beta - alpha
    if diff > 180:
      return alpha + 360
    elif diff < -180:
      return alpha - 360
    return alpha
  
  for i, row in df.iterrows():
    if i == 0:
      continue

    for col in cols:
      prev = df.loc[i - 1, col]
      curr = row[col]

      if abs(prev - curr) > 180:
        df.loc[i, col] = find_closest_equiv_angle(curr, prev) 

  for col in cols:
    df.loc[0, col] = find_closest_equiv_angle(df.loc[0, col], df.loc[1, col])  
    
  return df

def generate_euler_angles(poses_df: pd.DataFrame) -> pd.DataFrame:
  poses_df['roll'], poses_df['pitch'], poses_df['yaw'] = \
    zip(*poses_df.apply(lambda row: conv.get_euler_from_quaternion(*row[['qx','qy','qz','qw']], degrees=True), axis=1))

  stabilised_df = stabilise_euler_angles(poses_df, ['roll', 'pitch', 'yaw'])
  return stabilised_df

def get_ego_cam_from_top_marker(rvec, tvec):
  """
  Get the position of the ego camera in the BEV camera frame
  """
  ego_cam_in_ego_top_marker_frame = np.array([BEV_TO_CAM_EGO_X, BEV_TO_CAM_EGO_Y, BEV_TO_CAM_EGO_Z]) / 1000
  # Figure out whether the rotation should be transposed
  ego_top_marker_to_bev_rot = cv2.Rodrigues(rvec)[0]

  ego_cam_in_ego_top_marker_frame_rot_to_bev = ego_top_marker_to_bev_rot @ ego_cam_in_ego_top_marker_frame
  ego_top_marker_in_bev_frame = tvec

  ego_cam_in_bev_frame = ego_top_marker_in_bev_frame + ego_cam_in_ego_top_marker_frame_rot_to_bev
  return ego_cam_in_bev_frame

def get_opp_back_from_top_marker(rvec, tvec):
  """
  Get the position of the opponent back marker in the BEV camera frame
  """
  opp_back_in_opp_top_marker_frame = np.array([BEV_TO_BACK_OPP_X, BEV_TO_BACK_OPP_Y, BEV_TO_BACK_OPP_Z]) / 1000
  opp_top_marker_to_bev_rot = cv2.Rodrigues(rvec)[0]

  opp_back_in_opp_top_marker_frame_rot_to_bev = opp_top_marker_to_bev_rot @ opp_back_in_opp_top_marker_frame
  opp_top_marker_in_bev_frame = tvec

  opp_back_in_bev_frame = opp_top_marker_in_bev_frame + opp_back_in_opp_top_marker_frame_rot_to_bev
  return opp_back_in_bev_frame

def rotate_bev_frame_to_ego_cam_frame(rvec, tvec):
  """
  Rotate the pose vector in the BEV frame to the ego camera frame
  """
  roll, _, yaw = conv.get_euler_from_quaternion(*conv.get_quaternion_from_rodrigues(rvec))

  # Sometimes the marker may be detected as tilted in the pitch axis. 
  # Here we assume that the camera is always parallel to the ground
  bev_to_ego_top_marker_rot = cv2.Rodrigues(conv.get_rodrigues_from_euler(roll, 0, yaw))[0].T

  # Alternatively, this assumes that the camera and the top marker are aligned by 90 degree rotations
  # bev_to_ego_top_marker_rot = cv2.Rodrigues(rvec)[0].T
  
  return conv.get_rotation_matrix_from_quaternion(*EGO_TOP_MARKER_TO_EGO_CAM_QUAT).T @ bev_to_ego_top_marker_rot @ tvec

def get_ground_truth_rel_pose_ego_cam_frame(ego_top_marker_to_bev_rvec, ego_top_marker_in_bev_frame, opp_top_marker_to_bev_rvec, opp_top_marker_in_bev_frame):
  ego_cam_in_bev_frame = get_ego_cam_from_top_marker(ego_top_marker_to_bev_rvec, ego_top_marker_in_bev_frame)
  # Get position of opponent back marker in BEV frame
  opp_back_in_bev_frame = get_opp_back_from_top_marker(opp_top_marker_to_bev_rvec, opp_top_marker_in_bev_frame)

  # Get the relative pose of the opponent back marker in the BEV camera frame
  ground_truth_rel_pose_bev_frame = opp_back_in_bev_frame - ego_cam_in_bev_frame
  # Convert the relative pose to the ego camera frame
  ground_truth_rel_pose_ego_cam_frame = rotate_bev_frame_to_ego_cam_frame(ego_top_marker_to_bev_rvec, ground_truth_rel_pose_bev_frame)
  
  return ground_truth_rel_pose_ego_cam_frame

def apply_rolling_mean(df: pd.DataFrame, src: str, dest: str | None = None, window=20, min_periods=1, center=True):
  if dest is None:
    dest = src
  df.loc[:, dest] = df[src].rolling(window=window, min_periods=min_periods, center=center).mean()

def apply_savgol_filter(df: pd.DataFrame, src: str, dest: str | None = None, window=20, polyorder=5):
  if dest is None:
    dest = src
  df.loc[:, dest] = savgol_filter(df[src], window, polyorder)

def generate_savgol_tracking_data(tracking_df: pd.DataFrame, window=20, polyorder=5, inplace=False):
  if not inplace:
    tracking_df = tracking_df.copy()

  pos_columns = ['tx', 'ty', 'tz']
  rot_columns = ['roll', 'pitch', 'yaw']
  for col in pos_columns + rot_columns:
    apply_savgol_filter(tracking_df, col, window=window, polyorder=polyorder)

  stabilise_euler_angles(tracking_df, rot_columns)

  tracking_df["ax"], tracking_df["ay"], tracking_df["az"] = \
    zip(*tracking_df.apply(lambda row: conv.get_rodrigues_from_euler(
      *row[["roll", "pitch", "yaw"]], degrees=True), axis=1))
  tracking_df["qx"], tracking_df["qy"], tracking_df["qz"], tracking_df["qw"] = \
    zip(*tracking_df.apply(lambda row: conv.get_quaternion_from_euler(
      *row[["roll", "pitch", "yaw"]], degrees=True), axis=1))

  return tracking_df

def generate_rolling_mean_tracking_data(tracking_df: pd.DataFrame, window=20):
  pos_columns = ['tx', 'ty', 'tz']
  rot_columns = ['roll', 'pitch', 'yaw']
  for col in pos_columns + rot_columns:
    apply_rolling_mean(tracking_df, col, window=window)

  stabilise_euler_angles(tracking_df, rot_columns)

  tracking_df["ax"], tracking_df["ay"], tracking_df["az"] = \
    zip(*tracking_df.apply(lambda row: conv.get_rodrigues_from_euler(
      *row[["roll", "pitch", "yaw"]], degrees=True), axis=1))
  tracking_df["qx"], tracking_df["qy"], tracking_df["qz"], tracking_df["qw"] = \
    zip(*tracking_df.apply(lambda row: conv.get_quaternion_from_euler(
      *row[["roll", "pitch", "yaw"]], degrees=True), axis=1))

  return tracking_df

def compute_relative_pose(ego_bev_df: pd.DataFrame, opp_bev_df: pd.DataFrame):
  rel_poses = pd.DataFrame(columns=["time (sec)", "time_norm (sec)", "rel_x", "rel_y", "rel_z", "rel_roll", "rel_pitch", "rel_yaw"])

  for i in range(len(ego_bev_df)):
    ego_top_marker_to_bev_rvec = ego_bev_df.iloc[i][["ax", "ay", "az"]].to_numpy()
    ego_top_marker_in_bev_frame = ego_bev_df.iloc[i][["tx", "ty", "tz"]].to_numpy()

    opp_top_marker_to_bev_rvec = opp_bev_df.iloc[i][["ax", "ay", "az"]].to_numpy()
    opp_top_marker_in_bev_frame = opp_bev_df.iloc[i][["tx", "ty", "tz"]].to_numpy()

    ego_cam_in_bev_frame = get_ego_cam_from_top_marker(ego_top_marker_to_bev_rvec, ego_top_marker_in_bev_frame)
    opp_back_in_bev_frame = get_opp_back_from_top_marker(opp_top_marker_to_bev_rvec, opp_top_marker_in_bev_frame)

    ground_truth_rel_pose_bev_frame = opp_back_in_bev_frame - ego_cam_in_bev_frame
    ground_truth_rel_pose_ego_cam_frame = rotate_bev_frame_to_ego_cam_frame(ego_top_marker_to_bev_rvec, ground_truth_rel_pose_bev_frame)

    rel_poses = pd.concat([rel_poses, pd.DataFrame([{
      "time (sec)": ego_bev_df.loc[i, "time (sec)"],
      "time_norm (sec)": ego_bev_df.loc[i, "time_norm (sec)"],
      "rel_x": ground_truth_rel_pose_ego_cam_frame[0],
      "rel_y": ground_truth_rel_pose_ego_cam_frame[1],
      "rel_z": ground_truth_rel_pose_ego_cam_frame[2],
    }])])

  return rel_poses.reset_index(drop=True)

def fit_polynomial(df: pd.DataFrame, x_col: str, y_col: str, degree=1):
  # https://ostwalprasad.github.io/machine-learning/Polynomial-Regression-using-statsmodel.html
  polynomial_features= PolynomialFeatures(degree=degree)
  x = df[x_col].values.reshape(-1, 1)
  xp = polynomial_features.fit_transform(x)

  model = sm.OLS(df[y_col], xp).fit()
  model_derivative = np.polyder(model.params[::-1])
  model_2nd_derivative = np.polyder(model_derivative)

  return model, model_derivative, model_2nd_derivative