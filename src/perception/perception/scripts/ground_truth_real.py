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
EGO_TOP_MARKER_TO_EGO_CAM_ROT = conv.get_rotation_matrix_from_euler(-90, 0, 90, degrees=True)
OPP_TOP_MARKER_TO_OPP_BACK_MARKER_ROT = conv.get_rotation_matrix_from_euler(-90, 90, 90, degrees=True)

def read_single_run(bev_path, ego_path, window_before_start_time=1, selected_bev_cam="right"):
  start_time = int(np.loadtxt(f'{bev_path}/start_time.txt')) / 1e9
 
  ego_pose_df = remove_unused_frames(
    pd.read_csv(f'{bev_path}/bev/{selected_bev_cam}/ego_poses.csv'), start_time, window_before_start_time)

  opp_pose_df = remove_unused_frames(
    pd.read_csv(f'{bev_path}/bev/{selected_bev_cam}/opp_poses.csv'), start_time, window_before_start_time)

  tracking_df = remove_unused_frames(
    pd.read_csv(f'{ego_path}/opp_rel_poses.csv'), start_time, window_before_start_time)

  return ego_pose_df, opp_pose_df, tracking_df

def read_all_runs(bev_paths, ego_paths, window_before_start_time=1):
  run_data = []
  if len(bev_paths) != len(ego_paths):
    raise ValueError("The number of BEV and ego paths should be the same")

  for bev_path, ego_path in zip(bev_paths, ego_paths):
    start_time = int(np.loadtxt(f'{bev_path}/start_time.txt')) / 1e9
    ego_bev_right_df, opp_bev_right_df, tracking_df = read_single_run(bev_path, ego_path, window_before_start_time, selected_bev_cam="right")
    ego_bev_left_df, opp_bev_left_df, _ = read_single_run(bev_path, ego_path, window_before_start_time, selected_bev_cam="left")
    
    run_data.append({
      "start_time": start_time,
      "bev_path": bev_path,
      "ego_path": ego_path,
      "ego_bev_right_df": ego_bev_right_df,
      "opp_bev_right_df": opp_bev_right_df,
      "ego_bev_left_df": ego_bev_left_df,
      "opp_bev_left_df": opp_bev_left_df,
      "tracking_df": tracking_df
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

def stabilise_euler_angles(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
  """
  Stabilise the Euler angles by ensuring that the difference 
  between consecutive angles is less than 180 degrees
  """
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

def get_ego_cam_from_top_marker(bev_to_ego_top_marker_rvec, tvec):
  """
  Get the position of the ego camera in the BEV camera frame
  """
  ego_cam_in_ego_top_marker_frame = np.array([BEV_TO_CAM_EGO_X, BEV_TO_CAM_EGO_Y, BEV_TO_CAM_EGO_Z]) / 1000
  
  # Transforms points from the ego top marker frame to the BEV frame
  # Equivalently, aligns the BEV frame to the ego top marker frame
  bev_to_ego_top_marker_rot = cv2.Rodrigues(bev_to_ego_top_marker_rvec)[0]

  ego_cam_in_ego_top_marker_frame_rot_to_bev = bev_to_ego_top_marker_rot @ ego_cam_in_ego_top_marker_frame
  ego_top_marker_in_bev_frame = tvec

  ego_cam_in_bev_frame = ego_top_marker_in_bev_frame + ego_cam_in_ego_top_marker_frame_rot_to_bev
  return ego_cam_in_bev_frame

def get_opp_back_from_top_marker(bev_to_opp_top_marker_rvec, tvec):
  """
  Get the position of the opponent back marker in the BEV camera frame
  """
  opp_back_in_opp_top_marker_frame = np.array([BEV_TO_BACK_OPP_X, BEV_TO_BACK_OPP_Y, BEV_TO_BACK_OPP_Z]) / 1000
  bev_to_opp_top_marker_rot = cv2.Rodrigues(bev_to_opp_top_marker_rvec)[0]

  opp_back_in_opp_top_marker_frame_rot_to_bev = bev_to_opp_top_marker_rot @ opp_back_in_opp_top_marker_frame
  opp_top_marker_in_bev_frame = tvec

  opp_back_in_bev_frame = opp_top_marker_in_bev_frame + opp_back_in_opp_top_marker_frame_rot_to_bev
  return opp_back_in_bev_frame

def rotate_bev_frame_to_ego_cam_frame(bev_to_ego_top_marker_rvec, tvec):
  """
  Rotate the pose vector in the BEV frame to the ego camera frame
  """
  roll, _, yaw = conv.get_euler_from_quaternion(*conv.get_quaternion_from_rodrigues(bev_to_ego_top_marker_rvec), degrees=True)

  # Sometimes the marker may be detected as tilted in the pitch axis. 
  # Here we assume that the camera is fixed with respect to the BEV camera 
  # (might have to tune this to get better matches since camera is sensitive to movement
  # TODO: make bev_to_ego_cam_rot a fixed global variable
  ego_top_marker_to_bev_rot = cv2.Rodrigues(conv.get_rodrigues_from_euler(roll, -2, yaw, degrees=True))[0].T

  # Alternatively, this assumes that the camera and the top marker are aligned by 90 degree rotations
  # ego_top_marker_to_bev_rot = cv2.Rodrigues(bev_to_ego_top_marker_rvec)[0].T
  
  return EGO_TOP_MARKER_TO_EGO_CAM_ROT.T @ ego_top_marker_to_bev_rot @ tvec

def get_ground_truth_rel_pose_ego_cam_frame(bev_to_ego_top_marker_rvec, ego_top_marker_in_bev_frame, bev_to_opp_top_marker_rvec, opp_top_marker_in_bev_frame):
  ego_cam_in_bev_frame = get_ego_cam_from_top_marker(bev_to_ego_top_marker_rvec, ego_top_marker_in_bev_frame)
  # Get position of opponent back marker in BEV frame
  opp_back_in_bev_frame = get_opp_back_from_top_marker(bev_to_opp_top_marker_rvec, opp_top_marker_in_bev_frame)

  # Get the relative position  of the opponent back marker in the BEV camera frame
  rel_pos_bev_frame = opp_back_in_bev_frame - ego_cam_in_bev_frame
  # Convert the relative position to the ego camera frame
  rel_pos_ego_cam_frame = rotate_bev_frame_to_ego_cam_frame(bev_to_ego_top_marker_rvec, rel_pos_bev_frame)

  # Get the relative orientation of the opponent back marker in the ego camera frame
  # TODO: make bev_to_ego_cam_rot a fixed global variable
  bev_to_ego_cam_rot = cv2.Rodrigues(bev_to_ego_top_marker_rvec)[0] @ EGO_TOP_MARKER_TO_EGO_CAM_ROT
  bev_to_opp_back_rot = cv2.Rodrigues(bev_to_opp_top_marker_rvec)[0] @ OPP_TOP_MARKER_TO_OPP_BACK_MARKER_ROT
  ego_cam_to_opp_back_euler = conv.get_euler_from_rotation_matrix(bev_to_ego_cam_rot.T @ bev_to_opp_back_rot, degrees=True)
  
  return rel_pos_ego_cam_frame, ego_cam_to_opp_back_euler

def remove_nan_rows(run_data: list[dict]):
  for run in run_data:
    combined_right_df = pd.concat([run["ego_bev_right_df"], run["opp_bev_right_df"]], axis=1, keys=['ego', 'opp'])
    combined_left_df = pd.concat([run["ego_bev_left_df"], run["opp_bev_left_df"]], axis=1, keys=['ego', 'opp'])

    # Drop rows with NaN values in any of the columns
    combined_right_cleaned = combined_right_df.dropna().reset_index(drop=True)
    combined_left_cleaned = combined_left_df.dropna().reset_index(drop=True)

    run["ego_bev_right_df"] = combined_right_cleaned['ego']
    run["opp_bev_right_df"] = combined_right_cleaned['opp']

    run["ego_bev_left_df"] = combined_left_cleaned['ego']
    run["opp_bev_left_df"] = combined_left_cleaned['opp']

    last_bev_pose_time = max(
      min(run["ego_bev_right_df"]["time (sec)"].max(), 
          run["opp_bev_right_df"]["time (sec)"].max()),
      min(run["ego_bev_left_df"]["time (sec)"].max(),
          run["opp_bev_left_df"]["time (sec)"].max())
    )
    
    # Trim the tracking data to the last BEV pose time
    run["tracking_df"] = run["tracking_df"][run["tracking_df"]["time (sec)"] <= last_bev_pose_time]
    run["tracking_df"] = run["tracking_df"].dropna().reset_index(drop=True)

def process_run_data(run_data: list[dict]):
  for run in run_data:
    remove_nan_rows(run_data)

    run["tracking_df"] = remove_unused_frames(run["tracking_df"], run["start_time"])
    run["tracking_df"] = generate_euler_angles(run["tracking_df"])

    for df_name in ["ego_bev_right_df", "opp_bev_right_df", "ego_bev_left_df", "opp_bev_left_df"]:
      run[df_name] = remove_unused_frames(run[df_name], run["start_time"])
      run[df_name] = generate_euler_angles(run[df_name])

      run[f"{df_name} (savgol)"] = generate_savgol_poses_data(run[df_name])
      run[f"{df_name} (rolling_mean)"] = generate_rolling_mean_poses_data(run[df_name])

    run["rel_poses_left_df"] = compute_relative_pose(run["ego_bev_left_df"], run["opp_bev_left_df"])
    run["rel_poses_right_df"] = compute_relative_pose(run["ego_bev_right_df"], run["opp_bev_right_df"])

    # Make the angle difference between consecutive frames less than 180 degrees
    run["rel_poses_left_df"] = stabilise_euler_angles(run["rel_poses_left_df"], ["rel_roll", "rel_pitch", "rel_yaw"])
    run["rel_poses_right_df"] = stabilise_euler_angles(run["rel_poses_right_df"], ["rel_roll", "rel_pitch", "rel_yaw"])

  return run_data

def apply_rolling_mean(df: pd.DataFrame, src: str, dest: str | None = None, window=20, min_periods=1, center=True):
  if dest is None:
    dest = src
  df.loc[:, dest] = df[src].rolling(window=window, min_periods=min_periods, center=center).mean()

def apply_savgol_filter(df: pd.DataFrame, src: str, dest: str | None = None, window=20, polyorder=5):
  if dest is None:
    dest = src
  df.loc[:, dest] = savgol_filter(df[src], window, polyorder)

def generate_savgol_poses_data(poses_df: pd.DataFrame, window=20, polyorder=5, inplace=False):
  """
  Generate smoothed poses data using Savitzky-Golay filter
  """
  if not inplace:
    poses_df = poses_df.copy()

  pos_columns = ['tx', 'ty', 'tz']
  rot_columns = ['roll', 'pitch', 'yaw']
  for col in pos_columns + rot_columns:
    apply_savgol_filter(poses_df, col, window=window, polyorder=polyorder)

  poses_df = stabilise_euler_angles(poses_df, rot_columns)

  poses_df["ax"], poses_df["ay"], poses_df["az"] = \
    zip(*poses_df.apply(lambda row: conv.get_rodrigues_from_euler(
      *row[["roll", "pitch", "yaw"]], degrees=True), axis=1))
  poses_df["qx"], poses_df["qy"], poses_df["qz"], poses_df["qw"] = \
    zip(*poses_df.apply(lambda row: conv.get_quaternion_from_euler(
      *row[["roll", "pitch", "yaw"]], degrees=True), axis=1))

  return poses_df

def generate_rolling_mean_poses_data(poses_df: pd.DataFrame, window=20, inplace=False):
  """
  Generate smoothed poses data using rolling mean
  """
  if not inplace:
    poses_df = poses_df.copy()
    
  pos_columns = ['tx', 'ty', 'tz']
  rot_columns = ['roll', 'pitch', 'yaw']
  for col in pos_columns + rot_columns:
    apply_rolling_mean(poses_df, col, window=window)

  poses_df = stabilise_euler_angles(poses_df, rot_columns)

  poses_df["ax"], poses_df["ay"], poses_df["az"] = \
    zip(*poses_df.apply(lambda row: conv.get_rodrigues_from_euler(
      *row[["roll", "pitch", "yaw"]], degrees=True), axis=1))
  poses_df["qx"], poses_df["qy"], poses_df["qz"], poses_df["qw"] = \
    zip(*poses_df.apply(lambda row: conv.get_quaternion_from_euler(
      *row[["roll", "pitch", "yaw"]], degrees=True), axis=1))

  return poses_df

def compute_relative_pose(ego_bev_df: pd.DataFrame, opp_bev_df: pd.DataFrame):
  rel_poses = pd.DataFrame(columns=["time (sec)", "time_norm (sec)", "rel_x", "rel_y", "rel_z", "rel_roll", "rel_pitch", "rel_yaw"])

  for i in range(len(ego_bev_df)):
    ego_top_marker_to_bev_rvec = ego_bev_df.iloc[i][["ax", "ay", "az"]].to_numpy()
    ego_top_marker_in_bev_frame = ego_bev_df.iloc[i][["tx", "ty", "tz"]].to_numpy()

    opp_top_marker_to_bev_rvec = opp_bev_df.iloc[i][["ax", "ay", "az"]].to_numpy()
    opp_top_marker_in_bev_frame = opp_bev_df.iloc[i][["tx", "ty", "tz"]].to_numpy()

    ground_truth_rel_pose_ego_cam_frame, ground_truth_ego_cam_to_opp_back_euler = get_ground_truth_rel_pose_ego_cam_frame(
      ego_top_marker_to_bev_rvec, ego_top_marker_in_bev_frame,
      opp_top_marker_to_bev_rvec, opp_top_marker_in_bev_frame)
    
    rel_poses = pd.concat([rel_poses, pd.DataFrame([{
      "time (sec)": ego_bev_df.loc[i, "time (sec)"],
      "time_norm (sec)": ego_bev_df.loc[i, "time_norm (sec)"],
      "rel_x": ground_truth_rel_pose_ego_cam_frame[0],
      "rel_y": ground_truth_rel_pose_ego_cam_frame[1],
      "rel_z": ground_truth_rel_pose_ego_cam_frame[2],
      "rel_roll": ground_truth_ego_cam_to_opp_back_euler[0],
      "rel_pitch": ground_truth_ego_cam_to_opp_back_euler[1],
      "rel_yaw": ground_truth_ego_cam_to_opp_back_euler[2]
    }])])

  return rel_poses.reset_index(drop=True)

def fit_polynomial(x: np.ndarray, y: np.ndarray, degree=1):
  # https://ostwalprasad.github.io/machine-learning/Polynomial-Regression-using-statsmodel.html
  polynomial_features= PolynomialFeatures(degree=degree)
  x = x.reshape(-1, 1)
  xp = polynomial_features.fit_transform(x)

  model = sm.OLS(y, xp).fit()
  model_derivative = np.polyder(model.params[::-1])
  model_2nd_derivative = np.polyder(model_derivative)

  return model, model_derivative, model_2nd_derivative