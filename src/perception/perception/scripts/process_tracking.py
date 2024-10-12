import argparse
import os
from typing import Literal

import cv2
import numpy as np
import pandas as pd
from sympy import N
from perception.scripts import ground_truth_real
from perception.util.aruco import locate_aruco_poses
from perception.util.conversion import get_quaternion_from_rodrigues
import perception.util.conversion as conv

class TrackingProcessor():
  """
  This class processes the dumped tracking data for the ego to localize the opponent car
  """
  def __init__(self, process_dir, side_length=0.15, opp_back_aruco_id=15, corner_ref_method="subpix"):
    self.process_dir = f'{process_dir}'
    self.side_length = side_length

    self.opp_back_aruco_id = opp_back_aruco_id

    if corner_ref_method == "none":
      self.corner_ref_method = cv2.aruco.CORNER_REFINE_NONE
    elif corner_ref_method == "subpix":
      self.corner_ref_method = cv2.aruco.CORNER_REFINE_SUBPIX
    elif corner_ref_method == "apriltag":
      self.corner_ref_method = cv2.aruco.CORNER_REFINE_APRILTAG

    self.detector_params = cv2.aruco.DetectorParameters()
    self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

    # Define the half side length
    half_side_length = self.side_length / 2

    # Define the 4 corners of the ArUco marker
    self.marker_obj_points = np.array([[
        [-half_side_length, half_side_length, 0],
        [half_side_length, half_side_length, 0],
        [half_side_length, -half_side_length, 0],
        [-half_side_length, -half_side_length, 0]
    ]], dtype=np.float32)

    self.measurements = {}
    self.opp_rel_poses = []

  def process(self):
    for process_sub_dir, _, _ in os.walk(self.process_dir):
      print('Processing:', process_sub_dir)

      timestamps = [int(image_file.strip(".png")) for image_file in sorted(os.listdir(f"{process_sub_dir}")) if \
                    image_file.endswith(".png") or image_file.endswith(".jpg") or image_file.endswith(".jpeg")]
      
      if len(timestamps) == 0:
        print(f"Skipping {process_sub_dir} as no images found")
        continue
      if not os.path.exists(f"{process_sub_dir}/intrinsics.txt") or not os.path.exists(f"{process_sub_dir}/dist_coeffs.txt"):
        print(f"Skipping {process_sub_dir} as intrinsics.txt or dist_coeffs.txt is missing")
        continue
      
      intrinsics = np.loadtxt(f"{process_sub_dir}/intrinsics.txt")
      depth_intrinsics = np.loadtxt(f"{process_sub_dir}/../depth/intrinsics.txt")
      dist_coeffs = np.loadtxt(f"{process_sub_dir}/dist_coeffs.txt")

      # Make sure time index is in integer format
      df = pd.DataFrame(columns=["qx", "qy", "qz", "qw", "ax", "ay", "az", "tx", "ty", "tz", "depth_tx", "depth_ty", "depth_tz"], index=timestamps)
      df.index.name = "time"
      df.index = df.index.astype(int)

      previous_euler = None
      for time in df.index:
        # Load the images
        image = cv2.imread(f"{process_sub_dir}/{time}.png")
        aruco_poses, corners = locate_aruco_poses(image, self.aruco_dictionary, self.marker_obj_points, intrinsics, 
          dist_coeffs, output_all=False, return_corners=True, corner_ref_method=self.corner_ref_method, pnp_method=cv2.SOLVEPNP_SQPNP)
        
        if self.opp_back_aruco_id not in aruco_poses:
          df.loc[time] = [None] * 13
        else:
          # rvecs, tvecs, reproj_errors = aruco_poses[self.opp_back_aruco_id]
          rvec, tvec = aruco_poses[self.opp_back_aruco_id]
          # rvec, tvec, quaternion, euler = self.get_pose_continuity_constraint(rvecs, tvecs, reproj_errors, None)
          quaternion = get_quaternion_from_rodrigues(rvec)
          euler = conv.get_euler_from_quaternion(*quaternion, degrees=True)

          # Get the image coordinates of the center of the ArUco marker
          center_x = sum([corner[0] for corner in corners[self.opp_back_aruco_id][0]]) / 4
          center_y = sum([corner[1] for corner in corners[self.opp_back_aruco_id][0]]) / 4

          # Get the relative position of the center of the ArUco marker from the depth feed
          depth_relative_position = self.get_relative_position_from_depth_feed(center_x, center_y, time, depth_intrinsics, process_sub_dir)

          df.loc[time] = [*quaternion, *rvec.flatten(), *tvec.flatten(), *depth_relative_position]

          previous_euler = euler

      df.to_csv(f"{process_sub_dir}/opp_rel_poses.csv", index=True)
  
  def get_pose_continuity_constraint(self, rvecs, tvecs, reproj_errors, previous_euler=None)-> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if previous_euler is None:
      # get the pose that minimises the reprojection error
      min_error_idx = np.argmin(reproj_errors)
      rvec, tvec = rvecs[min_error_idx], tvecs[min_error_idx]
      quaternion = get_quaternion_from_rodrigues(rvec)
      euler = conv.get_euler_from_quaternion(*quaternion, degrees=True)
      return rvec, tvec, quaternion, euler
    
    chosen_rvec, chosen_tvec = rvecs[0], tvecs[0]
    chosen_quat = get_quaternion_from_rodrigues(chosen_rvec)
    chosen_euler = np.array(conv.get_euler_from_quaternion(*chosen_quat, degrees=True))
    chosen_euler = np.array([
      ground_truth_real.find_closest_equiv_angle(chosen_angle, previous_angle) \
      for chosen_angle, previous_angle in zip(chosen_euler, previous_euler)
    ])
    chosen_angle_dist = np.linalg.norm(np.array(chosen_euler) - np.array(previous_euler))

    for i in range(1, len(rvecs)):
      rvec, tvec = rvecs[i], tvecs[i]
      quaternion = np.array(get_quaternion_from_rodrigues(rvec))
      euler = conv.get_euler_from_quaternion(*quaternion, degrees=True)
      euler = np.array([
        ground_truth_real.find_closest_equiv_angle(chosen_angle, previous_angle) \
        for chosen_angle, previous_angle in zip(euler, previous_euler)
      ])
      dist = np.linalg.norm(np.array(euler) - np.array(previous_euler))

      if dist < chosen_angle_dist:
        chosen_rvec, chosen_tvec = rvec, tvec
        chosen_quat = quaternion
        chosen_euler = euler
        chosen_angle_dist = dist
    
    return chosen_rvec, chosen_tvec, chosen_quat, chosen_euler
  
  def get_relative_position_from_depth_feed(self, center_x, center_y, frame_timestamp, depth_intrinsics, process_sub_dir):
    # Use manually derived homography matrix from depth_feed.ipynb to align the color image with the depth image
    H = np.array([[ 4.3093820234694441e-01, -1.0743690101400583e-02, 1.9988026083233976e+02],
                  [-3.4476255716975181e-03,  4.4871665720989579e-01, 1.0526304884013102e+02],
                  [-6.2265656596329512e-06, -4.0521031852012964e-05, 9.8742302457834497e-01]])
    
    # Extrinsics taken from https://github.com/IntelRealSense/realsense-ros#extrinsics-from-sensor-a-to-sensor-b
    extrinsics_r = np.array([[ 0.9999583959579468,     0.008895332925021648, -0.0020127370953559875],
                             [-0.008895229548215866,   0.9999604225158691,    6.045500049367547e-05],
                             [ 0.0020131953060626984, -4.254872692399658e-05, 0.9999979734420776]])
    extrinsics_t = np.array([0.01485931035131216, 0.0010161789832636714, 0.0005317096947692335])
    
    depth_file = f"{frame_timestamp}.npy"

    depth_path = os.path.join(os.path.dirname(process_sub_dir), "depth", depth_file)
    if not os.path.exists(depth_path):
      return None
    depth_image = np.load(depth_path)

    # Get the position of the center of the marker in the depth image using the homography matrix
    depth_x = (H[0, 0] * center_x + H[0, 1] * center_y + H[0, 2]) / (H[2, 0] * center_x + H[2, 1] * center_y + H[2, 2])
    depth_y = (H[1, 0] * center_x + H[1, 1] * center_y + H[1, 2]) / (H[2, 0] * center_x + H[2, 1] * center_y + H[2, 2])

    # Get depth value at the center of the marker
    depth = depth_image[int(depth_y), int(depth_x)] / 1000  # Convert mm to m

    # Get the 3D coordinates of the center of the marker in the color camera frame
    relative_position = self.get_3d_coordinates((depth_x, depth_y), depth, depth_intrinsics, extrinsics_r, extrinsics_t)

    return relative_position

  def get_3d_coordinates(self, pixel, depth, intrinsics, extrinsics_r, extrinsics_t):
    u, v = pixel

    if depth == 0:
        return [None] * 3

    fx_depth = intrinsics[0, 0]
    fy_depth = intrinsics[1, 1]
    cx_depth = intrinsics[0, 2]
    cy_depth = intrinsics[1, 2]

    # Calculate 3D coordinates in the depth camera frame
    X_depth = (u - cx_depth) * depth / fx_depth
    Y_depth = (v - cy_depth) * depth / fy_depth

    # 3D point in the depth camera's frame
    P_depth = np.array([X_depth, Y_depth, depth])
    # Transform point from depth camera frame to color camera frame
    P_color = extrinsics_r @ P_depth + extrinsics_t

    return P_color[0], P_color[1], P_color[2]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("--run_dir", type=str, help="Directory containing a single run's data")
  group.add_argument("--latest", action="store_true", help="Process the latest run in the perception_debug directory")
  group.add_argument("--all", action="store_true", help="Process all runs in the perception_debug directory")

  parser.add_argument("--opp_back_aruco_id", type=int, default=15, help="ID of the ArUco marker on the back of the opponent car")
  parser.add_argument("--side_length", type=float, default=0.15, help="Side length of the ArUco markers")
  parser.add_argument("--corner_ref_method", type=str, choices=['none', 'subpix', 'apriltag'], required=True, help="Corner refinement method")

  args = parser.parse_args()

  DEBUG_DIR = "perception_debug"
  if (args.all or args.latest) and not os.path.exists(DEBUG_DIR):
      print("To use --all or --latest, the perception_debug directory must exist in the current terminal's working directory")
      exit()
    
  if args.all:
    for run_dir in os.listdir(DEBUG_DIR):
      if os.path.isdir(run_dir):
        node = TrackingProcessor(run_dir, side_length=args.side_length,
          opp_back_aruco_id=args.opp_back_aruco_id, corner_ref_method=args.corner_ref_method)
        node.process()
  elif args.latest:
    latest_dir = max([f.path for f in os.scandir(DEBUG_DIR) if f.is_dir()], key=os.path.getmtime)
    node = TrackingProcessor(latest_dir, side_length=args.side_length,
      opp_back_aruco_id=args.opp_back_aruco_id, corner_ref_method=args.corner_ref_method)
    node.process()
  elif args.run_dir:
    if not os.path.exists(args.run_dir):
      print(f"Directory {args.run_dir} does not exist")
      exit()
    node = TrackingProcessor(args.run_dir, side_length=args.side_length,
      opp_back_aruco_id=args.opp_back_aruco_id, corner_ref_method=args.corner_ref_method)
    node.process()