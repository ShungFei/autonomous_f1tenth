import argparse
import cv2
import os
import numpy as np
import pandas as pd

from perception.util.aruco import locate_aruco_corners, locate_aruco_poses
from perception.util.conversion import get_quaternion_from_rodrigues
import perception.util.conversion as conv

class BEVProcessor():
  """
  This class processes the dumped BEV images to localize the ego and opponent cars
  """
  def __init__(self, process_dir, side_length=0.15, ego_aruco_id=0, opp_aruco_id=1, reproj_error_threshold=1.0):
    self.process_dir = process_dir
    self.side_length = side_length

    self.ego_aruco_id = ego_aruco_id
    self.opp_aruco_id = opp_aruco_id

    self.detector_params = cv2.aruco.DetectorParameters()
    self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    self.term_criteria = (cv2.TERM_CRITERIA_EPS, 30, 0.01)

    # Define the half side length
    half_side_length = self.side_length / 2

    # Define the 4 corners of the ArUco marker
    self.marker_obj_points = np.array([[
        [-half_side_length, half_side_length, 0],
        [half_side_length, half_side_length, 0],
        [half_side_length, -half_side_length, 0],
        [-half_side_length, -half_side_length, 0]
    ]], dtype=np.float32)

    self.reproj_error_threshold = reproj_error_threshold

    self.measurements = {}

  def process_stereo(self):
    for process_sub_dir, dirs, files in os.walk(self.process_dir):
      print('Processing:', process_sub_dir)
      if "right" not in dirs or "left" not in dirs:
        print(f"Skipping {process_sub_dir} as right or left directory is missing")
        continue

      right_image_files = set([image_file for image_file in os.listdir(f"{process_sub_dir}/right") if \
                      image_file.endswith(".png") or image_file.endswith(".jpg") or image_file.endswith(".jpeg")])
      left_image_files = set([image_file for image_file in os.listdir(f"{process_sub_dir}/left") if \
                      image_file.endswith(".png") or image_file.endswith(".jpg") or image_file.endswith(".jpeg")])
      
      if not os.path.exists(f"{process_sub_dir}/right/intrinsics.txt") or not os.path.exists(f"{process_sub_dir}/right/dist_coeffs.txt") \
          or not os.path.exists(f"{process_sub_dir}/left/intrinsics.txt") or not os.path.exists(f"{process_sub_dir}/left/dist_coeffs.txt"):
        print(f"Skipping {process_sub_dir} as intrinsics.txt or dist_coeffs.txt is missing")
        continue

      right_intrinsics = np.loadtxt(f"{process_sub_dir}/right/intrinsics.txt")
      right_dist_coeffs = np.loadtxt(f"{process_sub_dir}/right/dist_coeffs.txt")

      left_intrinsics = np.loadtxt(f"{process_sub_dir}/left/intrinsics.txt")
      left_dist_coeffs = np.loadtxt(f"{process_sub_dir}/left/dist_coeffs.txt")

      extrinsics = np.loadtxt(f"{process_sub_dir}/extrinsics.txt")[:-1]
      # convert the translation vector to metres
      extrinsics[:, 3] /= 1000

      ego_poses = []
      opp_poses = []

      for image_file in sorted(right_image_files.intersection(left_image_files)):
        # check if the image ends with png or jpg or jpeg
        if (image_file.endswith(".png") or image_file.endswith(".jpg") or image_file.endswith(".jpeg")):
          # Load the images
          print("\n" + image_file)

          left_image = cv2.imread(f"{process_sub_dir}/left/{image_file}")
          right_image = cv2.imread(f"{process_sub_dir}/right/{image_file}")

          left_new_intrinsics, _ = cv2.getOptimalNewCameraMatrix(left_intrinsics, left_dist_coeffs, left_image.shape[:2][::-1], alpha=1)
          right_new_intrinsics, _ = cv2.getOptimalNewCameraMatrix(right_intrinsics, right_dist_coeffs, right_image.shape[:2][::-1], alpha=1)
          
          undistorted_left_image = cv2.undistort(left_image, left_intrinsics, left_dist_coeffs, None, newCameraMatrix=None)
          undistorted_right_image = cv2.undistort(right_image, right_intrinsics, right_dist_coeffs, None, newCameraMatrix=None)
          
          undistorted_left_gray_image = cv2.cvtColor(undistorted_left_image, cv2.COLOR_BGR2GRAY)
          undistorted_right_gray_image = cv2.cvtColor(undistorted_right_image, cv2.COLOR_BGR2GRAY)

          left_arucos = locate_aruco_corners(undistorted_left_image, self.aruco_dictionary)
          right_arucos = locate_aruco_corners(undistorted_right_image, self.aruco_dictionary)
          print('right arucos', right_arucos)
          # apply subpixel refinement to the detected corners
          for aruco_id in left_arucos:
            left_arucos[aruco_id] = cv2.cornerSubPix(undistorted_left_gray_image, left_arucos[aruco_id], (1, 1), (-1, -1), self.term_criteria)
          for aruco_id in right_arucos:
            right_arucos[aruco_id] = cv2.cornerSubPix(undistorted_right_gray_image, right_arucos[aruco_id], (1, 1), (-1, -1), self.term_criteria)
          # invert extrinsics
          rot_inv = extrinsics[:, :3].T
          extrinsics_inv = np.concatenate([rot_inv, rot_inv @ -extrinsics[:, 3].reshape(3,-1)], axis = -1)
          print('right_arucos', right_arucos)
          projection_left = left_intrinsics @ extrinsics_inv # np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
          projection_right = right_intrinsics @ np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1) # extrinsics

          os.makedirs(f"test", exist_ok=True)
          # draw the detected corners
          left_image = cv2.aruco.drawDetectedMarkers(undistorted_left_image, tuple(left_arucos.values()), np.array(list(right_arucos.keys())).reshape(-1, 1))
          right_image = cv2.aruco.drawDetectedMarkers(undistorted_right_image, tuple(right_arucos.values()), np.array(list(right_arucos.keys())).reshape(-1, 1))

          cv2.imwrite(f"test/{image_file}_left.png", left_image)
          cv2.imwrite(f"test/{image_file}_right.png", right_image)

          for aruco_id in left_arucos:
            if aruco_id not in right_arucos:
              continue
            print("id", aruco_id)
            left_marker = left_arucos[aruco_id]
            right_marker = right_arucos[aruco_id]

            # points_4d = cv2.triangulatePoints(projection_left, projection_right, left_marker[0].T, right_marker[0].T)
            points_4d = cv2.triangulatePoints(projection_right, projection_left, right_marker[0].T, left_marker[0].T)
            points_3d = cv2.convertPointsFromHomogeneous(points_4d.T).reshape(-1, 3)

            print('stereo points', points_3d)

            # _, rvec, tvec = cv2.solvePnP(self.marker_obj_points, right_marker[0], right_intrinsics, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            _, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(self.marker_obj_points, right_marker[0], 
                  right_intrinsics, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            # solve pnp for left camera
            _, rvecs_left, tvecs_left, reproj_errors_left = cv2.solvePnPGeneric(self.marker_obj_points, left_marker[0], 
                  left_intrinsics, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)

            # _, rvec_left, tvec_left = cv2.solvePnP(self.marker_obj_points, left_marker[0], left_intrinsics, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)

            # print('right pnp', [extrinsics @ np.vstack([tvec, 1]) for tvec in tvecs])
            print('right pnp', tvecs)
            # print(extrinsics[:, :3],extrinsics[:, 3], extrinsics[:, 3].reshape(-1,3), (tvecs_left[0] - extrinsics[:, 3]) )

            # transform the points to the right camera coordinate system
            # print('left pnp', tvecs_left)
            print('left pnp', [(extrinsics[:, :3].T @ (tvec_left - extrinsics[:, 3].reshape(3,-1))) for tvec_left in tvecs_left])

  def process(self):
    for process_sub_dir, _, _ in os.walk(self.process_dir):
      print('Processing:', process_sub_dir)
      image_files = [image_file for image_file in os.listdir(f"{process_sub_dir}") if \
                     image_file.endswith(".png") or image_file.endswith(".jpg") or image_file.endswith(".jpeg")]
      
      if len(image_files) == 0:
        print(f"Skipping {process_sub_dir} as no images found")
        continue
      if not os.path.exists(f"{process_sub_dir}/intrinsics.txt") or not os.path.exists(f"{process_sub_dir}/dist_coeffs.txt"):
        print(f"Skipping {process_sub_dir} as intrinsics.txt or dist_coeffs.txt is missing")
        continue

      intrinsics = np.loadtxt(f"{process_sub_dir}/intrinsics.txt")
      dist_coeffs = np.loadtxt(f"{process_sub_dir}/dist_coeffs.txt")

      ego_poses = []
      opp_poses = []

      for image_file in sorted(image_files):
        # check if the image ends with png or jpg or jpeg
        if (image_file.endswith(".png") or image_file.endswith(".jpg") or image_file.endswith(".jpeg")):
          # Load the images
          image = cv2.imread(f"{process_sub_dir}/{image_file}")
          aruco_poses = locate_aruco_poses(image, self.aruco_dictionary, self.marker_obj_points, intrinsics, dist_coeffs, output_all=True)
          if self.ego_aruco_id not in aruco_poses:
            ego_poses.append((image_file.strip(".png"), *([None] * 10)))
          else:
            rvecs, tvecs, reproj_errors = aruco_poses[self.ego_aruco_id]
            rvec, tvec, quat, roll, pitch, yaw = self.select_best_pnp_pose(rvecs, tvecs, reproj_errors)

            # print(f"rvec: {rvec}, tvec: {tvec}, quat: {quat}, roll: {roll}, pitch: {pitch}, yaw: {yaw}")

            ego_poses.append((image_file.strip(".png"), *quat, *rvec.flatten().tolist(), *tvec.flatten().tolist()))
          
          if self.opp_aruco_id not in aruco_poses:
            opp_poses.append((image_file.strip(".png"), *([None] * 10)))
          else:
            rvec, tvec, quat, roll, pitch, yaw = self.select_best_pnp_pose(*aruco_poses[self.opp_aruco_id])
            opp_poses.append((image_file.strip(".png"), *quat, *rvec.flatten().tolist(), *tvec.flatten().tolist()))
      
      # Save the poses to csv files
      ego_df = pd.DataFrame(ego_poses, 
                  columns=["time", "qx", "qy", "qz", "qw", "ax", "ay", "az", "tx", "ty", "tz"])
      ego_df.sort_values(by="time", inplace=True)
      ego_df.to_csv(f"{process_sub_dir}/ego_poses.csv", index=False)

      opp_df = pd.DataFrame(opp_poses,
                    columns=["time", "qx", "qy", "qz", "qw", "ax", "ay", "az", "tx", "ty", "tz"])
      opp_df.sort_values(by="time", inplace=True)
      opp_df.to_csv(f"{process_sub_dir}/opp_poses.csv", index=False)

  def select_best_pnp_pose(self, rvecs, tvecs, reproj_errors):
    chosen_rvec, chosen_tvec = rvecs[0], tvecs[0]
    chosen_quat = get_quaternion_from_rodrigues(chosen_rvec)
    chosen_roll, chosen_pitch, chosen_yaw = conv.get_euler_from_quaternion(*chosen_quat, degrees=True)
    chosen_angle_match = np.sqrt((180 - abs(chosen_roll)) ** 2 + abs(chosen_pitch) ** 2)

    for i in range(1, len(rvecs)):
      rvec, tvec, reproj_error = rvecs[i], tvecs[i], reproj_errors[i]
      quaternion = get_quaternion_from_rodrigues(rvec)
      roll, pitch, yaw = conv.get_euler_from_quaternion(*quaternion, degrees=True)

      # we expect the rotation in the y-axis to be closer to 0 and x-axis to be 180, so we choose the solution on this basis
      angle_match = np.sqrt((180 - abs(roll)) ** 2 + abs(pitch) ** 2)
      # print('test', abs(roll % 360 - 180), roll, pitch, yaw)
      # print('test2', abs(pitch))

      if angle_match < chosen_angle_match and reproj_error < self.reproj_error_threshold:
        chosen_rvec, chosen_tvec = rvec, tvec
        chosen_quat = quaternion
        chosen_roll, chosen_pitch, chosen_yaw = roll, pitch, yaw
        chosen_angle_match = angle_match

    return chosen_rvec, chosen_tvec, chosen_quat, chosen_roll, chosen_pitch, chosen_yaw
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("--run_dir", type=str, help="Directory containing the data for a single run")
  group.add_argument("--latest", action="store_true", help="Process the latest run in the perception_debug directory")
  group.add_argument("--all", action="store_true", help="Process all runs in the perception_debug directory")

  parser.add_argument("--ego_aruco_id", type=int, default=1, help="ID of the ego ArUco marker")
  parser.add_argument("--opp_aruco_id", type=int, default=14, help="ID of the opponent ArUco marker")
  parser.add_argument("--side_length", type=float, default=0.15, help="Side length of the ArUco markers")

  args = parser.parse_args()

  DEBUG_DIR = "perception_debug"
  if (args.all or args.latest) and not os.path.exists(DEBUG_DIR):
      print("To use --all or --latest, the perception_debug directory must exist in the current terminal's working directory")
      exit()
    
  if args.all:
    for run_dir in os.listdir(DEBUG_DIR):
      if os.path.isdir(run_dir):
        bev_dir = os.path.join(DEBUG_DIR, run_dir, "bev")
        node = BEVProcessor(bev_dir, side_length=args.side_length, ego_aruco_id=args.ego_aruco_id, opp_aruco_id=args.opp_aruco_id)
        node.process()
  elif args.latest:
    latest_dir = max([f.path for f in os.scandir(DEBUG_DIR) if f.is_dir()], key=os.path.getmtime)
    bev_dir = os.path.join(latest_dir, "bev")
    node = BEVProcessor(bev_dir, side_length=args.side_length, ego_aruco_id=args.ego_aruco_id, opp_aruco_id=args.opp_aruco_id)
    node.process()
  elif args.run_dir:
    # walk through the run directory
    if not os.path.exists(args.run_dir):
      print(f"Directory {args.run_dir} does not exist")
      exit()
    node = BEVProcessor(args.run_dir, side_length=args.side_length, ego_aruco_id=args.ego_aruco_id, opp_aruco_id=args.opp_aruco_id)
    node.process()