import argparse
import cv2
import os
import numpy as np
import pandas as pd

from perception.util.aruco import locate_arucos
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
    # NOTE: Localisation currently only done using the right camera
    # self.intrinsics = np.loadtxt(f"{self.process_dir}/right/intrinsics.txt") # 
    # self.dist_coeffs = np.loadtxt(f"{self.process_dir}/right/dist_coeffs.txt")

    self.measurements = {}

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

      for image_file in image_files:
        # check if the image ends with png or jpg or jpeg
        if (image_file.endswith(".png") or image_file.endswith(".jpg") or image_file.endswith(".jpeg")):
          # Load the images
          image = cv2.imread(f"{process_sub_dir}/{image_file}")
          arucos = locate_arucos(image, self.aruco_dictionary, self.marker_obj_points, intrinsics, dist_coeffs, output_all=True)

          if self.ego_aruco_id not in arucos:
            ego_poses.append((image_file.strip(".png"), *([None] * 10)))
          else:
            rvecs, tvecs, reproj_errors = arucos[self.ego_aruco_id]
            rvec, tvec, quat, roll, pitch, yaw = self.select_best_solution(rvecs, tvecs, reproj_errors)
            # print(f"rvec: {rvec}, tvec: {tvec}, quat: {quat}, roll: {roll}, pitch: {pitch}, yaw: {yaw}")

            ego_poses.append((image_file.strip(".png"), *quat, *rvec.flatten().tolist(), *tvec.flatten().tolist()))
          
          if self.opp_aruco_id not in arucos:
            opp_poses.append((image_file.strip(".png"), *([None] * 10)))
          else:
            rvec, tvec, quat, roll, pitch, yaw = self.select_best_solution(*arucos[self.opp_aruco_id])

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

  def select_best_solution(self, rvecs, tvecs, reproj_errors):
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