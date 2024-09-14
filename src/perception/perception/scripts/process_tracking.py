import argparse
import os

import cv2
import numpy as np
import pandas as pd
from perception.util.aruco import locate_arucos
from perception.util.conversion import (get_quaternion_from_rotation_matrix)

class TrackingProcessor():
  """
  This class processes the dumped tracking data for the ego to localize the opponent car
  """
  def __init__(self, process_dir, side_length=0.15, opp_back_aruco_id=15):
    self.process_dir = process_dir
    self.side_length = side_length

    self.opp_back_aruco_id = opp_back_aruco_id

    # change this to the folder where the images are stored

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

    self.intrinsics = np.loadtxt(f"{self.process_dir}/color/intrinsics.txt")
    self.dist_coeffs = np.loadtxt(f"{self.process_dir}/color/dist_coeffs.txt")

    self.measurements = {}
    self.opp_rel_poses = []

  def process(self):
    for image_file in os.listdir(self.process_dir):
      # check if the image ends with png or jpg or jpeg
      if (image_file.endswith(".png") or image_file.endswith(".jpg") or image_file.endswith(".jpeg")):
        # Load the images
        image = cv2.imread(f"{self.process_dir}/{image_file}")
        arucos = locate_arucos(image, self.aruco_dictionary, self.marker_obj_points, self.intrinsics, self.dist_coeffs)

        for id, (rvec, tvec) in arucos.items():
          rot_matrix, _ = cv2.Rodrigues(rvec)
          quaternion = get_quaternion_from_rotation_matrix(rot_matrix)

          if id == self.opp_back_aruco_id:
            self.opp_rel_poses.append((image_file.strip(".png"), *quaternion, *tvec.flatten().tolist()))
            
    pd.DataFrame(self.opp_rel_poses,
                 columns=["time", "qx", "qy", "qz", "qw", "tx", "ty", "tz"]).to_csv(f"{self.process_dir}/opp_rel_poses.csv", index=False)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("--run_dir", type=str, help="Directory containing a single run's data")
  group.add_argument("--latest", action="store_true", help="Process the latest run in the perception_debug directory")
  group.add_argument("--all", action="store_true", help="Process all runs in the perception_debug directory")

  parser.add_argument("--opp_back_aruco_id", type=int, default=15, help="ID of the ArUco marker on the back of the opponent car")
  parser.add_argument("--side_length", type=float, default=0.15, help="Side length of the ArUco markers")

  args = parser.parse_args()

  DEBUG_DIR = "perception_debug"
  if (args.all or args.latest) and not os.path.exists(DEBUG_DIR):
      print("To use --all or --latest, the perception_debug directory must exist in the current terminal's working directory")
      exit()
    
  if args.all:
    for run_dir in os.listdir(DEBUG_DIR):
      if os.path.isdir(run_dir):
        node = TrackingProcessor(run_dir, side_length=args.side_length, opp_back_aruco_id=args.opp_back_aruco_id)
        node.process()
  elif args.latest:
    latest_dir = max([f.path for f in os.scandir(DEBUG_DIR) if f.is_dir()], key=os.path.getmtime),
    node = TrackingProcessor(latest_dir, side_length=args.side_length, opp_back_aruco_id=args.opp_back_aruco_id)
    node.process()
  elif args.run_dir:
    if not os.path.exists(args.run_dir):
      print(f"Directory {args.run_dir} does not exist")
      exit()
    node = TrackingProcessor(args.run_dir, side_length=args.side_length, opp_back_aruco_id=args.opp_back_aruco_id)
    node.process()
