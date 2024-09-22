import argparse
import os

import cv2
import numpy as np
import pandas as pd
from sympy import N
from perception.util.aruco import locate_aruco_poses
from perception.util.conversion import get_quaternion_from_rodrigues

class TrackingProcessor():
  """
  This class processes the dumped tracking data for the ego to localize the opponent car
  """
  def __init__(self, process_dir, side_length=0.15, opp_back_aruco_id=15):
    self.process_dir = f'{process_dir}'
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

    self.measurements = {}
    self.opp_rel_poses = []

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

      opp_rel_poses = []

      for image_file in image_files:
        if (image_file.endswith(".png") or image_file.endswith(".jpg") or image_file.endswith(".jpeg")):
          # Load the images
          image = cv2.imread(f"{process_sub_dir}/{image_file}")
          arucos = locate_aruco_poses(image, self.aruco_dictionary, self.marker_obj_points, intrinsics, dist_coeffs)
          if self.opp_back_aruco_id not in arucos:
            opp_rel_poses.append((image_file.strip(".png"), *([None] * 10)))
          
          else:
            rvec, tvec = arucos[self.opp_back_aruco_id]
            quaternion = get_quaternion_from_rodrigues(rvec)

            opp_rel_poses.append((image_file.strip(".png"), *quaternion, *rvec.flatten().tolist(), *tvec.flatten().tolist()))

      df = pd.DataFrame(opp_rel_poses,
                  columns=["time", "qx", "qy", "qz", "qw", "ax", "ay", "az", "tx", "ty", "tz"])
      
      df.sort_values(by="time", inplace=True)
      df.to_csv(f"{process_sub_dir}/opp_rel_poses.csv", index=False)

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
    latest_dir = max([f.path for f in os.scandir(DEBUG_DIR) if f.is_dir()], key=os.path.getmtime)
    node = TrackingProcessor(latest_dir, side_length=args.side_length, opp_back_aruco_id=args.opp_back_aruco_id)
    node.process()
  elif args.run_dir:
    if not os.path.exists(args.run_dir):
      print(f"Directory {args.run_dir} does not exist")
      exit()
    node = TrackingProcessor(args.run_dir, side_length=args.side_length, opp_back_aruco_id=args.opp_back_aruco_id)
    node.process()