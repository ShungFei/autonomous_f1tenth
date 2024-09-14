import argparse
import cv2
import os
import numpy as np
import pandas as pd

from perception.util.aruco import locate_arucos
from perception.util.conversion import get_quaternion_from_rotation_matrix

class BEVProcessor():
  """
  This class processes the dumped BEV images to localize the ego and opponent cars
  """
  def __init__(self, process_dir, side_length=0.15, ego_aruco_id=0, opp_aruco_id=1):
    self.process_dir = process_dir
    self.side_length = side_length

    self.ego_aruco_id = ego_aruco_id
    self.opp_aruco_id = opp_aruco_id

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

    # NOTE: Localisation currently only done using the right camera
    self.intrinsics = np.loadtxt(f"{self.process_dir}/right/intrinsics.txt") # 
    self.dist_coeffs = np.loadtxt(f"{self.process_dir}/right/dist_coeffs.txt")

    self.measurements = {}
    self.ego_poses = []
    self.opp_poses = []

  def process(self):
    for image_file in sorted(os.listdir(f"{self.process_dir}/right")):
      # check if the image ends with png or jpg or jpeg
      if (image_file.endswith(".png") or image_file.endswith(".jpg") or image_file.endswith(".jpeg")):
        # Load the images
        image = cv2.imread(f"{self.process_dir}/right/{image_file}")
        arucos = locate_arucos(image, self.aruco_dictionary, self.marker_obj_points, self.intrinsics, self.dist_coeffs)

        for id, (rvec, tvec) in arucos.items():
          rot_matrix, _ = cv2.Rodrigues(rvec)
          quaternion = get_quaternion_from_rotation_matrix(rot_matrix)

          if id == self.ego_aruco_id:
            self.ego_poses.append((image_file.strip(".png"), *quaternion, *tvec.flatten().tolist()))

          elif id == self.opp_aruco_id:
            self.opp_poses.append((image_file.strip(".png"), *quaternion, *tvec.flatten().tolist()))
    
    # Save the poses to csv files
    pd.DataFrame(self.ego_poses, 
                 columns=["time", "qx", "qy", "qz", "qw", "tx", "ty", "tz"]).to_csv(f"{self.process_dir}/ego_poses.csv", index=False)
    pd.DataFrame(self.opp_poses,
                  columns=["time", "qx", "qy", "qz", "qw", "tx", "ty", "tz"]).to_csv(f"{self.process_dir}/opp_poses.csv", index=False)

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
    bev_dir = os.path.join(args.run_dir, "bev")
    if not os.path.exists(bev_dir):
      print(f"Directory {bev_dir} does not exist")
      exit()
    
    node = BEVProcessor(bev_dir, side_length=args.side_length, ego_aruco_id=args.ego_aruco_id, opp_aruco_id=args.opp_aruco_id)
    node.process()
  