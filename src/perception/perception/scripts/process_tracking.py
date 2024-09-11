import os
import time
from datetime import datetime
from math import sqrt

import cv2
import numpy as np
import pandas as pd
import rclpy
import seaborn as sns
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from matplotlib import pyplot as plt
from message_filters import (ApproximateTimeSynchronizer, Subscriber,
                             TimeSynchronizer)
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage

import perception.util.ground_truth as GroundTruth
from perception.util.aruco import locate_arucos
from perception.util.conversion import (get_quaternion_from_rotation_matrix,
                                        get_time_from_header)
from perception_interfaces.msg import StateEstimateStamped

class TrackingProcessor():
  """
  This class processes the dumped tracking data for the ego to localize the opponent car
  """
  def __init__(self, process_dir, side_length=0.15, opp_back_aruco_id=1):
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

    self.intrinsics = np.loadtxt(f"{self.process_dir}/intrinsics.txt")
    self.dist_coeffs = np.loadtxt(f"{self.process_dir}/dist_coeffs.txt")

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
            self.opp_rel_poses.append((image_file.strip(".jpg"), *quaternion, *tvec.flatten().tolist()))
            
    pd.DataFrame(self.opp_rel_poses,
                 columns=["time", "qx", "qy", "qz", "qw", "tx", "ty", "tz"]).to_csv(f"{self.process_dir}/opp_rel_poses.csv", index=False)

if __name__ == "__main__":
  rclpy.init()
  
  dir = os.path.join("perception_debug", "24_09_10_17:41:35", "bev")
  opp_back_aruco_id = 12
  node = TrackingProcessor(dir, opp_back_aruco_id=opp_back_aruco_id)
  node.process()

  rclpy.shutdown()