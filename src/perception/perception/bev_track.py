import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo 
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Vector3Stamped, PoseStamped
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from collections import deque

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from datetime import datetime
import cv2
import os
import numpy as np
from math import sqrt
import pyzed.sl as sl

import perception.util.ground_truth as GroundTruth
from perception.util.conversion import get_time_from_header, get_time_from_rosclock, get_quaternion_from_rotation_matrix
from perception.util.aruco import locate_aruco_poses

class BevTracker(Node):
  """
  Create an ImageSubscriber class, which is a subclass of the Node class.
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    super().__init__('bev_tracker')

    curr_time = datetime.now().strftime('%y_%m_%d_%H:%M:%S')
    fallback_debug_dir = f"perception_debug/{curr_time}"

    self.DEBUG_DIR = self.declare_parameter('debug_dir', fallback_debug_dir).get_parameter_value().string_value

    os.makedirs(f"{self.DEBUG_DIR}/bev", exist_ok=True)

    self.camera_name = self.declare_parameter('camera_name', "zed").get_parameter_value().string_value
    self.node_name = self.declare_parameter('node_name', "bev").get_parameter_value().string_value
    self.debug = self.declare_parameter('debug', False).get_parameter_value().bool_value
    self.fps = self.declare_parameter('fps', 60).get_parameter_value().integer_value
    self.exposure = self.declare_parameter('exposure', 30).get_parameter_value().integer_value
    self.gain = self.declare_parameter('gain', 55).get_parameter_value().integer_value
    self.side_length = self.declare_parameter('side_length', 0.15).get_parameter_value().double_value

    self.right_camera_info_sub = self.create_subscription(
      CameraInfo, 
      f'{self.camera_name}/{self.node_name}/right/camera_info', 
      self.camera_info_callback, 
      10
    )

    self.right_image_sub = self.create_subscription(
      Image,
      f'{self.camera_name}/{self.node_name}/right/image_rect_color',
      self.bev_track_callback,
      10
    )

    self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

    self.measurements = {}

    # Define the half side length
    half_side_length = self.side_length / 2

    # Define the 4 corners of the ArUco marker
    self.marker_obj_points = np.array([[
        [-half_side_length, half_side_length, 0],
        [half_side_length, half_side_length, 0],
        [half_side_length, -half_side_length, 0],
        [-half_side_length, -half_side_length, 0]
    ]], dtype=np.float32)

    # Used to convert between ROS and OpenCV images
    self.bridge = CvBridge()

    self.zed = sl.Camera()
    
    self.runtime_parameters = sl.RuntimeParameters()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720 # Use HD720 opr HD1200 video mode, depending on camera type.
    init_params.camera_fps = self.fps

    self.prev_time_capture = 0
    self.prev_time_process = 0

    # Open the camera
    err = self.zed.open(init_params)
    self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, self.exposure)
    self.zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, self.gain)

    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : "+repr(err)+". Exit program.")
        exit()

    self.camera_info = self.zed.get_camera_information()
    camera_info_raw_sl = self.camera_info.camera_configuration.calibration_parameters_raw
    right_camera_info_sl = camera_info_raw_sl.right_cam
    left_camera_info_sl = camera_info_raw_sl.left_cam
    extrinsics_sl = camera_info_raw_sl.stereo_transform

    self.right_intrinsics = np.array([
        [right_camera_info_sl.fx, 0, right_camera_info_sl.cx],
        [0, right_camera_info_sl.fy, right_camera_info_sl.cy],
        [0, 0, 1]
    ])
    self.right_dist_coeffs = np.array(right_camera_info_sl.disto)

    self.left_intrinsics = np.array([
        [left_camera_info_sl.fx, 0, left_camera_info_sl.cx],
        [0, left_camera_info_sl.fy, left_camera_info_sl.cy],
        [0, 0, 1]
    ])
    self.left_dist_coeffs = np.array(left_camera_info_sl.disto)
    self.extrinsics = extrinsics_sl.m

    self.reentrant_callback_group = ReentrantCallbackGroup()
    self.previous_capture_time = 0
    self.image_queue = deque()
    
    self.capture_timer = self.create_timer(1e-9, self.capture_callback_scuffed)

    # self.capture_timer = self.create_timer(1 / (self.fps), self.capture_callback)
    
    # self.process_image_timer = self.create_timer(1 / (self.fps - 10), self.process_image_callback, 
    #                                              callback_group=self.reentrant_callback_group)
  def capture_callback_scuffed(self):
    """
    Since the ZED Python API does not like us grabbing the frames in a ROS2 callback, we create a single time
    callback to execute a while loop
    """
    self.capture_timer.destroy()

    image_right = sl.Mat()
    image_left = sl.Mat()
    while True:
      # Grab an image, a RuntimeParameters object must be given to grab()
      if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
          # A new image is available if grab() returns SUCCESS
          self.zed.retrieve_image(image_right, sl.VIEW.RIGHT_UNRECTIFIED)
          self.zed.retrieve_image(image_left, sl.VIEW.LEFT_UNRECTIFIED)

          timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # Get the timestamp at the time the image was captured
          current_time = timestamp.get_nanoseconds()
          if current_time - self.previous_capture_time > 2e7:
            print(f"Current time: {current_time}, Time between two frames: {(current_time - self.previous_capture_time) / 1e9}")
          self.previous_capture_time = current_time
          self.image_queue.append((current_time, self.previous_capture_time, 
                                   image_left.numpy(force=True), image_right.numpy(force=True)))
          # cv2.imwrite(f"{self.DEBUG_DIR}/bev/{timestamp.get_milliseconds()}.png", np.copy(image.numpy()))
  
  def capture_callback(self):
    """
    Capture images from the Zed camera and place them in a queue (Zed SDK version)
    """
    image = sl.Mat()
    if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
      # A new image is available if grab() returns SUCCESS
      self.zed.retrieve_image(image, sl.VIEW.RIGHT_UNRECTIFIED)
      timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # Get the timestamp at the time the image was captured
      
      curr_time = timestamp.get_milliseconds()
      image_np = image.numpy()

      # arucos = locate_arucos(image_np, self.aruco_dictionary, self.marker_obj_points, self.right_intrinsics, self.right_dist_coeffs)
      # cv2.imwrite(f"{self.DEBUG_DIR}/bev/{curr_time}.png", image_np)
      self.image_queue.append((curr_time, self.prev_time_capture, image_np))
      print(curr_time - self.prev_time_capture)
      # print(arucos)

      self.prev_time_capture = curr_time

  def process_image_callback(self):
    """
    Process images from the Zed camera queue (Zed SDK version)
    """
    if len(self.image_queue) > 0:
      curr_time, prev_time_capture, image_np = self.image_queue.popleft()
      arucos = locate_aruco_poses(image_np, self.aruco_dictionary, self.marker_obj_points, self.right_intrinsics, self.right_dist_coeffs)

      self.measurements[curr_time] = arucos
      print(curr_time - prev_time_capture) #curr_time - self.prev_time_process

      # self.prev_time_process = curr_time

  def bev_track_callback(self, image_msg: Image):
    """
    Track the vehicles in bird's-eye view (ROS2 Wrapper version)
    ROS2 publisher for Zed appears to drop a lot of frames, so we will use the Zed SDK directly
    """
    curr_time = get_time_from_header(image_msg.header)
    current_frame = self.bridge.imgmsg_to_cv2(image_msg)
    arucos = locate_aruco_poses(current_frame, self.aruco_dictionary,
                           self.marker_obj_points, self.right_intrinsics, self.right_dist_coeffs)

    print(curr_time)

    self.measurements[curr_time] = arucos

    # current_frame = self.bridge.imgmsg_to_cv2(image)
    
    # # Save image to debug directory
    # cv2.imwrite(f"{self.DEBUG_DIR}/{get_time_from_header(image.header)}.png", current_frame)

    # if rvec is not None and tvec is not None:
    #   rot_matrix, _ = cv2.Rodrigues(rvec)
    #   quaternion = get_quaternion_from_rotation_matrix(rot_matrix)

  def camera_info_callback(self, data: CameraInfo):
    self.right_intrinsics = data.k.reshape(3, 3)
    self.right_dist_coeffs = np.array([data.d])
    
    # Only need the camera parameters once (assuming no change)
    self.destroy_subscription(self.right_camera_info_sub)
  
  def destroy_node(self):
    self.zed.close()
    os.makedirs(f"{self.DEBUG_DIR}/bev/left", exist_ok=True)
    os.makedirs(f"{self.DEBUG_DIR}/bev/right", exist_ok=True)

    # write the measurements to a file
    with open(f"{self.DEBUG_DIR}/bev/measurements.txt", "w") as f:
      for time, arucos in self.measurements.items():
        f.write(f"{time}: {arucos}\n")

    # write all the images in the queue to a file
    while len(self.image_queue) > 0:
      current_time, _, left_image, right_image = self.image_queue.popleft()
      cv2.imwrite(f"{self.DEBUG_DIR}/bev/left/{current_time}.png", left_image)
      cv2.imwrite(f"{self.DEBUG_DIR}/bev/right/{current_time}.png", right_image)

    # Write the camera intrinsics to a file
    np.savetxt(f"{self.DEBUG_DIR}/bev/right/intrinsics.txt", self.right_intrinsics)
    np.savetxt(f"{self.DEBUG_DIR}/bev/right/dist_coeffs.txt", self.right_dist_coeffs)

    np.savetxt(f"{self.DEBUG_DIR}/bev/left/intrinsics.txt", self.left_intrinsics)
    np.savetxt(f"{self.DEBUG_DIR}/bev/left/dist_coeffs.txt", self.left_dist_coeffs)
    np.savetxt(f"{self.DEBUG_DIR}/bev/extrinsics.txt", self.extrinsics)

    super().destroy_node()

  def image_callback(self, selected_camera: str, data: Image):
    """
    Callback function for images from the camera
    """
    # Convert ROS Image message to OpenCV image
    current_frame = self.bridge.imgmsg_to_cv2(data)
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
      image = current_frame,
      dictionary = self.aruco_dictionary)
    
    frame_copy = np.copy(current_frame)

    if len(marker_corners) > 0:
      frame_copy = cv2.aruco.drawDetectedMarkers(frame_copy, marker_corners, marker_ids)

      rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(marker_corners, self.side_length, self.right_intrinsics, self.right_dist_coeffs)

      for i in range(len(rvecs)):
        frame_copy = cv2.aruco.drawAxis(frame_copy, self.right_intrinsics, self.right_dist_coeffs, rvecs[i], tvecs[i], 0.1)
      
    # Write image with detected pose to video
    self.video_output.write(frame_copy)

    # Save image to debug directory
    cv2.imwrite(f"{self.DEBUG_DIR}/{selected_camera}/{get_time_from_header(data.header)}.png", current_frame)


def main(args=None):
  rclpy.init(args=args)
  
  bev_tracker = BevTracker()
  
  try:
    rclpy.spin(bev_tracker)
  except KeyboardInterrupt:
    pass
  finally:
    bev_tracker.destroy_node()
    rclpy.shutdown()
  
if __name__ == '__main__':
  main()