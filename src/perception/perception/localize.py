import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo 
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Vector3Stamped, PoseStamped
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from datetime import datetime
import cv2
import pyrealsense2 as rs
from queue import Queue

import os
import numpy as np
from math import sqrt

import perception.util.ground_truth as GroundTruth
from perception.util.conversion import get_time_from_header, get_quaternion_from_rotation_matrix, get_time_from_rosclock

class CarLocalizer(Node):
  """
  Create an ImageSubscriber class, which is a subclass of the Node class.
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    super().__init__('car_localizer')

    curr_time = datetime.now().strftime('%y_%m_%d_%H:%M:%S')
    fallback_debug_dir = f"perception_debug/{curr_time}"

    self.DEBUG_DIR = self.declare_parameter('debug_dir', fallback_debug_dir).get_parameter_value().string_value

    # Name of the cameras to use
    self.SELECTED_CAMERA = "color"
    self.SELECTED_DEPTH_CAMERA = "depth"

    os.makedirs(f"{self.DEBUG_DIR}/{self.SELECTED_CAMERA}", exist_ok=True)

    self.agent_name = self.declare_parameter('agent_name', "f1tenth").get_parameter_value().string_value
    self.camera_name = self.declare_parameter('camera_name', "d435").get_parameter_value().string_value

    # Currently unused
    self.opponent_name = self.declare_parameter('opponent_name', "opponent").get_parameter_value().string_value
    self.debug = self.declare_parameter('debug', False).get_parameter_value().bool_value

    self.get_logger().info(f"Subscribing to {self.agent_name}")

    if self.debug == True:
      self.video_output = cv2.VideoWriter(f"{self.DEBUG_DIR}/detection.avi", cv2.VideoWriter_fourcc(*'XVID'), 20.0, (1920, 1080))

      self.color_image_sub = self.create_subscription(
        Image, 
        f'{self.agent_name}/{self.camera_name}/{self.SELECTED_CAMERA}/image_raw', 
        lambda data: self.image_callback(self.SELECTED_CAMERA, data), 
        10)
      
    self.color_camera_info_sub = self.create_subscription(
      CameraInfo, 
      f'{self.agent_name}/{self.camera_name}/{self.SELECTED_CAMERA}/camera_info', 
      self.camera_info_callback, 
      10
    )

    self.opp_estimated_pose_pub = self.create_publisher(
      PoseStamped,
      f'{self.opponent_name}/pose_estimate',
      10
    )

    self.color_image_sub = Subscriber(
      self,
      Image,
      f'{self.agent_name}/{self.camera_name}/{self.SELECTED_CAMERA}/image_raw',
    )

    self.depth_image_sub = Subscriber(
      self,
      Image,
      f'{self.agent_name}/{self.camera_name}/{self.SELECTED_DEPTH_CAMERA}/image_rect_raw',
    )

    self.eval_sub = ApproximateTimeSynchronizer(
        # Add self.depth_image_sub if depth required
        [self.color_image_sub],
        10,
        0.1,
    )
    self.eval_sub.registerCallback(self.pose_pub_callback)
    self.color_intrinsics = None
    self.color_dist_coeffs = None
    self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

    # Define the side length of the ArUco marker
    self.side_length = 0.15

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

    self.previous_image_time = get_time_from_rosclock(self.get_clock())
    self.previous_pose_time = self.previous_image_time

    ## === USING THE PYREALSENSE2 API INSTEAD OF ROS2 WRAPPER === ##

    # Configure depth and color streams
    self.pipeline = rs.pipeline()
    self.config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
    pipeline_profile = self.config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    # Start streaming
    cfg = self.pipeline.start(self.config)
    profile = cfg.get_stream(rs.stream.color)
    color_intr = profile.as_video_stream_profile().get_intrinsics() 

    self.color_intrinsics = np.array([
        [color_intr.fx, 0, color_intr.ppx],
        [0, color_intr.fy, color_intr.ppy],
        [0, 0, 1]
    ])
    self.color_dist_coeffs = np.array(color_intr.coeffs)
    self.create_timer(1/30, self.rs_pose_pub_callback)

  def destroy_node(self):
    if self.debug == True:
      self.video_output.release()
    super().destroy_node()

  def rs_pose_pub_callback(self):
    """
    Publish the estimated pose of the opponent (PyRealsense2 Version)

    This currently just dumps the images to a folder (doesn't actually publish pose)
    """
    frames = self.pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    image_np = np.asanyarray(color_frame.get_data())
    current_time = color_frame.get_timestamp() / 1000

    # rvec, tvec = self.locate_aruco(image_np)
    cv2.imwrite(f"{self.DEBUG_DIR}/{self.SELECTED_CAMERA}/{current_time}.jpg", image_np)

    if current_time - self.previous_pose_time > 0.034:
      print(f"Current time: {current_time}, Time between two frames: {current_time - self.previous_pose_time}")

    self.previous_pose_time = current_time

  def pose_pub_callback(self, image: Image):
    """
    Publish the estimated pose of the opponent (ROS2 Version)
    """
    current_time = get_time_from_header(image.header)
    image_np = self.bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
    rvec, tvec = self.locate_aruco(image_np)
    
    if rvec is not None and tvec is not None:
      rot_matrix, _ = cv2.Rodrigues(rvec)
      quaternion = get_quaternion_from_rotation_matrix(rot_matrix)

      # Publish the estimated pose
      msg = PoseStamped()

      msg.header = image.header
      msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = tvec[0][0], tvec[1][0], tvec[2][0]
      msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = quaternion
      
      self.opp_estimated_pose_pub.publish(msg)
      
    if current_time - self.previous_pose_time > 0.034:
      print(f"Current time: {current_time}, Time between two frames: {current_time - self.previous_pose_time}")
    self.previous_pose_time = current_time

  def camera_info_callback(self, data: CameraInfo):
    self.color_intrinsics = data.k.reshape(3, 3)
    self.color_dist_coeffs = np.array([data.d])
    
    # Only need the camera parameters once (assuming no change)
    self.destroy_subscription(self.color_camera_info_sub)
  
  def locate_aruco(self, image: np.ndarray, bgr: bool=True):
    # probably unnecessary
    # if bgr:
    #   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Add subpixel refinement to marker detector
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
      image = image,
      parameters = detector_params,
      dictionary = self.aruco_dictionary)
    if len(marker_corners) > 0:
      # tvec contains position of marker in camera frame
      _, rvec, tvec = cv2.solvePnP(self.marker_obj_points, marker_corners[0], 
                         self.color_intrinsics, 0, flags=cv2.SOLVEPNP_IPPE_SQUARE)
      
      if self.debug == True:
        print('corners', marker_corners[0])
        print('rvec', rvec)
        print('tvec', tvec)
        print('distance', sqrt(np.sum((tvec)**2)))
      return rvec, tvec
    else:
      return None, None

  def image_callback(self, selected_camera: str, data: Image):
    """
    Callback function for images from the camera
    """
    # Convert ROS Image message to OpenCV image
    current_frame = self.bridge.imgmsg_to_cv2(data)
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    current_time = get_time_from_header(data.header)

    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
      image = current_frame,
      dictionary = self.aruco_dictionary)
    
    frame_copy = np.copy(current_frame)

    if len(marker_corners) > 0:
      frame_copy = cv2.aruco.drawDetectedMarkers(frame_copy, marker_corners, marker_ids)

      rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(marker_corners, self.side_length, self.color_intrinsics, self.color_dist_coeffs)

      for i in range(len(rvecs)):
        frame_copy = cv2.drawFrameAxes(frame_copy, self.color_intrinsics, self.color_dist_coeffs, rvecs[i], tvecs[i], 0.1)
        # add text to the image that shows the distance using tvec
        cv2.putText(frame_copy, f"Distance: {sqrt(np.sum((tvecs[i])**2))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        #add text to show the time between two frames
        cv2.putText(frame_copy, f"Time: {current_time - self.previous_image_time}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    self.previous_image_time = current_time
    # Write image with detected pose to video
    self.video_output.write(frame_copy)

    # Save image to debug directory
    cv2.imwrite(f"{self.DEBUG_DIR}/{selected_camera}/{get_time_from_header(data.header)}.jpg", current_frame)


def main(args=None):
  rclpy.init(args=args)
  
  car_localizer = CarLocalizer()
  
  try:
    rclpy.spin(car_localizer)
  except KeyboardInterrupt:
    pass
  
  car_localizer.destroy_node()
  
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()