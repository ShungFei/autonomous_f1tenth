import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo 
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Vector3Stamped, PoseStamped
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from datetime import datetime
from collections import deque
import cv2
import pyrealsense2 as rs
from queue import Queue
from perception.util.aruco import locate_arucos

import os
import numpy as np
from math import sqrt

import perception.util.ground_truth as GroundTruth
from perception.util.conversion import get_euler_from_quaternion, get_time_from_header, get_quaternion_from_rotation_matrix, get_time_from_rosclock

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
    os.makedirs(f"{self.DEBUG_DIR}/{self.SELECTED_DEPTH_CAMERA}", exist_ok=True)

    self.agent_name = self.declare_parameter('agent_name', "f1tenth").get_parameter_value().string_value
    self.camera_name = self.declare_parameter('camera_name', "d435").get_parameter_value().string_value

    # Currently unused
    self.opponent_name = self.declare_parameter('opponent_name', "opponent").get_parameter_value().string_value
    self.debug = self.declare_parameter('debug', False).get_parameter_value().bool_value
    self.opp_back_marker_id = self.declare_parameter('opp_back_marker_id', -1).get_parameter_value().integer_value
    self.auto_exposure = self.declare_parameter('auto_exposure', False).get_parameter_value().bool_value
    self.exposure_time = self.declare_parameter('exposure_time', 50).get_parameter_value().integer_value
    self.gain = self.declare_parameter('gain', 128).get_parameter_value().integer_value

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
    self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    # Start streaming
    cfg = self.pipeline.start(self.config)
    color_profile = cfg.get_stream(rs.stream.color)
    color_intr = color_profile.as_video_stream_profile().get_intrinsics() 

    self.color_intrinsics = np.array([
        [color_intr.fx, 0, color_intr.ppx],
        [0, color_intr.fy, color_intr.ppy],
        [0, 0, 1]
    ])
    self.color_dist_coeffs = np.array(color_intr.coeffs)
    
    depth_profile = cfg.get_stream(rs.stream.depth)
    depth_intr = depth_profile.as_video_stream_profile().get_intrinsics()

    self.depth_intrinsics = np.array([
        [depth_intr.fx, 0, depth_intr.ppx],
        [0, depth_intr.fy, depth_intr.ppy],
        [0, 0, 1]
    ])
    self.depth_dist_coeffs = np.array(depth_intr.coeffs)
    
    # Get depth scale
    depth_sensor = cfg.get_device().first_depth_sensor()
    color_sensor = cfg.get_device().first_color_sensor()

    color_sensor.set_option(rs.option.enable_auto_exposure, 1 if self.auto_exposure else 0)
    color_sensor.set_option(rs.option.exposure, self.exposure_time)
    color_sensor.set_option(rs.option.gain, self.gain)

    self.depth_scale = depth_sensor.get_depth_scale()

    print("Color Intrinsics: ", self.color_intrinsics)
    print("Depth Intrinsics: ", self.depth_intrinsics)
    print("Color Distortion Coefficients: ", self.color_dist_coeffs)
    print("Depth Distortion Coefficients: ", self.depth_dist_coeffs)
    print("Depth Scale: ", self.depth_scale)

    self.image_queue = deque()
    self.euler_window = []
    self.create_timer(1/30, self.rs_pose_pub_callback)

  def destroy_node(self):
    if self.debug == True:
      self.video_output.release()

    self.pipeline.stop()

    while self.image_queue:
      time, image_np = self.image_queue.popleft()
      cv2.imwrite(f"{self.DEBUG_DIR}/{self.SELECTED_CAMERA}/{time}.png", image_np)
      
    # Save camera parameters to debug directory
    np.savetxt(f"{self.DEBUG_DIR}/{self.SELECTED_CAMERA}/intrinsics.txt", self.color_intrinsics)
    np.savetxt(f"{self.DEBUG_DIR}/{self.SELECTED_CAMERA}/dist_coeffs.txt", self.color_dist_coeffs)

    np.savetxt(f"{self.DEBUG_DIR}/{self.SELECTED_DEPTH_CAMERA}/intrinsics.txt", self.depth_intrinsics)
    np.savetxt(f"{self.DEBUG_DIR}/{self.SELECTED_DEPTH_CAMERA}/dist_coeffs.txt", self.depth_dist_coeffs)
    np.savetxt(f"{self.DEBUG_DIR}/{self.SELECTED_DEPTH_CAMERA}/depth_scale.txt", np.array([self.depth_scale]))

    super().destroy_node()

  def rs_pose_pub_callback(self):
    """
    Publish the estimated pose of the opponent (PyRealsense2 Version)

    This currently just dumps the images to a folder (doesn't actually publish pose)
    """
    frames = self.pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    # bug if not doing deep copy of array????
    image_np = np.copy(np.asanyarray(color_frame.get_data()))
    depth_np = np.asanyarray(depth_frame.get_data())

    current_time = int(color_frame.get_timestamp() * 1e6) # From milliseconds to nano seconds

    # arucos = locate_arucos(image_np, self.aruco_dictionary, self.marker_obj_points, self.color_intrinsics, self.color_dist_coeffs)
    # self.show_angle_diffs(arucos, 26)
    self.image_queue.append((current_time, image_np))
    np.save(f"{self.DEBUG_DIR}/{self.SELECTED_DEPTH_CAMERA}/{current_time}.npy", depth_np)
    
    if current_time - self.previous_pose_time > 4e7:
      print(f"Current time: {current_time}, Time between two frames: {(current_time - self.previous_pose_time) / 1e9}")

    self.previous_pose_time = current_time

  def show_angle_diffs(self, arucos, wall_aruco_id: int):
    """Prints the differences from 0 with regards to the ideal euler orientation of an aruco marker against a wall"""
    if len(arucos) > 0:
      try:
        rvec = arucos[wall_aruco_id][0]
        rot_matrix, _ = cv2.Rodrigues(rvec)
        quaternion = get_quaternion_from_rotation_matrix(rot_matrix)
        x, y, z = get_euler_from_quaternion(*quaternion)
        x = x + math.pi if x < 0 else x - math.pi
        z = z + math.pi / 2
        self.euler_window.append((x, y, z))
        if len(self.euler_window) > 10:
          self.euler_window.pop(0)
        avg_x, avg_y, avg_z = sum([v[0] for v in self.euler_window]) / len(self.euler_window), sum([v[1] for v in self.euler_window]) / len(self.euler_window), sum([v[2] for v in self.euler_window]) / len(self.euler_window)\
        
        x_deg, y_deg, z_deg = math.degrees(x), math.degrees(y), math.degrees(z)
        avg_x_deg, avg_y_deg, avg_z_deg = math.degrees(avg_x), math.degrees(avg_y), math.degrees(avg_z)
        print(f"{x_deg:+.8f} {y_deg:+.8f} {z_deg:+.8f} Rolling mean: {avg_x_deg:+.8f} {avg_y_deg:+.8f} {avg_z_deg:+.8f}")
      except KeyError:
        pass

  def pose_pub_callback(self, image: Image):
    """
    Publish the estimated pose of the opponent (ROS2 Version)
    """
    current_time = 1e9 * image.header.stamp.sec + image.header.stamp.nanosec
    image_np = self.bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
    arucos = locate_arucos(image_np, self.aruco_dictionary, self.marker_obj_points, self.color_intrinsics, self.color_dist_coeffs)

    if len(arucos) > 0:
      if self.opp_back_marker_id in arucos:
        rvec, tvec = arucos[self.opp_back_marker_id]
    
        rot_matrix, _ = cv2.Rodrigues(rvec)
        quaternion = get_quaternion_from_rotation_matrix(rot_matrix)

        # Publish the estimated pose
        msg = PoseStamped()

        msg.header = image.header
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = tvec[0][0], tvec[1][0], tvec[2][0]
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = quaternion
        
        self.opp_estimated_pose_pub.publish(msg)
      
    if current_time - self.previous_pose_time > 34e6:
      print(f"Current time: {current_time}, Time between two frames: {(current_time - self.previous_pose_time) / 1e9}")
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
    cv2.imwrite(f"{self.DEBUG_DIR}/{selected_camera}/{get_time_from_header(data.header)}.png", current_frame)


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