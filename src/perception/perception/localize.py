import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo 
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Vector3Stamped, PoseStamped
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
import cv2
import os
import numpy as np
from math import sqrt

import perception.util.ground_truth as GroundTruth
from perception.util.conversion import get_time_from_header, get_quaternion_from_rotation_matrix
 
class CarLocalizer(Node):
  """
  Create an ImageSubscriber class, which is a subclass of the Node class.
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    super().__init__('car_localizer')
    
    self.debug_dir = "debug_images"
    # Create a directory to save images
    if not os.path.exists(self.debug_dir):
      os.makedirs(self.debug_dir)

    # Name of the cameras to use
    self.SELECTED_CAMERA = "color"
    self.SELECTED_DEPTH_CAMERA = "depth"

    self.agent_name = self.declare_parameter('agent_name', "f1tenth").get_parameter_value().string_value
    self.camera_name = self.declare_parameter('camera_name', "d435").get_parameter_value().string_value
    
    # Currently unused
    self.opponent_name = self.declare_parameter('opponent_name', "opponent").get_parameter_value().string_value
    self.debug = self.declare_parameter('debug', False).get_parameter_value().bool_value

    self.get_logger().info(f"Subscribing to {self.agent_name}")

    if self.debug == True:
      self.color_image_sub = self.create_subscription(
        Image, 
        f'{self.agent_name}/{self.camera_name}/{self.SELECTED_CAMERA}/image_raw', 
        self.image_callback, 
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
        [self.color_image_sub, self.depth_image_sub],
        10,
        0.1,
    )
    self.eval_sub.registerCallback(self.pose_pub_callback)

    self.color_intrinsics = None
    self.dist_coeffs = None
    self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

    # Define the side length of the ArUco marker
    side_length = 0.15

    # Define the half side length
    half_side_length = side_length / 2

    # Define the 4 corners of the ArUco marker
    self.marker_obj_points = np.array([[
        [-half_side_length, half_side_length, 0],
        [half_side_length, half_side_length, 0],
        [half_side_length, -half_side_length, 0],
        [-half_side_length, -half_side_length, 0]
    ]], dtype=np.float32)

    # Used to convert between ROS and OpenCV images
    self.bridge = CvBridge()

  def pose_pub_callback(self, image: Image, depth_image: Image):
    """
    Publish the estimated pose of the opponent
    """
    rvec, tvec = self.locate_aruco(image)

    if rvec is not None and tvec is not None:
      if self.debug == True:
        print('intrinsics', self.color_intrinsics)
        print('detected', tvec)

        # convert ros header time to seconds
        print('time:', get_time_from_header(image.header))
        print('estimated distance:', sqrt(np.sum((tvec)**2)))

      rot_matrix, _ = cv2.Rodrigues(rvec)
      quaternion = get_quaternion_from_rotation_matrix(rot_matrix)

      # Publish the estimated pose
      msg = PoseStamped()

      msg.header = image.header
      msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = tvec[0][0], tvec[1][0], tvec[2][0]
      msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = quaternion

      self.opp_estimated_pose_pub.publish(msg)

  def camera_info_callback(self, data: CameraInfo):
    self.color_intrinsics = data.k.reshape(3, 3)
    self.dist_coeffs = np.array([data.d])
    
    # Only need the camera parameters once (assuming no change)
    self.destroy_subscription(self.color_camera_info_sub)
  
  def locate_aruco(self, image: Image):
    current_frame = self.bridge.imgmsg_to_cv2(image)
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

    # Add subpixel refinement to marker detector
    detector_params = cv2.aruco.DetectorParameters_create()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
      image = current_frame,
      parameters = detector_params,
      dictionary = self.aruco_dictionary)
    if len(marker_corners) > 0:
      # tvec contains position of marker in camera frame
      _, rvec, tvec = cv2.solvePnP(self.marker_obj_points, marker_corners[0], 
                         self.color_intrinsics, 0, flags=cv2.SOLVEPNP_SQPNP)
      
      if self.debug == True:
        print('corners', marker_corners[0])
        print('rvec', rvec)
        print('tvec', tvec)
      return rvec, tvec
    else:
      return None, None

  def image_callback(self, data: Image):
    """
    Callback function for images from the camera
    """
    saved_counter = 0

    # Convert ROS Image message to OpenCV image
    current_frame = self.bridge.imgmsg_to_cv2(data)
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
      image = current_frame,
      dictionary = self.aruco_dictionary)
    
    frame_copy = np.copy(current_frame)

    if len(marker_corners) > 0:
      frame_copy = cv2.aruco.drawDetectedMarkers(frame_copy, marker_corners, marker_ids)

      # These two approaches should produce the same result
      rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(marker_corners, 75/720, self.color_intrinsics, self.dist_coeffs)
      _, rvec, tvec = cv2.solvePnP(self.marker_obj_points, marker_corners[0], 
                         self.color_intrinsics, np.array([0, 0, 0, 0, 0], dtype=np.float32), flags=cv2.SOLVEPNP_IPPE_SQUARE)
      
      for i in range(len(rvecs)):
        frame_copy = cv2.aruco.drawAxis(frame_copy, self.color_intrinsics, self.dist_coeffs, rvecs[i], tvecs[i], 0.1)
      
    # Save the image
    # while saved_counter < 100:
    #   cv2.imwrite(f"{self.debug_dir}/{data.header.stamp.nanosec}.jpg", current_frame)
    #   saved_counter += 1
      
    # Display image
    cv2.imshow("camera", frame_copy)

    cv2.waitKey(1)
  
def main(args=None):

  rclpy.init(args=args)
  
  car_localizer = CarLocalizer()
  
  rclpy.spin(car_localizer)
  
  car_localizer.destroy_node()
  
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()