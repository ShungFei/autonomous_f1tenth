import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo 
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped, Vector3Stamped, Vector3
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

import perception.util.ground_truth as GroundTruth

import cv2
import os
import numpy as np
from math import sqrt

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

    self.SELECTED_LEFT_CAMERA = "left"
    self.SELECTED_RIGHT_CAMERA = "right"

    self.agent_name = self.declare_parameter('agent_name', "f1tenth").get_parameter_value().string_value
    self.camera_name = self.declare_parameter('camera_name', "zed2").get_parameter_value().string_value
    self.opponent_name = self.declare_parameter('opponent_name', "opponent").get_parameter_value().string_value
    self.debug = self.declare_parameter('debug', False).get_parameter_value().bool_value

    self.get_logger().info(f"Subscribing to {self.agent_name}")

    self.left_camera_info_sub = self.create_subscription(
      CameraInfo, 
      f'{self.agent_name}/{self.camera_name}/{self.SELECTED_LEFT_CAMERA}/camera_info', 
      self.left_camera_info_callback, 
      10
    )

    self.right_camera_info_sub = self.create_subscription(
      CameraInfo, 
      f'{self.agent_name}/{self.camera_name}/{self.SELECTED_RIGHT_CAMERA}/camera_info', 
      self.right_camera_info_callback, 
      10
    )

    self.left_image_sub = self.create_subscription(
      Image, 
      f'{self.agent_name}/{self.camera_name}/{self.SELECTED_LEFT_CAMERA}/image_raw',
      self.image_callback, 
      10)
    
    self.right_image_sub = self.create_subscription(
      Image,
      f'{self.agent_name}/{self.camera_name}/{self.SELECTED_RIGHT_CAMERA}/image_raw',
      self.image_callback,
      10
    )

    self.camera_pose_pub = self.create_publisher(
      Vector3Stamped,
      f'{self.agent_name}/camera_pose',
      10
    )
    self.aruco_pose_pub = self.create_publisher(
      Vector3Stamped,
      f'{self.opponent_name}/aruco_pose',
      10
    )

    # Used to compute the ground truth pose of the agent's camera
    self.agent_pose_sub = self.create_subscription(
      TFMessage,
      f'{self.agent_name}/pose', 
      self.agent_pose_callback, 
      10
    )

    # Used to compute the ground truth pose of the AruCo marker
    self.opponent_pose_sub = self.create_subscription(
      TFMessage,
      f'{self.opponent_name}/pose',
      self.opponent_pose_callback,
      10
    )
    
    self.left_image_sub = Subscriber(
      self,
      Image,
      f'{self.agent_name}/{self.camera_name}/{self.SELECTED_LEFT_CAMERA}/image_raw',
    )

    self.right_image_sub = Subscriber(
      self,
      Image,
      f'{self.agent_name}/{self.camera_name}/{self.SELECTED_RIGHT_CAMERA}/image_raw',
    )

    self.camera_pose_sub = Subscriber(
      self,
      Vector3Stamped,
      f'{self.agent_name}/camera_pose',
    )

    self.aruco_pose_sub = Subscriber(
      self,
      Vector3Stamped,
      f'{self.opponent_name}/aruco_pose',
    )

    self.eval_sub = ApproximateTimeSynchronizer(
        [self.left_image_sub, self.right_image_sub, self.camera_pose_sub, self.aruco_pose_sub],
        10,
        0.1,
    )
    self.eval_sub.registerCallback(self.stereo_eval_callback)

    self.left_intrinsics = None
    self.left_projection = None
    self.left_dist_coeffs = None

    self.right_intrinsics = None
    self.right_projection = None
    self.right_dist_coeffs = None

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
    
  def compute_centroid(self, corners):
    """
    Compute the centroid of the ArUco marker
    """
    return np.mean(corners, axis=0)
  
  def stereo_eval_callback(self, left_image: Image, right_image: Image, camera_pose: Vector3Stamped, aruco_pose: Vector3Stamped):
    detected_aruco_pose = self.locate_aruco_stereo(left_image, right_image)
    aruco_pose = np.array([aruco_pose.vector.x, aruco_pose.vector.y, aruco_pose.vector.z])
    camera_pose = np.array([camera_pose.vector.x, camera_pose.vector.y, camera_pose.vector.z])

    if aruco_pose is not None and camera_pose is not None and detected_aruco_pose is not None:
      print('left_intrinsics', self.left_intrinsics)
      print('right_intrinsics', self.right_intrinsics)
      print('aruco', aruco_pose)
      print('camera', camera_pose)
      print('detected', detected_aruco_pose)
      print('distance', sqrt(np.sum((aruco_pose - camera_pose)**2)))
      print('distance', sqrt(np.sum((detected_aruco_pose)**2)))
  
  def left_camera_info_callback(self, data: CameraInfo):
    self.left_intrinsics = data.k.reshape(3, 3)
    self.left_projection = data.p.reshape(3, 4)
    self.left_dist_coeffs = np.array([data.d])

    # Only need the camera parameters once (assuming no change)
    self.destroy_subscription(self.left_camera_info_sub)
  
  def right_camera_info_callback(self, data: CameraInfo):
    self.right_intrinsics = data.k.reshape(3, 3)
    self.right_projection = data.p.reshape(3, 4)
    self.right_dist_coeffs = np.array([data.d])
    
    # Only need the camera parameters once (assuming no change)
    self.destroy_subscription(self.right_camera_info_sub)
    
  def agent_pose_callback(self, data: TFMessage):
    """
    Callback function for the agent's pose

    This publishes the 3D world position (x,y,z) of the camera on the agent vehicle
    """
    camera_world_pose, header = GroundTruth.get_camera_world_pose(data, self.agent_name, self.camera_name, self.SELECTED_LEFT_CAMERA)
    if camera_world_pose is not None:
      msg = Vector3Stamped()
      msg.header = header
      msg.vector.x, msg.vector.y, msg.vector.z = camera_world_pose

      self.camera_pose_pub.publish(msg)
  
  def opponent_pose_callback(self, data: TFMessage):
    """
    Callback function for the opponent's pose

    This publishes the 3D world position (x,y,z) of the ArUco marker on the opponent vehicle
    """
    aruco_world_pose, header = GroundTruth.get_aruco_world_pose(data, self.opponent_name)
    if aruco_world_pose is not None and header is not None:
      msg = Vector3Stamped()
      msg.header = header
      msg.vector.x, msg.vector.y, msg.vector.z = aruco_world_pose

      self.aruco_pose_pub.publish(msg)

  def locate_aruco_stereo(self, left_image: Image, right_image: Image):
    left_frame = self.bridge.imgmsg_to_cv2(left_image)
    right_frame = self.bridge.imgmsg_to_cv2(right_image)
    left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
    right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)

    parameters = cv2.aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    left_marker_corners, left_marker_ids, _ = cv2.aruco.detectMarkers(
      image = left_frame,
      parameters = parameters,
      dictionary = self.aruco_dictionary)
    right_marker_corners, right_marker_ids, _ = cv2.aruco.detectMarkers(
      image = right_frame,
      parameters = parameters,
      dictionary = self.aruco_dictionary)
    
    if len(left_marker_corners) > 0 and len(right_marker_corners) > 0:
      homo_points = cv2.triangulatePoints(self.left_projection, self.right_projection, left_marker_corners[0], right_marker_corners[0])
      points = cv2.convertPointsFromHomogeneous(homo_points.T)

      return self.compute_centroid(points)[0]
    else:
      return None

  def image_callback(self, data: Image):
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

      # These two approaches should produce the same result
      rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(marker_corners, 75/720, self.left_intrinsics, self.left_dist_coeffs)
      _, rvec, tvec = cv2.solvePnP(self.marker_obj_points, marker_corners[0], 
                         self.left_intrinsics, np.array([0, 0, 0, 0, 0], dtype=np.float32), flags=cv2.SOLVEPNP_IPPE_SQUARE)
      
      for i in range(len(rvecs)):
        frame_copy = cv2.aruco.drawAxis(frame_copy, self.left_intrinsics, self.left_dist_coeffs, rvecs[i], tvecs[i], 0.1)
      
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