import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo 
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped, Vector3Stamped, Vector3
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
import cv2
import os
import numpy as np
from math import sqrt

def get_euler_from_quaternion(qx, qy, qz, qw):
    """
    Convert a quaternion to an Euler angle.
     
    Input
        :param qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
     
    Output
        :return roll, pitch, yaw: The roll, pitch, and yaw angles in radians
    """
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    pitch = np.arcsin(2 * (qw * qy - qz * qx))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
     
    return roll, pitch, yaw

def get_rotation_matrix_from_quaternion(qx, qy, qz, qw):
    """
    Convert a quaternion to a rotation matrix.
     
    Input
        :param qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
     
    Output
        :return R: The rotation matrix
    """
    R = np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)]
    ])
    return R
 
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
    self.opponent_name = self.declare_parameter('opponent_name', "opponent").get_parameter_value().string_value
    self.debug = self.declare_parameter('debug', False).get_parameter_value().bool_value

    self.get_logger().info(f"Subscribing to {self.agent_name}")

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

    self.agent_pose_sub = self.create_subscription(
      TFMessage,
      f'{self.agent_name}/pose', 
      self.agent_pose_callback, 
      10
    )

    self.opponent_pose_sub = self.create_subscription(
      TFMessage,
      f'{self.opponent_name}/pose',
      self.opponent_pose_callback,
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
        [self.color_image_sub, self.depth_image_sub, self.camera_pose_sub, self.aruco_pose_sub],
        10,
        0.1,
    )
    self.eval_sub.registerCallback(self.eval_callback)

    self.color_intrinsics = None
    self.dist_coeffs = None
    self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

    # Define the side length of the ArUco marker
    side_length = 0.15 # 75/720

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

  def eval_callback(self, image: Image, depth_image: Image, camera_pose: Vector3Stamped, aruco_pose: Vector3Stamped):
    """
    Synced callback function for the camera image, depth image, camera pose, and ArUco pose for evaluation
    """
    detected_aruco_pose = self.locate_aruco(image)
    aruco_pose = np.array([aruco_pose.vector.x, aruco_pose.vector.y, aruco_pose.vector.z])
    camera_pose = np.array([camera_pose.vector.x, camera_pose.vector.y, camera_pose.vector.z])

    if aruco_pose is not None and camera_pose is not None and detected_aruco_pose is not None:
      print('intrinsics', self.color_intrinsics)
      print('aruco', aruco_pose)
      print('camera', camera_pose)
      print('detected', detected_aruco_pose)
      print('distance', sqrt(np.sum((aruco_pose - camera_pose)**2)))
      print('distance', sqrt(np.sum((detected_aruco_pose)**2)))

  def camera_info_callback(self, data: CameraInfo):
    self.color_intrinsics = data.k.reshape(3, 3)
    self.dist_coeffs = np.array([data.d])
    
    # Only need the camera parameters once (assuming no change)
    self.destroy_subscription(self.color_camera_info_sub)
  
  def get_camera_frame_id(self):
    """
    Get the frame ID of the camera in order to locate the transform in the TF tree
    """
    return f'{self.agent_name}/base_link/{self.agent_name}_{self.camera_name}_{self.SELECTED_CAMERA}'
  
  def get_aruco_frame_id(self):
    """
    Get the frame ID of the ArUco marker in order to locate the transform in the TF tree
    """
    return f'{self.opponent_name}/aruco_link'
  
  def get_camera_world_pose(self, data: TFMessage) -> tuple[np.ndarray, TransformStamped.header]:
    """
    Get the world pose of the camera from the agent vehicle's published transforms
    """
    agent_model_pos = [tf for tf in data.transforms if tf.child_frame_id == self.agent_name]
    camera_pos = [tf for tf in data.transforms if tf.child_frame_id == self.get_camera_frame_id()]

    if len(agent_model_pos) > 0 and len(camera_pos) > 0:
      agent_trans, agent_rot = agent_model_pos[0].transform.translation, agent_model_pos[0].transform.rotation
      camera_trans, camera_rot = camera_pos[0].transform.translation, camera_pos[0].transform.rotation

      camera_T = np.array([camera_trans.x, camera_trans.y, camera_trans.z])
      agent_T = np.array([agent_trans.x, agent_trans.y, agent_trans.z])
      R = get_rotation_matrix_from_quaternion(agent_rot.x, agent_rot.y, agent_rot.z, agent_rot.w)

      camera_world_pose = R @ camera_T + agent_T
      return camera_world_pose, camera_pos[0].header
    else:
      return None, None
  
  def get_aruco_world_pose(self, data: TFMessage) -> tuple[np.ndarray, TransformStamped.header]:
    """
    Get the world pose of the ArUco marker from the opponent vehicle's published transforms
    """
    opponent_model_pos = [tf for tf in data.transforms if tf.child_frame_id == self.opponent_name]
    opponent_aruco_pos = [tf for tf in data.transforms if tf.child_frame_id == self.get_aruco_frame_id()]
    
    if len(opponent_model_pos) > 0 and len(opponent_aruco_pos) > 0:
      opp_trans, opp_rot = opponent_model_pos[0].transform.translation, opponent_model_pos[0].transform.rotation
      aruco_trans, aruco_rot = opponent_aruco_pos[0].transform.translation, opponent_aruco_pos[0].transform.rotation

      aruco_T = np.array([aruco_trans.x, aruco_trans.y, aruco_trans.z])
      opp_T = np.array([opp_trans.x, opp_trans.y, opp_trans.z])
      R = get_rotation_matrix_from_quaternion(opp_rot.x, opp_rot.y, opp_rot.z, opp_rot.w)

      aruco_world_pose = R @ aruco_T + opp_T
      return aruco_world_pose, opponent_aruco_pos[0].header
    else:
      return None, None
    
  def agent_pose_callback(self, data: TFMessage):
    """
    Callback function for the agent's pose

    This publishes the 3D world position (x,y,z) of the camera on the agent vehicle
    """
    camera_world_pose, header = self.get_camera_world_pose(data)
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
    aruco_world_pose, header = self.get_aruco_world_pose(data)
    if aruco_world_pose is not None and header is not None:
      msg = Vector3Stamped()
      msg.header = header
      msg.vector.x, msg.vector.y, msg.vector.z = aruco_world_pose

      self.aruco_pose_pub.publish(msg)
  
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
      print('corners', marker_corners[0])
      print('rvec', rvec)
      print('tvec', tvec)
      return tvec
    else:
      return None

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