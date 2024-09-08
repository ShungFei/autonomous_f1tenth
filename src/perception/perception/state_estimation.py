from datetime import datetime
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped, PoseStamped
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag

from perception.util.conversion import get_time_from_header, get_euler_from_quaternion
from perception_interfaces.msg import OptionalPoseStamped, StateEstimateStamped

class StateEstimator(Node):
  def __init__(self):
    """
    Class constructor to set up the node
    """
    super().__init__('state_estimation')

    curr_time = datetime.now().strftime('%y_%m_%d_%H:%M:%S')
    fallback_debug_dir = f"perception_debug/{curr_time}"
    self.DEBUG_DIR = self.declare_parameter('debug_dir', fallback_debug_dir).get_parameter_value().string_value


    # self.agent_name = self.declare_parameter('agent_name', "f1tenth").get_parameter_value().string_value
    # self.camera_name = self.declare_parameter('camera_name', "d435").get_parameter_value().string_value
    self.opponent_name = self.declare_parameter('opponent_name', "opponent").get_parameter_value().string_value
    # self.debug = self.declare_parameter('debug', False).get_parameter_value().bool_value

    self.opp_estimated_pose_sub = self.create_subscription(
      OptionalPoseStamped,
      f'{self.opponent_name}/pose_estimate',
      self.pose_callback,
      10
    )

    self.state_estimate_pub = self.create_publisher(
      StateEstimateStamped,
      f'{self.opponent_name}/state_estimate',
      10
    )

    self.velocity_rolling_regression_pub = self.create_publisher(
      StateEstimateStamped,
      f'{self.opponent_name}/state_estimate/velocity/rolling_regression',
      10
    )

    self.velocity_baseline_pub = self.create_publisher(
      StateEstimateStamped,
      f'{self.opponent_name}/state_estimate/velocity/baseline',
      10
    )

  def initialize_kalman_filter(self, pose: OptionalPoseStamped = np.zeros(6), dt: float = 1 / 30):
    self.kf = KalmanFilter(dim_x=18, dim_z=6)

    # State
    self.kf.x = np.zeros(18)
    self.kf.x[:3] = pose[:3]
    self.kf.x[9:12] = pose[3:6]
    # State covariance
    self.kf.P = np.diag([1, 1, 1, 10, 10, 10, 25, 25, 25, 1, 1, 1, 10, 10, 10, 25, 25, 25])
    # Process noise
    q = Q_discrete_white_noise(dim=3, dt=dt, var=0.05)

    q_second_order = np.array([[q[0, 0], 0, 0, q[0, 1], 0, 0, q[0, 2], 0, 0],
                                [0, q[0, 0], 0, 0, q[0, 1], 0, 0, q[0, 2], 0],
                                [0, 0, q[0, 0], 0, 0, q[0, 1], 0, 0, q[0, 2]],
                                [q[1, 0], 0, 0, q[1, 1], 0, 0, q[1, 2], 0, 0],
                                [0, q[1, 0], 0, 0, q[1, 1], 0, 0, q[1, 2], 0],
                                [0, 0, q[1, 0], 0, 0, q[1, 1], 0, 0, q[1, 2]],
                                [q[2, 0], 0, 0, q[2, 1], 0, 0, q[2, 2], 0, 0],
                                [0, q[2, 0], 0, 0, q[2, 1], 0, 0, q[2, 2], 0],
                                [0, 0, q[2, 0], 0, 0, q[2, 1], 0, 0, q[2, 2]],
                                ])

    self.kf.Q = block_diag(q_second_order, q_second_order)
    
    # Measurement noise
    self.kf.R = np.eye(6) * 0.05

    # Transition matrix
    a_t = np.array([[1, 0, 0, dt, 0, 0, 0.5*dt**2, 0, 0],
                    [0, 1, 0, 0, dt, 0, 0, 0.5*dt**2, 0],
                    [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt**2],
                    [0, 0, 0, 1, 0, 0, dt, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, dt, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, dt],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1],
                    ])
  
    self.kf.F = np.zeros((18, 18), dtype=a_t.dtype)
    self.kf.F[:9, :9] = a_t
    self.kf.F[9:, 9:] = a_t

    # Measurement matrix
    self.kf.H = np.zeros((6, 18))
    self.kf.H[:3, :3] = np.eye(3)
    self.kf.H[3:, 9:12] = np.eye(3)


  def pose_callback(self, data: OptionalPoseStamped):
    """Estimate the velocity and acceleration of the opponent using the pose as measurement input to a Kalman filter"""
    time = get_time_from_header(data.header)

    position = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
    orientation = get_euler_from_quaternion(data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w)
    pose = np.concatenate((position, orientation))

    # Initialize the Kalman filter with pose if it has not already been initialized
    if not hasattr(self, 'kf'):
      self.initialize_kalman_filter(pose)
      return
    
    self.kf.predict()
    self.kf.update(pose)

    # Publish the estimated state
    msg = StateEstimateStamped()
    msg.header.stamp = data.header.stamp

    state_estimate = self.kf.x
    msg.position.x, msg.position.y, msg.position.z = state_estimate[:3]
    msg.linear_velocity.x, msg.linear_velocity.y, msg.linear_velocity.z = state_estimate[3:6]
    msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = state_estimate[6:9]
    msg.orientation.x, msg.orientation.y, msg.orientation.z = state_estimate[9:12]
    msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = state_estimate[12:15]
    msg.angular_acceleration.x, msg.angular_acceleration.y, msg.angular_acceleration.z = state_estimate[15:18]
    
    self.state_estimate_pub.publish(msg)

def velocity_rolling_window_regression(data, window_size):
  """
  Estimate position and velocity of the opponent using rolling window regression on the given pose data
  """
  

def main(args=None):

  rclpy.init(args=args)
  
  state_estimator = StateEstimator()

  rclpy.spin(state_estimator)

  state_estimator.destroy_node()
  
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()