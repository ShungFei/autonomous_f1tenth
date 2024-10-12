from collections import deque
from datetime import datetime
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped, PoseStamped
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag

from perception.util.conversion import get_time_from_header, get_euler_from_quaternion
from perception_interfaces.msg import StateEstimateStamped

class StateEstimator(Node):
  def __init__(self):
    """
    Class constructor to set up the node
    """
    super().__init__('state_estimation')

    curr_time = datetime.now().strftime('%y_%m_%d_%H:%M:%S')
    fallback_debug_dir = f"perception_debug/{curr_time}"
    self.DEBUG_DIR = self.declare_parameter('debug_dir', fallback_debug_dir).get_parameter_value().string_value
    self.opponent_name = self.declare_parameter('opponent_name', "opponent").get_parameter_value().string_value
    self.fps = self.declare_parameter('fps', 30).get_parameter_value().integer_value
    self.state_estimation_methods = self.declare_parameter('state_estimation_methods', ['kalman_filter_ca']).get_parameter_value().string_array_value
    self.window_size = self.declare_parameter('window_size', 5).get_parameter_value().integer_value
    self.debug = self.declare_parameter('debug', False).get_parameter_value().bool_value

    self.latest_poses : deque[PoseStamped] = deque(maxlen=self.window_size if "rolling_regression" in self.state_estimation_methods else 1)

    if self.debug:
      self.state_estimates_lists : dict[str, list[StateEstimateStamped]] = {}

    self.timer_sub = self.create_timer(1 / self.fps, self.timer_callback)

    self.opp_estimated_pose_sub = self.create_subscription(
      PoseStamped,
      f'{self.opponent_name}/pose_estimate',
      lambda data: self.latest_poses.append(data),
      10
    )
    
    self.kalman_ca_pub = self.create_publisher(
      StateEstimateStamped,
      f'{self.opponent_name}/state_estimate/kalman_filter_ca',
      10
    )

    # TODO
    # self.rolling_regression_pub = self.create_publisher(
    #   StateEstimateStamped,
    #   f'{self.opponent_name}/state_estimate/rolling_regression',
    #   10
    # )

  def initialize_kalman_filter_ca(self, pose: PoseStamped = np.zeros(6), dt: float = 1 / 30):
    kf = KalmanFilter(dim_x=18, dim_z=6)

    # State
    kf.x = np.zeros(18)
    kf.x[:3] = pose[:3]
    kf.x[9:12] = pose[3:6]
    # State covariance
    kf.P = np.diag([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    # Process noise
    q = Q_discrete_white_noise(dim=3, dt=dt, var=10)

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

    kf.Q = block_diag(q_second_order, q_second_order)
    
    # Measurement noise
    kf.R = np.diag([0.05, 0.05, 0.05, 4, 4, 4])

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
  
    kf.F = np.zeros((18, 18), dtype=a_t.dtype)
    kf.F[:9, :9] = a_t
    kf.F[9:, 9:] = a_t

    # Measurement matrix
    kf.H = np.zeros((6, 18))
    kf.H[:3, :3] = np.eye(3)
    kf.H[3:, 9:12] = np.eye(3)

    return kf


  def timer_callback(self):
    """Estimate the velocity and acceleration of the opponent using the latest pose as measurement input to a Kalman filter"""
    if len(self.latest_poses) == 0:
      return

    data = self.latest_poses[-1]
    position = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
    orientation = get_euler_from_quaternion(data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w)
    pose = np.concatenate((position, orientation))

    # Initialize the Kalman filter with pose if it has not already been initialized
    if not hasattr(self, 'kf'):
      self.kf = self.initialize_kalman_filter_ca(pose, 1 / self.fps)
      return
    else:
      self.kf.predict()

      sec, nanosec = self.get_clock().now().seconds_nanoseconds()
      time = nanosec + sec * 1e9

      # Only perform update step if the pose is not too old
      if time - get_time_from_header(data.header) < 1e9 / self.fps:
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
    
    self.kalman_ca_pub.publish(msg)

    if self.debug:
      self.state_estimates_lists['kalman_ca'].append(msg)

  def destroy_node(self):
    if self.debug:
      for method, state_estimate_list in self.state_estimates_lists:
        table = np.array(
          [
            get_time_from_header(state.header),
            state.position.x,
            state.position.y,
            state.position.z,
            state.linear_velocity.x,
            state.linear_velocity.y,
            state.linear_velocity.z,
            state.linear_acceleration.x,
            state.linear_acceleration.y,
            state.linear_acceleration.z,
            state.orientation.x,
            state.orientation.y,
            state.orientation.z,
            state.angular_velocity.x,
            state.angular_velocity.y,
            state.angular_velocity.z,
            state.angular_acceleration.x,
            state.angular_acceleration.y,
            state.angular_acceleration.z,
          ]
          for state in state_estimate_list
        )

        np.savetxt(f"{self.DEBUG_DIR}/state_estimates_{method}.csv", table, delimiter=",", header="time,position_x,position_y,position_z,linear_velocity_x,linear_velocity_y,linear_velocity_z,linear_acceleration_x,linear_acceleration_y,linear_acceleration_z,orientation_x,orientation_y,orientation_z,angular_velocity_x,angular_velocity_y,angular_velocity_z,angular_acceleration_x,angular_acceleration_y,angular_acceleration_z", comments="")

    super().destroy_node()
  

def main(args=None):

  rclpy.init(args=args)
  
  state_estimator = StateEstimator()

  try:
    rclpy.spin(state_estimator)
  except KeyboardInterrupt:
    pass

  state_estimator.destroy_node()
  
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()