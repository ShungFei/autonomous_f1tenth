from datetime import datetime
import rclpy
from rclpy.node import Node
import rclpy.time
import rclpy.timer
from ackermann_msgs.msg import AckermannDriveStamped
from builtin_interfaces.msg import Time

from perception.util.conversion import get_time_from_header, get_time_from_clock, get_time_from_rosclock

class CarTrajectory(Node):
  def __init__(self):
    super().__init__('car_trajectory')
    self.trajectory = {
      "eval_time": 3e9,
      "agent": 
        [
          {"time": 0.0, "linear": 0.5, "angular": 0.25},
          {"time": 2e9, "linear": 0.5, "angular": 0.0}
        ],
      "opponent": 
        [
          {"time": 0.0, "linear": 0.5, "angular": 0.0},
        ],
    }
    
    self.is_opponent = self.declare_parameter('is_opponent', False).get_parameter_value().bool_value
    self.agent_name = self.declare_parameter('agent_name', "f1tenth").get_parameter_value().string_value
    self.opponent_name = self.declare_parameter('opponent_name', "opponent").get_parameter_value().string_value

    self.agent_ackermann_pub = self.create_publisher(
      AckermannDriveStamped,
      f'/{self.agent_name}/drive',
      10
    )
    
    self.opp_ackermann_pub = self.create_publisher(
      AckermannDriveStamped,
      f'/{self.opponent_name}/drive',
      10
    )

    self.start_time = None
    self.is_start_time_passed = False
    self.agent_pose_sub = self.create_subscription(
      Time,
      '/start_time',
      self.wait_until_start_time,
      10
    )
    
    self.END_BUFFER_TIME = 1e9
    self.agent_checkpoint = 0
    self.opp_checkpoint = 0
    self.clock_sub = self.create_timer(0.01, self.clock_callback)

  def wait_until_start_time(self, start_time: Time):
    secs, nsecs = self.get_clock().now().seconds_nanoseconds()
    while secs < start_time.sec or (secs == start_time.sec and nsecs < start_time.nanosec):
      pass

    self.start_time = start_time.sec * 1e9 + start_time.nanosec
    self.is_start_time_passed = True

  def clock_callback(self, start_time: Time):
    time = self.get_clock().now().nanoseconds - self.start_time

    # Stop the vehicles after the evaluation time has passed
    if time >= self.trajectory["eval_time"]:
      self.agent_ackermann_pub.publish(self.create_ackermann_msg(0.0, 0.0))
      self.opp_ackermann_pub.publish(self.create_ackermann_msg(0.0, 0.0))
      if time >= self.trajectory["eval_time"] + self.END_BUFFER_TIME:
        raise SystemExit
      else:
        return
  
    if not self.is_opponent and self.agent_checkpoint < len(self.trajectory["agent"]):
      agent_timestep = self.trajectory["agent"][self.agent_checkpoint]
      if self.agent_checkpoint == len(self.trajectory["agent"]) - 1:
        next_agent_timestep = None
      else:
        next_agent_timestep = self.trajectory["agent"][self.agent_checkpoint + 1]
      if next_agent_timestep and time >= next_agent_timestep["time"]:
        self.agent_checkpoint += 1

      self.agent_ackermann_pub.publish(self.create_ackermann_msg(agent_timestep["linear"], agent_timestep["angular"]))
    
    if self.is_opponent and self.opp_checkpoint < len(self.trajectory["opponent"]):
      opp_timestep = self.trajectory["opponent"][self.opp_checkpoint]
      if self.opp_checkpoint == len(self.trajectory["opponent"]) - 1:
        next_opp_timestep = None
      else:
        next_opp_timestep = self.trajectory["opponent"][self.opp_checkpoint + 1]
      if next_opp_timestep and time >= next_opp_timestep["time"]:
        self.opp_checkpoint += 1

      self.opp_ackermann_pub.publish(self.create_ackermann_msg(opp_timestep["linear"], opp_timestep["angular"]))

  def create_ackermann_msg(self, linear, angular):
    """
    Create an AckermannDriveStamped message with the given linear and angular velocities
    """
    msg = AckermannDriveStamped()
    msg.drive.speed = linear
    msg.drive.steering_angle = self.twist_to_ackermann(angular, linear, self.wheel_base)

    return msg
 
def main(args=None):
  rclpy.init(args=args)
  
  car_trajectory = CarTrajectory()

  try:
    rclpy.spin(car_trajectory)
  except SystemExit:
    pass
  finally:
    car_trajectory.destroy_node()
    rclpy.shutdown()
