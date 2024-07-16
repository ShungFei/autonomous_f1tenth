import numpy as np
from std_msgs.msg import Header
from rosgraph_msgs.msg import Clock

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

def get_quaternion_from_rotation_matrix(R):
    """
    Convert a rotation matrix to a quaternion.
        
    Input
        :param R: The rotation matrix
        
    Output
        :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    if R[2, 2] < 0:
        if R[0, 0] > R[1, 1]:
            t = 1 + R[0, 0] - R[1, 1] - R[2, 2]
            q = [t, R[0, 1] + R[1, 0], R[2, 0] + R[0, 2], R[1, 2] - R[2, 1]]
        else:
            t = 1 - R[0, 0] + R[1, 1] - R[2, 2]
            q = [R[0, 1] + R[1, 0], t, R[1, 2] + R[2, 1], R[2, 0] - R[0, 2]]
    else:
        if R[0, 0] < -R[1, 1]:
            t = 1 - R[0, 0] - R[1, 1] + R[2, 2]
            q = [R[2, 0] + R[0, 2], R[1, 2] + R[2, 1], t, R[0, 1] - R[1, 0]]
        else:
            t = 1 + R[0, 0] + R[1, 1] + R[2, 2]
            q = [R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0], t]
    
    q = np.array(q) / (2 * np.sqrt(t))
    return q

def get_time_from_header(header: Header):
  """
  Convert the time from a ROS header to seconds.
    
  Input
      :param header: The ROS header
    
  Output
      :return time: The time in seconds
  """
  time = header.stamp.sec + header.stamp.nanosec * 1e-9
  return time

def get_time_from_clock(clock: Clock):
  """
  Convert the time from a ROS clock to seconds.
    
  Input
      :param clock: The ROS clock
    
  Output
      :return time: The time in seconds
  """
  time = clock.clock.sec + clock.clock.nanosec * 1e-9
  return time