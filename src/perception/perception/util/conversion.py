import numpy as np
from sympy import deg
from std_msgs.msg import Header
from rosgraph_msgs.msg import Clock
from rclpy.clock import Clock as ROSClock
from scipy.spatial.transform import Rotation

def get_euler_from_quaternion(qx, qy, qz, qw, degrees=False, positive=False):
  """
  Convert a quaternion to an Euler angle. (XYZ extrinsic rotation)
    
  Input
      :param qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    
  Output
      :return roll, pitch, yaw: The roll, pitch, and yaw angles in radians
  """

  rotation = Rotation.from_quat([qx, qy, qz, qw])
  roll, pitch, yaw = rotation.as_euler('xyz', degrees=degrees)
  
  return roll, pitch, yaw

def get_euler_from_rotation_matrix(R, degrees=False):
    """
    Convert a rotation matrix to an Euler angle. (XYZ extrinsic rotation)
        
    Input
        :param R: The rotation matrix
        
    Output
        :return roll, pitch, yaw: The roll, pitch, and yaw angles in radians
    """
    rotation = Rotation.from_matrix(R)
    roll, pitch, yaw = rotation.as_euler('xyz', degrees=degrees)

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

def get_rotation_matrix_from_euler(roll, pitch, yaw, degrees=False):
    """
    Convert an Euler angle to a rotation matrix. (XYZ extrinsic rotation)
        
    Input
        :param roll, pitch, yaw: The roll, pitch, and yaw angles in radians
        
    Output
        :return R: The rotation matrix
    """
    rotation = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    R = rotation.as_matrix()
    return R

def get_quaternion_from_euler(roll, pitch, yaw, degrees=False):
    """
    Convert an Euler angle to a quaternion. (XYZ extrinsic rotation)
        
    Input
        :param roll, pitch, yaw: The roll, pitch, and yaw angles in radians
        
    Output
        :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    rotation = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    qx, qy, qz, qw = rotation.as_quat()
    return qx, qy, qz, qw

def get_rodrigues_from_euler(roll, pitch, yaw, degrees=False):
    """
    Convert an Euler angle to a Rodrigues vector. (XYZ extrinsic rotation)
        
    Input
        :param roll, pitch, yaw: The roll, pitch, and yaw angles in radians
        
    Output
        :return rvec: The Rodrigues vector
    """
    rotation = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    rvec = rotation.as_rotvec()
    return rvec

def quaternion_conj(qx, qy, qz, qw):
  return np.array([-qx, -qy, -qz, qw]) / np.linalg.norm([qx, qy, qz, qw])

def get_quaternion_from_rodrigues(rvec, flatten=True):
    """
    Convert a Rodrigues vector to a quaternion.

    Input
        :param rvec: The Rodrigues vector
    
    Output
        :return q: The orientation in quaternion [x,y,z,w] format
    """
    if flatten:
        rvec = rvec.flatten()
    theta = np.linalg.norm(rvec)
    ax = rvec / theta

    qw = np.cos(theta / 2)
    qx = ax[0] * np.sin(theta / 2)
    qy = ax[1] * np.sin(theta / 2)
    qz = ax[2] * np.sin(theta / 2)

    q = np.array([qx, qy, qz, qw])
    return q

def get_axis_angle_from_rodrigues(rvec):
    """
    Convert a Rodrigues vector to an axis-angle representation.
    
    Input
        :param rvec: The Rodrigues vector
    
    Output
        :return ax, theta: The axis and angle of rotation
    """
    theta = np.linalg.norm(rvec)
    ax = rvec / theta

    ax_ang = np.append(ax, theta)

    return ax_ang

def get_quaternion_from_rotation_matrix(R):
    """
    Convert a rotation matrix to a quaternion.
    http://euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
        
    Input
        :param R: The rotation matrix
        
    Output
        :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    
    q = np.array([qx, qy, qz, qw])
    
    return q

def get_rodrigues_from_quaternion(qx, qy, qz, qw):
    """
    Convert a quaternion to a Rodrigues vector.
        
    Input
        :param qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
        
    Output
        :return rvec: The Rodrigues vector
    """
    theta = 2 * np.arccos(qw)
    x = qx / np.sqrt(1 - qw**2)
    y = qy / np.sqrt(1 - qw**2)
    z = qz / np.sqrt(1 - qw**2)
    rvec = np.array([x, y, z]) * theta
    return rvec
   
def get_time_from_header(header: Header, nanoseconds=False):
  """
  Convert the time from a ROS header to seconds.
    
  Input
      :param header: The ROS header
    
  Output
      :return time: The time in seconds
  """
  if nanoseconds:
    time = header.stamp.sec * 1e9 + header.stamp.nanosec
  else:
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

def get_time_from_rosclock(clock: ROSClock):
  """
  Convert the time from a ROS clock to seconds.
    
  Input
      :param clock: The ROS clock
    
  Output
      :return time: The time in seconds
  """
  sec, nanosec = clock.now().seconds_nanoseconds()
  time = sec + nanosec * 1e-9
  return time