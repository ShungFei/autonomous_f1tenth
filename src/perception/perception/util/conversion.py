import numpy as np
from std_msgs.msg import Header
from rosgraph_msgs.msg import Clock
from rclpy.clock import Clock as ROSClock

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

# TODO: this uses a different convention to get_euler_from_quaternion
def get_euler_from_rotation_matrix(R):
    """
    Convert a rotation matrix to an Euler angle.
        
    Input
        :param R: The rotation matrix
        
    Output
        :return roll, pitch, yaw: The roll, pitch, and yaw angles in radians
    """
    if R[1, 0] > 0.998:
        pitch = np.pi / 2
        yaw = np.arctan2(R[0, 2], R[2, 2])
        roll = 0
    elif R[1, 0] < -0.998:
        pitch = -np.pi / 2
        yaw = np.arctan2(R[0, 2], R[2, 2])
        roll = 0
    else:
        pitch = np.arcsin(R[1, 0])
        yaw = np.arctan2(-R[2, 0], R[0, 0])
        roll = np.arctan2(-R[1, 2], R[1, 1])
    
    return yaw, pitch, roll

# TODO: this uses a different convention to get_euler_from_quaternion
def get_euler_from_rodrigues(rvec):
    """
    Convert a Rodrigues vector to an Euler angle.
        
    Input
        :param rvec: The Rodrigues vector
        
    Output
        :return roll, pitch, yaw: The roll, pitch, and yaw angles in radians
    """
    theta = np.linalg.norm(rvec)
    ax = rvec / theta
    print(np.linalg.norm(ax))
    s = np.sin(theta)
    c = np.cos(theta)
    t = 1 - c

    if (ax[0] * ax[1] * t + ax[2] * s) > 0.998:
        pitch = np.pi / 2
        yaw = 2 * np.arctan2(ax[0] * np.sin(theta / 2), np.cos(theta / 2))
        roll = 0
    elif (ax[0] * ax[1] * t + ax[2] * s) < -0.998:
        pitch = -np.pi / 2
        yaw = -2 * np.arctan2(ax[0] * np.sin(theta / 2), np.cos(theta / 2))
        roll = 0
    else:
        pitch = np.arcsin(ax[0] * ax[1] * t + ax[2] * s)
        yaw = np.arctan2(ax[1]*s-ax[0]*ax[2]*t, 1-(ax[1]**2+ax[2]**2)*t)
        roll = np.arctan2(ax[0]*s-ax[1]*ax[2]*t, 1-(ax[0]**2+ax[2]**2)*t)
    
    return yaw, pitch, roll

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

def get_quaternion_from_rodrigues(rvec):
    """
    Convert a Rodrigues vector to a quaternion.

    Input
        :param rvec: The Rodrigues vector
    
    Output
        :return q: The orientation in quaternion [x,y,z,w] format
    """
    theta = np.linalg.norm(rvec)
    ax = rvec / theta

    qw = np.cos(theta / 2)
    qx = ax[0] * np.sin(theta / 2)
    qy = ax[1] * np.sin(theta / 2)
    qz = ax[2] * np.sin(theta / 2)

    q = np.array([qx, qy, qz, qw])
    return q


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