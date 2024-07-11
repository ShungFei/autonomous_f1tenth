import numpy as np

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