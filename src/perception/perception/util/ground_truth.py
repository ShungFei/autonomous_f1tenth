import numpy as np
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from .conversion import get_rotation_matrix_from_quaternion

def get_camera_frame_id(agent_name, camera_name, selected_camera):
  """
  Get the frame ID of the camera in order to locate the transform in the TF tree
  """
  return f'{agent_name}/base_link/{agent_name}_{camera_name}_{selected_camera}'
  
def get_aruco_frame_id(opponent_name):
  """
  Get the frame ID of the ArUco marker in order to locate the transform in the TF tree
  """
  return f'{opponent_name}/aruco_link'
  
def get_camera_world_pose(data: TFMessage, agent_name, camera_name, selected_camera) -> tuple[np.ndarray, TransformStamped.header]:
    """
    Get the world pose of the camera from the agent vehicle's published transforms
    """
    agent_model_pos = [tf for tf in data.transforms if tf.child_frame_id == agent_name]
    camera_pos = [tf for tf in data.transforms if tf.child_frame_id == get_camera_frame_id(agent_name, camera_name, selected_camera)]

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
  
def get_aruco_world_pose(data: TFMessage, opponent_name) -> tuple[np.ndarray, TransformStamped.header]:
  """
  Get the world pose of the ArUco marker from the opponent vehicle's published transforms
  """
  opponent_model_pos = [tf for tf in data.transforms if tf.child_frame_id == opponent_name]
  opponent_aruco_pos = [tf for tf in data.transforms if tf.child_frame_id == get_aruco_frame_id(opponent_name)]
  
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