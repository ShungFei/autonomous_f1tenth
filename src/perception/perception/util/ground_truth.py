import numpy as np
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from .conversion import get_rotation_matrix_from_quaternion
import perception.util.conversion as conv

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
  
def get_camera_world_pose(data: TFMessage, agent_name, camera_name, selected_camera) -> tuple[np.ndarray, np.ndarray, TransformStamped.header]:
    """
    Get the world pose of the camera from the agent vehicle's published transforms
    """
    agent_model_pos = [tf for tf in data.transforms if tf.child_frame_id == agent_name]
    camera_pos = [tf for tf in data.transforms if tf.child_frame_id == get_camera_frame_id(agent_name, camera_name, selected_camera)]

    if len(agent_model_pos) > 0 and len(camera_pos) > 0:
      agent_trans, agent_quat = agent_model_pos[0].transform.translation, agent_model_pos[0].transform.rotation
      camera_trans, camera_quat = camera_pos[0].transform.translation, camera_pos[0].transform.rotation

      camera_T = np.array([camera_trans.x, camera_trans.y, camera_trans.z])
      agent_T = np.array([agent_trans.x, agent_trans.y, agent_trans.z])
      agent_rot = get_rotation_matrix_from_quaternion(agent_quat.x, agent_quat.y, agent_quat.z, agent_quat.w)
      camera_rot = get_rotation_matrix_from_quaternion(camera_quat.x, camera_quat.y, camera_quat.z, camera_quat.w)
      
      # rotate the camera to the optical frame
      camera_rot = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]) @ camera_rot

      camera_world_pos = agent_rot @ camera_T + agent_T
      camera_world_rot = agent_rot @ camera_rot 
      camera_world_quat = conv.get_quaternion_from_rotation_matrix(camera_world_rot)

      return camera_world_pos, camera_world_quat, camera_pos[0].header
    else:
      return None, None, None
  
def get_aruco_world_pose(data: TFMessage, opponent_name) -> tuple[np.ndarray, np.ndarray, TransformStamped.header]:
  """
  Get the world pose of the ArUco marker from the opponent vehicle's published transforms
  """
  opponent_model_pos = [tf for tf in data.transforms if tf.child_frame_id == opponent_name]
  opponent_aruco_pos = [tf for tf in data.transforms if tf.child_frame_id == get_aruco_frame_id(opponent_name)]
  
  if len(opponent_model_pos) > 0 and len(opponent_aruco_pos) > 0:
    opp_trans, opp_quat = opponent_model_pos[0].transform.translation, opponent_model_pos[0].transform.rotation
    aruco_trans, aruco_quat = opponent_aruco_pos[0].transform.translation, opponent_aruco_pos[0].transform.rotation

    aruco_T = np.array([aruco_trans.x, aruco_trans.y, aruco_trans.z])
    opp_T = np.array([opp_trans.x, opp_trans.y, opp_trans.z])
    opp_rot = get_rotation_matrix_from_quaternion(opp_quat.x, opp_quat.y, opp_quat.z, opp_quat.w)
    aruco_rot = get_rotation_matrix_from_quaternion(aruco_quat.x, aruco_quat.y, aruco_quat.z, aruco_quat.w)
    aruco_rot = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]) @ aruco_rot
    
    aruco_world_pos = opp_rot @ aruco_T + opp_T
    aruco_world_rot = opp_rot @ aruco_rot

    aruco_world_quat = conv.get_quaternion_from_rotation_matrix(aruco_world_rot)

    return aruco_world_pos, aruco_world_quat, opponent_aruco_pos[0].header
  else:
    return None, None, None

def get_ground_truth_relative_pose(ego_pos, ego_rot, opp_pos, opp_rot) -> tuple[np.ndarray, np.ndarray]:
  """
  Get the relative pose of the ArUco marker with respect to the camera
  """
  if ego_pos is not None and opp_pos is not None:
    return ego_rot.T @ (opp_pos - ego_pos), ego_rot.T @ opp_rot
  else:
    return None, None