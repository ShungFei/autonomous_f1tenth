import cv2
import numpy as np

# COUNT = 0
def locate_aruco_corners(image: np.ndarray, aruco_dictionary) -> tuple[np.ndarray, np.ndarray]:
    # global COUNT
    """
    Returns the detected ArUco corners and IDs
    """
    # Add subpixel refinement to marker detector
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    # detector_params.cornerRefinementWinSize = 3
    # detector_params.cornerRefinementMaxIterations = 9999999
    # detector_params.cornerRefinementMinAccuracy = 0.01

    all_marker_corners, all_marker_ids, _ = cv2.aruco.detectMarkers(
      image = image,
      parameters = detector_params,
      dictionary = aruco_dictionary)
    all_marker_ids = all_marker_ids if all_marker_ids is not None else []

    # draw the markers
    # image = cv2.aruco.drawDetectedMarkers(image, all_marker_corners, all_marker_ids)

    # cv2.imwrite(f"aruco_{COUNT}.png", image)
    # COUNT += 1
    arucos = {}
    for id, marker in zip(all_marker_ids, all_marker_corners):
      arucos[id[0]] = marker
    
    return arucos

def locate_aruco_poses(image: np.ndarray, aruco_dictionary, marker_obj_points, intrinsics, dist_coeffs, output_all=False) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Returns a dictionary of detected ArUco markers and their poses
    """
    # Keep all of the original image by setting the alpha to 1
    new_intrinsics, _ = cv2.getOptimalNewCameraMatrix(intrinsics, dist_coeffs, (image.shape[1], image.shape[0]), alpha=1)
    undistorted_image = cv2.undistort(image, intrinsics, dist_coeffs, None, newCameraMatrix=new_intrinsics)
    # cv2.imwrite("undistorted.png", undistorted_image)
    # cv2.imwrite("original.png", image)
    aruco_corners = locate_aruco_corners(undistorted_image, aruco_dictionary)
    aruco_poses = {}

    for id, marker in aruco_corners.items():
        # tvec contains position of marker in camera frame
        if output_all:
          _, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(marker_obj_points, marker, 
                  new_intrinsics, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        
          aruco_poses[id] = (rvecs, tvecs, reproj_errors)
        else:
          _, rvec, tvec = cv2.solvePnP(marker_obj_points, marker, 
                  new_intrinsics, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)

          # TODO: handle multiple markers of the same ID
          aruco_poses[id] = (rvec, tvec)

    return aruco_poses
