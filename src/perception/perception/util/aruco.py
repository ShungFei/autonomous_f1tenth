import cv2
import numpy as np
  
def locate_arucos(image: np.ndarray, aruco_dictionary, marker_obj_points, intrinsics, dist_coeffs, output_all=False) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Returns a dictionary of detected ArUco markers and their poses
    """
    # Add subpixel refinement to marker detector
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    new_intrinsics, _ = cv2.getOptimalNewCameraMatrix(intrinsics, dist_coeffs, image.shape[:2][::-1], 1)
    undistorted_image = cv2.undistort(image, intrinsics, dist_coeffs, None, new_intrinsics)
    
    all_marker_corners, all_marker_ids, _ = cv2.aruco.detectMarkers(
      image = undistorted_image,
      parameters = detector_params,
      dictionary = aruco_dictionary)
    all_marker_ids = all_marker_ids if all_marker_ids is not None else []
    arucos = {}

    for id, marker in zip(all_marker_ids, all_marker_corners):
        # tvec contains position of marker in camera frame
        if output_all:
          _, rvecs, tvecs, reproj_error = cv2.solvePnPGeneric(marker_obj_points, marker, 
                  new_intrinsics, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        
          arucos[id[0]] = (rvecs, tvecs, reproj_error)
        else:
          _, rvec, tvec = cv2.solvePnP(marker_obj_points, marker, 
                  new_intrinsics, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)

          # TODO: handle multiple markers of the same ID
          arucos[id[0]] = (rvec, tvec)

    return arucos
