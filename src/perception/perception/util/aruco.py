import cv2
import numpy as np
  
def locate_arucos(image: np.ndarray, aruco_dictionary, marker_obj_points, intrinsics, dist_coeffs) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Returns a dictionary of detected ArUco markers and their poses
    """
    # Add subpixel refinement to marker detector
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    all_marker_corners, all_marker_ids, _ = cv2.aruco.detectMarkers(
      image = image,
      parameters = detector_params,
      dictionary = aruco_dictionary)
    all_marker_ids = all_marker_ids if all_marker_ids is not None else []
    arucos = {}

    for id, marker in zip(all_marker_ids, all_marker_corners):
        # tvec contains position of marker in camera frame
        _, rvec, tvec = cv2.solvePnP(marker_obj_points, marker, 
                            intrinsics, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        # print(id, marker, rvec, tvec)

        # TODO: handle multiple markers of the same ID
        arucos[id[0]] = (rvec, tvec)
        # if self.debug == True:
        #     print('id', id[0])
        #     print('corners', marker)
        #     print('rvec', rvec)
        #     print('tvec', tvec)
        #     print('distance', sqrt(np.sum((tvec)**2)))
    return arucos
