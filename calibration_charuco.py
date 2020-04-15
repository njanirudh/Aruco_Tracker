"""
Framework   : OpenCV Aruco
Description : Calibration of camera using charuco board.
Status      : Running
References  :
    1) https://github.com/kyle-bersani/opencv-examples/blob/master/CalibrationByCharucoBoard/CalibrateCamera.py
    2) https://calib.io/pages/camera-calibration-pattern-generator
    3) https://github.com/opencv/opencv_contrib/tree/master/modules/aruco/misc/pattern_generator
"""

import glob

import cv2
from cv2 import aruco

# This requires a set of images or a video taken with the camera you want to calibrate
# All images used should be the same size, which if taken with the same camera shouldn't be a problem
images = glob.glob('calib_images/charuco/*.jpg')

# ChAruco board variables
CHARUCOBOARD_ROWCOUNT = 7
CHARUCOBOARD_COLCOUNT = 9
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_50)

# Create constants to be passed into OpenCV and Aruco methods
# Length of the squares and markers must be in the same units.
CHARUCO_BOARD = aruco.CharucoBoard_create(
    squaresX=CHARUCOBOARD_COLCOUNT,
    squaresY=CHARUCOBOARD_ROWCOUNT,
    squareLength=0.19134,
    markerLength=0.1424,
    dictionary=ARUCO_DICT)

# Set parameters for detecting the markers precisely.
PARAMETERS = aruco.DetectorParameters_create()
PARAMETERS.adaptiveThreshWinSizeMin = 15
PARAMETERS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR

# Create the arrays and variables we'll use to store info like corners and IDs from images processed
corners_all = []  # Corners discovered in all images processed
ids_all = []  # Aruco ids corresponding to corners discovered
image_size = None  # Determined at runtime

# Loop through images glob'ed
for iname in images:
    # Open the image
    img = cv2.imread(iname)
    # Grayscale the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find aruco markers in the query image
    corners, ids, _ = aruco.detectMarkers(
        image=gray,
        dictionary=ARUCO_DICT,
        parameters=PARAMETERS)

    # Outline the aruco markers found in our query image
    img = aruco.drawDetectedMarkers(
        image=img,
        corners=corners)

    # Get charuco corners and ids from detected aruco markers
    response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=CHARUCO_BOARD)

    # If a Charuco board was found, let's collect image/corner points
    # Requiring at least 20 squares
    if response > 20:
        # Add these corners and ids to our calibration arrays
        corners_all.append(charuco_corners)
        ids_all.append(charuco_ids)

        # Draw the Charuco board we've detected to show our calibrator the board was properly detected
        img = aruco.drawDetectedCornersCharuco(
            image=img,
            charucoCorners=charuco_corners,
            charucoIds=charuco_ids)

        # If our image size is unknown, set it now (?)
        if not image_size:
            image_size = gray.shape[::-1]

        # Reproportion the image, maxing width or height at 1000
        proportion = max(img.shape) / 1000.0
        img = cv2.resize(img, (int(img.shape[1] / proportion), int(img.shape[0] / proportion)))

        # Pause to display each image, waiting for key press
        cv2.imshow('Charuco board', img)
        cv2.waitKey(300)
    else:
        print("Not able to detect a charuco board in image: {}".format(iname))

# Destroy any open CV windows
cv2.destroyAllWindows()

# Make sure at least one image was found
if len(images) < 1:
    # Calibration failed because there were no images, warn the user
    print(
        "Calibration was unsuccessful. No images of charucoboards were found. "
        "Add images of charucoboards and use or alter the naming conventions used in this file.")
    # Exit for failure
    exit()

# Make sure we were able to calibrate on at least one charucoboard by checking
# if we ever determined the image size
if not image_size:
    # Calibration failed because we didn't see any charucoboards of the PatternSize used
    print(
        "Calibration was unsuccessful. We couldn't detect charucoboards in any of the images supplied. "
        "Try changing the patternSize passed into Charucoboard_create(), or try different pictures of charucoboards.")
    # Exit for failure
    exit()

# Now that we've seen all of our images, perform the camera calibration
# based on the set of points we've discovered
calibration, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
    charucoCorners=corners_all,
    charucoIds=ids_all,
    board=CHARUCO_BOARD,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None)

# Print matrix and distortion coefficient to the console
print(cameraMatrix)
print(distCoeffs)

# ---------- Saving the calibration -----------------
cv_file = cv2.FileStorage("charuco_camera_calibration.yaml", cv2.FILE_STORAGE_WRITE)
cv_file.write("camera_matrix", cameraMatrix)
cv_file.write("dist_coeff", distCoeffs)
# note you *release* you don't close() a FileStorage object
cv_file.release()

# Print to console our success
print('Calibration successful. Calibration file used: {}'.format('calibration.pckl'))
