"""
Framework   : OpenCV Aruco
Description : Calibration using checkerboard 
Status      : Running
References  :
    1) https://stackoverflow.com/questions/31249037/calibrating-webcam-using-python-and-opencv-error?rq=1
    2) https://calib.io/pages/camera-calibration-pattern-generator
"""

import numpy as np
import cv2
import glob

# Wait time to show calibration in 'ms'
WAIT_TIME = 100

# termination criteria for iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# generalizable checkerboard dimensions
# https://stackoverflow.com/questions/31249037/calibrating-webcam-using-python-and-opencv-error?rq=1
cbrow = 6
cbcol = 7

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# IMPORTANT : Object points must be changed to get real physical distance.
objp = np.zeros((cbrow * cbcol, 3), np.float32)
objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('calib_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (cbcol, cbrow), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(WAIT_TIME)

cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# ---------- Saving the calibration -----------------
cv_file = cv2.FileStorage("calib_images/test.yaml", cv2.FILE_STORAGE_WRITE)
cv_file.write("camera_matrix", mtx)
cv_file.write("dist_coeff", dist)

# note you *release* you don't close() a FileStorage object
cv_file.release()


