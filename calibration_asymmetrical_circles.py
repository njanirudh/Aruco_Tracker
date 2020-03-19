"""
Framework   : OpenCV Aruco
Description : Calibration of camera using Asymmetrical Circular Pattern.
Status      : Not Working
References  :
    1)
"""

import glob

import cv2
from cv2 import aruco
import numpy as np

images = glob.glob('calib_images/asymmetric_circles/*.jpg')

# Wait time to show calibration in 'ms'
WAIT_TIME = 100

# Asymmetrical circle board variables
ASYM_CIRCLE_ROWCOUNT = 7
ASYM_CIRCLE_COLCOUNT = 9

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# IMPORTANT : Object points must be changed to get real physical distance.
objp = np.zeros((ASYM_CIRCLE_ROWCOUNT * ASYM_CIRCLE_COLCOUNT, 3), np.float32)
objp[:, :2] = np.mgrid[0:ASYM_CIRCLE_COLCOUNT, 0:ASYM_CIRCLE_ROWCOUNT].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Creating a blob detector for finding the circular blobs.
BLOB_DETECTOR = cv2.SimpleBlobDetector_create()

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, centers = cv2.findCirclesGrid(gray,(ASYM_CIRCLE_COLCOUNT,ASYM_CIRCLE_ROWCOUNT),
                                          cv2.CALIB_CB_ASYMMETRIC_GRID, BLOB_DETECTOR)

    if ret == True:
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (ASYM_CIRCLE_COLCOUNT,ASYM_CIRCLE_ROWCOUNT),
                                        centers,ret)
        cv2.imshow('img',img)
        cv2.waitKey(WAIT_TIME)

cv2.destroyAllWindows()
