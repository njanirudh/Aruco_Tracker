"""
Framework   : OpenCV Aruco
Description : Extracting calibration from file.
Status      : Running
References  :
    1) https://answers.opencv.org/question/31207/how-do-i-load-an-opencv-generated-yaml-file-in-python/
"""

import cv2

# File storage in OpenCV
cv_file = cv2.FileStorage("calib_images/test.yaml", cv2.FILE_STORAGE_READ)

# Note : we also have to specify the type
# to retrieve otherwise we only get a 'None'
# FileNode object back instead of a matrix
camera_matrix = cv_file.getNode("camera_matrix").mat()
dist_matrix = cv_file.getNode("dist_coeff").mat()

print("camera_matrix : ", camera_matrix.tolist())
print("dist_matrix : ", dist_matrix.tolist())

cv_file.release()
