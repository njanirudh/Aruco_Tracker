import numpy as np
import cv2
import cv2.aruco as aruco
import glob

cap = cv2.VideoCapture(0)

# FILE_STORAGE_READ
cv_file = cv2.FileStorage("calib_images/test.yaml", cv2.FILE_STORAGE_READ)

# Note :: we also have to specify the type to retrieve other wise we only get a
# fileNode object back instead of a matrix
camera_matrix = cv_file.getNode("camera_matrix").mat()
dist_matrix = cv_file.getNode("dist_coeff").mat()

print("camera_matrix : ", camera_matrix.tolist())
print("dist_matrix : ", dist_matrix.tolist())


while (True):
    ret, frame = cap.read()
    # operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    #lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)


    font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)


    if np.all(ids != None):

        rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_matrix) #Estimate pose of each marker and return the values rvet and tvec---different from camera coefficients
        #(rvec-tvec).any() # get rid of that nasty numpy value array error

        for i in range(0, ids.size):
            aruco.drawAxis(frame, camera_matrix, dist_matrix, rvec[i], tvec[i], 0.1)  # Draw Axis
        aruco.drawDetectedMarkers(frame, corners) #Draw A square around the markers


        ###### DRAW ID #####
        strg = ''
        for i in range(0, ids.size):
            strg += str(ids[i][0])+', '

        cv2.putText(frame, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)


    else:
        ##### DRAW "NO IDS" #####
        cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()