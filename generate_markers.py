# -*- coding: utf-8 -*-
import cv2
import cv2.aruco as aruco

if __name__ == "__main__":
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    n = 50
    for i in range(n):
        image = aruco.drawMarker(dictionary, i, 150)
        fileName = str(i) + ".png"
        cv2.imwrite("img/" + fileName, image)