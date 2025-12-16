import cv2

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

marker_id = 0
marker_size_px = 800

img = aruco.generateImageMarker(dictionary, marker_id, marker_size_px)

cv2.imwrite("aruco_4x4_id0.png", img)
