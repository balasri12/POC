import cv2
import os
import numpy as np

for file in os.listdir('images/'):
    
#     frame = cv2.imread(r'img_blk/'+file)
    original = cv2.imread(r'images/'+file)
    
    img = original.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    img[thresh == 255] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    erosion = cv2.erode(img, kernel, iterations = 1)
    edges = cv2.Canny(img,100,200) 

    # Display edges in a frame 
    #cv2.imshow('Edges') 
    
    pts = np.argwhere(edges>0)
    y1_gy,x1_gy = pts.min(axis=0)
    y2_gy,x2_gy = pts.max(axis=0)

    ## crop the region
    cropped = img[y1_gy:y2_gy, x1_gy:x2_gy]
    
    
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    # define range of white color in HSV
    # change it according to your need !
    sensitivity = 80
    lower_white = np.array([0,0,255-sensitivity], dtype=np.uint8)
    upper_white = np.array([255,sensitivity,255], dtype=np.uint8)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel)

    res = cv2.bitwise_and(cropped,cropped, mask= mask)
    edges = cv2.Canny(res,100,200) 

    pts = np.argwhere(edges>0)
    y1_mk,x1_mk = pts.min(axis=0)
    y2_mk,x2_mk = pts.max(axis=0)
    
    step_1 = original[y1_gy:y2_gy, x1_gy:x2_gy]
    step_2 = step_1[y1_mk:y2_mk, x1_mk:x2_mk]
    

    ## crop the region
    #cropped = cropped[y1_mk:y2_mk, x1_mk:x2_mk]
    cv2.imwrite('img_out/'+file,step_2)