
# ------------ FUNCTION TO CALCULATE AREA OF DISC------------------------------

# import computer vision library(cv2) in this code
import cv2
import numpy as np

#img_path = "Fuzzy_C_strange_point_disk_IS.jpg"


def cal_area(img_path):
    if img_path == "Fuzzy_C_strange_point_cup_IS.jpg":
        image = cv2.imread(img_path)
        # structuring element
        kernel = np.ones((15, 15), np.uint8)
        # dilation
        img_dilation = cv2.dilate(image, kernel, iterations=2)
        cv2.imshow('Input', image)
        cv2.imshow('Dilation', img_dilation)
        cv2.waitKey(10)
        # getting boundary points
        contours, hierarchy = cv2.findContours(
            img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # src,retrival,approx

        for cnt in contours:
            # center, maj and minor axis and angle is returned
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(img_dilation, ellipse, (255, 0, 0), 1)

        # center, maj and minor axis and angle is returned to cal area
        (x, y), (MA, ma), angle = ellipse
        cv2.imwrite("cupcdr.jpg", img_dilation)
        area_cup = MA*ma*3.14
        return area_cup

    else:
        image = cv2.imread(img_path, 0)
        kernel = np.ones((15, 15), np.uint8)
        img_dilation = cv2.dilate(image, kernel, iterations=2)
        cv2.imshow('Input', image)
        cv2.imshow('Dilation', img_dilation)
        cv2.waitKey(10)
        contours, hierarchy = cv2.findContours(
            img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n = len(contours)
        for cnt in contours:
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(img_dilation, ellipse, (255, 0, 0), 1)

        # Calculate x,y coordinates and Major axis, Minor axis
        (x, y), (MA, ma), angle = ellipse

        cv2.imwrite("diskcdr.jpg", img_dilation)

    area_disc = MA*ma*3.14
    return area_disc
