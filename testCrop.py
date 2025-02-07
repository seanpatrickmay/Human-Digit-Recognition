import os
import json
import cv2
import numpy as np
import albumentations as A

if __name__ == '__main__':
    image = cv2.imread("data/FULL_IMAGES/test1.jpg")
    cropped_image = image[600:1000, 100:350]
    cv2.imwrite("data/FULL_IMAGES/test1.5.jpg", cropped_image)
