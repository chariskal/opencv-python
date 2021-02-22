import cv2
import numpy as np
from advanced_functions import find_contour_features

img = cv2.imread("img.jpg")
img, centroids, _,_,_ = find_contour_features(img)

cv2.imshow("image", img)
cv2.waitKey(0)

