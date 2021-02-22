import numpy as np
import pytz
import math
from math import isnan
import cv2
import datetime

def zero2one(num):
    """
    Function Description: Replace 0 value to 1 for division
    """
    if isnan(num):
        print("Number is NaN ...")

    if num == 0:
        return 1.0
    else:
        return num

def almostzero(num):
    """
    Function Description: Replace 0 value to a very close to zero value
    """
    if isnan(num):
        print("Number is NaN ...")

    if num == 0:
        return 0.00000001
    else:
        return num


def draw_point(img, p, color = (0, 0, 255)):
    """
    Function Description: Draw a filled point p to an image img
    """
    cv2.circle(img, p, 2, color, -1, cv2.LINE_AA, 0)


def draw_rect(img, rect, color = (0, 255, 0)):
    """
    Function Description: Draw a bounding box on image img
    """
    cv2.rectangle(img,(rect[0],rect[1]),(rect[2],rect[3]),(0,255,0),2)


def draw_centroid(img, rect, color = (255, 255, 0)):
    """
    Function Description: Draw centroid of rectangle
    """
    x = np.int0(rect[2] - rect[0])/2
    y = np.int0(rect[3] - rect[1])/2
    cv2.circle(img, (x, y), 2, color, -1, cv2.LINE_AA, 0)


def is_in_rect(rect, point):
    """
    Function Description: Check if a point is inside a rectangle
    """
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def is_in_circle(c, R, point):
    """
    Function Description: Check if a point is inside a circle of center c and radius R
    """
    if pow(point[0] - c[0], 2) + pow(point[1] - c[1], 2) < pow(R, 2):
        return True
    return False


def five_num_sum(data):
    """
    Function Description : calculates the 5 num statistical summary of data
    """
    quartiles = np.percentile(data, [25, 50, 75])
    return [data.min(), quartiles[0], quartiles[1], quartiles[2], data.max()]


def get_time():
    """
    Function Description : returns the current UTC time
    """
    cur_time = pytz.utc.localize(datetime.datetime.utcnow())
    return cur_time