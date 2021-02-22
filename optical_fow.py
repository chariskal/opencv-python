#!/usr/bin/python
"""
############################################################################
# Description:
# Author: Charis Kalavritinos
############################################################################
"""

# pylint: disable=line-too-long
import sys
import argparse
import os
import joblib
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from numpy import uint8
from numpy import float32
from scipy import stats



class SparseOFlow:
    """
    Class description: custom Sparse Optical Flow (OF) using a predefined grid and Lucas-Kanade
    NOTES:
    """

    def __init__(self, img, step=10):
        height, width = img.shape[:2]
        y, x = np.mgrid[step/2:height:step, step/2:width:step].reshape(2, -1).astype(int)
        y = y.astype(int)
        x = x.astype(int)
        # Parameters for Lucas-Kanade optical flow algorithm
        self.lk_params = dict(winSize=(16, 16), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        self.new_points = []

        temp = np.vstack([x, y]).T
        self.old_points = np.zeros([x.shape[0], 1, 2])
        self.old_points[..., 0] = temp[..., 0].reshape(x.shape[0], 1)
        self.old_points[..., -1] = temp[..., -1].reshape(x.shape[0], 1)
        self.old_points = np.array(self.old_points, dtype=float32)

    def compute_flow(self, frame_processed_old, frame_processed):
        """
        Method Description: calculate OF using LK Image pyramids
        """
        self.new_points = cv.calcOpticalFlowPyrLK(frame_processed_old, frame_processed, self.old_points, None, **self.lk_params)

    def draw_flow(self, img, clip=32, quiver=(0, 0, 255)):
        """
        Method Description: OF grid visualization

        """
        xnew = self.new_points[:, 0, 0]
        xold = self.old_points[:, 0, 0]
        ynew = self.new_points[:, 0, 1]
        yold = self.old_points[:, 0, 1]

        for i in range(xnew.shape[0]):
            if xold[i]-xnew[i] > clip:
                xnew[i] = xold[i] + clip

            elif xold[i]-xnew[i] < -clip:
                xnew[i] = xold[i] -clip

            if yold[i]-ynew[i] > clip:
                ynew[i] = yold[i] + clip

            elif yold[i]-ynew[i] < -clip:
                ynew[i] = yold[i] -clip

        vis = img.copy()
        lines = np.vstack([xold, yold, xnew, ynew]).T.reshape(-1, 2, 2)   # sth*2*2
        lines = np.int32(lines + 0.5)
        cv.polylines(vis, lines, 0, quiver)
        for (x1, y1), (_x2, _y2) in lines:
            cv.circle(vis, (x1, y1), 1, (0, 0, 255), -1)
        #    cv.circle(vis, (_x2, _y2), 1, (255, 0, 255), -1)
        #xnew=np.int32(xnew + 0.5)
        #ynew=np.int32(ynew + 0.5)

        #for i in range(xnew.shape[0]):
        #    cv.circle(vis, (xnew[i],ynew[i]),1,(0,255,0),-1)

        #cv.destroyAllWindows()
        cv.imshow('flow', vis)
        return xnew, ynew, xold, yold


class DenseOFlow:
    """
    Class description: dense OF using Farneback.
    NOTES:  old_points are always fixed at the same fixed position

    """
    def __init__(self, img, step=10):
        height, width = img.shape[:2]
        self.y, self.x = np.mgrid[step/2:height:step, step/2:width:step].reshape(2, -1).astype(int)
        self.y = self.y.astype(int)
        self.x = self.x.astype(int)
        self.flow = []

    def compute_flow(self, frame_processed_old, frame_processed):
        """
        Method Description: Calculate OF using Dense Farneback
        """
        self.flow = cv.calcOpticalFlowFarneback(frame_processed_old, frame_processed, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    def draw_flow(self, img, quiver=(0, 0, 255)):
        """
        Method Description : OF grid visualization
        """
        fx, fy = self.flow[self.y, self.x].T
        xnew = self.x + fx
        ynew = self.y + fy
        lines = np.vstack([self.x, self.y, self.x + fx, self.y + fy]).T.reshape(-1, 2, 2)   # sth*2*2
        lines = np.int32(lines + 0.5)
        vis = img                                                   #cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv.polylines(vis, lines, 0, quiver)
        for (x1, y1), (_x2, _y2) in lines:
            cv.circle(vis, (x1, y1), 1, (0, 0, 255), -1)

        #magnitude, angle = cv.cartToPolar(self.flow[..., 0], self.flow[..., 1])
        #hsv[..., 0] = angle * 180 / np.pi / 2          # Hue = angle
        #hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX) # magnitude = value
        #rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        #cv.imshow("Optical flow in HSV Colorspace: Hue=Angle, Value=Magnitude", rgb)

        cv.imshow('flow', vis)
        return xnew, ynew, self.x, self.y

