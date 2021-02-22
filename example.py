#!/usr/bin/python
"""
############################################################################
# Description:
#
# Author
############################################################################

"""
# pylint: disable=line-too-long
import time
import csv
import sys
import argparse
import os
import json
import datetime
from math import log2
import pytz
import joblib
import requests

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from numpy import uint8
from numpy import float32
from scipy import stats
from imutils.video import FileVideoStream
from basic_functions import get_time
from basic_functions import five_num_sum
from optical_fow import DenseOFlow
from optical_fow import SparseOFlow

def processing(frame, width=420, height=300):
    """
    Function Description : image processing techniques for each video frame
    """
    def adjust_gamma(frame, gamma=1.8):
        inv_gamma = 1.0/gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv.LUT(frame, table)

    kernel = np.ones((5, 5), uint8)                                     # kernel for morph.opening

    frame_resized = cv.resize(frame, (width, height))                   # resize the frame
    frame_grayscale = cv.cvtColor(frame_resized, cv.COLOR_BGR2GRAY)     # grayscale frame
    frame_gamma_adjusted = adjust_gamma(frame_grayscale)                # adjust the gamma

    frame_opened = cv.morphologyEx(frame_gamma_adjusted, cv.MORPH_OPEN, kernel)

    frame_smoothed = cv.medianBlur(frame_opened, 5)                     # reduce noise using median filtering
    frame_eq = cv.equalizeHist(frame_opened)                            # Histogram equalization
    mn = int(np.mean(frame_eq))                                         # mean extraction
    frame_mean_extracted = cv.subtract(frame_eq, mn*np.ones_like(frame_eq))
    return frame_resized, frame_smoothed


def main():
    """
    Function Description : main function

    """
    # Argument Parser. Handle arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("--display", type=bool, default=True,
                    help="[bool]Display mutliple figures. Default: True")
    ap.add_argument("--livestream", type=bool, default=False,
                    help="[bool]Use web cam or video. Default: False")
    ap.add_argument("--save_fgs", type=bool, default=False,
                    help="[bool]Save figures into files. Default: False")
    ap.add_argument("--flow", type=str, choices=['dense', 'sparse'], default='dense',
                    help="[str]Optical flow argument. Accepted values: dense, sparse")
    ap.add_argument("-o", "--output", type=bool, default=False,
                    help="[bool]Save video output as file. Default: False")
    ap.add_argument("-i", "--source", type=str, default="./source/videos/demo.mp4",
                    help="[str]Path to video. Default:'./source/videos/demo.mp4'")
    args = vars(ap.parse_args())

    display_fgrs = args["display"]
    live_or_video = args["livestream"]          # choose input source
    save_fgrs = args["save_fgs"]
    save_vid = args["output"]
    dense_or_sparse = args["flow"]

    source_path = args["source"]

    fvs_or_cap = 0
    print('Call with -h or --help for help on how to run...')
    print("#")
    print("# Video: ", source_path)
    print("#")

    print('###########  Arguments  ############')
    print("   Display Figures: ", display_fgrs)
    print("   Save figures to files: ", save_fgrs)
    print("")
    print("Press 'Esc' anytime to close windows ...")
    print("")

    if fvs_or_cap == 0:
        if live_or_video:
            cap = cv.VideoCapture(0)
        else:
            cap = cv.VideoCapture(source_path)
        if not cap.isOpened():                                  # check for errors while opening vidcapture
            print("VideoCapture failed...")
    else:
        fvs = FileVideoStream(source_path).start()
        time.sleep(1.0)

    if fvs_or_cap == 0:
        ret, frame0 = cap.read()
        if not ret:
            print("Error opening frame!")
    else:
        frame0 = fvs.read()                                     # get first frame

    frame0_resized, frame_processed_old = processing(frame0)    # image processing for better results

    # Parameters
    step = 10
    height, width = frame0_resized.shape[:2]

    cnt = 0                                             # counts frames
    font = cv.FONT_HERSHEY_SIMPLEX                      # font of above message
    clr = (255, 255, 255)                               # color of message text

    CE_total = []
    D_total = []
    magnitude = np.zeros([round(height/step), round(width/step)])
    angle = np.zeros([round(height/step), round(width/step)])
    Ex = np.zeros([0, round(width/step)])
    Ey = np.zeros([round(height/step), 0])

    if save_vid:
        out = cv.VideoWriter('outpy.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (896, 504))

    while cap.isOpened():
    #while fvs.more():                                  # main video reading loop !
        ret, frame = cap.read()
        #frame = fvs.read()                             # read frame
        if np.shape(frame) == ():
            continue
        frame_resized, frame_processed = processing(frame)
        cnt = cnt+1

        # Create the flow object ...
        if dense_or_sparse == 'dense':
            flow_object = DenseOFlow(frame_resized.copy())
        else:
            flow_object = SparseOFlow(frame_resized.copy())
        flow_object.compute_flow(frame_processed_old, frame_processed)
        xnew, ynew, xold, yold = flow_object.draw_flow(frame_resized)

        # Manually calculate OF magnitude and angle
        x = xnew-xold
        y = ynew-yold
        magn = np.sqrt(x*x+y*y)
        x[x == 0] = 0.001                                   # replace zero values to very small number
        ang = np.arctan2(y, x)

        i = 0
        for yi in range(round(height/step)):
            for xi in range(round(width/step)):
                magnitude[yi, xi] = magn[i]
                angle[yi, xi] = ang[i]
                i = i+1

        magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)     # normalize values
        if magnitude.max() == 0:
            magn_norm = magnitude
        else:
            magn_norm = magnitude/(1.0*magnitude.max())
        angle = angle * 180 / np.pi / 2

        # Calculate magnitude and angle histograms
        [magn_hist, mbins] = np.histogram(magn_norm, bins=20, range=[0, 1], density=None)
        [angle_hist, abins] = np.histogram(angle, bins=36, range=[-180, 180], density=None)

        if False:                                       # plot the histograms
            plt.figure()
            width = 0.8*(mbins[1]-mbins[0])
            center = (mbins[:-1]+mbins[1:])/2
            plt.bar(center, magn_hist, align='center', width=width)
            plt.figure()
            width = 0.8*(abins[1]-abins[0])
            center = (abins[:-1]+abins[1:])/2
            plt.bar(center, angle_hist, align='center', width=width)
            plt.show()
        # Find the trimmed mean of magnitude in frame
        magn_mean = stats.trim_mean(magnitude, 0.1)
        m = np.mean(magn_norm)

        # Calculate the x,y magnitude positional histograms based on the previously calculated mean
        temp = magn_norm.copy()
        temp[temp <= m] = 0
        temp[temp > m] = 1
        positional_x = np.sum(temp, axis=0)
        positional_y = np.sum(temp, axis=1)


        # Create statistical joint histograms
        hsa = np.zeros([mbins.shape[0]-1, abins.shape[0]-1])
        for i in range(round(height/step)):
            for j in range(round(width/step)):
                pos1 = int(20*(np.floor(max(0, magn_norm[i, j]-0.00001)/0.05)*0.05))
                pos2 = int((np.floor(max(-180, angle[i, j]-0.00001)/5)*5)/5)
                #print('magn:',magn_norm[i,j],'pos1:',pos1,'pos2:',pos2)
                hsa[pos1, pos2] = hsa[pos1, pos2]+1

        psa = hsa/sum(sum(hsa))*100
        psa[psa == 0] = 0.00000000000000001
        D = 0
        for i in range(psa.shape[0]):
            for j in range(psa.shape[1]):
                D = D+(psa[i, j]*log2(psa[i, j]))
        D = -1*D

        hor = round(height/step/3)
        ver = round(width/step/3)
        hxy = np.zeros([hor, ver])
        for i in range(0, int(height/step), 3):
            for j in range(0, int(width/step), 3):
                hxy[int(i/3), int(j/3)]=sum(sum(temp[i:i+2, j:j+2]))
                #print(int(i/3),int(j/3))
                #print(i,j)
        pxy = hxy/sum(sum(temp))*100
        pxy[pxy == 0] = 0.00000000000000001
        CE = 0
        for i in range(pxy.shape[0]):
            for j in range(pxy.shape[1]):
                CE = CE+pxy[i, j]*log2(pxy[i, j])
        CE = -1*CE
        BMFS = -1*(D+100)*(D+100)*CE
        PAOF = (D+100)*(D+100)/CE

        CE_total = np.append(CE_total, CE)
        D_total = np.append(D_total, D)
        #print('CE',CE,'D',D)

        # Calculate particle Entropy using the positional x,y magn
        fx = positional_x/sum(positional_x)
        fy = positional_y/sum(positional_y)
        fx[fx == 0] = 0.000001
        fy[fy == 0] = 0.000001
        Ex_tmp = -1*fx*[log2(i) for i in fx]
        Ey_tmp = -1*fy*[log2(i) for i in fy]

        Ex_tmp = Ex_tmp.reshape([Ex_tmp.shape[0], 1]).T
        Ex = np.append(Ex, Ex_tmp, axis=0)
        Ey_tmp = Ey_tmp.reshape([Ey_tmp.shape[0], 1])
        Ey = np.append(Ey, Ey_tmp, axis=1)

        # Store the means of the frames in a 4 sec time period

        if display_fgrs:
            magn = cv.resize(magnitude, (8*magnitude.shape[1], 8*magnitude.shape[0]), interpolation=cv.INTER_NEAREST)
            cv.imshow("Optical flow Magnitude", cv.applyColorMap(magn.astype(uint8), cv.COLORMAP_JET))
            angl = cv.normalize(angle, None, 0, 255, cv.NORM_MINMAX)
            angl = cv.resize(angl, (8*angl.shape[1], 8*angl.shape[0]), interpolation=cv.INTER_NEAREST)
            cv.imshow("Optical flow Angle", cv.applyColorMap(angl.astype(uint8), cv.COLORMAP_JET))
            psa2 = cv.normalize(psa, None, 0, 255, cv.NORM_MINMAX)
            psa2 = cv.resize(psa2, (8*psa2.shape[1], 8*psa2.shape[0]), interpolation=cv.INTER_NEAREST)
            pxy2 = cv.normalize(pxy, None, 0, 255, cv.NORM_MINMAX)
            pxy2 = cv.resize(pxy2, (8*pxy2.shape[1], 8*pxy2.shape[0]), interpolation=cv.INTER_NEAREST)
            cv.imshow("Pxy: X axis vs Y axis", cv.applyColorMap(pxy2.astype(uint8), cv.COLORMAP_JET))
            cv.imshow("Psa: Magnitude vs Angle", cv.applyColorMap(psa2.astype(uint8), cv.COLORMAP_JET))

            cv.imshow('processed', frame_processed)
            img = cv.putText(cv.resize(frame, (896, 504)), (40, 460), font, 1.2, clr, 2, cv.LINE_AA)
            if save_vid:
                out.write(img)
            cv.imshow("Swimming Behaviour", img)

        frame_processed_old = frame_processed.copy()
        k = cv.waitKey(30) & 0xff                   # if 'Esc' is pressed then quit
        if k == 27:
            break

    Ex = cv.normalize(Ex, None, 0, 255, cv.NORM_MINMAX)
    Ex = cv.resize(Ex, (2*Ex.shape[1], 2*Ex.shape[0]), interpolation=cv.INTER_NEAREST)
    cv.imshow('Positional Entropy Ex', cv.applyColorMap(Ex.astype(uint8), cv.COLORMAP_JET))

    Ey = cv.normalize(Ey, None, 0, 255, cv.NORM_MINMAX)
    Ey = cv.resize(Ey, (2*Ey.shape[1], 2*Ey.shape[0]), interpolation=cv.INTER_NEAREST)
    cv.imshow('Positional Entropy Ey', cv.applyColorMap(Ey.astype(uint8), cv.COLORMAP_JET))

    plt.figure()
    plt.title('High D: Intense emergent behavior; Low D: Normal behavior')
    plt.ylabel('Entropy of magnitude corresponding to angle (D)')
    plt.xlabel('frames')
    plt.plot(D_total, label='D', color='red')

    plt.figure()
    plt.title('High CE: Scattering; Low CE: Gathering')
    plt.ylabel('Crowd Entropy (CE)')
    plt.xlabel('frames')
    plt.plot(CE_total, label='CE', color='green')


    plt.show()
    cv.destroyAllWindows()                          # close all windows opencv has opened

    print("End ...")
    if save_vid:
        out.release()
    cap.release()                                   # release capture handle

if __name__ == '__main__':
    main()
