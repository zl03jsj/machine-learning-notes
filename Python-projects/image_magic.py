# -*- coding: UTF-8 -*-
import cv2
import numpy as np


imgfile = "bird.jpg"
birdImg = cv2.imread(imgfile)

# 图像锐化滤波器Sharpness Filter
sharpness_filter_3_3 = np.linspace(-1, -1, 9).reshape(3, 3)
sharpness_filter_3_3[1, 1] = 9
print sharpness_filter_3_3

sharpness_filter_5_5 = np.linspace(-1, -1, 25).reshape(5, 5)
sharpness_filter_5_5[1:4, 1:4] = 2
sharpness_filter_5_5[2, 2] = 8
print sharpness_filter_5_5

def multiple_sum(a, b):
    return (a * b).sum()


def gaussianImage(img, filter) :
    for x in np.arange(0, img.shape[0]) :

        for y in np.arange(0, img.shape[1]) :


