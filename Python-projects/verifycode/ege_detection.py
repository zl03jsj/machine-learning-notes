import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image


def fun2(img):
    mser = cv.MSER_create(_min_area=300)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    regions, boxes = mser.detectRegions(gray)
    for box in boxes:
        if box[2] > 60 or box[3] > 60: continue
        x, y, w, h = box
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    plt.imshow(img, 'brg')
    plt.show()

def fun1():
    im = cv.imread('img/origin2/001.jpg', flags=cv.IMREAD_COLOR)
    cv.imshow('original image', im)

    thresholds = np.array(im)
    cv.threshold(im, 120, 255, cv.THRESH_BINARY_INV, thresholds)
    cv.imshow("thresholds", thresholds)
    cv.waitKey()

    original_to_gray = cv.cvtColor(thresholds, code=cv.COLOR_BGR2GRAY)
    gray = np.array(original_to_gray)
    cv.imshow('original image convert to gray', gray)

    dst = cv.Laplacian(gray, ddepth=cv.CV_32FC3, ksize=3)
    cv.imshow('gray laplaced to dst', dst)

    cv.convertScaleAbs(dst, gray)
    cv.imshow('dst convert scale to gray', gray)

    mser = cv.MSER_create(_min_area=70)
    regions, boxes = mser.detectRegions(gray)
    for box in boxes:
        if (box[2] > 50 or box[2] <20) or (box[3] > 50 or box[3] <20) : continue
        x, y, w, h = box
        cv.rectangle(original_to_gray, (x, y), (x + w, y + h), (255, 0, 0), 2, )
    cv.imshow('final image', original_to_gray)
    cv.waitKey()

#
#
fun1()

# import cv2
# import cv2.cv as cv
#
# def fun1():
#     im = cv.LoadImage('img/origin2/002.jpg', cv.CV_LOAD_IMAGE_COLOR)
#     # Laplace on a gray scale picture
#     gray = cv.CreateImage(cv.GetSize(im), 8, 1)
#     cv.CvtColor(im, gray, cv.CV_BGR2GRAY)
#
#     aperture = 3
#
#     dst = cv.CreateImage(cv.GetSize(gray), cv.IPL_DEPTH_32F, 1)
#     cv.Laplace(gray, dst, aperture)
#
#     cv.ConvertScale(dst, gray)
#     cv.ShowImage('dst convert to gray', gray)
#
#     # ------------------------------------
#     # Laplace on color
#     thresholded = cv.CloneImage(im)
#     cv.Threshold(im, thresholded, 50, 255, cv.CV_THRESH_BINARY_INV)
#     planes = [cv.CreateImage(cv.GetSize(im), 8, 1) for i in range(3)]
#     laplace = cv.CreateImage(cv.GetSize(im), cv.IPL_DEPTH_16S, 1)
#     colorlaplace = cv.CreateImage(cv.GetSize(im), 8, 3)
#
#     cv.Split(im, planes[0], planes[1], planes[2], None)  # Split channels to apply laplace on each
#     for plane in planes:
#         cv.Laplace(plane, laplace, 3)
#         cv.ConvertScaleAbs(laplace, plane, 1, 0)
#
#     cv.Merge(planes[0], planes[1], planes[2], None, colorlaplace)
#
#     cv.ShowImage('Laplace Color', colorlaplace)
#     # -------------------------------------
#     cv.WaitKey(0)
#
#
# fun1()
