import numpy as np

import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image


def ege_canny():
    low_threshold = 120
    max_low_threshold = 255
    ratio = 3
    kernel_size = 3

    img = cv.imread('img/origin/004.bmp')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    def canny_threshold(cur_threshold):
        detected_edges = cv.GaussianBlur(gray, (3, 3), 0)
        detected_edges = cv.Canny(detected_edges, cur_threshold,
                                  cur_threshold * ratio, apertureSize=kernel_size)
        # just add some colours to edges from original image.
        dst = cv.bitwise_and(img, img, mask=detected_edges)
        cv.imshow('canny demo', dst)

    cv.namedWindow('canny demo')
    cv.createTrackbar('Min threshold', 'canny demo', low_threshold, max_low_threshold, canny_threshold)

    canny_threshold(low_threshold)  # initialization

    if cv.waitKey(0) == 27:
        cv.destroyAllWindows()


if __name__ == '__main__':
    ege_canny()


# def fun2(img):
#     mser = cv.MSER_create(_min_area=300)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     regions, boxes = mser.detectRegions(gray)
#     for box in boxes:
#         if box[2] > 60 or box[3] > 60: continue
#         x, y, w, h = box
#         cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#     plt.imshow(img, 'brg')
#     plt.show()
#
#
# def fun1():
#     im = cv.imread('img/origin2/007.bmp', flags=cv.IMREAD_COLOR)
#     # cv.imshow('original image', im)
#
#     # im = cv.GaussianBlur(im, (3, 3), 3.0)
#     # cv.imshow("GaussianBlur", im)
#     # cv.waitKey()
#
#     gray = np.array(im)
#     cv.threshold(im, 7, 255, cv.THRESH_BINARY_INV, gray)
#     # cv.imshow("thresholds", gray)
#
#     gray = cv.cvtColor(gray, code=cv.COLOR_BGR2GRAY)
#     cv.imshow('original image convert to gray', gray)
#
#     dst = cv.Laplacian(gray, ddepth=cv.CV_32F, ksize=3)
#     cv.imshow('gray laplaced to dst', dst)
#
#     cv.convertScaleAbs(dst, gray)
#     cv.imshow('dst convert scale to gray', gray)
#     cv.waitKey()
#
#     # print(cv.__version__)
#     mser = cv.MSER_create(_min_area=50)
#     regions, boxes = mser.detectRegions(gray)
#
#     for box in boxes:
#         if (box[2] > 60 or box[2] < 20) or (box[3] > 60 or box[3] < 20): continue
#         x, y, w, h = box
#         cv.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
#     cv.imshow('detected regions', im)
#     cv.waitKey()
#
#
# fun1()

# import cv2.cv as cv
# import cv2
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
#     mser = cv2.MSER(_min_area=300)
#
#     cvmat = cv.GetMat(gray)
#     gray_arr = np.asarray(cvmat)
#     regions = mser.detect(gray_arr)
#
#     hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
#
#     cv2.polylines(gray, hulls, 1, (0, 255, 0))
#
#     # for box in boxes:
#     #     if box[2] > 60 or box[3] > 60: continue
#     #     x, y, w, h = box
#     #     cv.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
#
#     plt.imshow(gray, 'brg')
#     plt.show()
#     # ------------------------------------
#     # Laplace on color
#     # thresholded = cv.CloneImage(im)
#     # cv.Threshold(im, thresholded, 50, 255, cv.CV_THRESH_BINARY_INV)
#     # planes = [cv.CreateImage(cv.GetSize(im), 8, 1) for i in range(3)]
#     # laplace = cv.CreateImage(cv.GetSize(im), cv.IPL_DEPTH_16S, 1)
#     # colorlaplace = cv.CreateImage(cv.GetSize(im), 8, 3)
#     #
#     # cv.Split(im, planes[0], planes[1], planes[2], None)  # Split channels to apply laplace on each
#     # for plane in planes:
#     #     cv.Laplace(plane, laplace, 3)
#     #     cv.ConvertScaleAbs(laplace, plane, 1, 0)
#
#     # cv.Merge(planes[0], planes[1], planes[2], None, colorlaplace)
#
#     # cv.ShowImage('Laplace Color', colorlaplace)
#     # -------------------------------------
#     # cv.WaitKey(0)
#
#
# fun1()
