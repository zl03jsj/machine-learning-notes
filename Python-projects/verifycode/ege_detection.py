import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image

res_path = 'img/origin/'


def show_canny():
    low_threshold = 50
    max_low_threshold = 255
    ratio = 3
    kernel_size = 3
    img = cv.imread('img/origin2/002.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    def canny_threshold(cur_threshold):
        detected_edges = cv.GaussianBlur(gray, (3, 3), 0)
        detected_edges = cv.Canny(detected_edges, cur_threshold, cur_threshold * ratio, apertureSize=kernel_size)
        dst = cv.bitwise_and(img, img, mask=detected_edges)  # just add some colours to edges from original image.
        cv.imshow('canny demo', dst)

    cv.namedWindow('canny demo')
    cv.createTrackbar('Min threshold', 'canny demo', low_threshold, max_low_threshold, canny_threshold)

    canny_threshold(low_threshold)  # initialization
    if cv.waitKey(0) == 27:
        cv.destroyAllWindows()


def mser_text_detect(img):
    gray = cv.cvtColor(img, code=cv.COLOR_BGR2GRAY)
    mser = cv.MSER_create(_min_area=250)
    regions, boxes = mser.detectRegions(gray)
    for box in boxes:
        x, y, w, h = box
        if w > 60 or w < 20 or h > 60 or h < 20: continue
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=1)
    cv.imshow('final image', img)
    cv.waitKey()


def canny__(img):
    gray = cv.cvtColor(img, code=cv.COLOR_BGR2GRAY)
    cv.imshow('gray', gray)
    detected_edges = cv.GaussianBlur(gray, (3, 3), 0)
    # detected_edges = gray
    detected_edges = cv.Canny(detected_edges, 120, 120 * 3, apertureSize=3)
    # just add some colours to edges from original image.
    # img = cv.bitwise_and(img, img, mask=detected_edges)
    return detected_edges


def laplace__(img):
    thresholds = np.array(img)
    cv.threshold(img, 110, 150, cv.THRESH_BINARY_INV, thresholds)
    original_to_gray = cv.cvtColor(thresholds, code=cv.COLOR_BGR2GRAY)
    gray = np.array(original_to_gray)
    dst = cv.Laplacian(gray, ddepth=cv.CV_32FC3, ksize=3)
    cv.convertScaleAbs(dst, gray)


def do_ege_detect():
    img = cv.imread(res_path + '003.jpg', flags=cv.IMREAD_COLOR)
    img = canny__(img)
    gray = cv.cvtColor(img, code=cv.COLOR_BGR2GRAY)

    sub_image = gray[0:50, 230:gray.shape[1] - 10]
    cv.imwrite(res_path + 'sub_image_1.jpg', sub_image)

    sub_image = gray[50:gray.shape[0], 0:gray.shape[1]]
    cv.imwrite(res_path + 'sub_image_2.jpg', sub_image)

    cv.imshow('image', gray)
    cv.waitKey()


if __name__ == '__main__':
    # show_canny()
    do_ege_detect()



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
