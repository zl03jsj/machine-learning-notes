import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image

res_path = 'img/origin2/'

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


def canny__(img):
    gray = cv.cvtColor(img, code=cv.COLOR_BGR2GRAY)
    detected_edges = cv.GaussianBlur(gray, (3, 3), 0)
    detected_edges = cv.Canny(detected_edges, 120, 120 * 3, apertureSize=3)
    # just add some colours to edges from original image.
    img = cv.bitwise_and(img, img, mask=detected_edges)
    return img


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


def laplace__(img):
    thresholds = np.array(img)
    cv.threshold(img, 110, 150, cv.THRESH_BINARY_INV, thresholds)
    original_to_gray = cv.cvtColor(thresholds, code=cv.COLOR_BGR2GRAY)
    gray = np.array(original_to_gray)
    dst = cv.Laplacian(gray, ddepth=cv.CV_32FC3, ksize=3)
    cv.convertScaleAbs(dst, gray)

def do_ege_detect():
    img = cv.imread('img/origin2/003.jpg', flags=cv.IMREAD_COLOR)
    img = canny__(img)
    gray = cv.cvtColor(img, code=cv.COLOR_BGR2GRAY)

    sub_image = gray[0:50, 230:gray.shape[1]-10]
    cv.imwrite(res_path + 'sub_image_1.jpg', sub_image)

    sub_image= gray[50:gray.shape[0], 0:gray.shape[1]]
    cv.imwrite(res_path + 'sub_image_2.jpg', sub_image)

    cv.imshow('image', gray)
    cv.waitKey()


if __name__ == '__main__':
    # show_canny()
    do_ege_detect()
