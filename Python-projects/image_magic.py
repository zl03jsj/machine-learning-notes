# -*- coding: UTF-8 -*-
import cv2
import numpy as np


# 图像锐化滤波器Sharpness Filter
sharpness_filter_3_3 = np.linspace(-1, -1, 9).reshape(3, 3)
sharpness_filter_3_3[1, 1] = 9
print sharpness_filter_3_3

sharpness_filter_5_5 = np.linspace(-1, -1, 25).reshape(5, 5)
sharpness_filter_5_5[1:4, 1:4] = 2
sharpness_filter_5_5[2, 2] = 8
print sharpness_filter_5_5


def multiple_sum(f, sub_img):
    print "-------------------"
    print f
    print sub_img
    r = (f * sub_img[:, :, 0]).sum()
    g = (f * sub_img[:, :, 1]).sum()
    b = (f * sub_img[:, :, 2]).sum()
    color = np.array([r, g, b], dtype=np.uint8)
    print "-------------------"
    return color


def do_image_magic(img, f):
    print '-----------do_image_magic-----------'

    print 'img shape = ', img.shape
    print 'filter cub shape = ', f.shape

    fx = f.shape[0]
    fy = f.shape[1]

    if 0 == (fx % 2) or 0 == (fy % 2):
        print "filter cub shape can not be even,should be odd number!!!!"
        return

    fx /= 2
    fy /= 2

    tmp = np.arange(0, img.shape[1], dtype=object)
    for x in np.arange(0, img.shape[0]):
        if x > 0:
            img[0, 0:img.shape[1]] = tmp

        begin_x, end_x = x-fx, x+fx

        if begin_x < 0: begin_x = 0
        if end_x > img.shape[0]: end_x = img.shape[0]

        begin_fx = f.shape[0] - (end_x - begin_x + 1)
        end_fx = begin_fx + (end_x - begin_x + 1)

        for y in np.arange(0, img.shape[1]):
            begin_y, end_y = y-fy, y+fy
            if begin_y < 0: begin_y = 0
            if end_y > img.shape[1]: end_y = img.shape[y]

            begin_fy = f.shape[1] - (end_y - begin_y + 1)
            end_fy = begin_fy + (end_y - begin_y + 1)

            print "filter range:([%d-%d], [%d-%d]), img range:([%d-%d], [%d-%d])" % (begin_fx, end_fx, begin_fy, end_fy, begin_x, end_x, begin_y, end_y)

            sub_filter = f[begin_fx:end_fx + 1, begin_fy:end_fy + 1]
            sub_img = img[begin_x:end_x + 1, begin_y:end_y + 1]

            tmp[y] = multiple_sum(sub_filter, sub_img)
    print '-----------do_image_magic-----------'

imagefile = "bird.jpg"
birdimage = cv2.imread(imagefile)

do_image_magic(birdimage, sharpness_filter_3_3)


cv2.imshow("image magic show!!!!", birdimage)
cv2.waitKey(6*1000)
cv2.destroyAllWindows()
