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
    r = (f * sub_img[:, :, 0]).sum()
    g = (f * sub_img[:, :, 1]).sum()
    b = (f * sub_img[:, :, 2]).sum()
    color = np.array([r, g, b], dtype=np.uint8)
    return color


def do_image_magic(img, f):
    print '-----------do_image_magic-----------'

    print 'img shape = ', img.shape
    print 'filter cub shape = ', f.shape

    if f.shape[0]!=f.shape[1] or 0==(f.shape[0] % 2):
        print "filter cub shape can not be even,should be odd number!!!!"
        return

    r = f.shape[0]/2

    tmp = np.zeros((img.shape[1] * (r+1), 3), dtype=np.uint8).reshape(r+1, img.shape[1], 3)
    print tmp.shape
    tx = 0

    for x in np.arange(0, img.shape[0]):
        begin_x, end_x = x-r, x+r

        if begin_x < 0: begin_x = 0
        if end_x >= img.shape[0]: end_x = img.shape[0] - 1

        begin_fx = f.shape[0] - (end_x - begin_x + 1)
        end_fx = begin_fx + (end_x - begin_x + 1)

        tx = x if x <= r else r
        for y in np.arange(0, img.shape[1]):
            begin_y, end_y = y-r, y+r
            if begin_y < 0: begin_y = 0
            if end_y >= img.shape[1]: end_y = img.shape[1] - 1

            begin_fy = f.shape[1] - (end_y - begin_y + 1)
            end_fy = begin_fy + (end_y - begin_y + 1)

            sub_filter = f[begin_fx:end_fx + 1, begin_fy:end_fy + 1]
            sub_img = img[begin_x:end_x + 1, begin_y:end_y + 1]

            tmp[tx, y, :] = multiple_sum(sub_filter, sub_img)

        if x >= r:
            img[x-r, 0:img.shape[1], :] = tmp[0, 0:tmp.shape[1], :]
            for i in [1, r]:
                tmp[i-1, 0:tmp.shape[1], :] = tmp[i, 0:tmp.shape[1], :]

    for x in np.arange(img.shape[0]-r, img.shape[0]):
        img[x, 0:img.shape[1], :] = tmp[img.shape[0]-x, 0:tmp.shape[1], :]

    print '-----------do_image_magic-----------'

imagefile = "snipaste_20170714_162153.jpg"
birdimage = cv2.imread(imagefile)

do_image_magic(birdimage, sharpness_filter_3_3)

cv2.imshow("image magic show!!!!", birdimage)
cv2.waitKey()
cv2.destroyAllWindows()
