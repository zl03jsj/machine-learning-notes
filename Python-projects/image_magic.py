# -*- coding: UTF-8 -*-
import cv2
import numpy as np

# 图像锐化滤波器Sharpness Filter
sharpness_filter_1_1 = np.linspace(0, 0, 9).reshape(3, 3)
sharpness_filter_1_1[1, 1] = 1
print sharpness_filter_1_1

sharpness_filter_3_3 = np.linspace(-1, -1, 9).reshape(3, 3)
sharpness_filter_3_3[1, 1] = 8
print sharpness_filter_3_3

sharpness_filter_5_5 = np.linspace(-1, -1, 25).reshape(5, 5)
sharpness_filter_5_5[1:4, 1:4] = 2
sharpness_filter_5_5[2, 2] = 8
print sharpness_filter_5_5

def multiple_sum(f, sub_img):
    r = min(abs((f * sub_img[:, :, 0]).sum()), 255)
    g = min(abs((f * sub_img[:, :, 1]).sum()), 255)
    b = min(abs((f * sub_img[:, :, 2]).sum()), 255)
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

    tmp = np.zeros((img.shape[0] * (r+1), 3), dtype=np.uint8).reshape(img.shape[0], r+1,  3)
    tx = 0

    for x in np.arange(0, img.shape[1]):
        begin_x = max(0, x-r)
        end_x = min(img.shape[1]-1, x+r)

        begin_fx = max(0, r-x)
        end_fx = min(f.shape[1]-1, (img.shape[1] - x))

        tx = x if x <= r else r
        print "x = %d, tx=%d" %(x, tx)

        for y in np.arange(0, img.shape[0]):
            begin_y = max(0, y-r)
            end_y = min(img.shape[1]-1, y+r)

            begin_fy = max(0, r-y)
            end_fy = min(f.shape[0]-1, (img.shape[0] - y))

            sub_filter = f[begin_fy:end_fy + 1, begin_fx:end_fx + 1]
            sub_img = img[begin_y:end_y + 1, begin_x:end_x + 1]

            tmp[y, tx, :] = multiple_sum(sub_filter, sub_img)

        if x >= r:
            print "copy pixels from (%d:%d, %d) to img(%d:%d, %d)" %(0, tmp.shape[0], 0, 0, img.shape[0], x-r)
            img[0:img.shape[0], x-r, :] = tmp[0:tmp.shape[0], 0, :]
            for i in np.arange(1, r+1):
                tmp[0:tmp.shape[0], i-1, :] = tmp[0:tmp.shape[0], i, :]

    for x in np.arange(img.shape[1]-r, img.shape[1]):
        print "copy pixels from (%d:%d, %d) to img(%d:%d, %d)" %(0, tmp.shape[0], img.shape[1]-x, 0, img.shape[0], x)
        img[0:img.shape[0], x, :] = tmp[0:tmp.shape[0], img.shape[1]-x, :]

    print '-----------do_image_magic-----------'

imagefile = "flower.jpg"
birdimage = cv2.imread(imagefile)

cv2.imshow("original image", birdimage)

do_image_magic(birdimage, sharpness_filter_3_3)

cv2.imshow("image magic show!!!!", birdimage)
cv2.waitKey()
cv2.destroyAllWindows()
