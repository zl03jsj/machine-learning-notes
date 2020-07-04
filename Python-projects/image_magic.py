# -*- coding: UTF-8 -*-
import cv2
import numpy as np

# 图像锐化滤波器Sharpness Filter
sharpness_filter_1 = np.linspace(0, 0, 25).reshape(5, 5)
sharpness_filter_1[3, 3] = 1

sharpness_filter_2 = np.linspace(-1, -1, 9).reshape(3, 3)
sharpness_filter_2[1, 1] = 9

sharpness_filter_3 = np.linspace(1, 1, 9).reshape(3, 3)
sharpness_filter_3[1, 1] = -7

sharpness_filter_4 = np.linspace(-1, -1, 25).reshape(5, 5)
sharpness_filter_4[1:4, 1:4] = 2
sharpness_filter_4[2, 2] = 1

sharpness_filter_5 = np.zeros(25).reshape(5, 5)
sharpness_filter_5[2, 0:3] = [-1, -1, 2]

sharpness_filter_6 = np.zeros(25).reshape(5, 5)
sharpness_filter_6[0:5, 2] = [-1, -1, 4, -1, -1]

sharpness_filter_7 = np.zeros(25).reshape(5, 5)
sharpness_filter_7[0, 0] = -1
sharpness_filter_7[1, 1] = -2
sharpness_filter_7[2, 2] = 6
sharpness_filter_7[3, 3] = -2
sharpness_filter_7[4, 4] = -1

sharpness_filter_8 = np.linspace(-1, -1, 9).reshape(3, 3)
sharpness_filter_8[1, 1] = 8

sharpness_filter_9 = np.array([[-1, -1, 0],
                               [-1,  0, 1],
                               [ 0,  1, 1]])

sharpness_filter_10 = np.array([[-1, -1, -1, -1, 0],
                                [-1, -1, -1,  0, 1],
                                [-1, -1,  0,  1, 1],
                                [-1,  0,  1,  1, 1],
                                [ 0,  1,  1,  1, 1]])

sharpness_filter_11 = np.linspace(0.04, 0.04, 25).reshape(5, 5)

#need to divied 273
sharpness_filter_12 = np.array([[1.0/273,  4.0/273,  7.0/273,  4.0/273, 1.0/273],
                                [4.0/273, 16.0/273, 26.0/273, 16.0/273, 4.0/273],
                                [7.0/273, 26.0/273, 41.0/273, 26.0/273, 7.0/273],
                                [4.0/273, 16.0/273, 26.0/273, 16.0/273, 4.0/273],
                                [1.0/273,  4.0/273,  7.0/273,  4.0/273, 1.0/273]])



filters = [sharpness_filter_1, sharpness_filter_2, sharpness_filter_3,
           sharpness_filter_4, sharpness_filter_5,  sharpness_filter_6,
           sharpness_filter_7, sharpness_filter_8,  sharpness_filter_9,
           sharpness_filter_10, sharpness_filter_11,  sharpness_filter_12]

def multiple_sum(f, sub_img):
    r = min(max(0, (f * sub_img[:, :, 0]).sum()), 255)
    g = min(max(0, (f * sub_img[:, :, 1]).sum()), 255)
    b = min(max(0, (f * sub_img[:, :, 2]).sum()), 255)
    color = np.array([r, g, b], dtype=np.uint8)
    return color


def do_image_magic(img, f):
    print ('-----------do_image_magic-----------')

    print ('img shape = '), img.shape
    print ('filter cub shape = ', f.shape)
    print (f)

    if f.shape[0]!=f.shape[1] or 0==(f.shape[0] % 2):
        print ("filter cub shape can not be even,should be odd number!!!!")
        return

    r = int(f.shape[0]/2)

    tmp = np.zeros((img.shape[0] * (r+1), 3), dtype=np.uint8).reshape(img.shape[0], r+1,  3)
    tx = 0

    for x in np.arange(0, img.shape[1]):
        begin_x = max(0, x-r)
        end_x = min(img.shape[1]-1, x+r)

        begin_fx = max(0, r-x)
        end_fx = min(f.shape[1]-1, (img.shape[1] -1  - x + r))

        tx = x if x <= r else r

        for y in np.arange(0, img.shape[0]):
            begin_y = max(0, y-r)
            end_y = min(img.shape[0]-1, y+r)

            begin_fy = max(0, r-y)
            end_fy = min(f.shape[0]-1, (img.shape[0] - 1 - y + r))

            sub_filter = f[begin_fy:end_fy + 1, begin_fx:end_fx + 1]
            sub_img = img[begin_y:end_y + 1, begin_x:end_x + 1]

            # print "calculate color at position = (%d, %d)" % (x, y)
            tmp[y, tx, :] = multiple_sum(sub_filter, sub_img)

        if x >= r:
            img[0:img.shape[0], x-r, :] = tmp[0:tmp.shape[0], 0, :]
            for i in np.arange(1, r+1):
                tmp[0:tmp.shape[0], i-1, :] = tmp[0:tmp.shape[0], i, :]

    for x in np.arange(img.shape[1]-r, img.shape[1]):
        img[0:img.shape[0], x, :] = tmp[0:tmp.shape[0], img.shape[1]-x, :]

    print ('-----------do_image_magic-----------')

def transporion_image(img):
    w = img.shape[1]
    h = img.shape[0]
    tmp_image = np.zeros((w, h, 3), np.uint8)
    for i in np.arange(0, w):
        tmp_image[i, 0:h, :] = img[0:h, i, :]
    return tmp_image

def fusion_image(img1, img2):
    w = min(img1.shape[1], img2.shape[1])
    h = min(img1.shape[0], img2.shape[1])
    for x in range(0, w, 1):
        for y in range(0, h, 1):
            img2[y, x, :] = (img2[y, x, :] * 0.9) + img1[y, x, :] * 0.1

imagefile = "lenna.jpg"
image = cv2.imread(imagefile)
cityimage = cv2.imread("city.jpg")

tmpimage = transporion_image(image)

# fusion_image(image, cityimage)

# cv2.imshow("magic image", cityimage)
i = 1
for f in filters:
    tmpimage = image.copy()
    do_image_magic(tmpimage, f)
    cv2.imshow("filter=" + str(i), tmpimage)
    i += 1

cv2.imshow("image magic", image)

do_image_magic(image, sharpness_filter_12)
do_image_magic(image, sharpness_filter_12)

cv2.imshow("image magic show!!!![filter=", image)

cv2.waitKey()
cv2.destroyAllWindows()

