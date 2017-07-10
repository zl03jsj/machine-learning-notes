import cv2
import numpy as np


def color_caculate(multiple, color) :
    color = color % 256
    if multiple == 0 : return color

    if multiple > 0 : total = (255-color)
    else : total = color

    return color + multiple * total


def showImgUseLight(imgfile, multiple) :

    img = cv2.imread(imgfile)
    shape = img.shape
    for x in np.arange(0, shape[0]):
        for y in np.arange(0, shape[1]):
            r, g, b = img[x, y, 0], img[x, y, 1], img[x, y, 2]
            r = color_caculate(multiple, r)
            g = color_caculate(multiple, g)
            b = color_caculate(multiple, b)
            img[x, y, 0], img[x, y, 1], img[x, y, 2] = r, g, b

    cv2.imshow("double darked image", img)
    cv2.waitKey(10*1000)
    cv2.destroyAllWindows()

showImgUseLight("bird.jpg", 0.8)
