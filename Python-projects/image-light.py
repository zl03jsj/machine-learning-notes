import cv2
import numpy as np


def text_image(img, pos, txt, size):
    cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_PLAIN, float(size), (127, 127, 127), thickness=2)


def color_calculate(multiple, color):
    color = color % 256
    if multiple == 0:
        return color

    total = ((255-color) if (multiple > 0) else color)
    color += multiple * total
    return color


def light_image(img, multiple):
    shape = img.shape
    for x in np.arange(0, shape[0]):
        for y in np.arange(0, shape[1]):
            img[x, y, 0] = color_calculate(multiple, img[x, y, 0])
            img[x, y, 1] = color_calculate(multiple, img[x, y, 1])
            img[x, y, 2] = color_calculate(multiple, img[x, y, 2])


def gray_image(img):
    for x in np.arange(0, img.shape[0]):
        for y in np.arange(0, img.shape[1]):
            color = (np.int16(img[x, y, 0]) + np.int16(img[x, y, 1]) + np.int16(img[x, y, 2]) + 2)/3
            img[x, y] = color


def v2_image(img, upvalue):
    for x in np.arange(0, img.shape[0]):
        for y in np.arange(0, img.shape[1]):
            color = (np.int16(img[x, y, 0]) + np.int16(img[x, y, 1]) + np.int16(img[x, y, 2]) + 2)/3
            color = 255 if color > upvalue else 0
            img[x, y] = color


def noise_image(img, count):
    for n in range(0, count):
        x = np.random.randint(0, img.shape[0], dtype=np.int16)
        y = np.random.randint(0, img.shape[1], dtype=np.int16)
        img[x, y] = np.random.randint(0, 255)


image = cv2.imread('bird.jpg')
text_image(image, (20, 20),  "mathine learning", 2.0)
text_image(image, (20, 100), "support vector machines(SVMs), is an algorithm of machine learning", 1.0)
# light_image(image, 0.3)
# gray_image(image)
# image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# v2_image(image, 80)
# noise_image(image, 300000)


cv2.imshow("image magic", image)

cv2.waitKey(6 * 1000)
cv2.destroyAllWindows()

