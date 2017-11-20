import numpy as np
import cv2

sz1, sz2 = 400, 600
img = np.zeros((sz1, sz2, 3), np.uint8)

pos1 = np.random.randint(sz1, size=(200, 5))
pos2 = np.random.randint(sz2, size=(200, 5))

for i in range(200):
    img[pos1[i], pos2[i], 0] = np.random.randint(0, 255)
    img[pos1[i], pos2[i], 1] = np.random.randint(0, 255)
    img[pos1[i], pos2[i], 2] = np.random.randint(0, 255)

## print pos1[0]
## print pos2[0]
## print img[pos1[0], pos2[0], 0]

cv2.imshow("snow image", img)
cv2.waitKey()
cv2.destroyAllWindows()

