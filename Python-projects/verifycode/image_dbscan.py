import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time

# %matplotlib inline

from sklearn.datasets import load_sample_image

'''
X1, y1 = datasets.make_circles(n_samples=5000, factor=.6,
                               noise=.05)
X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2, 1.2]], cluster_std=[[.1]],
                             random_state=9)

plt.scatter(X1[:, 0], X1[:, 1], marker='o')
plt.show()
plt.scatter(X2[:, 0], X2[:, 1], marker='o')
plt.show()

X = np.concatenate([X1, X2])
plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.show()

# y_pred = DBSCAN(eps=0.1).fit_predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# plt.show()


# y_pred = DBSCAN(eps=0.1, min_samples=10).fit_predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# plt.show()
'''

X1, y1 = datasets.make_circles(n_samples=5000, factor=.6, noise=.05)
X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2, 1.2]], cluster_std=[[.1]], random_state=9)
X = np.concatenate([X1, X2])

image = cv.imread('img/origin/004.bmp')

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
image = np.array(image, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(image.shape)
assert d == 3
image_array = np.reshape(image, (w * h, d))

print("Fitting model on a small sub-sample of the data")
t0 = time()
image_array_sample = shuffle(image_array, random_state=0)[:1000]
y_pred = DBSCAN(esp=0.1).fit(image_array_sample, y=None, sample_weight=None)

plt.scatter(image_array_sample[:, 0], image_array_sample[:, 1], c=y_pred)




















#
