from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
from verifycode.ege_detection import canny__


# %matplotlib inline

def random_color():
    color = (np.random.uniform(0, 1),
             np.random.uniform(0, 1),
             np.random.uniform(0, 1), 1.0)
    return color


def make_points(img):
    if len(img.shape) != 2:
        gray = cv.cvtColor(img, code=cv.COLOR_BGR2GRAY)
    else:
        gray = img
    rows, cols = gray.shape
    points = []
    for i in np.arange(0, rows):
        for j in np.arange(0, cols):
            if gray[i][j] == 0: continue
            points.append([i, j])
    return points


def main():
    image = cv.imread('img/origin/011.bmp')
    image = canny__(image)

    cv.imshow('ege_detected image', image)

    points = make_points(image)
    samples = StandardScaler().fit_transform(points)

    db = DBSCAN(eps=0.05, min_samples=8).fit(samples)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    points = np.array(points)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)

    colors = [plt.cm.Spectral(each)
             for each in np.linspace(0, 1, len(unique_labels))]
    #
    # colors = []
    # for _ in np.arange(15): colors.append(random_color())


    for k, c in zip(unique_labels, colors):
        if k == -1: c = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = points[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 1], -xy[:, 0], '.', color=c)  # , markerfacecolor=tuple(col),markeredgecolor='k', markersize=14)

        xy = points[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 1], -xy[:, 0], '.', color=c)  # , markerfacecolor=tuple(col),markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    # plt.waitforbuttonpress()


if __name__ == '__main__':
    start = time.clock()
    main()
    end = time.clock()
    print('finish all in %s' % str(end - start))



















#
