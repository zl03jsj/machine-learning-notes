#-*- coding: utf-8 -*-


import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array

def loadImage(fileName):
    with Image.open(fileName) as image:
        image = np.array(image)
        image = convert2gray(image)
        return filterImage(image)

# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        # gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img

def filterImage(image_gray):
    threshold = 120;
    rows,cols = image_gray.shape
    for row in range(rows):
        for col in range(cols):
            if ((row > 50 or col < 180) and (row < 50 or row > 180)):
                image_gray[row,col] = 255
            elif image_gray[row,col] >= threshold:
                image_gray[row,col] = 255
            else:
                image_gray[row,col] = 0
    return image_gray

def make_points(image_gray):
    points = []
    rows,cols = image_gray.shape
    for row in range(rows):
        for col in range(cols):
            if image_gray[row,col] == 0:
                points.append([row,col])
    return points

def main():
    #dataSet = loadDataSet('./data/788points.txt', splitChar=',')
    #dataSet = loadImage('./images/51_capt_20171110_104630(120,98;43,128;309,108;187,108).bmp')
    dataSet = loadImage('./images/51_capt_20171110_104630(120,98;43,128;309,108;187,108).bmp')
    Image.fromarray(dataSet).show()
    X = make_points(dataSet)
    X1 = StandardScaler().fit_transform(X)
    #dataSet = np.mat(dataSet).transpose()
    print(X)
    print(X1)
    db =  DBSCAN(eps=0.1, min_samples=8).fit(X1)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)
    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt

    X = np.array(X)
    print(X)
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    #colors = [plt.cm.Spectral(each)
    #          for each in np.linspace(0, 1, len(unique_labels))]
    colors = []
    for each in np.linspace(0, 1, 100):
        color = plt.cm.Spectral(each)
        if color not in colors:
            colors.append(color)
        if len(colors) >= len(unique_labels):
            break;

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 1],-xy[:, 0], '.')#, markerfacecolor=tuple(col),markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 1], -xy[:, 0], '.')#, markerfacecolor=tuple(col),markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

if __name__ == '__main__':
    start = time.clock()
    main()
    end = time.clock()
    print('finish all in %s' % str(end - start))