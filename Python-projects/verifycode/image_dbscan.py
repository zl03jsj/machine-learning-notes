from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from verifycode.ege_detection import canny__

__commands__ = {1:'template match', 2:'feature match', 3:'maching learning'}

# %matplotlib inline

def random_color():
    color = (np.random.uniform(0, 1),
             np.random.uniform(0, 1),
             np.random.uniform(0, 1), 1.0)
    return color


def make_points(img):
    gray = get_gray(img)
    rows, cols = gray.shape
    points = []
    for i in np.arange(0, rows):
        for j in np.arange(0, cols):
            if gray[i][j] == 0: continue
            points.append([i, j])
    return points


def image_is_51_verifycode(image):
    shape = image.shape
    return False if (shape[0] != 236 or shape[1] != 350) else True


def image51_split(image):
    if not image_is_51_verifycode(image): return None
    template_images = [image[19:43, 206:235], image[19:43, 235:265], image[19:43, 265:294], image[19:43, 294:323]]
    # cv2.imwrite('./template_image.jpg', image[19:43, 206:323])
    return template_images, image[60:175, 10:340]


def image51_pre_solve(image):
    if not image_is_51_verifycode(image):
        return
    shape = image.shape
    image[0:45, 0:180] = 0


def filter_image(image_gray):
    threshold = 120
    rows, cols = image_gray.shape
    for row in range(rows):
        for col in range(cols):
            if row in range(0, 45) and col in range(0, 180):
                image_gray[row, col] = 0
            elif image_gray[row, col] >= threshold:
                image_gray[row, col] = 0
            else:
                image_gray[row, col] = 255
    return image_gray


def filter_cluster(labels, points, max_shape, expend_=0):
    unique_labels = set(labels)

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    cluster_boxs = []
    used_clusters = []
    for k in unique_labels:
        if -1 == k: continue
        class_member_mask = (labels == k)
        cluster_points = points[class_member_mask]

        xs = cluster_points[:, 1]
        ys = cluster_points[:, 0]

        l = xs.min()
        r = xs.max()
        t = ys.min()
        b = ys.max()
        if r - l < 10 or b - t < 10 or r - l > 40 or b - t > 40: continue

        l = max(0, l-expend_)
        r = min(max_shape[1], r+expend_)
        t = max(0, t-expend_)
        b = min(max_shape[0], b+expend_)

        box = [(l, t), (r, b)]

        cluster_boxs.append(box)
        used_clusters.append(k)

    return used_clusters, cluster_boxs


# def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
#     x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
#
#     # 3 conv layer
#     w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
#     b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
#     conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
#     conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#     conv1 = tf.nn.dropout(conv1, keep_prob)
#
#     w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
#     b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
#     conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
#     conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#     conv2 = tf.nn.dropout(conv2, keep_prob)
#
#     w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
#     b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
#     conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
#     conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#     conv3 = tf.nn.dropout(conv3, keep_prob)
#
#     # Fully connected layer
#     w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
#     b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
#     dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
#     dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
#     dense = tf.nn.dropout(dense, keep_prob)
#
#     w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
#     b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
#     out = tf.add(tf.matmul(dense, w_out), b_out)
#     # out = tf.nn.softmax(out)
#     return out

def do_feather_match(imgs, tmplates) :
    return


def get_gray(image):
    if len(image.shape) != 2: return cv2.cv2.Color(image, code=cv2.COLOR_BGR2GRAY)
    else: return image.copy()


def get_sift(img):
    gray = get_gray(img)
    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(gray, None)
    # fea_det = cv2.Feature2D_create('SIFT')
    # des_ext = cv2.DescriptorExtrator_create('SIFT')
    # keypoints = fea_det.detect(gray)
    # kp, des = des_ext.compute(gray, keypoints)
    img = cv2.drawKeypoints(gray, kp)
    plt.imshow(img),plt.show()
    return kp, des


def opencv_sift_match(img1, img2):
    kp1, des1 = get_sift(img1)
    kp2, des2 = get_sift(img2)




def do_templagte_match(img, template):
    # img2 = img.copy()
    w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    for meth in methods:
        img2 = img.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(img2, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img2, top_left, bottom_right, 255, 2)

        plt.subplot(211), plt.imshow(template, cmap='gray')

        plt.subplot(223), plt.imshow(res, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])

        plt.subplot(224), plt.imshow(img2, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)

        plt.show()



def main():
    image = cv2.imread('img/origin/009.bmp')

    gray = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
    filter_image(gray)

    templates, target_image = image51_split(gray)
    # cv2.imshow('gray', gray)
    # cv2.imshow('templates', templates[0])
    cv2.imshow('target_image', target_image)

    # if __commands__=='do template match':
    #     for tmplate in templates:
    #         do_templagte_match(traget_image, tmplate)
    # elif __commands__=='do featcher match':
    #     do_feather_match()

    points = make_points(target_image)
    samples = StandardScaler().fit_transform(points)

    db = DBSCAN(eps=0.15, min_samples=7).fit(samples)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    labels = db.labels_
    unique_labels = set(labels)

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    points = np.array(points)

    used_labels, cluster_boxs = filter_cluster(labels, points, target_image.shape, 3)
    source_images = []

    for box in cluster_boxs:
        tmp_image = gray[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        source_images.append(tmp_image)
        # cv2.rectangle(target_image, box[0], box[1], thickness=1, color=np.random.randint(0, high=256, size=(3,)).tolist())

    # 这里是把聚类绘制出来...
    # colors = [random_color() for _ in np.arange(n_clusters_)]
    # colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    # for k, c in zip(unique_labels, colors):
    #     if k not in used_labels: continue
    #     class_member_mask = (labels == k)
    #     xy = points[class_member_mask]
    #     plt.plot(xy[:, 1], -xy[:, 0], '*', color=c)
    #
    # plt.title('Estimated number of clusters: %d' % n_clusters_)
    # plt.show()

    get_sift(source_images[0])

if __name__ == '__main__':
    start = time.clock()
    main()
    end = time.clock()
    print('finish all in %s' % str(end - start))
