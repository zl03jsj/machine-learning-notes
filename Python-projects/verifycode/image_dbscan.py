import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import verifycode.ege_detection
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

__commands__ = {1: 'template match', 2: 'feature match', 3: 'maching learning'}

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

        l = max(0, l - expend_)
        r = min(max_shape[1], r + expend_)
        t = max(0, t - expend_)
        b = min(max_shape[0], b + expend_)

        box = [[l, t], [r, b]]

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

def do_feather_match(imgs, tmplates):
    return


def get_gray(image):
    if len(image.shape) != 2:
        return cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
    else:
        return image.copy()


def get_sift(img):
    gray = get_gray(img)
    surf = cv2.xfeatures2d.SURF_create()
    kp, des = surf.detectAndCompute(gray, None)
    # fea_det = cv2.Feature2D_create('SIFT')
    # des_ext = cv2.DescriptorExtrator_create('SIFT')
    # keypoints = fea_det.detect(gray)
    # kp, des = des_ext.compute(gray, keypoints)
    # out_image = img.copy()
    # cv2.drawKeypoints(img, kp, out_image, color=255, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    # plt.subplot(121), plt.imshow(img)
    # plt.title('original image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(out_image)
    # plt.title('image with feature points'), plt.xticks([]), plt.yticks([])
    # plt.show()
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


def filter_matches(kp1, kp2, matches, ratio=0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs, len(mkp1)


def explore_match(win, img1, img2, kp_pairs, status=None, H=None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool)
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)
            cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)
            cv2.line(vis, (x2 - r, y2 - r), (x2 + r, y2 + r), col, thickness)
            cv2.line(vis, (x2 - r, y2 + r), (x2 + r, y2 - r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    cv2.imshow(win, vis)


def get_most_match_image(tmplate, targets_arr):
    if 0 == len(targets_arr): return None
    tmp_kp, tmp_des = get_sift(tmplate)
    most_match_index = 0
    most_match_count = 0
    for index, target in zip(np.arange(0, len(targets_arr)), targets_arr):
        cv2.imshow('image', target['image'])
        cv2.waitKey()
        cv2.destroyAllWindows()

        tar_kp, tar_des = get_sift(target['image'])
        bf = cv2.BFMatcher(cv2.NORM_L1)
        matches = bf.match(tmp_des, tar_des)
        _, _, _, count = filter_matches(tmp_kp, tar_kp, matches, ratio=0.5)
        if count > most_match_count:
            most_match_index = index
            most_match_count = count
    return most_match_index


def get_shape_match_rate(img1, img2):
    cv2.imshow('image1', img1)
    cv2.imshow('image2', img2)
    cv2.waitKey()

    gray1 = get_gray(img1)
    gray2 = get_gray(img2)

    ret, thresh = cv2.threshold(gray1, 127, 255, 0)
    ret, thresh2 = cv2.threshold(gray2, 127, 255, 0)

    _, contours, hierarchy = cv2.findContours(thresh, 2, 1)
    cnt1 = contours[0]
    _, contours, hierarchy = cv2.findContours(thresh2, 2, 1)
    cnt2 = contours[0]

    ret = cv2.matchShapes(cnt1, cnt2, 1, 0.0)
    # _, contours1, hierarchy1 = cv2.findContours(gray1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # _, contours2, hierarchy2 = cv2.findContours(gray2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # ret = cv2.matchShapes(contours1[0], contours2[0], cv2.CONTOURS_MATCH_I1, 0.0)
    return ret

def get_most_matched_shape(template, targets):
    match_rate = 1000
    matched_index = 0
    for index, tgt in zip(np.arange(0, len(targets)), targets):
        rate = get_shape_match_rate(template, tgt['image'])
        print('match rage', rate)
        if rate < match_rate:
            match_rate = rate
            matched_index = index
    return matched_index


def main():
    image = cv2.imread('img/origin/009.bmp')
    origin_templates, origin_target = image51_split(image)

    gray = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
    filter_image(gray)
    cv2.imshow('image', gray)
    cv2.waitKey()

    gray_templates, gray_target = image51_split(gray)
    # cv2.imshow('gray', gray)
    # cv2.imshow('templates', templates[0])
    # if __commands__=='do template match':
    #     for tmplate in templates:
    #         do_templagte_match(traget_image, tmplate)
    # elif __commands__=='do featcher match':
    #     do_feather_match()

    points = make_points(gray_target)
    samples = StandardScaler().fit_transform(points)

    db = DBSCAN(eps=0.15, min_samples=7).fit(samples)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    labels = db.labels_
    unique_labels = set(labels)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    points = np.array(points)

    used_labels, cluster_boxs = filter_cluster(labels, points, gray_target.shape, 3)
    # cv2.imshow('gray_target', gray_target)
    # cv2.waitKey()
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

    subimage_info_arr = []
    for box in cluster_boxs:
        tmp_image = gray_target[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        point = ((box[0][0] + box[1][0]) // 2, (box[0][1] + box[1][1]) // 2)
        subimage_info_arr.append({'image': tmp_image, 'box': box, 'point': point})

    matched_points = []
    for template in gray_templates:
        index = get_most_match_image(template, subimage_info_arr)
        matched_points.append(subimage_info_arr[index]['point'])
        origin_target = cv2.circle(origin_target, subimage_info_arr[index]['point'], radius=10, color=(0, 0, 255),
                                   thickness=4)
        cv2.imshow('image', origin_target)
        cv2.waitKey()


def main_2():
    image = cv2.imread('img/origin/009.bmp')
    cv2.imshow('original image', image)

    gray = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
    filter_image(gray)

    _, image = image51_split(image)

    gray_templates, gray_targets = image51_split(gray)
    points = make_points(gray_targets)

    samples = StandardScaler().fit_transform(points)
    db = DBSCAN(eps=0.15, min_samples=7).fit(samples)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    unique_labels = set(labels)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    points = np.array(points)

    used_labels, cluster_boxs = filter_cluster(labels, points, gray_targets.shape, 3)

    # eged_gray = verifycode.ege_detection.canny__(gray_targets)

    subimage_info_arr = []
    for box in cluster_boxs:
        tmp_image = gray_targets[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        point = ((box[0][0] + box[1][0]) // 2, (box[0][1] + box[1][1]) // 2)
        subimage_info_arr.append({'image': tmp_image, 'box': box, 'point': point})

    matched_points = []

    for template in gray_templates:
        index = get_most_matched_shape(template, subimage_info_arr)
        matched_points.append(subimage_info_arr[index]['point'])
        image = cv2.circle(image, subimage_info_arr[index]['point'], radius=10, color=(0, 0, 255),
                               thickness=4)
        # cv2.imshow('tmeplate', template)
        # cv2.imshow('matched image', subimage_info_arr[index]['image'])
        # cv2.waitKey()

    cv2.imshow('image', image)
    cv2.waitKey()

if __name__ == '__main__':
    start = time.clock()
    main_2()
    end = time.clock()
    print('finish all in %s' % str(end - start))
