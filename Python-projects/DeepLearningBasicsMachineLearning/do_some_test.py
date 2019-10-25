from skimage import data, transform
import numpy as np
import matplotlib.pyplot as plt

def rotate_3d():
    img = data.camera()
    theta = np.deg2rad(-10)
    s, c = np.sin(theta), np.cos(theta)
    w, h = img.shape
    mtx_3d = np.array([[1, 0, -w/2], [0, 1, -h/2], [0, 0,  1], [0, 0,  1]])
    mtx_r = np.array([[c, 0, -s, 0], [0, 1,  0, 0], [s, 0,  c, 0], [0, 0,  0, 1]])
    mtx_t = np.array([[1, 0, 0, -20], [0, 1, 0, 0], [0, 0, 1, 250], [0, 0, 0, 1]])
    mtx_2d = np.array([[350, 0,   w/2, 0], [0,   350, h/2, 0], [0,   0,   1,   0]])
    mtx_total = np.dot(mtx_2d, np.dot(mtx_t, np.dot(mtx_r, mtx_3d)))
    img_rot = transform.warp(img, mtx_total, cval=1)
    f, (ax0, ax1) = plt.subplots(1, 2)
    ax0.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    ax1.imshow(img_rot, cmap=plt.cm.gray, interpolation='nearest')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
points = np.ones(5)  # Draw 3 points for each line
text_style = dict(horizontalalignment='right', verticalalignment='center',
                  fontsize=12, fontdict={'family': 'monospace'})
marker_style = dict(color='cornflowerblue', linestyle=':', marker='o',
                    markersize=15, markerfacecoloralt='gray')
def format_axes(ax):
    ax.margins(0.2)
    ax.set_axis_off()
def nice_repr(text):
    return repr(text).lstrip('u')
fig, ax = plt.subplots()
# Plot all fill styles.
for y, fill_style in enumerate(Line2D.fillStyles):
    ax.text(-0.5, y, nice_repr(fill_style), **text_style)
    ax.plot(y * points, fillstyle=fill_style, **marker_style)
    format_axes(ax)
    ax.set_title('fill style')
plt.show()
