from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.io import imread, imsave
from skimage.measure import label, regionprops


def imbinarize(im):
    """
    图像二值化
    """
    gray = rgb2gray(im)
    bw = gray < threshold_otsu(gray)
    return bw


def imcrop(im, bbox):
    """
    裁剪图像
    """
    min_row, min_col, max_row, max_col = bbox
    return im[min_row:max_row, min_col:max_col, :]


if __name__ == "__main__":
    im = imread("img/origin2/001.jpg")
    bw = imbinarize(im)

    # 连通域分析
    labeled = label(bw)
    props = regionprops(labeled)
    cnt = 0
    for region in props:
        if region.area > 64:
            cnt += 1
            imsave("{}.jpg".format(cnt), imcrop(im, region.bbox))