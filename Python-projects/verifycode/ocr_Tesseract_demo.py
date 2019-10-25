from PIL import Image, ImageFilter, ImageEnhance
from skimage.filters import threshold_otsu
import skimage.morphology as sm
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import re
import os
import time
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s", filename='train_output.log')


class ShopCert(object):
    def cut_region(self, img):
        """
        先按规则缩小搜索范围
        """
        w, h = img.size
        if h<1200:
            factor = max(1, 1600.0/h)
            newsize = int(w*factor), int(h*factor)
            img = img.resize(newsize, Image.ANTIALIAS)
        if w<h:
            box = (w*0.4, h*0.18, w*0.96, h*0.6)
        else:
            box = (w*0.1, h*0.18, w*0.96, h*0.9)
        return img.crop(box)


    def detect_text(self, img):
        """
        检测字符区域
        """
        Image._show(img)
        img_m = np.array(img.convert('L'))
        img_m = 1 * (img_m < threshold_otsu(img_m))

        img_m = sm.binary_closing(img_m, np.ones((5, 10)))
        img_m = sm.remove_small_objects(img_m, 600)
        label_img = sm.label(img_m)
        img_list = []
        for region in regionprops(label_img):
            minr, minc, maxr, maxc = region.bbox
            w, h = (maxc-minc), (maxr-minr)
            if h > w * 0.2:
                continue
            box = minc-5, minr-3, maxc+5, maxr+3
            img_list.append(img.crop(box))
        return img_list



    def clear_noise(self, box):
        """
        降噪处理
        """
        box = box.convert('L')
        #       box = box.point(lambda x: 0 if x<50 else x)
        box = box.point(lambda x: 200 if x>200 else x)
        box = ImageEnhance.Contrast(box).enhance(2.5)
        return box

    def predict(self, fname, lang='eng'):
        """
        ocr 识别
        """
        img = Image.open(fname)
        # 先大致缩小范围
        region = self.cut_region(img)

        # 候选字符区域
        # region = self.clear_noise(region)
        box_list = self.detect_text(region)
        # 遍历识别
        for box in box_list:
            box = self.clear_noise(box)
            w, h = box.size
            if float(w)/h > 12.5:
                res = pytesseract.image_to_string(box, lang='chi_sim', config='-psm 7')
            else:
                res = pytesseract.image_to_string(box, lang='eng', config='-psm 7')
            res = re.sub('\s', '', res)  # 去除中间空白
            res = re.findall(r'[0-9][A-Z0-9]{13,20}', res)  # 13-20位
            for line in res:
                line = line.strip()
                if line.find(u'年')>1:
                    continue
                print('line', line)
                if len(line)> 14:
                    box.save('img/clearNoise/%s_%s.jpg' % (fname.split('/')[-1].split('.')[0], line))
                    return line
                else:
                    print('error line', line)
        return 'error'


def show_pic(path='img/origin2/'):
    fnames = [os.path.join(path, fname) for fname in os.listdir(path)]
    for i, fname in enumerate(fnames, 0):
        print(fname)
        img = Image.open(fname)
        # img.save('./tesseract-train/cert.normal.exp%d.ttf' % i)
        img = ImageEnhance.Contrast(img).enhance(2.0)
        img = img.filter(ImageFilter.MedianFilter).convert('L')
        plt.figure(figsize=(10, 12), dpi=300)
        plt.imshow(img, plt.cm.gray)
        plt.title(fname.split('/')[-1]+'_%d' % i)
        plt.show()

if __name__ == '__main__':
    test = ShopCert()
    path = 'img/origin2/'
    fnames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('jpg')]
    fnames.sort()

    arguments = 'mode: L; enhance:2.0; h:0.5; dh:0.15'
    logging.info('%s' % arguments)
    logging.info("%s: %s" % ('imgname', 'result'))
    start_time = time.time()
    cnt = 0
    for idx, fname in enumerate(fnames, 1):
        print(idx, fname)
        y_true = fname.split('/')[-1].split('_')[0]
        y_pred = test.predict(fname)
        if y_true == y_pred:
            cnt +=1
            print(fname)
        else:
            print('***' * 20)
            print('error')
        logging.info("%s: %s" % (fname, y_pred))
        print('y_true', y_true)
        print('y_pred', y_pred)
        acc = float(cnt)/idx
        print(acc, cnt)
        print('==' * 20, idx)
        logging.info('%.3f %d/%d' % (acc, cnt, idx))
    print('cost time: ', time.time() - start_time)
    logging.info('accuracy: %.2f' % acc)
