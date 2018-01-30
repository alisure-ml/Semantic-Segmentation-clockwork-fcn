import sys

import numpy as np
import matplotlib.pyplot as plt
import random
import sys
sys.path.append("/home/z840/caffe-tmp/caffe/python")

import caffe

from datasets.cityscapes import cityscapes

from lib import run_net
from lib import score_util
from lib import plot_util


class DataCityscapes(object):

    def __init__(self):
        # paint
        plt.rcParams['image.cmap'] = 'gray'
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['figure.figsize'] = (12, 12)

        # config
        caffe.set_device(0)
        caffe.set_mode_gpu()

        # data
        self.cityscapes = cityscapes('C:/ALISURE/Data/cityscapes')
        self.class_count = len(self.cityscapes.classes)
        self.valset = self.cityscapes.get_dset('val')

        # net
        self.net = self.get_net()
        pass

    @staticmethod
    def get_net():
        return caffe.Net('../model/cityscapes-fcn8s.prototxt', '../model/cityscapes-fcn8s-heavy.caffemodel', caffe.TEST)

    def run_sigle_item(self, item):
        im, gt = self.cityscapes.load_image('val', *item), self.cityscapes.load_label('val', *item)
        out = run_net.segrun(self.net, self.cityscapes.preprocess(im))
        return im, gt, out

    def test_one(self):
        # 选择一个数据测试
        item = random.choice(self.valset)
        im, gt, out = self.run_sigle_item(item)
        plot_util.segshow(im, self.cityscapes.palette(gt), self.cityscapes.palette(out))

    def test_all(self):
        # 测试所有数据
        hist = np.zeros((self.class_count, self.class_count))
        for i, item in enumerate(self.valset):
            if i % 100 == 0:
                print('running {}/{}'.format(i, len(self.valset)))
                sys.stdout.flush()
            im, gt, out = self.run_sigle_item(item)
            hist += score_util.score_out_gt(out, gt, n_cl=self.class_count)

        acc, cl_acc, iu, fw_iu = score_util.get_scores(hist)
        print('val results: acc {:.3f} class acc {:.3f} iu {:.3f} fw iu {:.3f}'.format(acc, cl_acc, iu, fw_iu))

    def run(self):
        self.test_one()
        self.test_all()
        pass

    pass


if __name__ == '__main__':

    data_city_scapes = DataCityscapes()
    data_city_scapes.run()

    pass
