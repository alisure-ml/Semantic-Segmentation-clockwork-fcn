import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from collections import namedtuple

import caffe

from lib import run_net
from lib import score_util

from datasets.pascal_voc import Pascal

PV = Pascal('C:\\ALISURE\\Data\\voc\\VOCdevkit\\VOC2012')
val_set = PV.get_data_set()


def show_demo():
    image_name_0 = val_set[0]
    im, label = PV.load_image(image_name_0), PV.load_label(image_name_0)
    im_t, label_t = PV.make_translated_frames(im, label, shift=32, num_frames=6)

    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['figure.figsize'] = (12, 12)

    plt.figure()
    for i, im in enumerate(im_t):
        plt.subplot(3, len(im_t), i + 1)
        plt.imshow(im)
        plt.axis('off')
        plt.subplot(3, len(label_t), len(im_t) + i + 1)
        plt.imshow(PV.palette(label_t[i]))
        plt.axis('off')

    plt.subplot(3, len(label_t), 2 * len(im_t) + 2)
    plt.imshow(PV.palette(label))
    plt.axis('off')
    plt.subplot(3, len(label_t), 2 * len(im_t) + 5)
    plt.imshow(PV.make_boundaries(label, thickness=2))
    plt.axis('off')

    plt.show()


class_number = len(PV.classes)
num_frames = 6
thickness = 5
shifts = (16, 32)


Method = namedtuple('Method', 'method arch weights infer_func, input_offset')
fcn = Method('fcn', '../nets/voc-fcn8s.prototxt',
             '../nets/voc-fcn8s-heavy.caffemodel', run_net.segrun, 2)
baseline_3stage = Method('baseline_3stage', '../nets/voc-fcn-pool3.prototxt',
                         '../nets/voc-fcn-pool3.caffemodel', run_net.segrun, 2)
baseline_2stage = Method('baseline_2stage', '../nets/voc-fcn-pool4.prototxt',
                         '../nets/voc-fcn-pool4.caffemodel', run_net.segrun, 2)
pipeline_3stage = Method('pipeline_3stage', '../nets/stage-voc-fcn8s.prototxt',
                         '../nets/voc-fcn8s-heavy.caffemodel', run_net.pipeline_3stage_forward, 0)
pipeline_2stage = Method('pipeline_2stage', '../nets/stage-voc-fcn8s.prototxt',
                         '../nets/voc-fcn8s-heavy.caffemodel', run_net.pipeline_2stage_forward, 1)


def score_translations(method, shift, arch, weights, infer, offset):
    """
    Score the translated "video" of PASCAL VOC seg11valid images
    taking care of the net architecture and weights, the particular inference method,
    and the input offset needed to align every frame and pipeline methods.
    """
    net = caffe.Net(arch, weights, caffe.TEST)
    hist, hist_b = np.zeros((class_number, class_number)), np.zeros((class_number, class_number))
    for index, image_name in enumerate(val_set[0: 10]):
        print("{} begin {}".format(time.strftime("%H:%M:%S", time.localtime()), index))

        im, label = PV.load_image(image_name), PV.load_label(image_name)
        im_frames, label_frames = PV.make_translated_frames(im, label, shift=shift, num_frames=num_frames)
        im_frames, label_frames = im_frames[offset:], label_frames[offset:]

        # prepare pipelines: feed initial inputs then skip accordingly
        if method == 'pipeline_3stage':
            run_net.pipeline_fill_3stage(net, PV.pre_process(im_frames[0]), PV.pre_process(im_frames[1]))
            im_frames, label_frames = im_frames[2:], label_frames[2:]
        elif method == 'pipeline_2stage':
            run_net.pipeline_fill_2stage(net, PV.pre_process(im_frames[0]))
            im_frames, label_frames = im_frames[1:], label_frames[1:]

        for im_t, label_t in zip(im_frames, label_frames):
            print("{} begin {} .....".format(time.strftime("%H:%M:%S", time.localtime()), index))
            out = infer(net, PV.pre_process(im_t))
            Image.fromarray(out * 12).convert("L").show()

            hist += score_util.score_out_gt(out, label_t, n_cl=class_number)
            bdry = PV.make_boundaries(label_t, thickness=thickness)
            hist_b += score_util.score_out_gt_bdry(out, label_t, bdry, n_cl=class_number)
        pass

    for name, h in zip(('seg', 'bdry'), (hist, hist_b)):
        accP, cl_accP, mean_iuP, fw_iuP = score_util.get_scores(h)
        print('{}: {}, shift {}'.format(method, name, shift))
        print('acc\t\t cl acc\t\t mIU\t\t fwIU')
        print('{:f}\t {:f}\t {:f}\t {:f}\t'.format(100*accP, 100*cl_accP, 100*mean_iuP, 100*fw_iuP))

for shift in shifts:
    for m in (fcn, baseline_3stage, pipeline_3stage, baseline_2stage, pipeline_2stage):
        score_translations(m.method, shift, m.arch, m.weights, m.infer_func, m.input_offset)


"""
fcn: seg, shift 16
acc		 cl acc		 mIU		 fwIU
91.974863	 82.881608	 70.022842	 85.902034	
fcn: bdry, shift 16
acc		 cl acc		 mIU		 fwIU
63.948030	 65.065930	 49.555515	 53.667815	

baseline_3stage: seg, shift 16
acc		 cl acc		 mIU		 fwIU
60.286632	 13.705269	 4.690409	 43.286493	
baseline_3stage: bdry, shift 16
acc		 cl acc		 mIU		 fwIU
11.166320	 11.496480	 1.818804	 3.798124	

pipeline_3stage: seg, shift 16
acc		 cl acc		 mIU		 fwIU
88.349069	 74.970788	 55.175040	 79.954080	
pipeline_3stage: bdry, shift 16
acc		 cl acc		 mIU		 fwIU
56.349989	 56.221222	 41.709843	 46.512476	

baseline_2stage: seg, shift 16
acc		 cl acc		 mIU		 fwIU
69.464357	 36.060632	 13.589065	 53.310369	
baseline_2stage: bdry, shift 16
acc		 cl acc		 mIU		 fwIU
29.001448	 26.982958	 8.294242	 19.001877	

pipeline_2stage: seg, shift 16
acc		 cl acc		 mIU		 fwIU
90.942986	 80.925445	 67.604536	 84.123599	
pipeline_2stage: bdry, shift 16
acc		 cl acc		 mIU		 fwIU
61.421430	 61.866688	 46.878984	 51.400744	

baseline_3stage: seg, shift 32
acc		 cl acc		 mIU		 fwIU
59.179855	 13.635292	 4.642013	 41.671485	
baseline_3stage: bdry, shift 32
acc		 cl acc		 mIU		 fwIU
11.052108	 11.406026	 1.778673	 3.790397	

pipeline_3stage: seg, shift 32
acc		 cl acc		 mIU		 fwIU
82.182183	 64.902390	 49.651163	 70.833176	
pipeline_3stage: bdry, shift 32
acc		 cl acc		 mIU		 fwIU
50.599466	 48.977500	 35.428999	 41.064723	

"""
