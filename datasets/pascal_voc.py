import copy
import numpy as np
import skimage.morphology as skm

from PIL import Image


class Pascal:

    def __init__(self, data_path):
        self.dir = data_path  # C:\\ALISURE/Data/voc/VOCdevkit/VOC2012
        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person',
                        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.class_number = len(self.classes)
        self.voc_palette = Image.open('{}/SegmentationClass/{}.png'.format(self.dir, '2008_000666')).palette
        pass

    def get_data_set(self):
        """
        Load seg11valid, the non-intersecting set of PASCAL VOC 2011 segval and SBD train.
        """
        seg_set_dir = '{}/ImageSets/Segmentation'.format(self.dir)
        return open('{}/trainval.txt'.format(seg_set_dir)).read().splitlines()  # seg11valid.txt

    def load_image(self, idx):
        return Image.open('{}/JPEGImages/{}.jpg'.format(self.dir, idx))

    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        label = np.array(Image.open('{}/SegmentationClass/{}.png'.format(self.dir, idx)), dtype=np.uint8)
        return label[np.newaxis, ...]

    @staticmethod
    def pre_process(im):
        """
        Preprocess loaded image (by load_image) for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:, :, ::-1]
        mean = (104.00698793, 116.66876762, 122.67891434)
        in_ -= np.array(mean)
        in_ = in_.transpose((2, 0, 1))  # 将通道维变到前面
        return in_

    def palette(self, label_im):
        """
        Transfer the VOC color palette to an output mask
        """
        if label_im.ndim == 3:
            label_im = label_im[0]
        label = Image.fromarray(label_im, mode='P')
        label.palette = copy.copy(self.voc_palette)
        return label

    # 从一张图片上得到图片和注解序列
    @staticmethod
    def make_translated_frames(im, label, shift=None, num_frames=None):
        """
        Extract corresponding image and label crops.
        Shift by `shift` pixels at a time for a total of `num_frames`
        so that the total translation is `shift * (num_frames - 1)`.
        im should be prepared by preprocess_voc and gt by load_gt
        """
        assert(shift is not None and num_frames is not None)
        im = np.asarray(im)
        im_crops = []
        label_crops = []
        # find largest dimension, fit crop to shift and frames
        max_dim, shift_idx = np.max(im.shape), np.argmax(im.shape)
        crop_dim = max_dim - shift * (num_frames - 1)
        crop_shape = list(im.shape)
        crop_shape[shift_idx] = crop_dim
        # determine shifts
        crop_shifts = np.arange(0, max_dim - crop_dim + 1, shift)
        for sh in crop_shifts:
            # TODO(shelhamer) there has to be a better way
            crop_idx = [slice(None)] * 3
            crop_idx[shift_idx] = slice(sh, sh + crop_dim)
            im_crops.append(im[crop_idx])
            label_crops.append(label[[0] + crop_idx[:-1]])
        # output is (# frames, channels, spatial)
        im_crops = np.asarray(im_crops)
        label_crops = np.asarray(label_crops)[:, np.newaxis, :, :]
        return im_crops, label_crops

    @staticmethod
    def make_boundaries(label, thickness=None):
        """
        Input is an image label, output is a numpy array mask encoding the boundaries of the objects
        Extract pixels at the true boundary by dilation - erosion of label.
        Don't just pick the void label: it is not exclusive to the boundaries.
        """
        assert(thickness is not None)
        mask = np.logical_and(label > 0, label != 255)[0]
        selem = skm.disk(thickness)
        boundaries = np.logical_xor(skm.dilation(mask, selem), skm.erosion(mask, selem))  # 得到宽度为thickness的边界
        return boundaries

    pass
