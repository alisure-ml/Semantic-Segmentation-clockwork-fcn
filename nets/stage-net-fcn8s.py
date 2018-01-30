import sys
import caffe
from caffe.coord_map import crop
from caffe.proto import caffe_pb2
from caffe import layers as L, params as P


def conv_relu(bottom, num_output, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=num_output, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True)


def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def fcn(split, num_classes=None):
    n = caffe.NetSpec()

    # 输入
    n.data = L.Input(shape=[dict(dim=[1, 3, 500, 500])])

    """ pipeline 2 stage 阶段一 """
    """ pipeline 3 stage 阶段一 """

    # conv1
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, pad=100)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64)
    n.pool1 = max_pool(n.relu1_2)

    # conv2
    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128)
    n.pool2 = max_pool(n.relu2_2)

    # conv3
    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256)
    n.pool3 = max_pool(n.relu3_3)

    # score 1
    n.score_pool3 = L.Convolution(n.pool3, num_output=num_classes, kernel_size=1, pad=0,
                                  param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

    """ pipeline 3 stage 阶段二 """

    # conv4
    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
    n.pool4 = max_pool(n.relu4_3)

    # score 2
    n.score_pool4 = L.Convolution(n.pool4, num_output=num_classes, kernel_size=1, pad=0,
                                  param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

    """ pipeline 2 stage 阶段二 """
    """ pipeline 3 stage 阶段三 """

    # conv5
    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512)
    n.pool5 = max_pool(n.relu5_3)

    # fc6
    n.fc6, n.relu6 = conv_relu(n.pool5, 4096, ks=7, pad=0)
    # fc7
    n.fc7, n.relu7 = conv_relu(n.relu6, 4096, ks=1, pad=0)

    # score 3
    n.score_fr = L.Convolution(n.relu7, num_output=num_classes, kernel_size=1, pad=0,
                               param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

    # deconv 1 : 2x
    n.upscore2 = L.Deconvolution(n.score_fr,
                                 convolution_param=dict(num_output=num_classes, group=num_classes,
                                                        kernel_size=4, stride=2, bias_term=False),
                                 param=[dict(lr_mult=0)])

    """ pipeline 2 stage 融合 """
    """ pipeline 3 stage 融合 """

    # fuse 1
    n.score_pool4c = crop(n.score_pool4, n.upscore2)
    n.fuse_pool4 = L.Eltwise(n.upscore2, n.score_pool4c, operation=P.Eltwise.SUM)

    # deconv 2 : 2x
    n.upscore_pool4 = L.Deconvolution(n.fuse_pool4,
                                      convolution_param=dict(num_output=num_classes, group=num_classes,
                                                             kernel_size=4, stride=2, bias_term=False),
                                      param=[dict(lr_mult=0)])

    # fuse 2
    n.score_pool3c = crop(n.score_pool3, n.upscore_pool4)
    n.fuse_pool3 = L.Eltwise(n.upscore_pool4, n.score_pool3c, operation=P.Eltwise.SUM)

    # deconv 3 : 8x
    n.upscore8 = L.Deconvolution(n.fuse_pool3,
                                 convolution_param=dict(num_output=num_classes, group=num_classes,
                                                        kernel_size=16, stride=8, bias_term=False),
                                 param=[dict(lr_mult=0)])

    # 最终输出
    n.score = crop(n.upscore8, n.data)

    return n.to_proto()


def make_nets():
    with open('../model/stage-voc-fcn8s.prototxt', 'w') as f:
        f.write(str(fcn('deploy', 21)))
    with open('../model/stage-nyud-fcn8s.prototxt', 'w') as f:
        f.write(str(fcn('deploy', 40)))
    with open('../model/stage-cityscapes-fcn8s.prototxt', 'w') as f:
        f.write(str(fcn('deploy', 19)))


if __name__ == '__main__':
    make_nets()
