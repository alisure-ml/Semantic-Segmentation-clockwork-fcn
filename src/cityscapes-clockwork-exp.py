import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import caffe

from lib import run_net
from lib import score_util

from datasets.cityscapes import cityscapes


net = caffe.Net('../nets/stage-cityscapes-fcn8s.prototxt',
                '../nets/cityscapes-fcn8s-heavy.caffemodel',
                caffe.TEST)

CS = cityscapes('/x/cityscapes')
n_cl = len(CS.classes)
split = 'val'
label_frames = CS.list_label_frames(split)


hist_perframe = np.zeros((n_cl, n_cl))
for i, idx in enumerate(label_frames):
    if i % 100 == 0:
        print('running {}/{}'.format(i, len(label_frames)))
    city = idx.split('_')[0]
    # idx is city_shot_frame
    im = CS.load_image(split, city, idx)
    out = run_net.segrun(net, CS.preprocess(im))
    label = CS.load_label(split, city, idx)
    hist_perframe += score_util.fast_hist(label.flatten(), out.flatten(), n_cl)

accP, cl_accP, mean_iuP, fw_iuP = score_util.get_scores(hist_perframe)
print('Oracle: Per frame')
print('acc\t\t cl acc\t\t mIU\t\t fwIU')
print('{:f}\t {:f}\t {:f}\t {:f}\t'.format(100*accP, 100*cl_accP, 100*mean_iuP, 100*fw_iuP))


hist_baseline = np.zeros((n_cl, n_cl))
for i, idx in enumerate(label_frames):
    if i % 100 == 0:
        print('running {}/{}'.format(i, len(label_frames)))
    city = idx.split('_')[0]
    all_frames = CS.collect_frame_sequence(split, idx, 19) # list of Images including labeled frame
    label = CS.load_label(split, city, idx) # label for CURRENT frame
    choice = random.random() # in [0,1)
    if choice < 0.5:
        preceding_frame = all_frames[-2] # do previous frame
        out = run_net.segrun(net, CS.preprocess(preceding_frame))
        hist_baseline += score_util.fast_hist(label.flatten(), out.flatten(), n_cl)
    else:
        curr_frame = all_frames[-1]
        out = run_net.segrun(net, CS.preprocess(curr_frame))
        hist_baseline += score_util.fast_hist(label.flatten(), out.flatten(), n_cl)

acc, cl_acc, mean_iu, fw_iu = score_util.get_scores(hist_baseline)
print('Baseline: Full FCN every other frame')
print('acc\t\t cl acc\t\t mIU\t\t fwIU')
print('{:f}\t {:f}\t {:f}\t {:f}\t'.format(100*acc, 100*cl_acc, 100*mean_iu, 100*fw_iu))

hist_altern = np.zeros((n_cl, n_cl))
for i, idx in enumerate(label_frames):
    if i % 100 == 0:
        print('running {}/{}'.format(i, len(label_frames)))
    city = idx.split('_')[0]
    all_frames = CS.collect_frame_sequence(split, idx, 19)  # list of Images including labeled frame
    label = CS.load_label(split, city, idx)
    curr_frame = all_frames[-1]
    choice = random.random()  # in [0,1)

    if choice < 0.5:
        # Push previous frame through the net
        preceding_frame = all_frames[-2]  # do previous frame
        _ = run_net.segrun(net, CS.preprocess(preceding_frame))
        # Update lower layers on current frame and get prediction
        out = run_net.clockwork_forward(net, CS.preprocess(curr_frame))
        hist_altern += score_util.fast_hist(label.flatten(), out.flatten(), n_cl)
    else:
        out = run_net.segrun(net, CS.preprocess(curr_frame))
        hist_altern += score_util.fast_hist(label.flatten(), out.flatten(), n_cl)

acc, cl_acc, mean_iu, fw_iu = score_util.get_scores(hist_altern)
print('Alternating Clockwork')
print('acc\t\t cl acc\t\t mIU\t\t fwIU')
print('{:f}\t {:f}\t {:f}\t {:f}\t'.format(100 * acc, 100 * cl_acc, 100 * mean_iu, 100 * fw_iu))

# collect all preceding frames in the Cityscapes sequence surrounding each annotated frame
SEQ_LEN = 19


def scoremap_diff(prev_scores, scores):
    prev_seg = prev_scores.argmax(axis=0).astype(np.uint8).copy()
    curr_seg = scores.argmax(axis=0).astype(np.uint8).copy()
    diff = np.array(prev_seg != curr_seg).mean()
    return diff


def adaptive_clockwork_cityscapes(thresh):
    hist = np.zeros((n_cl, n_cl))
    num_frames = 0  # number of frames in total
    num_update_frames = 0  # number of frames when clock fires
    for idx in CS.list_label_frames('val'):
        city = idx.split('_')[0]
        # run on sequence of preceding frames, fully processing the first frame
        frames = CS.collect_frame_sequence('val', idx, SEQ_LEN)
        first_frame, frames = frames[0], frames[1:]
        _ = run_net.segrun(net, CS.preprocess(first_frame))
        prev_score = net.blobs['score_pool4'].data[0].copy()
        num_frames += 1
        for f in frames:
            num_frames += 1
            # Run to pool4 on current frame
            run_net.feed_net(net, CS.preprocess(f))
            net.forward(start='conv1_1', end='score_pool4')
            curr_score = net.blobs['score_pool4'].data[0].copy()

            # Decide whether or not to update to fc7
            if scoremap_diff(prev_score, curr_score) >= thresh:
                net.forward(start='conv5_1', end='upscore2')
                prev_score = net.blobs['score_pool4'].data[0].copy()
                num_update_frames += 1

        # Compute full merge score on the annotated frame (the last frame)
        net.forward(start='score_pool4c')
        out = net.blobs['score'].data[0].argmax(axis=0).astype(np.uint8)
        label = CS.load_label('val', city, idx)
        hist += score_util.score_out_gt(out, label, n_cl=n_cl)

    acc, cl_acc, mean_iu, fw_iu = score_util.get_scores(hist)
    print('Adaptive Clockwork: Threshold', thresh, ' Updated {:d}/{:d} frames ({:2.1f}%)'.format(num_update_frames, num_frames, 100.0*num_update_frames/num_frames))
    print('acc\t cl acc\t mIU\t fwIU')
    print('{:2.1f}\t {:2.1f}\t {:2.1f}\t {:2.1f}\t'.format(100*acc, 100*cl_acc, 100*mean_iu, 100*fw_iu))
    return acc, cl_acc, mean_iu, fw_iu

for thresh in (0.25, 0.35, 0.47):
    adaptive_clockwork_cityscapes(thresh)
