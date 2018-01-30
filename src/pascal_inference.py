import cv2
import time
import sys
sys.path.append("/home/z840/caffe-tmp/caffe/python")
import caffe
import numpy as np
from PIL import Image
from lib import run_net
import scipy.misc as misc
from collections import namedtuple
from datasets.pascal_voc import Pascal


class ModelName(object):
    name_fcn = "fcn"
    name_baseline_3stage = "baseline_3stage"
    name_baseline_2stage = "baseline_2stage"
    name_pipeline_2stage = "pipeline_2stage"
    name_pipeline_3stage = "pipeline_3stage"
    pass


class RunPascal(object):

    def __init__(self, model_name, capture_para, width, height):
        self.model_name = model_name
        self.method, self.net = self._get_model()
        self.capture = self._init_capture(capture_para, width, height)

        self.count = 0
        pass

    # 得到模型
    def _get_model(self):
        Method = namedtuple('Method', 'arch weights infer_func')
        if self.model_name == ModelName.name_fcn:
            method = Method('../model/voc-fcn8s.prototxt',
                            '../model/voc-fcn8s-heavy.caffemodel', run_net.segrun)
        elif self.model_name == ModelName.name_baseline_3stage:
            method = Method('../model/voc-fcn-pool3.prototxt',
                            '../model/voc-fcn-pool3.caffemodel', run_net.segrun)
        elif self.model_name == ModelName.name_baseline_2stage:
            method = Method('../model/voc-fcn-pool4.prototxt',
                            '../model/voc-fcn-pool4.caffemodel', run_net.segrun)
        elif self.model_name == ModelName.name_pipeline_2stage:
            method = Method('../model/stage-voc-fcn8s.prototxt',
                            '../model/voc-fcn8s-heavy.caffemodel', run_net.pipeline_2stage_forward)
        else:
            method = Method('../model/stage-voc-fcn8s.prototxt',
                            '../model/voc-fcn8s-heavy.caffemodel', run_net.pipeline_3stage_forward)
        return method, caffe.Net(method.arch, method.weights, caffe.TEST)

    # 初始化Capture
    @staticmethod
    def _init_capture(capture_para, width, height):
        cap = cv2.VideoCapture(capture_para)
        cap.set(3, width)
        cap.set(4, height)
        return cap

    # 释放Capture
    def _release_capture(self):
        self.capture.release()
        cv2.destroyAllWindows()
        pass

    # 得到一帧图像
    def _get_frame(self):
        if self.capture.isOpened():
            _ret, _frame = self.capture.read()
            if _ret:
                _frame = cv2.flip(_frame, 1)
                cv2.waitKey(1)
            else:
                raise Exception("...........")
        else:
            raise Exception(".....................")
        return _frame

    # 着色
    @staticmethod
    def _color(frame):
        im = np.asarray(frame, np.uint8)
        draw_palette = Image.fromarray(im, 'P')
        draw_palette.putpalette(np.load('../datasets/palette.npy').tolist())
        im = misc.fromimage(draw_palette)
        return im

    # 初始化执行
    def _init_run(self):
        if self.model_name == ModelName.name_pipeline_3stage:
            run_net.pipeline_fill_3stage(self.net, Pascal.pre_process(self._get_frame()),
                                         Pascal.pre_process(self._get_frame()))
        elif self.model_name == ModelName.name_pipeline_2stage:
            run_net.pipeline_fill_2stage(self.net, Pascal.pre_process(self._get_frame()))
        pass

    # 执行
    def run(self):
        self._init_run()

        while self.capture.isOpened():
            self.count += 1
            print("{} begin {}".format(time.strftime("%H:%M:%S", time.localtime()), self.count))

            frame = self._get_frame()
            out = self.method.infer_func(self.net, Pascal.pre_process(frame))
            out = self._color(out)
            cv2.imshow("capture", frame)
            cv2.imshow("out", out)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            pass

        self._release_capture()
        pass

    pass

if __name__ == '__main__':

    caffe.set_device(0)
    caffe.set_mode_gpu()

    run_pascal = RunPascal(model_name=ModelName.name_pipeline_3stage, capture_para=0, width=600, height=400)
    run_pascal.run()

    pass
