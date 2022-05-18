import os
import cv2
import numpy as np
import glob
import os.path as osp
from insightface.model_zoo import model_zoo


class LandmarkModel():
    def __init__(self, name, root='./checkpoints'):
        self.models = {}
        root = os.path.expanduser(root)
        onnx_files = glob.glob(osp.join(root, name, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            if onnx_file.find('_selfgen_')>0:
                continue
            model = model_zoo.get_model(onnx_file)
            if model.taskname not in self.models:
                print('find model:', onnx_file, model.taskname)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']


    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640), mode ='None'):
        self.det_thresh = det_thresh
        self.mode = mode
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size)
            else:
                model.prepare(ctx_id)


    def get(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img, threshold=self.det_thresh, max_num=max_num, metric='default')
        if bboxes.shape[0] == 0:
            return None
        det_score = bboxes[..., 4]

        # select the face with the hightest detection score
        best_index = np.argmax(det_score)

        kps = None
        if kpss is not None:
            kps = kpss[best_index]
        return kps

    def gets(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img, threshold=self.det_thresh, max_num=max_num, metric='default')
        return kpss

    def gets(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img, threshold=self.det_thresh, max_num=max_num, metric='default')
        return kpss