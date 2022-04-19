
import paddle
import argparse
import cv2
import numpy as np
import os
from models.model import FaceSwap, l2_norm
from models.arcface import IRBlock, ResNet
from utils.align_face import back_matrix, dealign, align_img
from utils.util import paddle2cv, cv2paddle
from utils.prepare_data import LandmarkModel
from tqdm import tqdm

def get_id_emb(id_net, id_img):
    id_img = cv2.resize(id_img, (112, 112))
    id_img = cv2paddle(id_img)
    mean = paddle.to_tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
    std = paddle.to_tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
    id_img = (id_img - mean) / std

    id_emb, id_feature = id_net(id_img)
    id_emb = l2_norm(id_emb)

    return id_emb, id_feature


def video_test(args):

    paddle.set_device("gpu" if args.use_gpu else 'cpu')
    faceswap_model = FaceSwap(args.use_gpu)

    id_net = ResNet(block=IRBlock, layers=[3, 4, 23, 3])
    id_net.set_dict(paddle.load('./checkpoints/arcface.pdparams'))

    id_net.eval()

    weight = paddle.load('./checkpoints/MobileFaceSwap_224.pdparams')

    landmarkModel = LandmarkModel(name='landmarks')
    landmarkModel.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))
    id_img = cv2.imread(args.source_img_path)

    landmark = landmarkModel.get(id_img)
    if landmark is None:
        print('**** No Face Detect Error ****')
        exit()
    aligned_id_img, _ = align_img(id_img, landmark)

    id_emb, id_feature = get_id_emb(id_net, aligned_id_img)

    faceswap_model.set_model_param(id_emb, id_feature, model_weight=weight)
    faceswap_model.eval()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture()
    cap.open(args.target_video_path)
    videoWriter = cv2.VideoWriter(os.path.join(args.output_path, os.path.basename(args.target_video_path)), fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    all_f = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in tqdm(range(int(all_f))):
        ret, frame = cap.read()
        landmark = landmarkModel.get(frame)
        if landmark is not None:
            att_img, back_matrix = align_img(frame, landmark)
            att_img = cv2paddle(att_img)
            res, mask = faceswap_model(att_img)
            res = paddle2cv(res)
            mask = np.transpose(mask[0].numpy(), (1, 2, 0))
            res = dealign(res, frame, back_matrix, mask)
            frame = res
        else:
            print('**** No Face Detect Error ****')
        videoWriter.write(frame)
    cap.release()
    videoWriter.release()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="MobileFaceSwap Test")

    parser = argparse.ArgumentParser(description="MobileFaceSwap Test")
    parser.add_argument('--source_img_path', type=str, help='path to the source image')
    parser.add_argument('--target_video_path', type=str, help='path to the target video')
    parser.add_argument('--output_path', type=str, default='results', help='path to the output videos')
    parser.add_argument('--image_size', type=int, default=224,help='size of the test images (224 SimSwap | 256 FaceShifter)')
    parser.add_argument('--merge_result', type=bool, default=True, help='output with whole image')
    parser.add_argument('--use_gpu', type=bool, default=False)

    args = parser.parse_args()
    video_test(args)