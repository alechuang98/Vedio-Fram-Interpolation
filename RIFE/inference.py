import os
import pathlib
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
import numpy as np

from raft_flow import flow_viz
from PIL import Image
from raft_flow.raft import RAFT
from raft_flow.utils import InputPadder

def getFt0Ft1(image0: np.ndarray, image1: np.ndarray, model):
    image0 = torch.from_numpy(image0).permute(2, 0, 1).float()[None].to('cuda')
    image0 = F.interpolate(image0, scale_factor= 0.5, mode="bilinear", align_corners=False)
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()[None].to('cuda')
    image1 = F.interpolate(image1, scale_factor= 0.5, mode="bilinear", align_corners=False)
    padder = InputPadder(image0.shape)
    image0, image1 = padder.pad(image0, image1)

    _, F01 = model(image0, image1, iters=20, test_mode=True)
    F01 = F01[0].permute(1, 2, 0).cpu().numpy()

    _, F10 = model(image1, image0, iters=20, test_mode=True)
    F10 = F10[0].permute(1, 2, 0).cpu().numpy()

    t = 0.5
    Ft0 = -(1 - t) * t * F01 + t ** 2 * F10
    Ft1 = (1 - t) ** 2 * F01 - t * (1 - t) * F10

    Ft0 = Ft0 / 2
    Ft1 = Ft1 / 2

    return torch.from_numpy(np.concatenate((Ft0, Ft1), axis=2)[np.newaxis, :]).permute(0, 3, 1, 2).to('cuda')

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

class inference():
    def __init__(self, modelDir='train_log.large', raftModelPath='raft_flow/raft-things.pth'):
        if modelDir == 'train_log.large':
            from model.RIFE2F15C_m import Model
            model = Model()
            model.load_model(modelDir, -1)
            print("Loaded RIFE-large model.")
        else:
            from train_log.RIFE_HDv3 import Model
            model = Model()
            model.load_model(modelDir, -1)
            print("Loaded v3.x HD model.")

        model.eval()
        model.device()
        self.model = model

        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help="restore checkpoint")
        parser.add_argument('--path', help="dataset for evaluation")
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        args = parser.parse_args()

        raft_model = torch.nn.DataParallel(RAFT(args))
        raft_model.load_state_dict(torch.load(raftModelPath))
        raft_model = raft_model.module
        raft_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        raft_model.eval()
        self.raft_model = raft_model
    
    def get_frame(self, img0Path, img1Path, exp, ratio=0, rthreshold=0.0001, rmaxcycles=20):
        img0 = cv2.imread(img0Path, cv2.IMREAD_UNCHANGED)
        img1 = cv2.imread(img1Path, cv2.IMREAD_UNCHANGED)
        
        img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

        n, c, h, w = img0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        if ratio:
            img_list = [img0]
            img0_ratio = 0.0
            img1_ratio = 1.0
            if ratio <= img0_ratio + rthreshold / 2:
                middle = img0
            elif ratio >= img1_ratio - rthreshold / 2:
                middle = img1
            else:
                tmp_img0 = img0
                tmp_img1 = img1
                for inference_cycle in range(rmaxcycles):
                    pil_img0 = Image.fromarray(cv2.cvtColor((tmp_img0[0] * 255).byte().cpu().numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB))
                    pil_img1 = Image.fromarray(cv2.cvtColor((tmp_img1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB))
                    flow = getFt0Ft1(np.array(pil_img0), np.array(pil_img1), self.raft_model)

                    middle = self.model.inference(tmp_img0, tmp_img1, flow)
                    middle_ratio = ( img0_ratio + img1_ratio ) / 2
                    if ratio - (rthreshold / 2) <= middle_ratio <= ratio + (rthreshold / 2):
                        break
                    if ratio > middle_ratio:
                        tmp_img0 = middle
                        img0_ratio = middle_ratio
                    else:
                        tmp_img1 = middle
                        img1_ratio = middle_ratio
            img_list.append(middle)
            img_list.append(img1)
        else:
            img_list = [img0, img1]
            for i in range(exp):
                tmp = []
                for j in range(len(img_list) - 1):
                    pil_img0 = Image.fromarray(cv2.cvtColor((img_list[j][0] * 255).byte().cpu().numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB))
                    pil_img1 = Image.fromarray(cv2.cvtColor((img_list[j + 1][0] * 255).byte().cpu().numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB))
                    flow = getFt0Ft1(np.array(pil_img0), np.array(pil_img1), self.raft_model)
                    # TODO
                    mid = self.model.inference(img_list[j], img_list[j + 1], flow)
                    tmp.append(img_list[j])
                    tmp.append(mid)
                tmp.append(img1)
                img_list = tmp

        img_list = [(img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w] for i in range(len(img_list))]
        return img_list