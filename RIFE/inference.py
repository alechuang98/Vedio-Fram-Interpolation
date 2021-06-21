import os
import pathlib
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

class inference():
    def __init__(self, modelDir='train_log'):
        try:
            try:
                from model.RIFE_HDv2 import Model
                model = Model()
                model.load_model(modelDir, -1)
                print("Loaded v2.x HD model.")
            except:
                from train_log.RIFE_HDv3 import Model
                model = Model()
                model.load_model(modelDir, -1)
                print("Loaded v3.x HD model.")
        except:
            from model.RIFE_HD import Model
            model = Model()
            model.load_model(modelDir, -1)
            print("Loaded v1.x HD model")
        model.eval()
        model.device()
        self.model = model
    
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
                    middle = self.model.inference(tmp_img0, tmp_img1)
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
                    mid = self.model.inference(img_list[j], img_list[j + 1])
                    tmp.append(img_list[j])
                    tmp.append(mid)
                tmp.append(img1)
                img_list = tmp

        img_list = [(img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w] for i in range(len(img_list))]
        return img_list