import os
import sys
import cv2
import glob
import torch
import argparse
import numpy as np

from PIL import Image
from raft.raft import RAFT
from raft.utils import flow_viz
from frat.utils.utils import InputPadder

sys.path.append('core')

def getFt0Ft1(image0: np.ndarray, image1: np.ndarray):
    padder = InputPadder(image0.shape)
    image0, image1 = padder.pad(image0, image1)

    _, F01 = model(image0, image1, iters=20, test_mode=True)

    F01 = F01[0].permute(1, 2, 0).cpu().numpy()
    F01 = flow_viz.flow_to_image(F01)

    _, F10 = model(image1, image0, iters=20, test_mode=True)

    F10 = F10[0].permute(1, 2, 0).cpu().numpy()
    F10 = flow_viz.flow_to_image(F10)

    t = 0.5
    Ft0 = -(1 - t) * t * F01 + t ^ 2 * F10
    Ft1 = (1 - t) ^ 2 * F01 - t * (1 - t) * F10

    return Ft0, Ft1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    DEVICE = 'cuda'
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()