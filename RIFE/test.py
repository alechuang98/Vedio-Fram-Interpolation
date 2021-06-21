import cv2
import numpy as np
import math
from skimage.metrics import structural_similarity
import os
import glob
import pathlib
from tqdm import tqdm
from inference import inference


def psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return PSNR

def ssim(img1, img2):
    return structural_similarity(img1.astype(np.float32)/255., img2.astype(np.float32)/255., gaussian_weights=True, sigma=1.5, use_sample_covariance=False, multichannel=True)

inf = inference()

if __name__ == '__main__':

    """ 0_center_frame """
    sequences = ['7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
    total_psnr = 0
    total_ssim = 0
    for sq in tqdm(sequences):
        # read inputs
        I0 = '../data/testing/0_center_frame/'+sq+'/input/frame10.png'
        I1 = '../data/testing/0_center_frame/'+sq+'/input/frame11.png'
        out_path = 'output/0_center_frame/'+sq+'/frame10i11.jpg'

        img = inf.get_frame(I0, I1, exp=1)[1]

        directory = "/".join(out_path.split('/')[:-1])
        if not os.path.exists(directory):
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])


    """ 1_30fps_to_240fps """
    sequences = ['3', '4']
    total_psnr = 0
    total_ssim = 0
    for sq in tqdm(sequences):
        for i in range(12):
            # read inputs
            I0 = '../data/testing/1_30fps_to_240fps/'+sq+'/'+str(i)+'/input/{:0>5d}.jpg'.format(i*8)
            I1 = '../data/testing/1_30fps_to_240fps/'+sq+'/'+str(i)+'/input/{:0>5d}.jpg'.format(i*8+8)
            out_dir = 'output/1_30fps_to_240fps/'+sq+'/'+str(i)+'/'
            # print(I0, I1)
            img = inf.get_frame(I0, I1, exp=3)

            if not os.path.exists(out_dir):
                pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        
            for j in range(1, 8):
                cv2.imwrite(out_dir+'{:0>5d}.jpg'.format(j+i*8), img[j], [cv2.IMWRITE_JPEG_QUALITY, 100])


    """ 2_24fps_to_60fps """
    sequences = ['3', '4']
    total_psnr = 0
    total_ssim = 0
    cnt = 0
    for sq in tqdm(sequences):
        for i in range(8):
            # read inputs
            I0 = '../data/testing/2_24fps_to_60fps/'+sq+'/'+str(i)+'/input/{:0>5d}.jpg'.format(i*10)
            I1 = '../data/testing/2_24fps_to_60fps/'+sq+'/'+str(i)+'/input/{:0>5d}.jpg'.format(i*10+10)
            out_dir = 'output/2_24fps_to_60fps/'+sq+'/'+str(i)+'/'
            # print(I0, I1)

            if not os.path.exists(out_dir):
                pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        
            for j in range(i*10+1, i*10+10):
                if j % 4 != 0:
                    continue
                cnt += 1
                img = inf.get_frame(I0, I1, exp=0, ratio=(j%10)/10)[1]
                cv2.imwrite(out_dir+'{:0>5d}.jpg'.format(j), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    