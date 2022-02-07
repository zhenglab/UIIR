from PIL import Image
import numpy as np
import os
import torch
import argparse
import cv2
# import pytorch_ssim_enhancement as pytorch_ssim
import torchvision.transforms.functional as tf
import torchvision
import torch.nn.functional as f
from skimage import data, img_as_float
from skimage.measure import compare_mse as mse
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
import lpips
from tqdm import tqdm
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='test', help='train or test ?')
    parser.add_argument('--dataroot', type=str, default='./datasets/', help='dataset_dir')
    parser.add_argument('--dataname', type=str, default='euvp', help='dataset_dir')
    parser.add_argument('--results_root', type=str, default='mnist', help='dataset_dir')
    parser.add_argument('--dataset_name', type=str, default='ihd', help='dataset_name')
    parser.add_argument("--data_mode", type=str, default="ihd", help="ihd||pix2pix")
    parser.add_argument('--result_name', type=str, default="reflectance", help='use_gen_real')
    parser.add_argument('--model', type=str, default="our", help='use_gen_real')

    return parser.parse_args()


def main():
    cuda = True if torch.cuda.is_available() else False
    IMAGE_SIZE = np.array([256,256])
    opt = parse_args()
    if opt is None:
        exit()
    files = opt.dataroot+opt.dataname+'_test.txt'
    comp_paths = []
    cleared_paths = []
    mask_paths = []
    real_paths = []
    with open(files,'r') as f:
            for line in f.readlines():
                name_str = line.rstrip()
                if opt.model == 'our':
                    cleared_path = os.path.join(opt.results_root,name_str.replace(".jpg", "_"+opt.result_name+".jpg").replace(".png", "_"+opt.result_name+".jpg"))
                    # print(cleared_path)
                    if os.path.exists(cleared_path):
                        real_path = os.path.join(opt.results_root,name_str.replace(".jpg", "_real.jpg").replace(".png", "_real.jpg"))
                        # real_path = os.path.join(opt.dataroot,'real',name_str)
                        real_paths.append(real_path)
                        cleared_paths.append(cleared_path)
                elif opt.model == 'ori':
                    cleared_path = os.path.join(opt.results_root,'inputs',name_str)
                    # print(cleared_path)
                    if os.path.exists(cleared_path):
                        real_path = os.path.join(opt.dataroot,'real',name_str)
                        # real_path = os.path.join(opt.comp1_root,name_str.replace(".jpg", "_real.jpg"))

                        real_paths.append(real_path)
                        cleared_paths.append(cleared_path)
    count = 0


    mse_scores = 0
    psnr_scores = 0
    ssim_scores = 0
    image_size = 256


    for i, cleared_path in enumerate(tqdm(cleared_paths)):
        count += 1

        cleared = Image.open(cleared_path).convert('RGB')
        real = Image.open(real_paths[i]).convert('RGB')
        if real.size[0] != image_size:
            cleared = tf.resize(cleared,[image_size,image_size], interpolation=Image.BICUBIC)
            real = tf.resize(real,[image_size,image_size], interpolation=Image.BICUBIC)

        cleared_np = np.array(cleared, dtype=np.float32)
        real_np = np.array(real, dtype=np.float32)

        cleared = tf.to_tensor(cleared_np).unsqueeze(0).cuda()
        real = tf.to_tensor(real_np).unsqueeze(0).cuda()

        ssim_score = ssim(real_np,cleared_np,multichannel=True)

        mse_score = torch.nn.functional.mse_loss(cleared, real)

        psnr_score = psnr(real_np, cleared_np, data_range=255)
        mse_score = mse_score.item()

        psnr_scores += psnr_score
        mse_scores += mse_score

       
        ssim_scores += ssim_score

    mse_scores_mu = mse_scores/count
    psnr_scores_mu = psnr_scores/count
    ssim_scores_mu = ssim_scores/count


    print("MSE %0.2f | PSNR %0.2f | SSIM %0.4f" % (mse_scores_mu,psnr_scores_mu,ssim_scores_mu))

if __name__ == '__main__':
    main()
