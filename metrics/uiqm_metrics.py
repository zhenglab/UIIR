#!/usr/bin/env python
"""
# > Script for measuring quantitative performances in terms of
#    - Structural Similarity Metric (SSIM) 
#    - Peak Signal to Noise Ratio (PSNR)
#    - Underwater Image Quality Measure (UIQM)
# Maintainer: Jahid (email: islam034@umn.edu)
# Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)
"""
## python libs
import os
import numpy as np
from glob import glob
from os.path import join
from ntpath import basename
from PIL import Image
## local libs
import argparse
from tqdm import tqdm
from uiqm_utils import getUIQM
import cv2


# measurement in a common dimension
im_w, im_h = 256, 256

## data paths
REAL_im_dir = "data/sample_test_ufo/lrd/"  # real/input im-dir with {f.ext}
GEN_im_dir  = "data/output/keras_out/"  # generated im-dir with {f_SESR/EN.ext}
GTr_im_dir  = "data/sample_test_ufo/hr/"  # ground truth im-dir with {f.ext}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='test', help='train or test ?')
    parser.add_argument('--dataroot', type=str, default='./datasets/', help='dataset_dir')
    parser.add_argument('--dataname', type=str, default='euvp', help='dataset_dir')
    parser.add_argument('--results_root', type=str, default='mnist', help='dataset_dir')
    parser.add_argument('--result_name', type=str, default="reflectance", help='use_gen_real')
    return parser.parse_args()


## mesures uqim for all images in a directory
def measure_UIQMs(dir_name, file_ext=None):
    """
      # measured in RGB
      Assumes:
        * dir_name contain generated images 
        * to evaluate on all images: file_ext = None 
        * to evaluate images that ends with "_SESR.png" or "_En.png"  
            * use file_ext = "_SESR.png" or "_En.png" 
    """
    # paths = sorted(glob(join(dir_name, "*.*")))
    # if file_ext:
    #     paths = [p for p in paths if p.endswith(file_ext)]
    files = opt.dataroot+opt.dataname+'_test.txt'
    paths = []
    with open(files,'r') as f:
        for line in f.readlines():
            name_str = line.rstrip()
            path = os.path.join(opt.results_root,name_str.replace(".png", "_"+opt.result_name+".jpg").replace(".jpg", "_"+opt.result_name+".jpg"))
            # path = os.path.join(opt.results_root,name_str)
            paths.append(path)
    uqims = []
    uciqes = []
    for img_path in tqdm(paths):
        im = Image.open(img_path).resize((im_w, im_h))
        uqims.append(getUIQM(np.array(im)))
        # uciqes.append(getUCIQE(img_path))
    #     ref = np.array(im.convert('LA'))[:,:,0] # ref
    #     uciqes.append(niqe(ref))
    # return np.array(uqims), np.array(uciqes)
    return np.array(uqims)


def measure_SSIM(gtr_dir, gen_dir):
    """
      # measured in RGB
      Assumes:
        * gtr_dir contain ground-truths {filename.ext}
        * gen_dir contain generated images {filename_SESR.png} 
    """
    gtr_paths = sorted(glob(join(gtr_dir, "*.*")))
    gen_paths = sorted(glob(join(gen_dir, "*.*")))
    ssims = []
    for gtr_path in gtr_paths:
        fname = basename(gtr_path).split('.')[0]
        gen_path = join(gen_dir, fname + '_SESR.png') # for SESR
        #gen_path = join(gen_dir, fname + '_En.png') # enhancement
        if gen_path in gen_paths:
            r_im = Image.open(gtr_path).resize((im_w, im_h))
            g_im = Image.open(gen_path).resize((im_w, im_h))
            # get ssim on RGB channels (SOTA norm)
            ssim = getSSIM(np.array(r_im), np.array(g_im))
            ssims.append(ssim)
    return np.array(ssims)


def measure_PSNR(gtr_dir, gen_dir):
    """
      # measured in lightness channel 
      Assumes:
        * gtr_dir contain ground-truths {filename.ext}
        * gen_dir contain generated images {filename_SESR.png}
    """
    gtr_paths = sorted(glob(join(gtr_dir, "*.*")))
    gen_paths = sorted(glob(join(gen_dir, "*.*")))
    psnrs = []
    for gtr_path in gtr_paths:
        fname = basename(gtr_path).split('.')[0]
        gen_path = join(gen_dir, fname + '_SESR.png') # for SESR
        #gen_path = join(gen_dir, fname + '_En.png') # enhancement
        if gen_path in gen_paths:
            r_im = Image.open(gtr_path).resize((im_w, im_h))
            g_im = Image.open(gen_path).resize((im_w, im_h))
            # get psnt on L channel (SOTA norm)
            r_im = r_im.convert("L"); g_im = g_im.convert("L")
            psnr = getPSNR(np.array(r_im), np.array(g_im))
            psnrs.append(psnr)
    return np.array(psnrs)

### compute SSIM and PSNR
# SSIM_measures = measure_SSIM(GTr_im_dir, GEN_im_dir)
# PSNR_measures = measure_PSNR(GTr_im_dir, GEN_im_dir)
# print ("SSIM >> Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures)))
# print ("PSNR >> Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures)))

if __name__ == '__main__':
    # main()
    opt = parse_args()
    gen_uqims = measure_UIQMs(opt.dataroot, file_ext="_reflectance.jpg")  # or file_ext="_SESR.png"
    print ("Generated UQIM >> Mean: {0} std: {1}".format(np.mean(gen_uqims), np.std(gen_uqims)))