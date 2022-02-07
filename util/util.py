"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
# from PIL import Image
from PIL import Image,ImageDraw, ImageFont
import matplotlib.font_manager as fm # to create font
import torch.nn as nn
import os
import torch.nn.functional as F
from util.tools import *

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 0) / 1.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    image_numpy = np.clip(image_numpy, 0,255)
    return image_numpy.astype(imtype)

def retry_load_images(image_paths, retry=10, backend="pytorch"):
    """
    This function is to load images with support of retrying for failed load.

    Args:
        image_paths (list): paths of images needed to be loaded.
        retry (int, optional): maximum time of loading retrying. Defaults to 10.
        backend (str): `pytorch` or `cv2`.

    Returns:
        imgs (list): list of loaded images.
    """
    for i in range(retry):
        imgs = []
        for image_path in image_paths:
            with PathManager.open(image_path, "rb") as f:
                img_str = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(img_str, flags=cv2.IMREAD_COLOR)
            imgs.append(img)

        if all(img is not None for img in imgs):
            if backend == "pytorch":
                imgs = torch.as_tensor(np.stack(imgs))
            return imgs
        else:
            logger.warn("Reading failed. Will retry.")
            time.sleep(1.0)
        if i == retry - 1:
            raise Exception("Failed to load images {}".format(image_paths))

def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path,quality=95) #added by Mia (quality)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def calc_unmask_mean(feat,mask, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C, H, W = size
    feat_unmask = feat*(1-mask)
    feat_unmask_sum = feat.view(N, C, -1).sum(dim=2)
    mask_pixel_sum = mask.view(mask.size(0), mask.size(1), -1).sum(dim=2)
    feat_unmask_mean = feat_unmask_sum.div(H*W-mask_pixel_sum).view(N, C, 1, 1)
    return feat_unmask_mean

def saveprint(opt, name, message):
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, '{}_opt.txt'.format(name))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

def clip_by_tensor(t,t_min,t_max=None):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t=t.float()
    t_min=t_min.float()
    
 
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    if t_max is not None:
        t_max=t_max.float()
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss

def gradient(input_tensor, direction):
    # input_tensor = input_tensor.permute(0, 3, 1, 2)
    b,c,h, w = input_tensor.size()

    smooth_kernel_x = torch.reshape(torch.Tensor([[0., 0.], [-1., 1.]]), (1, 1, 2, 2)).repeat(1,c,1,1).to(input_tensor.get_device())
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)

    assert direction in ['x', 'y']
    if direction == "x":
        kernel = smooth_kernel_x
    else:
        kernel = smooth_kernel_y

    out = F.conv2d(input_tensor, kernel, padding=(1, 1))
    out = torch.abs(out[:, :, 0:h, 0:w])
    return out
    # return out.permute(0, 2, 3, 1)

def gradient_sobel(input_tensor, direction):
    # input_tensor = input_tensor.permute(0, 3, 1, 2)
    h, w = input_tensor.size()[2], input_tensor.size()[3]

    smooth_kernel_x = torch.reshape(torch.Tensor([[-1., 0., 1], [-2.,0, 2.], [-1, 0, 1]]), (1, 1, 3, 3)).to(input_tensor.get_device())
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)

    assert direction in ['x', 'y']
    if direction == "x":
        kernel = smooth_kernel_x
    else:
        kernel = smooth_kernel_y

    out = F.conv2d(input_tensor, kernel, padding=(1, 1))
    # out = torch.abs(out[:, :, 0:h, 0:w])
    return out


def ave_gradient(input_tensor, direction):
    return (F.avg_pool2d(gradient(input_tensor, direction), 3, stride=1, padding=1))



def smooth(input_l, input_r=None):
    rgb_weights = torch.Tensor([0.2989, 0.5870, 0.1140])
    input_l = torch.tensordot(input_l, rgb_weights.to(input_l.device), dims=([1], [-1]))
    input_l = torch.unsqueeze(input_l, 1)
    return torch.norm(gradient_sobel(input_l, 'x'), 1) + torch.norm(gradient_sobel(input_l, 'y'), 1)

def calImageGradient(image):
    if image.size(1) >1:
        image = rgbtogray(image)
    gradient_x = gradient(image, 'x')
    gradient_y = gradient(image, 'y')
    gradient_i = gradient_x + gradient_y
    return gradient_i

def calRobustRetinexG(image):
    
    gradient_i = calImageGradient(image)
    k = 1 + 10 * torch.exp(-torch.abs(gradient_i).div(10))
    return gradient_i * k

def rgbtogray(image):
    rgb_weights = torch.Tensor([0.2989, 0.5870, 0.1140]).to(image.get_device())
    input_r = torch.tensordot(image, rgb_weights, dims=([-3], [-1]))
    input_r = input_r.unsqueeze(-3)
    return input_r
    

def compute_smooth_loss(pred_disp):
        def gradient(pred):
            D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            return D_dx, D_dy
        dx, dy = gradient(pred_disp)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        return torch.mean(torch.abs(dx2)) + \
               torch.mean(torch.abs(dxdy)) + \
               torch.mean(torch.abs(dydx)) + \
               torch.mean(torch.abs(dy2))

def exposure_loss(gen, mask):
    mask_image_mean = calc_unmask_mean(gen, mask)
    mean = F.adaptive_avg_pool2d(gen, 16)
    d = torch.mean(torch.pow(mean- mask_image_mean,2))
    return d

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    # images = torch.nn.ZeroPad2d(paddings)(images)
    images = torch.nn.ReflectionPad2d(paddings)(images)
    return images


def gredient_xy(images):
    gradient_x = gradient(images, 'x')
    gradient_y = gradient(images, 'y')
    gredient_images = gradient_x + gradient_y
    return gredient_images



