import numpy as np
import scipy.misc as misc
import os, sys
import torch
from PIL import Image
import math

def getProc_Data_Dir(cur_dir, split_name):
    return os.path.join(cur_dir, 'DF_'+ split_name +'_data')

def getProc_HDF5Data(data_dir, split_name):
    return os.path.join(data_dir, 'DF_'+ split_name +'.hdf5')

def get_split(split='train'):
    split_train = 'train'
    split_test = 'test'
    cur_dir = os.getcwd()

    Proc_train_data_dir = getProc_Data_Dir(cur_dir, split_name = split_train)
    Proc_test_data_dir =  getProc_Data_Dir(cur_dir, split_name = split_test)

    train_file = getProc_HDF5Data(Proc_train_data_dir, split_name = split_train)
    test_file = getProc_HDF5Data(Proc_test_data_dir, split_name = split_test)

    with open(os.path.join(Proc_train_data_dir,'total_pairs.txt'), 'r') as f:
        total_pairs_train = int(f.readline().split(':')[-1])

    with open(os.path.join(Proc_test_data_dir,'total_pairs.txt'), 'r') as f:
        total_pairs_test = int(f.readline().split(':')[-1])

    if split=='train':
        return {'filename': train_file, 'total_data': total_pairs_train}
    return {'filename': test_file, 'total_data': total_pairs_test}

def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def discriminator_loss(logits_real, logits_fake,is_wgan=False):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    if is_wgan:
        return -(torch.mean(logits_real)-torch.mean(logits_fake))
    true = torch.ones_like(logits_real)
    fake = torch.zeros_like(logits_fake)
    loss = bce_loss(logits_real, true) + bce_loss(logits_fake, fake)
    return loss


def generator_loss(logits_fake,is_wgan=False):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    if is_wgan:
        return -torch.mean(logits_fake)
    true = torch.ones_like(logits_fake)
    loss = bce_loss(logits_fake, true)
    return loss

def denorm_img(img):
    return torch.clamp((img+1) * 127.5, 0, 255)

def process_image(image, mean_pixel, norm):
    return (image - mean_pixel) / norm

def unprocess_image(image, mean_pixel, norm):
    return image * norm + mean_pixel
