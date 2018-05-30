import numpy as np
import scipy.misc as misc
import os, sys
import torch

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

def discriminator_loss(logits_real, logits_fake, dtype):
    """
    Computes the discriminator loss described above.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    size = logits_real.shape[0]
    true_labels = torch.ones(size).type(dtype)
    fake_labels = torch.zeros(size).type(dtype)
    loss = bce_loss(logits_real, true_labels) + bce_loss(logits_fake, 0)
    return loss


def generator_loss(logits_fake, dtype):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    size = logits_fake.shape[0]
    true_labels = torch.ones(size).type(dtype)
    loss = bce_loss(logits_fake, true_labels)
    return loss

def denorm_img(img):
    return torch.clamp((img+1) * 127.5, 0, 255)


def save_image(image, image_height, image_width, save_dir, name=""):
    """
    Save image by unprocessing assuming mean 127.5
    """
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image += 1
    image *= 127.5
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = np.reshape(image, (image_height, image_width, -1))
    misc.imsave(os.path.join(save_dir, 'pred_img_'+name+'.jpg'), image)


def process_image(image, mean_pixel, norm):
    return (image - mean_pixel) / norm

def unprocess_image(image, mean_pixel, norm):
    return image * norm + mean_pixel