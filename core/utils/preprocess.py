__author__ = 'yunbo'

import numpy as np
import torch


def reshape_patch(img_tensor, patch_size):
    assert 5 == img_tensor.ndim
    batch_size = img_tensor.shape[0]
    seq_length = img_tensor.shape[1]
    img_height = img_tensor.shape[2]
    img_width = img_tensor.shape[3]
    num_channels = img_tensor.shape[4]

    a = torch.reshape(img_tensor, [batch_size, seq_length,
                                img_height // patch_size, patch_size,
                                img_width // patch_size, patch_size,
                                num_channels])
    b = torch.permute(a, [0, 1, 2, 4, 3, 5, 6])
    patch_tensor = np.reshape(b, [batch_size, seq_length,
                                  img_height // patch_size,
                                  img_width // patch_size,
                                  patch_size * patch_size * num_channels])
    return patch_tensor


def reshape_patch_back(patch_tensor, patch_size):
    assert 5 == patch_tensor.ndim
    batch_size = np.shape(patch_tensor)[0]
    seq_length = np.shape(patch_tensor)[1]
    patch_height = np.shape(patch_tensor)[2]
    patch_width = np.shape(patch_tensor)[3]
    channels = np.shape(patch_tensor)[4]
    img_channels = channels // (patch_size * patch_size)
    a = np.reshape(patch_tensor, [batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  img_channels])
    b = np.transpose(a, [0, 1, 2, 4, 3, 5, 6])
    img_tensor = np.reshape(b, [batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                img_channels])
    return img_tensor
