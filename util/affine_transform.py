from typing import List

import torch
import torch.nn.functional as F
import numpy as np
import napari

from util import se3

def set_border_value(img: torch.Tensor, value=None):
    if value is None:
        value = img.min()
    img[:, :, 0, :, :] = value
    img[:, :, -1, :, :] = value
    img[:, :, :, 0, :] = value
    img[:, :, :, -1, :] = value
    img[:, :, :, :, 0] = value
    img[:, :, :, :, -1] = value
    return img


def transform_image(img: torch.Tensor, transform, device):
    """

    Parameters
    ----------
    img
    transform
    device

    Returns
    -------
    transformed image
    """
    grid = F.affine_grid(transform[:, :3, :], img.shape).to(device)
    x_trans = F.grid_sample(img, grid, padding_mode='border')
    # x_trans = torch.tensor(x_trans.view(1,9,256,256))
    return x_trans


def create_random_affine(n, img_shape=torch.tensor([128.0, 128.0, 128.0]), dtype=torch.float,
                         device=torch.device('cpu')):
    """
    creates a random rotation (in axis-angle presentation) and translation and returns the affine matrix, and the 6D pose
    Parameters
    ----------
    img_shape : need to normalize the translation to the img_shape since F.affine_grid will expect it like that
    n : batch size
    dtype
    device

    Returns
    -------
    affine
    vector
    """
    rotation = torch.rand((n, 3), dtype=dtype) * 0.2
    translation = torch.rand((n, 3), dtype=dtype) * 5
    translation /= torch.tensor(img_shape, dtype=translation.dtype, device=translation.device)
    vector = torch.cat((rotation, translation), dim=1)
    affines = torch.zeros((n, 4, 4), dtype=dtype)
    for i in range(n):
        affine = se3.vector_to_matrix(vector[i, :])
        affines[i, ...] = affine
    return affines.to(device), vector.to(device)


def tensor_vector_to_matrix(t: torch.Tensor):
    affines = torch.zeros((t.shape[0], 4, 4), dtype=t.dtype)
    for i in range(t.shape[0]):
        affine = se3.vector_to_matrix(t[i, :].cpu())
        affines[i, ...] = affine
    return affines.to(t.device)

def show_volumes(img_list: List[torch.Tensor]):
    img_list_np = []
    for t in img_list:
        img_list_np.append(t[0, ...].cpu().squeeze().numpy())
    with napari.gui_qt():
        napari.view_image(np.stack(img_list_np))
