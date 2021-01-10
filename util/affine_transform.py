from typing import List

import torch
import torch.nn.functional as F
import numpy as np
try:
    import napari
except:
    print("failed to load napari")

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
    grid = F.affine_grid(transform[:, :3, :], img.shape, align_corners=False).to(device)
    x_trans = F.grid_sample(img, grid, padding_mode='border', align_corners=False)
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
    rotation = (2 * torch.rand((n, 3), dtype=dtype) - 1) * 0.4
    translation = (2 * torch.rand((n, 3), dtype=dtype) - 1) * 0.2
    vector = torch.cat((rotation, translation), dim=1)
    affines = torch.zeros((n, 4, 4), dtype=dtype)
    for i in range(n):
        affine = se3.vector_to_matrix(vector[i, :])
        affines[i, ...] = affine
    # vector[:, -3:] *= torch.tensor(img_shape, dtype=translation.dtype, device=translation.device)
    return affines.to(device), vector.to(device)


def tensor_vector_to_matrix(t: torch.Tensor):
    affines = torch.zeros((t.shape[0], 4, 4), dtype=t.dtype)
    for i in range(t.shape[0]):
        affine = se3.vector_to_matrix(t[i, :].cpu())
        affines[i, ...] = affine
    return affines.to(t.device)

def tensor_matrix_to_vector(t: torch.Tensor):
    vectors = torch.zeros((t.shape[0], 6), dtype=t.dtype)
    for i in range(t.shape[0]):
        vector = se3.matrix_to_vector(t[i, :].cpu())
        vectors[i, ...] = vector
    return vectors.to(t.device)

def show_volumes(img_list: List[torch.Tensor]):
    img_list_np = []
    for t in img_list:
        img_list_np.append(t[0, ...].cpu().squeeze().numpy())
    try:
        with napari.gui_qt():
            napari.view_image(np.stack(img_list_np))
    except:
        print("failed to load napari")

