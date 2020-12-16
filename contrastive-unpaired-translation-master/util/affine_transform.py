from typing import List
import torch
import torch.nn.functional as F
from pytorch3d.transforms import euler_angles_to_matrix
import torchgeometry as tgm
import scipy
from util import se3
#
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
   # img = img.unsqueeze(dim=0)
    #img = img.view(-1, 1, 9, 256, 256)
    img = img.view(-1, 1, 80, 80, 80)

    grid = F.affine_grid(transform[:, :3, :], img.shape).to(device)
    x_trans = F.grid_sample(img, grid, mode='nearest', padding_mode='border')
    x_trans = torch.tensor(x_trans.view(-1,1, 80, 80, 80))
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
    # +- 20 degrees rotation
    rotation = (2 * torch.rand((n, 3), dtype=dtype) - 1) * 0.436332
    # +- 5 mm translation
    translation = (2 * torch.rand((n, 3), dtype=dtype) - 1) * 0.0625
    vector = torch.cat((rotation, translation), dim=1)
    affines = torch.zeros((n, 4, 4), dtype=dtype)
    for i in range(n):
        affine = torch.tensor(se3.vector_to_matrix((vector[i, :])))
        affines[i, ...] = affine
    return affines.to(device), vector.to(device)

def create_identity_affine(n,dtype=torch.float):
    rotation = torch.eye(3)
    translation = (torch.zeros((3, n), dtype=dtype))

    vector = torch.cat((rotation, translation), dim=1)
    last_row = torch.eye(4)[3,:]

    affines = torch.cat((vector, last_row.unsqueeze(dim=0)), dim=0)
    return affines

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


def torch_vector_to_matrix(t: torch.Tensor, device):
    matrix = euler_angles_to_matrix(t[:,0:3], convention=('XYZ'))
  #  transl_vec = torch.div(t[:,3:], original_mri.shape[-3:])

    rotmatrix = torch.cat((matrix,t[:,3:].view(-1,3,1)), dim=2)

    zeros = torch.zeros((t.shape[0],1,3)).to(device)
    ones = torch.ones((t.shape[0],1,1)).to(device)
    adding = torch.cat((zeros,ones), dim =2 )
   # print(f' adding {adding.shape} matrix {adding}')

    rotmatrix = torch.cat((rotmatrix,adding), dim=1)
   # print(f'rotmatrix  {rotmatrix} and shape { rotmatrix.shape}')

    return rotmatrix

def inverse_matrix(t = torch.Tensor):
    return tgm.inverse_transformation(t)

def inverse_matrix_test(t = torch.Tensor):
    return scipy.linalg.inv(t.view(4,4).cpu())