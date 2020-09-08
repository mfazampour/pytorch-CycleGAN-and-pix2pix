import unittest
from data import create_dataset
from options.test_options import TestOptions
from scipy import ndimage
from scipy import linalg
from torchio.transforms import (
    RandomAffine,
)
import torch.nn.functional as F

import matplotlib.pyplot as plt
import math
import transforms3d
import torchio
import numpy as np
from random import uniform
from skimage.transform import warp, AffineTransform

from models.reg_model import transform_image

# import pytorch3d

import torch


class TestTransforms(unittest.TestCase):
    def test_translations(self):
        # self.device = torch.device("cpu")
        opt = TestOptions().parse()  # define shared options
        opt.dataset_mode = 'multichannel'
        dataset = create_dataset(opt)
        opt.device = torch.device('cpu')
        opt.degree = 0
        opt.transl = 5

        for i, data in enumerate(dataset):
            if i == 0:
                original_mri = data['A']

                random_radX = uniform(math.radians(-opt.degree), math.radians(opt.degree))
                random_radY = uniform(math.radians(-opt.degree), math.radians(opt.degree))
                random_radZ = uniform(math.radians(-opt.degree), math.radians(opt.degree))

                random_translY = uniform(-opt.transl,opt.transl)
                random_translX = uniform(-opt.transl,opt.transl)
                random_translZ = uniform(-opt.transl,opt.transl)

                rot_matr = transforms3d.euler.euler2mat(random_radX,
                                                        random_radY,
                                                        random_radZ)

                random_transl_vec = np.divide((random_translX,random_translY,random_translZ), original_mri.shape[-3:])
                transform = np.zeros((3, 4))
                transform[:, :3] = rot_matr
                transform[:, 3] = random_transl_vec

                ## Original ---> Deformed
                transform_batch = torch.tensor(transform, dtype=torch.float, device=opt.device).view(1, 3, 4)
                random_deformed_mri = transform_image(img=original_mri, transform=transform_batch, device=opt.device)

                ##  Deformed ---> Original
                transform_stacked = np.vstack([transform, [0,0,0,1]])
                transform_inv = linalg.inv(transform_stacked)
                transform_inv = np.delete(transform_inv,3,0)
                transform_batch_inv = torch.tensor(transform_inv, dtype=torch.float, device=opt.device).view(1, 3, 4)
                inverted_deformed_mri = transform_image(img=random_deformed_mri, transform=transform_batch_inv, device=opt.device)


                print(f'transform matrix: \n {transform}')
                print(f'inverted transform matrix: \n {transform_inv}')

                print(f'min {original_mri.min().numpy()} and max {original_mri.max().numpy()} of torchio transformed ')
                print(f'min {inverted_deformed_mri.min().numpy()} and max {inverted_deformed_mri.max().numpy()} of grid transformed ')
                print(torch.mean(torch.abs(original_mri - inverted_deformed_mri)))



if __name__ == '__main__':
    unittest.main()
