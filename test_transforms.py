import math
import sys
import unittest
from random import uniform

import napari
import numpy as np
import torch
import transforms3d
from scipy import linalg

from data import create_dataset
from models.reg_model import transform_image
from options.test_options import TestOptions


# import pytorch3d

class TestTransforms(unittest.TestCase):
    def test_translations(self):
        sys.argv[1:] += ['--dataroot=/home/kixcodes/Documents/python/'
                         'pytorch-CycleGAN-and-pix2pix-multichannel_images/datasets/facades/reg_patient/']
        print(f'args: {sys.argv[1:]}')
        # self.device = torch.device("cpu")
        opt = TestOptions().parse()  # define shared options
        opt.dataset_mode = 'multichannel'
        opt.visualize = False
        dataset = create_dataset(opt)
        opt.device = torch.device('cpu')
        opt.degree = 20
        opt.transl = 5

        error = []

        for i, data in enumerate(dataset):
            original_mri = data['A']

            random_radX = uniform(math.radians(-opt.degree), math.radians(opt.degree))
            random_radY = uniform(math.radians(-opt.degree), math.radians(opt.degree))
            random_radZ = uniform(math.radians(-opt.degree), math.radians(opt.degree))

            random_translY = uniform(-opt.transl, opt.transl)
            random_translX = uniform(-opt.transl, opt.transl)
            random_translZ = uniform(-opt.transl, opt.transl)

            rot_matr = transforms3d.euler.euler2mat(random_radX,
                                                    random_radY,
                                                    random_radZ)

            random_transl_vec = np.divide((random_translX, random_translY, random_translZ), original_mri.shape[-3:])
            transform = np.zeros((3, 4))
            transform[:, :3] = rot_matr
            transform[:, 3] = random_transl_vec

            ## Original ---> Deformed
            transform_batch = torch.tensor(transform, dtype=torch.float, device=opt.device).view(1, 3, 4)
            random_deformed_mri = transform_image(img=original_mri, transform=transform_batch, device=opt.device)

            ##  Deformed ---> Original
            transform_stacked = np.vstack([transform, [0, 0, 0, 1]])
            transform_inv = linalg.inv(transform_stacked)
            transform_inv = np.delete(transform_inv, 3, 0)
            transform_batch_inv = torch.tensor(transform_inv, dtype=torch.float, device=opt.device).view(1, 3, 4)
            inverted_deformed_mri = transform_image(img=random_deformed_mri, transform=transform_batch_inv,
                                                    device=opt.device)

            error_ = torch.mean(torch.abs(original_mri - inverted_deformed_mri))
            print(f'for image pair {i} error is {error_}')

            error += [error_]

            if opt.visualize:
                with napari.gui_qt():
                    viewer = napari.view_image(np.stack([original_mri.squeeze().numpy(),
                                                         random_deformed_mri.squeeze().numpy(),
                                                         inverted_deformed_mri.squeeze().numpy(),
                                                         original_mri.squeeze().numpy() -
                                                         inverted_deformed_mri.squeeze().numpy()]))

        print(f'mean of error is {np.mean(error)}')
        assert np.mean(error) < 0.05


if __name__ == '__main__':
    unittest.main()
