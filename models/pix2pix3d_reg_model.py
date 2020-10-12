import numpy as np
import torch
import os
import argparse
from collections import OrderedDict
import GPUtil
from typing import Tuple

import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from .pix2pix3d_model import Pix2Pix3dModel
from . import networks3d
from . import networks
from util import affine_transform


class Pix2Pix3dRegModel(Pix2Pix3dModel):

    @staticmethod
    def modify_commandline_options(parser: argparse.ArgumentParser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        Parameters
        ----------
        parser
        """
        parser.set_defaults(norm='batch', netG='unet_128', dataset_mode='volume')
        parser.add_argument('--netSeg', type=str, default='unet_128', help='Type of network used for segmentation')
        parser.add_argument('--num_classes', type=int, default=2, help='num of classes for segmentation')
        parser.add_argument('--show_volumes', type=bool, default=False, help='visualize transformed volumes w napari')

        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--no_lsgan', type=bool, default=False)
            parser.add_argument('--lambda_Reg', type=float, default=0.5, help='weight for the registration loss')
            parser.add_argument('--lr_Reg', type=float, default=0.00001, help='learning rate for the reg. network opt.')
            # parser.add_argument('--visualize_volume', type=bool, default=False)

        return parser

    def __init__(self, opt):
        """Initialize the class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        For visualization, we set 'visual_names' as 'real_A' (input real image),
        'real_B_rgb' (ground truth RGB image), and 'fake_B_rgb' (predicted RGB image)
        We convert the Lab image 'real_B' (inherited from Pix2pixModel) to a RGB image 'real_B_rgb'.
        we convert the Lab image 'fake_B' (inherited from Pix2pixModel) to a RGB image 'fake_B_rgb'.
        """
        super().__init__(opt)
        self.isTrain = opt.isTrain

        # specify the training losses you want to print out. The training/test scripts will call
        # <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'Reg_fake', 'Reg_real']

        # specify the images you want to save/display. The training/test scripts will call
        # <BaseModel.get_current_visuals>
        # the empty slice is added since the visualization would be 3 * 4
        self.visual_names += ['reg_A_center_sag', 'diff_A_center_sag', 'reg_B_center_sag', 'diff_B_center_sag']
        self.visual_names += ['reg_A_center_cor', 'diff_A_center_cor', 'reg_B_center_cor', 'diff_B_center_cor']
        self.visual_names += ['reg_A_center_axi', 'diff_A_center_axi', 'reg_B_center_axi', 'diff_B_center_axi']
        self.visual_names += ['diff_orig_center_sag', 'diff_orig_center_cor', 'diff_orig_center_axi', 'empty_img_4']
        self.visual_names += ['deformed_center_sag', 'deformed_center_cor', 'deformed_center_axi', 'empty_img_5']
        # specify the models you want to save to the disk. The training/test scripts will call
        # <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D', 'Reg']
        else:  # during test time, only load G
            self.model_names = ['G', 'Reg']

        # We are using DenseNet for rigid registration
        # self.netReg = networks3d.define_reg_model(n_input_channels=2, num_classes=6, gpu_ids=self.gpu_ids)
        self.netReg = networks3d.define_reg_model(model_type='NormalNet', num_classes=6, gpu_ids=self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionReg = networks.RegistrationLoss()
            self.optimizer_Reg = torch.optim.Adam(self.netReg.parameters(), lr=opt.lr_Reg, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_Reg)

    # def name(self):
    #     return 'Pix2Pix3dModel'

    def clean_tensors(self):
        all_members = self.__dict__.keys()
        # print(f'{all_members}')
        # GPUtil.showUtilization()
        for item in all_members:
            if isinstance(self.__dict__[item], torch.Tensor):
                self.__dict__[item] = None
        torch.cuda.empty_cache()
        # GPUtil.showUtilization()

    def set_input(self, input):
        self.clean_tensors()
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        affine, self.gt_vector = affine_transform.create_random_affine(self.real_B.shape[0],
                                                                       self.real_B.shape[-3:],
                                                                       self.real_B.dtype,
                                                                       device=self.real_B.device)
        self.transformed_B = affine_transform.transform_image(self.real_B, affine, self.real_B.device)
        if self.opt.show_volumes:
            affine_transform.show_volumes([self.real_B, self.transformed_B])

    def forward(self):
        super(Pix2Pix3dRegModel, self).forward()
        self.reg_A_params = self.netReg(torch.cat([self.fake_B, self.transformed_B], dim=1))
        self.reg_B_params = self.netReg(torch.cat([self.real_B, self.transformed_B], dim=1))

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)

        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        # Third, Seg(G(A)) =
        # self.loss_G_Reg = self.criterionReg(self.reg_A_params, self.gt_vector) * self.opt.lambda_Reg

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1  # + self.loss_G_Reg
        self.loss_G.backward()

    def normalize_vector_angle(self, output):
        norm = torch.norm(output[:3])
        k = torch.floor(norm / (2 * np.pi))
        output_n = torch.clone(output)
        output_n[:3] = output[:3] * (norm - (k * 2 * np.pi)) / norm
        return output_n

    def backward_Reg(self):
        """
        Calculate Segmentation loss to update the segmentation networks
        Returns
        -------
        """
        reg_A_params = self.netReg(torch.cat([self.fake_B.detach(), self.transformed_B], dim=1))
        self.loss_Reg_fake = self.criterionReg(reg_A_params,
                                               self.gt_vector) * self.opt.lambda_Reg  # to be of the same order as loss_G_Seg
        self.loss_Reg_real = self.criterionReg(self.reg_B_params,
                                               self.gt_vector) * self.opt.lambda_Reg  # to be of the same order as loss_G_Seg
        # self.loss_Reg_real = torch.nn.functional.mse_loss(self.reg_B_params, self.gt_vector * 10)

        reg_B_params_n = self.normalize_vector_angle(self.reg_B_params)
        loss_Reg_real_n = self.criterionReg(reg_B_params_n, self.gt_vector) * self.opt.lambda_Reg
        tmp_affine_B = affine_transform.tensor_vector_to_matrix(reg_B_params_n.detach())
        tmp_affine_gt = affine_transform.tensor_vector_to_matrix(self.gt_vector)

        tmp_vector_B = affine_transform.tensor_matrix_to_vector(tmp_affine_B)
        print(f'--------- abs loss {torch.mean(torch.abs(reg_B_params_n - self.gt_vector)).detach().cpu().numpy()}, '
              f'geodesic loss {self.loss_Reg_real}, '
              f'diff of rotation matrix {torch.mean(torch.abs(tmp_affine_gt - tmp_affine_B)).cpu().numpy()}'
              f'geodesic loss normalized {loss_Reg_real_n}')
        print(f'--------- inference first row {reg_B_params_n[0, ...].flatten().detach().cpu().numpy()},'
              f' GT first row {self.gt_vector[0, ...].flatten().cpu().numpy()}'
              f' tmp vector B {tmp_vector_B[0, ...].flatten().detach().cpu().numpy()}')

        self.loss_Reg = self.loss_Reg_real  # + self.loss_Reg_fake
        self.loss_Reg.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netReg, False)
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

        # update registration networks
        self.set_requires_grad(self.netReg, True)
        self.optimizer_Reg.zero_grad()
        self.backward_Reg()
        self.optimizer_Reg.step()

    def get_transformed_images(self) -> Tuple[torch.Tensor, torch.Tensor]:
        reg_A = affine_transform.transform_image(self.fake_B,
                                                 affine_transform.tensor_vector_to_matrix(self.reg_A_params.detach()),
                                                 device=self.real_B.device)

        reg_B = affine_transform.transform_image(self.real_B,
                                                 affine_transform.tensor_vector_to_matrix(self.reg_B_params.detach()),
                                                 device=self.real_B.device)
        return reg_A, reg_B

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        super(Pix2Pix3dRegModel, self).compute_visuals()

        reg_A, reg_B = self.get_transformed_images()

        diff_A = reg_A - self.transformed_B
        diff_B = reg_B - self.transformed_B
        diff_orig = self.real_B - self.transformed_B

        n_c = self.real_A.shape[2]

        self.reg_A_center_sag = reg_A[:, :, int(n_c / 2), ...]
        self.diff_A_center_sag = diff_A[:, :, int(n_c / 2), ...]
        self.reg_B_center_sag = reg_B[:, :, int(n_c / 2), ...]
        self.diff_B_center_sag = diff_B[:, :, int(n_c / 2), ...]
        self.diff_orig_center_sag = diff_orig[:, :, int(n_c / 2), ...]
        self.deformed_center_sag = self.transformed_B[:, :, int(n_c / 2), ...]

        n_c = self.real_A.shape[3]
        self.reg_A_center_cor = reg_A[:, :, :, int(n_c / 2), ...]
        self.diff_A_center_cor = diff_A[:, :, :, int(n_c / 2), ...]
        self.reg_B_center_cor = reg_B[:, :, :, int(n_c / 2), ...]
        self.diff_B_center_cor = diff_B[:, :, :, int(n_c / 2), ...]
        self.diff_orig_center_cor = diff_orig[:, :, :, int(n_c / 2), ...]
        self.deformed_center_cor = self.transformed_B[:, :, :, int(n_c / 2), ...]

        n_c = self.real_A.shape[4]
        self.reg_A_center_axi = reg_A[..., int(n_c / 2)]
        self.diff_A_center_axi = diff_A[..., int(n_c / 2)]
        self.reg_B_center_axi = reg_B[..., int(n_c / 2)]
        self.diff_B_center_axi = diff_B[..., int(n_c / 2)]
        self.diff_orig_center_axi = diff_orig[..., int(n_c / 2)]
        self.deformed_center_axi = self.transformed_B[..., int(n_c / 2)]

        self.empty_img_4 = torch.zeros_like(self.real_A_center_axi)
        self.empty_img_5 = torch.zeros_like(self.real_A_center_axi)
