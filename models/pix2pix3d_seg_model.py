import numpy as np
import torch
import os
import argparse
from collections import OrderedDict

from torch.utils.tensorboard import SummaryWriter

import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from .pix2pix3d_model import Pix2Pix3dModel
from . import networks3d

from . import networks

os.environ['VXM_BACKEND'] = 'pytorch'
from voxelmorph import voxelmorph as vxm  # just need it for tensorboard visualization helper functions

class Pix2Pix3dSegModel(Pix2Pix3dModel):

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
        parser.add_argument('--num-classes', type=int, default=2, help='num of classes for segmentation')

        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--no-lsgan', type=bool, default=False)
            parser.add_argument('--lambda_Seg', type=float, default=0.5, help='weight for the segmentation loss')
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
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'Seg_real', 'Seg_fake']

        # specify the images you want to save/display. The training/test scripts will call
        # <BaseModel.get_current_visuals>
        # the empty slice is added since the visualization would be 3 * 4
        self.visual_names += ['mask_A_center_sag', 'seg_A_center_sag', 'seg_B_center_sag', 'empty_img_4']
        self.visual_names += ['mask_A_center_cor', 'seg_A_center_cor', 'seg_B_center_cor', 'empty_img_5']
        self.visual_names += ['mask_A_center_axi', 'seg_A_center_axi', 'seg_B_center_axi', 'empty_img_6']
        # specify the models you want to save to the disk. The training/test scripts will call
        # <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D', 'SegA', 'SegB']
        else:  # during test time, only load G
            self.model_names = ['G', 'SegA', 'SegB']

        # TODO add a function called to define_Seg
        self.netSegA = networks3d.define_G(opt.input_nc, opt.num_classes, opt.ngf,
                                           opt.netSeg, opt.norm, not opt.no_dropout,
                                           gpu_ids=self.gpu_ids, is_seg_net=True)

        self.netSegB = networks3d.define_G(opt.input_nc, opt.num_classes, opt.ngf,
                                           opt.netSeg, opt.norm, not opt.no_dropout,
                                           gpu_ids=self.gpu_ids, is_seg_net=True)

        if self.isTrain:
            # define loss functions
            self.criterionSeg = networks.DiceLoss()
            self.optimizer_Seg = torch.optim.Adam(list(self.netSegA.parameters()) + list(self.netSegB.parameters()),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_Seg)

    # def name(self):
    #     return 'Pix2Pix3dModel'

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.mask_A = input['A_mask'].to(self.device)

    def forward(self):
        super(Pix2Pix3dSegModel, self).forward()
        self.seg_A = self.netSegA(self.real_A)
        self.seg_B = self.netSegB(self.fake_B)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)

        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        # Third, Seg(G(A)) =
        self.loss_G_Seg = self.criterionSeg(self.seg_B, self.mask_A) * self.opt.lambda_Seg

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_Seg
        self.loss_G.backward()

    def backward_Seg(self):
        """
        Calculate Segmentation loss to update the segmentation networks
        Returns
        -------
        """
        self.loss_Seg_real = self.criterionSeg(self.seg_A, self.mask_A) * self.opt.lambda_Seg # to be of the same order as loss_G_Seg
        self.seg_A = torch.argmax(self.seg_A, dim=1, keepdim=True)

        seg_B = self.netSegB(self.fake_B.detach())
        self.loss_Seg_fake = self.criterionSeg(seg_B, self.mask_A) * self.opt.lambda_Seg # to be of the same order as loss_G_Seg
        self.seg_B = torch.argmax(self.seg_B, dim=1, keepdim=True)

        self.loss_Seg = self.loss_Seg_real + self.loss_Seg_fake
        self.loss_Seg.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netSegB, False)
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
        # update segmentation networks
        self.set_requires_grad(self.netSegB, True)
        self.optimizer_Seg.zero_grad()
        self.backward_Seg()
        self.optimizer_Seg.step()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        super(Pix2Pix3dSegModel, self).compute_visuals()

        n_c = self.real_A.shape[2]
        # average over channel to get the real and fake image
        self.mask_A_center_sag = self.mask_A[:, :, int(n_c / 2), ...]
        self.seg_A_center_sag = self.seg_A[:, :, int(n_c / 2), ...]
        self.seg_B_center_sag = self.seg_B[:, :, int(n_c / 2), ...]

        n_c = self.real_A.shape[3]
        self.mask_A_center_cor = self.mask_A[:, :, :, int(n_c / 2), ...]
        self.seg_A_center_cor = self.seg_A[:, :, :, int(n_c / 2), ...]
        self.seg_B_center_cor = self.seg_B[:, :, :, int(n_c / 2), ...]

        n_c = self.real_A.shape[4]
        self.mask_A_center_axi = self.mask_A[..., int(n_c / 2)]
        self.seg_A_center_axi = self.seg_A[..., int(n_c / 2)]
        self.seg_B_center_axi = self.seg_B[..., int(n_c / 2)]

        self.empty_img_4 = torch.zeros_like(self.real_A_center_axi)
        self.empty_img_5 = torch.zeros_like(self.real_A_center_axi)
        self.empty_img_6 = torch.zeros_like(self.real_A_center_axi)


    def log_tensorboard(self, writer: SummaryWriter, losses: OrderedDict = None, global_step: int = 0,
                        save_gif=True, use_image_name=False, mode=''):
        axs, fig = vxm.torch.utils.init_figure(3, 6)
        vxm.torch.utils.set_axs_attribute(axs)
        vxm.torch.utils.fill_subplots(self.real_A.cpu(), axs=axs[0, :], img_name='A')
        vxm.torch.utils.fill_subplots(self.fake_B.detach().cpu(), axs=axs[1, :], img_name='fake')
        vxm.torch.utils.fill_subplots(self.real_B.cpu(), axs=axs[2, :], img_name='B')
        vxm.torch.utils.fill_subplots(self.mask_A.cpu(), axs=axs[3, :], img_name='Mask A')
        vxm.torch.utils.fill_subplots(self.seg_A.detach().cpu(), axs=axs[4, :], img_name='Seg Fake')
        vxm.torch.utils.fill_subplots(self.seg_B.cpu(), axs=axs[5, :], img_name='Seg B')

        writer.add_figure(tag='volumes', figure=fig, global_step=global_step)

        for key in losses:
            writer.add_scalar(f'losses/{key}', scalar_value=losses[key], global_step=global_step)