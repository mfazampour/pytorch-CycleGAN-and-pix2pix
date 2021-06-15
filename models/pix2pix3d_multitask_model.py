
import argparse
import os
import sys
from collections import OrderedDict
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter

from util import distance_landmarks
from monai.metrics import compute_meandice
import torch
from util import affine_transform
from . import networks
from . import networks3d
from .pix2pix3d_model import Pix2Pix3dModel
from .multitask_parent import Multitask

os.environ['VXM_BACKEND'] = 'pytorch'
from voxelmorph import voxelmorph as vxm

class Pix2Pix3dMultiTaskModel(Pix2Pix3dModel, Multitask):

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

        parser = Pix2Pix3dModel.modify_commandline_options(parser, is_train)
        parser = Multitask.modify_commandline_options(parser, is_train)

        parser.set_defaults(norm='batch', netG='unet_128', dataset_mode='volume')
        return parser

    def __init__(self, opt: argparse.Namespace):
        """Initialize the class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        For visualization, we set 'visual_names' as 'real_A' (input real image),
        'real_B_rgb' (ground truth RGB image), and 'fake_B_rgb' (predicted RGB image)
        We convert the Lab image 'real_B' (inherited from Pix2pixModel) to a RGB image 'real_B_rgb'.
        we convert the Lab image 'fake_B' (inherited from Pix2pixModel) to a RGB image 'fake_B_rgb'.
        """
        super(Pix2Pix3dMultiTaskModel, self).__init__(opt)
        self.add_visdom_names(self.loss_names, self.visual_names)

        self.loss_functions = ['backward_G', 'compute_D_loss']

        self.add_networks(opt, self.model_names, self.loss_functions, self.gpu_ids)

        if self.isTrain:
            self.add_optimizers(self.optimizers)

    def set_input(self, input):
        self.clean_tensors()
        super(Pix2Pix3dModel, self).set_input(input)
        # AtoB = self.opt.direction == 'AtoB'
        # self.real_A = input['A' if AtoB else 'B'].to(self.device)
        # self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.set_mt_input(input, real_B=self.real_B, shape=self.real_B.shape,
                          dtype=self.real_B.dtype, device=self.real_B.device)
        self.init_loss_tensors()

    def forward(self):
        super().forward()
        fixed = self.real_B
        self.mt_forward(self.fake_B, self.real_B, fixed, self.real_A)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G = 0.0
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)

        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        self.loss_G = (self.loss_G_GAN + self.loss_G_L1) * self.first_phase_coeff
        self.loss_G = self.mt_g_backward(self.fake_B, self.loss_G)

        if torch.is_grad_enabled():
            self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A), rigid registration params, DVF and segmentation mask
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.compute_D_loss()
        if torch.is_grad_enabled():
            self.loss_D.backward()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights

        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netRigidReg, False)
        self.set_requires_grad(self.netDefReg, False)
        self.set_requires_grad(self.netSeg, False)
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # update G's weights

        # update rigid registration network
        if self.opt.use_rigid_branch:
            self.set_requires_grad(self.netRigidReg, True)
            self.optimizer_RigidReg.zero_grad()
            self.backward_RigidReg()
            self.optimizer_RigidReg.step()

        # update deformable registration and segmentation network
        if (1 - self.first_phase_coeff) == 0:
            return
        for _ in range(self.opt.vxm_iteration_steps):
            self.set_requires_grad(self.netDefReg, True)
            self.set_requires_grad(self.netSeg, True)
            self.optimizer_DefReg.zero_grad()
            self.optimizer_Seg.zero_grad()
            self.backward_DefReg_Seg()  # only back propagate through fake_B once
            self.optimizer_DefReg.step()
            self.optimizer_Seg.step()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        super().compute_visuals()
        self.compute_mt_visuals(self.real_B, self.real_A.shape)

    def update_learning_rate(self, epoch=0):
        super().update_learning_rate(epoch)
        if epoch >= self.opt.epochs_before_reg:
            self.first_phase_coeff = 0


