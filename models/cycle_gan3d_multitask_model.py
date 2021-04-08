
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
from .cycle_gan3d_model import CycleGAN3dModel
from .multitask_parent import Multitask

os.environ['VXM_BACKEND'] = 'pytorch'
from voxelmorph import voxelmorph as vxm

class CycleGan3dMultiTaskModel(CycleGAN3dModel, Multitask):

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
        parser = CycleGAN3dModel.modify_commandline_options(parser, is_train)
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
        super(CycleGan3dMultiTaskModel, self).__init__(opt)

        self.add_visdom_names(self.loss_names, self.visual_names)

        self.loss_functions = ['backward_G', 'compute_D_loss']

        self.add_networks(opt, self.model_names, self.loss_functions, self.gpu_ids)

        if self.isTrain:
            self.add_optimizers(self.optimizers)

    def set_input(self, input):
        self.clean_tensors()
        super(CycleGan3dMultiTaskModel, self).set_input(input)
        self.set_mt_input(input, real_B=self.real_B, shape=self.real_B.shape,
                          dtype=self.real_B.dtype, device=self.real_B.device)
        self.init_loss_tensors()

    def forward(self):
        super().forward()
        fixed = self.real_B
        fixed = fixed * 0.5 + 0.5
        self.mt_forward(self.fake_B, self.real_B, fixed, self.real_A)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        self.loss_G_GAN = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B

        self.loss_G = self.loss_G_GAN * self.first_phase_coeff
        self.loss_G = self.mt_g_backward(self.fake_B, self.loss_G)

        if torch.is_grad_enabled():
            self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A), rigid registration params, DVF and segmentation mask
        # update G
        # D requires no gradients when optimizing G
        self.set_requires_grad(self.netRigidReg, False)
        self.set_requires_grad(self.netDefReg, False)
        self.set_requires_grad(self.netSeg, False)
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # update G's weights

        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

        # update rigid registration network
        if self.opt.use_rigid_branch:
            self.optimizer_RigidReg.zero_grad()
            self.backward_RigidReg()
            self.optimizer_RigidReg.step()

        # update deformable registration and segmentation network
        if (1 - self.first_phase_coeff) == 0:
            return
        for _ in range(self.opt.vxm_iteration_steps):
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

    def log_tensorboard(self, writer: SummaryWriter, losses: OrderedDict = None, global_step: int = 0,
                        save_gif=True, use_image_name=False, mode=''):
        super().log_tensorboard(writer=writer, losses=losses, global_step=global_step,
                                save_gif=save_gif, use_image_name=use_image_name, mode=mode)

        self.log_mt_tensorboard(self.real_A, self.real_B, self.fake_B, writer, losses, global_step,
                                use_image_name, mode)
