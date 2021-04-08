import os
from collections import OrderedDict

import torch
from torch.utils.tensorboard import SummaryWriter

from .base_model import BaseModel
from . import networks
from . import networks3d

os.environ['VXM_BACKEND'] = 'pytorch'
from voxelmorph import voxelmorph as vxm


class SegmentationModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(norm='batch', dataset_mode='volume')
        parser.add_argument('--netSeg', type=str, default='unet_small', help='Type of network used for registration')
        parser.add_argument('--num-classes', type=int, default=2, help='num of classes for segmentation')
        if is_train:
            pass
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)
        self.loss_names = ['G']
        self.visual_names = []  #'data_A', 'mask_A', 'seg_A']
        self.model_names = ['G']
        self.netG = networks3d.define_G(opt.input_nc, opt.num_classes, opt.ngf,
                                        opt.netSeg, opt.norm, use_dropout=not opt.no_dropout,
                                        gpu_ids=self.gpu_ids, is_seg_net=True)
        if self.isTrain:
            self.criterionSeg = networks.DiceLoss()
            self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.data_A = input['A'].to(self.device)
        self.mask_A = input['A_mask'].to(self.device).type(self.data_A.dtype)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.seg_A = self.netG(self.data_A)
        if not self.isTrain:
            self.seg_A = torch.argmax(self.seg_A, dim=1, keepdim=True)

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.loss_G = self.criterionSeg(self.seg_A, self.mask_A)
        self.loss_G.backward()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results
        self.optimizer.zero_grad()   # clear network G's existing gradients
        self.backward()              # calculate gradients for network G
        self.optimizer.step()        # update gradients for network G

    def log_tensorboard(self, writer: SummaryWriter, losses: OrderedDict = None, global_step: int = 0,
                        save_gif=True, use_image_name=False, mode=''):
        for key in losses:
            writer.add_scalar(f'losses/{key}', scalar_value=losses[key], global_step=global_step)
        seg_A = torch.argmax(self.seg_A, dim=1, keepdim=True)
        axs, fig = vxm.torch.utils.init_figure(3, 4)
        vxm.torch.utils.set_axs_attribute(axs)
        vxm.torch.utils.fill_subplots(self.mask_A.cpu(), axs=axs[0, :], img_name='Mask MR')
        vxm.torch.utils.fill_subplots(seg_A.detach().cpu(), axs=axs[1, :], img_name='Seg MR')

        overlay = self.data_A.detach().repeat(1, 3, 1, 1, 1) * 0.5 + 0.5
        overlay[:, 0:1, ...] += 0.5 * seg_A.detach()
        overlay *= 0.8
        overlay[overlay > 1] = 1
        vxm.torch.utils.fill_subplots(overlay.cpu(), axs=axs[2, :], img_name='Seg mask overlay', cmap=None)

        overlay = self.data_A.repeat(1, 3, 1, 1, 1) * 0.5 + 0.5
        overlay[:, 0:1, ...] += 0.5 * self.mask_A.detach()
        overlay *= 0.8
        overlay[overlay > 1] = 1
        vxm.torch.utils.fill_subplots(overlay.cpu(), axs=axs[3, :], img_name='Input mask overlay', cmap=None)

        writer.add_figure(tag='Segmentation', figure=fig, global_step=global_step)