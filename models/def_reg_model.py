import os
import sys

import numpy as np
import torch
from collections import OrderedDict

from monai.metrics import compute_meandice

from .base_model import BaseModel
from . import networks3d
from torch.utils.tensorboard import SummaryWriter
from monai.visualize import img2tensorboard
from util import affine_transform

os.environ['VXM_BACKEND'] = 'pytorch'
from voxelmorph import voxelmorph as vxm


from . import networks


class DefRegModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        parser.set_defaults(norm='batch', netG='unet_128', dataset_mode='volume', batch_size=2)

        # voxelmorph params
        parser.add_argument('--cudnn-nondet', action='store_true',
                            help='disable cudnn determinism - might slow down training')
        # network architecture parameters
        parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
        parser.add_argument('--dec', type=int, nargs='+',
                            help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
        parser.add_argument('--int-steps', type=int, default=7, help='number of integration steps (default: 7)')
        parser.add_argument('--int-downsize', type=int, default=2,
                            help='flow downsample factor for integration (default: 2)')
        parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')
        parser.add_argument('--use-probs', action='store_true', help='enable probabilities')
        parser.add_argument('--kl-lambda', type=float, default=10,
                            help='prior lambda regularization for KL loss (default: 10)')
        parser.add_argument('--flow-logsigma-bias', type=float, default=-10,
                            help='negative value for initialization of the logsigma layer bias value')

        parser.add_argument('--show_volumes', type=bool, default=False, help='visualize transformed volumes w napari')
        parser.add_argument('--num-classes', type=int, default=2, help='num of classes for segmentation')

        if is_train:
            parser.add_argument('--vxm_iteration_steps', type=int, default=5,
                                help='number of steps to train the registration network for each simulated US')
            parser.add_argument('--similarity', type=str, default='NCC', choices=['NCC', 'MIND'],
                                help='type of the similarity used for training voxelmorph')
            parser.add_argument('--epochs_before_reg', type=int, default=0,
                                help='number of epochs to train the network before reg loss is used')
            parser.add_argument('--image-loss', default='lcc',
                                help='image reconstruction loss - can be mse or ncc (default: mse)')

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
        self.loss_names = ['G','DefReg']
        # specify the models you want to save to the disk. The training/test scripts will call
        # <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['DefReg']
        else:  # during test time, only load G
            self.model_names = ['DefReg']

        # extract shape from sampled input
        inshape = opt.inshape
        # device handling
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in opt.gpu_ids])
        # enabling cudnn determinism appears to speed up training by a lot
        torch.backends.cudnn.deterministic = not opt.cudnn_nondet
        # unet architecture
        enc_nf = opt.enc if opt.enc else [16, 32, 32, 32]
        dec_nf = opt.dec if opt.dec else [32, 32, 32, 32, 32, 16, 16]
        # configure new model
        self.netDefReg = vxm.networks.VxmDense(
            inshape=inshape,
            nb_unet_features=[enc_nf, dec_nf],
            bidir=opt.bidir,
            int_steps=opt.int_steps,
            int_downsize=opt.int_downsize,
            use_probs=opt.use_probs,
            flow_logsigma_bias=opt.flow_logsigma_bias
        )
        self.netDefReg = networks3d.init_net(self.netDefReg, gpu_ids=self.opt.gpu_ids)

        self.transformer_label = networks3d.init_net(vxm.layers.SpatialTransformer(size=opt.inshape, mode='nearest'),
                                                     gpu_ids=opt.gpu_ids)
        self.resizer = networks3d.init_net(vxm.layers.ResizeTransform(vel_resize=0.5, ndims=3), gpu_ids=opt.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks3d.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                            opt.netD, opt.n_layers_D, norm=opt.norm,
                                            use_sigmoid=use_sigmoid, gpu_ids=self.gpu_ids)

        if self.isTrain:
            # define loss functions

            self.criterionDefReg = getattr(sys.modules['models.networks'], self.opt.similarity + 'Loss')()
            self.criterionDefRgl = networks.GradLoss('l2', loss_mult=self.opt.int_downsize)
            self.optimizer_DefReg = torch.optim.Adam(self.netDefReg.parameters(), lr=self.opt.lr)
            self.optimizers.append(self.optimizer_DefReg)

            self.transformer = vxm.layers.SpatialTransformer((1, 1, 80, 80, 80))
            # resize = vxm.layers.ResizeTransform(0.5, 1)

    # def name(self):
    #     return 'Pix2Pix3dModel'

    def set_input(self, input):

        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)

        self.patient = input['Patient']

        if input['B_mask_available'][0] and input['B_mask_available'][1]:  # TODO in this way it only works with batch size 2!
            self.mask_B = input['B_mask'].to(self.device).type(self.real_A.dtype)
        else:
            self.mask_B = None

        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fixed = self.real_B[0:1, ...] * 0.5 + 0.5
        self.moving = self.real_B[1:2, ...] * 0.5 + 0.5

        if self.mask_B is not None:
            self.mask_fixed = self.mask_B[0:1, ...]
            self.mask_moving = self.mask_B[1:0, ...]
        def_reg_output = self.netDefReg(self.moving, self.fixed, registration=not self.isTrain)

        if self.opt.bidir and self.isTrain:
            (self.deformed_moving, self.deformed_fixed, self.dvf) = def_reg_output
        else:
            (self.deformed_moving, self.dvf) = def_reg_output

        if self.isTrain:
            self.dvf = self.resizer(self.dvf)
        self.compute_gt_dice()

    def get_current_landmark_distances(self):
        return torch.tensor([0]), torch.tensor([0]), torch.tensor([0])

    def compute_landmark_loss(self):
        pass

    def compute_gt_dice(self):
        """
        calculate the dice score between the deformed mask and the ground truth if we have it
        Returns
        -------

        """
        if self.mask_B is not None:
            self.mask_def = self.transformer_label(self.mask_moving, self.dvf.detach())
            shape = list(self.mask_B.shape)
            n = self.opt.num_classes  # number of classes
            shape[1] = n
            one_hot_fixed = torch.zeros(shape, device=self.mask_B.device)
            one_hot_deformed = torch.zeros(shape, device=self.mask_B.device)
            one_hot_moving = torch.zeros(shape, device=self.mask_B.device)
            for i in range(n):
                one_hot_fixed[:, i, self.mask_fixed[1, 0, ...] == i] = 1
                one_hot_deformed[:, i, self.mask_def[0, 0, ...] == i] = 1
                one_hot_moving[:, i, self.mask_moving[0, 0, ...] == i] = 1


            self.loss_warped_dice = compute_meandice(one_hot_deformed, one_hot_fixed, include_background=False)
            self.loss_moving_dice = compute_meandice(one_hot_deformed, one_hot_moving, include_background=False)
            loss_beginning_dice = compute_meandice( one_hot_moving, one_hot_fixed,include_background=False)
            self.diff_dice = loss_beginning_dice - self.loss_warped_dice

        else:
            self.loss_warped_dice = None
            self.loss_moving_dice = None
            self.diff_dice = None

    def backward_defReg(self):
        def_reg_output = self.netDefReg(self.moving, self.fixed)  # fake_B is the moving image here

        if self.opt.bidir:
            (self.deformed_moving, self.deformed_fixed, self.dvf) = def_reg_output
        else:
            (self.deformed_moving, self.dvf) = def_reg_output

        self.loss_DefReg_real = self.criterionDefReg(self.deformed_moving, self.fixed)  # TODO add weights same as vxm!
        self.loss_DefReg = self.loss_DefReg_real
        if self.opt.bidir:
            self.loss_DefReg_fake = self.criterionDefReg(self.deformed_fixed, self.moving)
            self.loss_DefReg += self.loss_DefReg_fake
        else:
            self.loss_DefReg_fake = torch.tensor([0.0])
        self.loss_DefRgl = self.criterionDefRgl(self.dvf, None)
        self.loss_DefReg += self.loss_DefRgl

        self.loss_G = self.loss_DefReg
        # self.loss_DefReg.backward()
        if torch.is_grad_enabled():
            self.loss_DefReg.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update Def Reg
        self.optimizer_DefReg.zero_grad()  # set G's gradients to zero
        self.backward_defReg()  # calculate graidents for G
        self.optimizer_DefReg.step()  # udpate G's weights


    def log_tensorboard(self, writer: SummaryWriter, losses: OrderedDict = None, global_step: int = 0,
                        save_gif=True, use_image_name=False, mode=''):

        self.add_deformable_figures(mode=mode, global_step=global_step, writer=writer, use_image_name=use_image_name)
        if losses is not None:
            for key in losses:
                writer.add_scalar(f'losses/{key}', scalar_value=losses[key], global_step=global_step)

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def add_deformable_figures(self, mode, global_step, writer, use_image_name=False):
        n = 7 if self.mask_B is None else 9
        axs, fig = vxm.torch.utils.init_figure(3, n)
        vxm.torch.utils.set_axs_attribute(axs)
        vxm.torch.utils.fill_subplots(self.dvf.cpu()[:, 0:1, ...], axs=axs[0, :], img_name='Def. X', cmap='RdBu',
                                      fig=fig, show_colorbar=True)
        vxm.torch.utils.fill_subplots(self.dvf.cpu()[:, 1:2, ...], axs=axs[1, :], img_name='Def. Y', cmap='RdBu',
                                      fig=fig, show_colorbar=True)
        vxm.torch.utils.fill_subplots(self.dvf.cpu()[:, 2:3, ...], axs=axs[2, :], img_name='Def. Z', cmap='RdBu',
                                      fig=fig, show_colorbar=True)
        vxm.torch.utils.fill_subplots(self.deformed_moving.detach().cpu(), axs=axs[3, :], img_name='Deformed')
        vxm.torch.utils.fill_subplots((self.deformed_moving.detach() - self.moving).abs().cpu(),
                                      axs=axs[4, :], img_name='Deformed - moving')
        overlay = self.moving.repeat(1, 3, 1, 1, 1)
        overlay *= 0.8
        overlay[:, 0:1, ...] += self.fixed.detach() * 0.8
        overlay[overlay > 1] = 1
        vxm.torch.utils.fill_subplots(overlay.cpu(), axs=axs[5, :], img_name='Moving overlay', cmap=None)
        overlay = self.deformed_moving.repeat(1, 3, 1, 1, 1)
        overlay *= 0.8
        overlay[:, 0:1, ...] += self.fixed.detach() * 0.6
        overlay[overlay > 1] = 1
        vxm.torch.utils.fill_subplots(overlay.cpu(), axs=axs[6, :], img_name='Deformed overlay', cmap=None)
        if self.mask_B is not None:
            overlay = self.mask_fixed.repeat(1, 3, 1, 1, 1)
            overlay[:, 0:1, ...] = self.mask_moving.detach()
            overlay[:, 2, ...] = 0
            vxm.torch.utils.fill_subplots(overlay.cpu(), axs=axs[7, :], img_name='mask moving on fixed', cmap=None)

            overlay = self.mask_fixed.repeat(1, 3, 1, 1, 1)
            overlay[:, 0:1, ...] = self.mask_def.detach()
            overlay[:, 2, ...] = 0
            vxm.torch.utils.fill_subplots(overlay.cpu(), axs=axs[8, :], img_name='mask warped on fixed', cmap=None)
        #
        if use_image_name:
            tag = mode + f'{self.patient}/Deformable'
        else:
            tag = mode + 'Deformable'
        writer.add_figure(tag=tag, figure=fig, global_step=global_step)





