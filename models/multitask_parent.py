import argparse
import os
import sys
from collections import OrderedDict
from typing import Tuple
import util.util as util
from .patchnce import PatchNCELoss

import torch
from torch.utils.tensorboard import SummaryWriter

from util import affine_transform
from util import distance_landmarks
from . import networks
from . import networks3d
from .cut3d_model import CUT3dModel

os.environ['VXM_BACKEND'] = 'pytorch'
from voxelmorph import voxelmorph as vxm

from monai.metrics import compute_meandice


class Multitask:

    @staticmethod
    def modify_commandline_options(parser: argparse.ArgumentParser, is_train=True):
        # rigid and segmentation
        parser.add_argument('--netReg', type=str, default='NormalNet', help='Type of network used for registration')
        parser.add_argument('--netSeg', type=str, default='unet_128', help='Type of network used for segmentation')
        parser.add_argument('--num-classes', type=int, default=2, help='num of classes for segmentation')

        # voxelmorph params
        parser.add_argument('--cudnn-nondet', action='store_true', help='disable cudnn determinism - might slow down training')
        # network architecture parameters
        parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
        parser.add_argument('--dec', type=int, nargs='+', help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
        parser.add_argument('--int-steps', type=int, default=7, help='number of integration steps (default: 7)')
        parser.add_argument('--int-downsize', type=int, default=2, help='flow downsample factor for integration (default: 2)')
        parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')
        parser.add_argument('--use-probs', action='store_true', help='enable probabilities')
        parser.add_argument('--kl-lambda', type=float, default=10, help='prior lambda regularization for KL loss (default: 10)')
        parser.add_argument('--flow-logsigma-bias', type=float, default=-10, help='negative value for initialization of the logsigma layer bias value')

        # others
        parser.add_argument('--use_rigid_branch', action='store_true', help='train the rigid registration network')
        parser.add_argument('--reg_idt_B', action='store_true', help='use idt_B from CUT model instead of real B')
        parser.add_argument('--show_volumes', type=bool, default=False, help='visualize transformed volumes w napari')

        if is_train:
            parser.add_argument('--lambda_Reg', type=float, default=0.5, help='weight for the registration loss')
            parser.add_argument('--lr_Reg', type=float, default=0.0001, help='learning rate for the reg. network opt.')
            parser.add_argument('--lambda_Seg', type=float, default=0.5, help='weight for the segmentation loss')
            parser.add_argument('--lr_Seg', type=float, default=0.0001, help='learning rate for the reg. network opt.')
            parser.add_argument('--lambda_Def', type=float, default=1.0, help='weight for the segmentation loss')
            parser.add_argument('--lr_Def', type=float, default=0.0001, help='learning rate for the reg. network opt.')
            parser.add_argument('--vxm_iteration_steps', type=int, default=5, help='number of steps to train the registration network for each simulated US')
            parser.add_argument('--similarity', type=str, default='NCC', choices=['NCC', 'MIND'], help='type of the similarity used for training voxelmorph')
            parser.add_argument('--epochs_before_reg', type=int, default=0, help='number of epochs to train the network before reg loss is used')
            parser.add_argument('--image-loss', default='mse', help='image reconstruction loss - can be mse or ncc (default: mse)')
        return parser


    def __init__(self, opt):
        self.isTrain = opt.isTrain
        self.set_visdom_names()

        if self.isTrain:
            self.criterionRigidReg = networks.RegistrationLoss()
            self.optimizer_RigidReg = torch.optim.Adam(self.netRigidReg.parameters(), lr=opt.lr_Reg,
                                                       betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_RigidReg)

            self.criterionDefReg = getattr(sys.modules['models.networks'], opt.similarity + 'Loss')()
            self.criterionDefRgl = networks.GradLoss('l2', loss_mult=opt.int_downsize)
            self.optimizer_DefReg = torch.optim.Adam(self.netDefReg.parameters(), lr=opt.lr_Def)
            self.optimizers.append(self.optimizer_DefReg)

            self.criterionSeg = networks.DiceLoss()
            self.optimizer_Seg = torch.optim.Adam(self.netSeg.parameters(), lr=opt.lr_Seg, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_Seg)
            self.transformer = vxm.layers.SpatialTransformer((1, 1, 80, 80, 80))
            # resize = vxm.layers.ResizeTransform(0.5, 1)

            self.first_phase_coeff = 1
            self.loss_landmarks = 0
            self.loss_landmarks_deformed = 0

            self.first_phase_coeff = 1 if self.opt.epochs_before_reg > 0 else 0

    def add_networks(self, opt):
        # specify the models you want to save to the disk. The training/test scripts will call
        # <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names += ['RigidReg', 'DefReg', 'Seg']
        else:  # during test time, only load G
            self.model_names += ['RigidReg', 'DefReg', 'Seg']

        self.loss_functions += ['backward_RigidReg', 'backward_DefReg_Seg']

        # We are using DenseNet for rigid registration -- actually DenseNet didn't provide any performance improvement
        # TODO change this to obelisk net
        self.netRigidReg = networks3d.define_reg_model(model_type=self.opt.netReg, n_input_channels=2,
                                                       num_classes=6, gpu_ids=self.gpu_ids, img_shape=opt.inshape)
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
        self.netDefReg = networks3d.init_net(self.netDefReg, gpu_ids=self.gpu_ids)

        self.netSeg = networks3d.define_G(opt.input_nc, opt.num_classes, opt.ngf,
                                          opt.netSeg, opt.norm, use_dropout=not opt.no_dropout,
                                          gpu_ids=self.gpu_ids, is_seg_net=True)
        self.transformer_label = networks3d.init_net(vxm.layers.SpatialTransformer(size=opt.inshape, mode='nearest'),
                                                     gpu_ids=self.gpu_ids)
        self.transformer_intensity = networks3d.init_net(vxm.layers.SpatialTransformer(size=opt.inshape),
                                                         gpu_ids=self.gpu_ids)
        self.resizer = networks3d.init_net(vxm.layers.ResizeTransform(vel_resize=0.5, ndims=3), gpu_ids=self.gpu_ids)


    def add_visdom_names(self):
        # specify the training losses you want to print out. The training/test scripts will call
        # <BaseModel.get_current_losses>
        self.loss_names += ['RigidReg_fake', 'RigidReg_real']
        self.loss_names += ['DefReg_real', 'DefReg_fake', 'Seg_real', 'Seg_fake']
        self.loss_names += ['landmarks']
        # specify the images you want to save/display. The training/test scripts will call
        # <BaseModel.get_current_visuals>

        # rigid registration
        self.visual_names += ['diff_A_center_sag', 'diff_A_center_cor', 'diff_A_center_axi']
        self.visual_names += ['diff_B_center_sag', 'diff_B_center_cor', 'diff_B_center_axi']
        self.visual_names += ['diff_orig_center_sag', 'diff_orig_center_cor', 'diff_orig_center_axi']
        self.visual_names += ['deformed_center_sag', 'deformed_center_cor', 'deformed_center_axi']

        # segmentation
        self.visual_names += ['mask_A_center_sag', 'mask_A_center_cor', 'mask_A_center_axi']
        self.visual_names += ['seg_A_center_sag', 'seg_A_center_cor', 'seg_A_center_axi']
        self.visual_names += ['seg_B_center_sag', 'seg_B_center_cor', 'seg_B_center_axi']

        # deformable registration
        self.visual_names += ['dvf_center_sag', 'dvf_center_cor', 'dvf_center_axi']
        self.visual_names += ['deformed_B_center_sag', 'deformed_B_center_cor', 'deformed_B_center_axi']