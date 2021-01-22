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


class CUT3DMultiTaskModel(CUT3dModel):

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
        parser.set_defaults(norm='batch', dataset_mode='volume')
        parser.add_argument('--netReg', type=str, default='NormalNet', help='Type of network used for registration')
        parser.add_argument('--show_volumes', type=bool, default=False, help='visualize transformed volumes w napari')

        parser.add_argument('--epochs_before_reg', type=int, default=0,
                            help='number of epochs to train the network before reg loss is used')
        parser.add_argument('--cudnn-nondet', action='store_true',
                            help='disable cudnn determinism - might slow down training')
        # network architecture parameters
        parser.add_argument('--enc', type=int, nargs='+',
                            help='list of unet encoder filters (default: 16 32 32 32)')
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
        parser.add_argument('--netSeg', type=str, default='unet_128', help='Type of network used for segmentation')
        parser.add_argument('--num-classes', type=int, default=2, help='num of classes for segmentation')
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                            help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--use_rigid_branch', action='store_true', help='train the rigid registration network')
        parser.add_argument('--reg_idt_B', action='store_true', help='use idt_B from CUT model instead of real B')

        parser.set_defaults(pool_size=0)  # no image pooling
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--no-lsgan', type=bool, default=False)
            parser.add_argument('--lambda_Reg', type=float, default=0.5, help='weight for the registration loss')
            parser.add_argument('--lr_Reg', type=float, default=0.0001, help='learning rate for the reg. network opt.')
            parser.add_argument('--lambda_Seg', type=float, default=0.5, help='weight for the segmentation loss')
            parser.add_argument('--lr_Seg', type=float, default=0.0001, help='learning rate for the reg. network opt.')
            parser.add_argument('--lambda_Def', type=float, default=1.0, help='weight for the segmentation loss')
            parser.add_argument('--lr_Def', type=float, default=0.0001, help='learning rate for the reg. network opt.')
            parser.add_argument('--vxm_iteration_steps', type=int, default=1,
                                help='number of steps to train the registration network for each simulated US')
            parser.add_argument('--similarity', type=str, default='NCC', choices=['NCC', 'MIND'],
                                help='type of the similarity used for training voxelmorph')

            # loss hyperparameters
            parser.add_argument('--image-loss', default='mse',
                                help='image reconstruction loss - can be mse or ncc (default: mse)')
        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

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
        super().__init__(opt)
        self.isTrain = opt.isTrain

        self.set_visdom_names()

        self.set_networks(opt)
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

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
            self.distance_landmarks_b = 0

        self.first_phase_coeff = 1 if self.opt.epochs_before_reg > 0 else 0

    def get_model(self, name):
        if name == "RigidReg":
            return self.netRigidReg
        if name == "DefReg":
            return self.netDefReg
        if name == "Seg":
            return self.netSeg
        if name == "G":
            return self.netG
        if name == 'F':
            return self.netF
        if name == 'D':
            return self.netD

    def set_networks(self, opt):
        # specify the models you want to save to the disk. The training/test scripts will call
        # <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D', 'F', 'RigidReg', 'DefReg', 'Seg']
        else:  # during test time, only load G
            self.model_names = ['G', 'RigidReg', 'DefReg', 'Seg']

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

        # define networks (both generator and discriminator)
        self.netG = networks3d.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
                                        opt.init_type, init_gain=opt.init_gain, no_antialias=opt.no_antialias,
                                        no_antialias_up=opt.no_antialias_up, gpu_ids=self.gpu_ids, opt=opt)

        self.netF = networks3d.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type,
                                        init_gain=opt.init_gain, no_antialias=opt.no_antialias,
                                        gpu_ids=self.gpu_ids, opt=opt)

        self.netD = networks3d.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type,
                                        init_gain=opt.init_gain, no_antialias=opt.no_antialias,
                                        gpu_ids=self.gpu_ids, opt=opt)

    def set_visdom_names(self):
        # specify the training losses you want to print out. The training/test scripts will call
        # <BaseModel.get_current_losses>
        self.loss_names = ['G', 'G_NCE', 'D_real', 'D_fake', 'RigidReg_fake', 'RigidReg_real']
        self.loss_names += ['DefReg_real', 'DefReg_fake', 'Seg_real', 'Seg_fake']
        if self.opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
        # specify the images you want to save/display. The training/test scripts will call
        # <BaseModel.get_current_visuals>

        self.visual_names = ['real_A_center_sag', 'real_A_center_cor', 'real_A_center_axi']
        self.visual_names += ['fake_B_center_sag', 'fake_B_center_cor', 'fake_B_center_axi']
        self.visual_names += ['real_B_center_sag', 'real_B_center_cor', 'real_B_center_axi']

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
        if self.opt.nce_idt and self.isTrain:
            self.visual_names += ['idt_B_center_sag', 'idt_B_center_cor', 'idt_B_center_axi']

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
        self.landmarks_A = input['A_landmark'].to(self.device).unsqueeze(dim=0)
        self.landmarks_B = input['B_landmark'].to(self.device).unsqueeze(dim=0)
        affine, self.gt_vector = affine_transform.create_random_affine(self.real_B.shape[0],
                                                                       self.real_B.shape[-3:],
                                                                       self.real_B.dtype,
                                                                       device=self.real_B.device)
        self.transformed_B = affine_transform.transform_image(self.real_B, affine, self.real_B.device)
        self.transformed_LM_B = affine_transform.transform_image(self.landmarks_B, affine, self.landmarks_B.device)

        if self.opt.show_volumes:
            affine_transform.show_volumes([self.real_B, self.transformed_B])

        self.mask_A = input['A_mask'].to(self.device).type(self.real_A.dtype)

        ###
        self.loss_DefReg_real = torch.tensor([0.0])
        self.loss_DefReg = torch.tensor([0.0])
        self.loss_DefReg_fake = torch.tensor([0.0])

        self.loss_DefReg_fake = torch.tensor([0.0])
        self.loss_DefRgl = torch.tensor([0.0])
        self.loss_Seg_real = torch.tensor([0.0])
        self.loss_Seg_fake = torch.tensor([0.0])
        self.loss_Seg = torch.tensor([0.0])
        self.loss_G_NCE = torch.tensor([0.0])
        self.loss_RigidReg_fake = torch.tensor([0.0])
        self.loss_RigidReg_real = torch.tensor([0.0])
        self.loss_RigidReg = torch.tensor([0.0])

    def forward(self):
        super().forward()
        self.reg_A_params = self.netRigidReg(torch.cat([self.fake_B, self.transformed_B], dim=1))
        self.reg_B_params = self.netRigidReg(torch.cat([self.real_B, self.transformed_B], dim=1))

        fixed = self.idt_B.detach() if self.opt.reg_idt_B else self.real_B
        fixed = fixed * 0.5 + 0.5
        def_reg_output = self.netDefReg(self.fake_B * 0.5 + 0.5, fixed, registration=not self.isTrain)
        if self.opt.bidir:
            (self.deformed_fake_B, self.deformed_B, self.dvf) = def_reg_output
        else:
            (self.deformed_fake_B, self.dvf) = def_reg_output

        if self.isTrain:
            self.dvf = self.resizer(self.dvf)
        self.deformed_A = self.transformer_intensity(self.real_A, self.dvf.detach())
        self.mask_A_deformed = self.transformer_label(self.mask_A, self.dvf.detach())
        self.seg_B = self.netSeg(self.idt_B.detach() if self.opt.reg_idt_B else self.real_B)
        self.seg_fake_B = self.netSeg(self.fake_B)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        self.loss_G_NCE = self.compute_G_loss()

        # Third, rigid registration
        loss_G_Reg = self.criterionRigidReg(self.reg_A_params, self.gt_vector) * self.opt.lambda_Reg

        # fourth, Def registration:
        if self.opt.bidir:
            loss_DefReg_fake = self.criterionDefReg(self.deformed_fake_B, self.real_B)
        else:
            loss_DefReg_fake = 0

        # fifth, segmentation
        loss_G_Seg = self.criterionSeg(self.seg_fake_B, self.mask_A) * self.opt.lambda_Seg

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_NCE * self.first_phase_coeff + \
                      loss_DefReg_fake * (1 - self.first_phase_coeff) + \
                      loss_G_Seg * (1 - self.first_phase_coeff)

        if self.opt.use_rigid_branch:
            self.loss_G += loss_G_Reg * (1 - self.first_phase_coeff)

        self.loss_G.backward()

    def backward_RigidReg(self):
        """
        Calculate Segmentation loss to update the segmentation networks
        Returns
        -------
        """
        reg_A_params = self.netRigidReg(torch.cat([self.fake_B.detach(), self.transformed_B], dim=1))
        self.loss_RigidReg_fake = self.criterionRigidReg(reg_A_params,
                                                         self.gt_vector) * self.opt.lambda_Reg  # to be of the same order as loss_G_Seg
        self.loss_RigidReg_real = self.criterionRigidReg(self.reg_B_params,
                                                         self.gt_vector) * self.opt.lambda_Reg  # to be of the same order as loss_G_Seg

        self.loss_RigidReg = self.loss_RigidReg_real + self.loss_RigidReg_fake * (1 - self.first_phase_coeff)
        self.loss_RigidReg.backward()

    def bacward_DefReg_Seg(self):
        fixed = self.idt_B.detach() if self.opt.reg_idt_B else self.real_B
        fixed = fixed * 0.5 + 0.5
        def_reg_output = self.netDefReg(self.fake_B.detach() * 0.5 + 0.5, fixed)  # fake_B is the moving image here
        if self.opt.bidir:
            (deformed_fake_B, deformed_B, dvf) = def_reg_output
        else:
            (deformed_fake_B, dvf) = def_reg_output

        self.loss_DefReg_real = self.criterionDefReg(deformed_fake_B, self.real_B)  # TODO add weights same as vxm!
        self.loss_DefReg = self.loss_DefReg_real
        if self.opt.bidir:
            self.loss_DefReg_fake = self.criterionDefReg(deformed_B, self.fake_B.detach())
            self.loss_DefReg += self.loss_DefReg_fake
        else:
            self.loss_DefReg_fake = torch.tensor([0.0])
        self.loss_DefRgl = self.criterionDefRgl(dvf, None)
        self.loss_DefReg += self.loss_DefRgl
        self.loss_DefReg *= (1 - self.first_phase_coeff)

        seg_B = self.netSeg(self.idt_B.detach() if self.opt.reg_idt_B else self.real_B)
        dvf_resized = self.resizer(dvf)
        mask_A_deformed = self.transformer_label(self.mask_A, dvf_resized)
        self.loss_Seg_real = self.criterionSeg(seg_B, mask_A_deformed)

        seg_fake_B = self.netSeg(self.fake_B.detach())
        self.loss_Seg_fake = self.criterionSeg(seg_fake_B, self.mask_A)

        self.loss_Seg = (self.loss_Seg_real + self.loss_Seg_fake) * (1 - self.first_phase_coeff)
        # self.loss_DefReg.backward()
        (self.loss_DefReg * self.opt.lambda_Def + self.loss_Seg * self.opt.lambda_Seg).backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A), rigid registration params, DVF and segmentation mask
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.loss_D = super().compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()  # update D's weights

        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netRigidReg, False)
        self.set_requires_grad(self.netDefReg, False)
        self.set_requires_grad(self.netSeg, False)
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
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
            self.bacward_DefReg_Seg()  # only back propagate through fake_B once
            self.optimizer_DefReg.step()
            self.optimizer_Seg.step()

    def get_transformed_images(self) -> Tuple[torch.Tensor, torch.Tensor]:
        reg_A = affine_transform.transform_image(self.real_B,
                                                 affine_transform.tensor_vector_to_matrix(self.reg_A_params.detach()),
                                                 device=self.real_B.device)

        reg_B = affine_transform.transform_image(self.real_B,
                                                 affine_transform.tensor_vector_to_matrix(self.reg_B_params.detach()),
                                                 device=self.real_B.device)
        return reg_A, reg_B

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        super().compute_visuals()

        reg_A, reg_B = self.get_transformed_images()
        self.transformed_LM_B = affine_transform.transform_image(self.transformed_LM_B,
                                                                 affine_transform.tensor_vector_to_matrix(
                                                                     self.reg_B_params.detach()),
                                                                 device=self.transformed_LM_B.device)

        self.distance_landmarks_b = distance_landmarks.get_distance_lmark(self.landmarks_B, self.transformed_LM_B,
                                                                          self.transformed_LM_B.device)

        self.diff_A = reg_A - self.transformed_B
        self.diff_B = reg_B - self.transformed_B
        self.diff_orig = self.real_B - self.transformed_B
        self.seg_fake_B = torch.argmax(self.seg_fake_B, dim=1, keepdim=True)
        self.seg_B = torch.argmax(self.seg_B, dim=1, keepdim=True)

        n_c = self.real_A.shape[2]

        self.reg_A_center_sag = reg_A[:, :, int(n_c / 2), ...]
        self.diff_A_center_sag = self.diff_A[:, :, int(n_c / 2), ...]
        self.reg_B_center_sag = reg_B[:, :, int(n_c / 2), ...]
        self.diff_B_center_sag = self.diff_B[:, :, int(n_c / 2), ...]
        self.diff_orig_center_sag = self.diff_orig[:, :, int(n_c / 2), ...]
        self.deformed_center_sag = self.transformed_B[:, :, int(n_c / 2), ...]

        n_c = self.real_A.shape[3]
        self.reg_A_center_cor = reg_A[:, :, :, int(n_c / 2), ...]
        self.diff_A_center_cor = self.diff_A[:, :, :, int(n_c / 2), ...]
        self.reg_B_center_cor = reg_B[:, :, :, int(n_c / 2), ...]
        self.diff_B_center_cor = self.diff_B[:, :, :, int(n_c / 2), ...]
        self.diff_orig_center_cor = self.diff_orig[:, :, :, int(n_c / 2), ...]
        self.deformed_center_cor = self.transformed_B[:, :, :, int(n_c / 2), ...]

        n_c = self.real_A.shape[4]
        self.reg_A_center_axi = reg_A[..., int(n_c / 2)]
        self.diff_A_center_axi = self.diff_A[..., int(n_c / 2)]
        self.reg_B_center_axi = reg_B[..., int(n_c / 2)]
        self.diff_B_center_axi = self.diff_B[..., int(n_c / 2)]
        self.diff_orig_center_axi = self.diff_orig[..., int(n_c / 2)]
        self.deformed_center_axi = self.transformed_B[..., int(n_c / 2)]

        n_c = self.real_A.shape[2]
        # average over channel to get the real and fake image
        self.mask_A_center_sag = self.mask_A[:, :, int(n_c / 2), ...]
        self.seg_A_center_sag = self.seg_fake_B[:, :, int(n_c / 2), ...]
        self.seg_B_center_sag = self.seg_B[:, :, int(n_c / 2), ...]

        n_c = self.real_A.shape[3]
        self.mask_A_center_cor = self.mask_A[:, :, :, int(n_c / 2), ...]
        self.seg_A_center_cor = self.seg_fake_B[:, :, :, int(n_c / 2), ...]
        self.seg_B_center_cor = self.seg_B[:, :, :, int(n_c / 2), ...]

        n_c = self.real_A.shape[4]
        self.mask_A_center_axi = self.mask_A[..., int(n_c / 2)]
        self.seg_A_center_axi = self.seg_fake_B[..., int(n_c / 2)]
        self.seg_B_center_axi = self.seg_B[..., int(n_c / 2)]

        n_c = int(self.real_A.shape[2] / 2)
        self.dvf_center_sag = self.dvf[:, :, n_c, ...]
        self.deformed_B_center_sag = self.deformed_A[:, :, n_c, ...]

        n_c = int(self.real_A.shape[3] / 2)
        self.dvf_center_cor = self.dvf[..., n_c, :]
        self.deformed_B_center_cor = self.deformed_A[..., n_c, :]

        n_c = int(self.real_A.shape[4] / 2)
        self.dvf_center_axi = self.dvf[:, :, ..., n_c]
        self.deformed_B_center_axi = self.deformed_A[..., n_c]

    def update_learning_rate(self, epoch=0):
        super().update_learning_rate(epoch=epoch)
        if epoch >= self.opt.epochs_before_reg:
            self.first_phase_coeff = 0

    def log_tensorboard(self, writer: SummaryWriter, losses: OrderedDict, global_step: int = 0):
        super(CUT3DMultiTaskModel, self).log_tensorboard(writer=writer, losses=losses, global_step=global_step)

        self.add_rigid_figures(global_step, writer)

        self.add_segmentation_figures(global_step, writer)

        self.add_deformable_figures(global_step, writer)

        writer.add_scalar('landmarks/', scalar_value=self.distance_landmarks_b, global_step=global_step)

    def add_deformable_figures(self, global_step, writer):
        axs, fig = vxm.torch.utils.init_figure(3, 8)
        vxm.torch.utils.set_axs_attribute(axs)
        vxm.torch.utils.fill_subplots(self.dvf.cpu()[:, 0:1, ...], axs=axs[0, :], img_name='Def. X', cmap='RdBu',
                                      fig=fig, show_colorbar=True)
        vxm.torch.utils.fill_subplots(self.dvf.cpu()[:, 1:2, ...], axs=axs[1, :], img_name='Def. Y', cmap='RdBu',
                                      fig=fig, show_colorbar=True)
        vxm.torch.utils.fill_subplots(self.dvf.cpu()[:, 2:3, ...], axs=axs[2, :], img_name='Def. Z', cmap='RdBu',
                                      fig=fig, show_colorbar=True)
        vxm.torch.utils.fill_subplots(self.deformed_A.detach().cpu(), axs=axs[3, :], img_name='Deformed')
        vxm.torch.utils.fill_subplots((self.deformed_A.detach() - self.real_A).abs().cpu(),
                                      axs=axs[4, :], img_name='Deformed - moving')
        overlay = self.real_A.repeat(1, 3, 1, 1, 1) * 0.5 + 0.5
        overlay *= 0.8
        overlay[:, 0:1, ...] += (0.5 * self.real_B.detach() + 0.5) * 0.8
        overlay[overlay > 1] = 1
        vxm.torch.utils.fill_subplots(overlay.cpu(), axs=axs[5, :], img_name='Moving overlay', cmap=None)
        overlay = self.deformed_A.repeat(1, 3, 1, 1, 1) * 0.5 + 0.5
        overlay *= 0.8
        overlay[:, 0:1, ...] += (0.5 * self.real_B.detach() + 0.5) * 0.8
        overlay[overlay > 1] = 1
        vxm.torch.utils.fill_subplots(overlay.cpu(), axs=axs[6, :], img_name='Deformed overlay', cmap=None)
        overlay = self.mask_A.repeat(1, 3, 1, 1, 1)
        overlay[:, 0:1, ...] = self.mask_A_deformed.detach()
        overlay[:, 2, ...] = 0
        vxm.torch.utils.fill_subplots(overlay.cpu(), axs=axs[7, :], img_name='Def. mask overlay', cmap=None)
        writer.add_figure(tag='Deformable', figure=fig, global_step=global_step)

    def add_rigid_figures(self, global_step, writer):
        axs, fig = vxm.torch.utils.init_figure(3, 4)
        vxm.torch.utils.set_axs_attribute(axs)
        vxm.torch.utils.fill_subplots(self.diff_A.cpu(), axs=axs[0, :], img_name='Diff A')
        vxm.torch.utils.fill_subplots(self.diff_B.cpu(), axs=axs[1, :], img_name='Diff B')
        vxm.torch.utils.fill_subplots(self.diff_orig.cpu(), axs=axs[2, :], img_name='Diff orig')
        vxm.torch.utils.fill_subplots(self.transformed_B.detach().cpu(), axs=axs[3, :], img_name='Transformed')
        writer.add_figure(tag='Rigid', figure=fig, global_step=global_step)

    def add_segmentation_figures(self, global_step, writer):
        axs, fig = vxm.torch.utils.init_figure(3, 7)
        vxm.torch.utils.set_axs_attribute(axs)
        vxm.torch.utils.fill_subplots(self.mask_A.cpu(), axs=axs[0, :], img_name='Mask MR')
        vxm.torch.utils.fill_subplots(self.seg_fake_B.detach().cpu(), axs=axs[1, :], img_name='Seg fake US')

        overlay = self.fake_B.detach().repeat(1, 3, 1, 1, 1) * 0.5 + 0.5
        overlay[:, 0:1, ...] += 0.5 * self.seg_fake_B.detach()
        overlay *= 0.8
        overlay[overlay > 1] = 1
        vxm.torch.utils.fill_subplots(overlay.cpu(), axs=axs[2, :], img_name='Fake mask overlay', cmap=None)
        vxm.torch.utils.fill_subplots(self.mask_A_deformed.detach().cpu(), axs=axs[3, :], img_name='Deformed mask')

        overlay = self.real_B.repeat(1, 3, 1, 1, 1) * 0.5 + 0.5
        overlay[:, 0:1, ...] += 0.5 * self.mask_A_deformed.detach()
        overlay *= 0.8
        overlay[overlay > 1] = 1
        vxm.torch.utils.fill_subplots(overlay.cpu(), axs=axs[4, :], img_name='Def. mask overlay', cmap=None)
        vxm.torch.utils.fill_subplots(self.seg_B.detach().cpu(), axs=axs[5, :], img_name='Seg. US')

        overlay = self.real_B.repeat(1, 3, 1, 1, 1) * 0.5 + 0.5
        overlay[:, 0:1, ...] += 0.5 * self.seg_B.detach()
        overlay *= 0.8
        overlay[overlay > 1] = 1
        vxm.torch.utils.fill_subplots(overlay.cpu(), axs=axs[6, :], img_name='Seg. US overlay', cmap=None)

        writer.add_figure(tag='Segmentation', figure=fig, global_step=global_step)
