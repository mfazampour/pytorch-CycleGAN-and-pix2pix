import argparse
import os
from collections import OrderedDict
from typing import Tuple

import torch
from torch.utils.tensorboard import SummaryWriter

from util import affine_transform
from . import networks
from . import networks3d
from .pix2pix3d_model import Pix2Pix3dModel

os.environ['VXM_BACKEND'] = 'pytorch'
from voxelmorph import voxelmorph as vxm

class Pix2Pix3dMultiTaskModel(Pix2Pix3dModel):

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
        parser.add_argument('--netReg', type=str, default='NormalNet', help='Type of network used for registration')
        parser.add_argument('--show_volumes', type=bool, default=False, help='visualize transformed volumes w napari')

        parser.add_argument('--L1_epochs', type=int, default=15,
                            help='number of epochs to train the network with the L1 loss before reg loss is used')
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

        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--no-lsgan', type=bool, default=False)
            parser.add_argument('--lambda_Reg', type=float, default=0.5, help='weight for the registration loss')
            parser.add_argument('--lr-Reg', type=float, default=0.00001, help='learning rate for the reg. network opt.')
            parser.add_argument('--lambda_Seg', type=float, default=0.5, help='weight for the segmentation loss')
            parser.add_argument('--lambda_Def', type=float, default=1.0, help='weight for the segmentation loss')
            parser.add_argument('--lr-Def', type=float, default=0.00001, help='learning rate for the reg. network opt.')

            # loss hyperparameters
            parser.add_argument('--image-loss', default='mse',
                                help='image reconstruction loss - can be mse or ncc (default: mse)')

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

        if self.isTrain:
            # define loss functions
            self.criterionRigidReg = networks.RegistrationLoss()
            self.optimizer_RigidReg = torch.optim.Adam(self.netRigidReg.parameters(), lr=opt.lr_Reg,
                                                       betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_RigidReg)

            self.criterionDefReg = torch.nn.MSELoss()  # TODO change later to sth more meaningful (LCC?!)
            self.criterionDefRgl = networks.GradLoss('l2', loss_mult=opt.int_downsize)
            self.optimizer_DefReg = torch.optim.Adam(self.netDefReg.parameters(), lr=opt.lr_Def)
            self.optimizers.append(self.optimizer_DefReg)

            self.criterionSeg = networks.DiceLoss()
            self.optimizer_Seg = torch.optim.Adam(self.netSeg.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_Seg)

        self.first_phase_coeff = 1

    def set_networks(self, opt):
        # specify the models you want to save to the disk. The training/test scripts will call
        # <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D', 'RigidReg', 'DefReg', 'Seg']
        else:  # during test time, only load G
            self.model_names = ['G', 'RigidReg', 'DefReg', 'Seg']

        # We are using DenseNet for rigid registration -- actually DenseNet didn't provide any performance improvement
        # TODO change this to obelisk net
        self.netRigidReg = networks3d.define_reg_model(model_type=self.opt.netReg, n_input_channels=2,
                                                       num_classes=6, gpu_ids=self.gpu_ids)
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
                                          opt.netSeg, opt.norm, not opt.no_dropout,
                                          gpu_ids=self.gpu_ids, is_seg_net=True)
        self.transformer = networks3d.init_net(vxm.layers.SpatialTransformer(size=opt.inshape, mode='nearest'),
                                               gpu_ids=self.gpu_ids)
        self.resizer = networks3d.init_net(vxm.layers.ResizeTransform(vel_resize=0.5, ndims=3), gpu_ids=self.gpu_ids)

    def set_visdom_names(self):
        # specify the training losses you want to print out. The training/test scripts will call
        # <BaseModel.get_current_losses>
        self.loss_names = ['G', 'G_GAN', 'G_L1', 'D_real', 'D_fake', 'RigidReg_fake', 'RigidReg_real']
        self.loss_names += ['DefReg_real', 'DefReg_fake', 'Seg_real', 'Seg_fake']
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

        self.mask_A = input['A_mask'].to(self.device)

        ###
        self.loss_DefReg_real = torch.tensor([0.0])
        self.loss_DefReg = torch.tensor([0.0])
        self.loss_DefReg_fake = torch.tensor([0.0])

        self.loss_DefReg_fake = torch.tensor([0.0])
        self.loss_DefRgl = torch.tensor([0.0])
        self.loss_Seg_real = torch.tensor([0.0])
        self.loss_Seg_fake = torch.tensor([0.0])

        self.loss_Seg = torch.tensor([0.0])

    def forward(self):
        super().forward()
        self.reg_A_params = self.netRigidReg(torch.cat([self.fake_B, self.transformed_B], dim=1))
        self.reg_B_params = self.netRigidReg(torch.cat([self.real_B, self.transformed_B], dim=1))

        def_reg_output = self.netDefReg(self.real_B, self.fake_B, registration=not self.isTrain)
        if self.opt.bidir:
            (self.deformed_B, self.deformed_fake_B, self.dvf) = def_reg_output
        else:
            (self.deformed_B, self.dvf) = def_reg_output

        if self.isTrain:
            self.dvf = self.resizer(self.dvf)
        self.seg_B = self.netSeg(self.deformed_B)
        self.seg_fake_B = self.netSeg(self.fake_B)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)

        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        # Third, rigid registration =
        loss_G_Reg = self.criterionRigidReg(self.reg_A_params, self.gt_vector) * self.opt.lambda_Reg

        # fourth, Def registration:
        if self.opt.bidir:
            loss_DefReg_fake = self.criterionDefReg(self.deformed_fake_B, self.real_B)
        else:
            loss_DefReg_fake = 0

        # fifth, segmentation
        loss_G_Seg = self.criterionSeg(self.seg_fake_B, self.mask_A) * self.opt.lambda_Seg

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + \
                      self.loss_G_L1 * self.first_phase_coeff + \
                      loss_G_Reg * (1 - self.first_phase_coeff) + \
                      loss_DefReg_fake * (1 - self.first_phase_coeff) + \
                      loss_G_Seg * (1 - self.first_phase_coeff)

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
        def_reg_output = self.netDefReg(self.real_B, self.fake_B.detach())
        if self.opt.bidir:
            (deformed_B, deformed_fake_B, dvf) = def_reg_output
        else:
            (deformed_B, dvf) = def_reg_output

        self.loss_DefReg_real = self.criterionDefReg(deformed_B, self.fake_B.detach())  # TODO add weights same as vxm!
        self.loss_DefReg = self.loss_DefReg_real
        if self.opt.bidir:
            self.loss_DefReg_fake = self.criterionDefReg(deformed_fake_B, self.real_B)
            self.loss_DefReg += self.loss_DefReg_fake
        else:
            self.loss_DefReg_fake = torch.tensor([0.0])
        self.loss_DefRgl = self.criterionDefRgl(dvf, None)
        self.loss_DefReg += self.loss_DefRgl
        self.loss_DefReg *= (1 - self.first_phase_coeff)

        seg_B = self.netSeg(deformed_B)
        self.loss_Seg_real = self.criterionSeg(seg_B, self.mask_A)

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
        self.backward_D()  # calculate gradients for D
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
        self.set_requires_grad(self.netRigidReg, True)
        self.optimizer_RigidReg.zero_grad()
        self.backward_RigidReg()
        self.optimizer_RigidReg.step()

        # update deformable registration and segmentation network
        if (1 - self.first_phase_coeff) == 0:
            return
        self.set_requires_grad(self.netDefReg, True)
        self.set_requires_grad(self.netSeg, True)
        self.optimizer_DefReg.zero_grad()
        self.optimizer_Seg.zero_grad()
        self.bacward_DefReg_Seg()
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

        self.diff_A = reg_A - self.transformed_B
        self.diff_B = reg_B - self.transformed_B
        self.diff_orig = self.real_B - self.transformed_B

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

        self.seg_fake_B = torch.argmax(self.seg_fake_B, dim=1, keepdim=True)
        self.seg_B = torch.argmax(self.seg_B, dim=1, keepdim=True)

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
        self.deformed_B_center_sag = self.deformed_B[:, :, n_c, ...]

        n_c = int(self.real_A.shape[3] / 2)
        self.dvf_center_cor = self.dvf[:, :, ..., n_c, :]
        self.deformed_B_center_cor = self.deformed_B[..., n_c, :]

        n_c = int(self.real_A.shape[4] / 2)
        self.dvf_center_axi = self.dvf[:, :, ..., n_c]
        self.deformed_B_center_axi = self.deformed_B[..., n_c]

    def update_learning_rate(self, epoch=0):
        super().update_learning_rate(epoch)
        if epoch >= self.opt.L1_epochs:
            self.first_phase_coeff = 0

    def log_tensorboard(self, writer: SummaryWriter, losses: OrderedDict, global_step: int = 0):
        axs, fig = vxm.torch.utils.init_figure(3, 12)
        vxm.torch.utils.set_axs_attribute(axs)
        vxm.torch.utils.fill_subplots(self.real_A.cpu(), axs=axs[0, :], img_name='A')
        vxm.torch.utils.fill_subplots(self.fake_B.detach().cpu(), axs=axs[1, :], img_name='fake')
        vxm.torch.utils.fill_subplots(self.real_B.cpu(), axs=axs[2, :], img_name='B')
        vxm.torch.utils.fill_subplots(self.diff_A.cpu(), axs=axs[3, :], img_name='Diff A')
        vxm.torch.utils.fill_subplots(self.diff_B.cpu(), axs=axs[4, :], img_name='Diff B')
        vxm.torch.utils.fill_subplots(self.diff_orig.cpu(), axs=axs[5, :], img_name='Diff Orig')
        vxm.torch.utils.fill_subplots(self.transformed_B.detach().cpu(), axs=axs[6, :], img_name='Transformed')
        vxm.torch.utils.fill_subplots(self.mask_A.cpu(), axs=axs[7, :], img_name='Mask A')
        vxm.torch.utils.fill_subplots(self.seg_fake_B.detach().cpu(), axs=axs[8, :], img_name='Seg Fake')
        vxm.torch.utils.fill_subplots(self.seg_B.cpu(), axs=axs[9, :], img_name='Seg B')
        vxm.torch.utils.fill_subplots(self.dvf.cpu().detach().cpu(), axs=axs[10, :], img_name='DVF', cmap=None)
        vxm.torch.utils.fill_subplots(self.deformed_B.detach().cpu(), axs=axs[11, :], img_name='Deformed')

        writer.add_figure(tag='volumes', figure=fig, global_step=global_step)

        for key in losses:
            writer.add_scalar(f'losses/{key}', scalar_value=losses[key], global_step=global_step)

