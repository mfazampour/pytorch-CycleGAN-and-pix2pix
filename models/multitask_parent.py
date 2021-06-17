import argparse
import os
import sys
from collections import OrderedDict
from typing import Tuple

import torch
from torch.utils.tensorboard import SummaryWriter

import util.util as util
from util import affine_transform
from util import distance_landmarks
from . import networks
from . import networks3d
from util import tensorboard

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
            parser.add_argument('--augment_segmentation', action='store_true', help='Augment data before segmenting')
        return parser


    def __init__(self, opt):
        self.isTrain = opt.isTrain
        self.opt = opt

    def add_networks(self, opt, model_names, loss_functions, gpu_ids):
        # specify the models you want to save to the disk. The training/test scripts will call
        # <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            model_names += ['RigidReg', 'DefReg', 'Seg']
        else:  # during test time, only load G
            model_names += ['RigidReg', 'DefReg', 'Seg']

        loss_functions += ['backward_RigidReg', 'backward_DefReg_Seg']

        # We are using DenseNet for rigid registration -- actually DenseNet didn't provide any performance improvement
        # TODO change this to obelisk net
        self.netRigidReg = networks3d.define_reg_model(model_type=self.opt.netReg, n_input_channels=2,
                                                       num_classes=6, gpu_ids=gpu_ids, img_shape=opt.inshape)
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
        self.netDefReg = networks3d.init_net(self.netDefReg, gpu_ids=gpu_ids)

        self.netSeg = networks3d.define_G(opt.input_nc, opt.num_classes, opt.ngf,
                                          opt.netSeg, opt.norm, use_dropout=not opt.no_dropout,
                                          gpu_ids=gpu_ids, is_seg_net=True)
        self.transformer_label = networks3d.init_net(vxm.layers.SpatialTransformer(size=opt.inshape, mode='nearest'),
                                                     gpu_ids=gpu_ids)
        self.transformer_intensity = networks3d.init_net(vxm.layers.SpatialTransformer(size=opt.inshape),
                                                         gpu_ids=gpu_ids)
        self.resizer = networks3d.init_net(vxm.layers.ResizeTransform(vel_resize=0.5, ndims=3), gpu_ids=gpu_ids)

    def add_visdom_names(self, loss_names, visual_names):
        # specify the training losses you want to print out. The training/test scripts will call
        # <BaseModel.get_current_losses>
        loss_names += ['RigidReg_fake', 'RigidReg_real']
        loss_names += ['DefReg_real', 'DefReg_fake', 'Seg_real', 'Seg_fake']
        loss_names += ['landmarks']
        loss_names += ['diff_dice', 'moving_dice', 'warped_dice']
        # specify the images you want to save/display. The training/test scripts will call
        # <BaseModel.get_current_visuals>

        # rigid registration
        visual_names += ['diff_A_center_sag', 'diff_A_center_cor', 'diff_A_center_axi']
        visual_names += ['diff_B_center_sag', 'diff_B_center_cor', 'diff_B_center_axi']
        visual_names += ['diff_orig_center_sag', 'diff_orig_center_cor', 'diff_orig_center_axi']
        visual_names += ['deformed_center_sag', 'deformed_center_cor', 'deformed_center_axi']

        # segmentation
        visual_names += ['mask_A_center_sag', 'mask_A_center_cor', 'mask_A_center_axi']
        visual_names += ['seg_A_center_sag', 'seg_A_center_cor', 'seg_A_center_axi']
        visual_names += ['seg_B_center_sag', 'seg_B_center_cor', 'seg_B_center_axi']

        # deformable registration
        visual_names += ['dvf_center_sag', 'dvf_center_cor', 'dvf_center_axi']
        visual_names += ['deformed_B_center_sag', 'deformed_B_center_cor', 'deformed_B_center_axi']

    def add_optimizers(self, optimizers):
        self.criterionRigidReg = networks.RegistrationLoss()
        self.optimizer_RigidReg = torch.optim.Adam(self.netRigidReg.parameters(), lr=self.opt.lr_Reg,
                                                   betas=(self.opt.beta1, 0.999))
        optimizers.append(self.optimizer_RigidReg)

        self.criterionDefReg = getattr(sys.modules['models.networks'], self.opt.similarity + 'Loss')()
        self.criterionDefRgl = networks.GradLoss('l2', loss_mult=self.opt.int_downsize)
        self.optimizer_DefReg = torch.optim.Adam(self.netDefReg.parameters(), lr=self.opt.lr_Def)
        optimizers.append(self.optimizer_DefReg)

        self.criterionSeg = networks.DiceLoss()
        self.optimizer_Seg = torch.optim.Adam(self.netSeg.parameters(), lr=self.opt.lr_Seg,
                                              betas=(self.opt.beta1, 0.999))
        optimizers.append(self.optimizer_Seg)
        self.transformer = vxm.layers.SpatialTransformer((1, 1, 80, 80, 80))
        # resize = vxm.layers.ResizeTransform(0.5, 1)

        self.first_phase_coeff = 1
        self.loss_landmarks = 0
        self.loss_landmarks_deformed = 0

        self.first_phase_coeff = 1 if self.opt.epochs_before_reg > 0 else 0

    def clean_tensors(self):
        all_members = self.__dict__.keys()
        # print(f'{all_members}')
        # GPUtil.showUtilization()
        for item in all_members:
            if isinstance(self.__dict__[item], torch.Tensor):
                self.__dict__[item] = None
        torch.cuda.empty_cache()
        # GPUtil.showUtilization()

    def set_mt_input(self, input, real_B, shape, device, dtype=torch.float32):
        self.patient = input['Patient']
        self.landmarks_A = input['A_landmark'].to(device).unsqueeze(dim=0)
        self.landmarks_B = input['B_landmark'].to(device).unsqueeze(dim=0)

        affine, self.gt_vector = affine_transform.create_random_affine(shape[0],
                                                                       shape[-3:],
                                                                       dtype,
                                                                       device=device)

        self.deformed_B = affine_transform.transform_image(real_B, affine, device)
        self.deformed_LM_B = affine_transform.transform_image(self.landmarks_B, affine, self.landmarks_B.device)

        if self.opt.show_volumes:
            affine_transform.show_volumes([real_B, self.deformed_B])

        self.mask_A = input['A_mask'].to(device).type(dtype)
        if input['B_mask_available'][0]:  # TODO in this way it only works with batch size 1
            self.mask_B = input['B_mask'].to(device).type(dtype)
            self.deformed_mask_B = affine_transform.transform_image(self.mask_B, affine, self.mask_B.device)
        else:
            self.mask_B = None


    def init_loss_tensors(self):
        self.loss_DefReg_real = torch.tensor([0.0])
        self.loss_DefReg = torch.tensor([0.0])
        self.loss_DefReg_fake = torch.tensor([0.0])
        self.loss_DefRgl = torch.tensor([0.0])

        self.loss_Seg_real = torch.tensor([0.0])
        self.loss_Seg_fake = torch.tensor([0.0])
        self.loss_Seg = torch.tensor([0.0])


        self.loss_RigidReg_fake = torch.tensor([0.0])
        self.loss_RigidReg_real = torch.tensor([0.0])
        self.loss_RigidReg = torch.tensor([0.0])
      #  self.diff_dice= torch.tensor([0.0])
        self.loss_landmarks_diff = torch.tensor([0.0])
        self.loss_landmarks_deformed= torch.tensor([0.0])
        self.loss_landmarks = torch.tensor([0.0])

    def mt_forward(self, fake_B, real_B, fixed, real_A):
        self.reg_A_params = self.netRigidReg(torch.cat([fake_B, self.deformed_B], dim=1))
        self.reg_B_params = self.netRigidReg(torch.cat([real_B, self.deformed_B], dim=1))

        def_reg_output = self.netDefReg(fake_B * 0.5 + 0.5, fixed * 0.5 + 0.5, registration=not self.isTrain)

        if self.opt.bidir and self.isTrain:
            (self.deformed_fake_B, self.deformed_B, self.dvf) = def_reg_output
        else:
            (self.deformed_fake_B, self.dvf) = def_reg_output

        if self.isTrain:
            self.dvf = self.resizer(self.dvf)
        self.deformed_A = self.transformer_intensity(real_A, self.dvf.detach())
        self.mask_A_deformed = self.transformer_label(self.mask_A, self.dvf.detach())

        if self.opt.augment_segmentation:
            self.augmented_mask, affine = affine_transform.apply_random_affine(
                torch.cat([self.mask_A, self.mask_A_deformed], dim=0), rotation=0.5, translation=0.1, batchsize=2)
            # self.augmented_B, affine = affine_transform.apply_random_affine(
            #     torch.cat([fake_B, fixed], dim=0), rotation=0.5, translation=0.1, batchsize=2)
            # self.augmented_mask, _ = affine_transform.apply_random_affine(
            #     torch.cat([self.mask_A, self.mask_A_deformed], dim=0), affine=affine, batchsize=2)
            self.augmented_fake, _ = affine_transform.apply_random_affine(fake_B, affine=affine[self.opt.batch_size:, ...])
            self.augmented_real, _ = affine_transform.apply_random_affine(fixed, affine=affine[:self.opt.batch_size, ...])
            # seg_ = self.netSeg(self.augmented_B)
            self.seg_B = self.netSeg(self.augmented_real)
            self.seg_fake_B = self.netSeg(self.augmented_fake)
        else:
            self.seg_B = self.netSeg(fixed)
            self.seg_fake_B = self.netSeg(fake_B)
        self.Landmark_A_dvf = self.transformer_label(self.landmarks_A, self.dvf.detach())
        self.compute_gt_dice()
        self.compute_landmark_loss()

        self.fake_B = fake_B
        self.fixed = fixed
        self.real_A = real_A
        self.real_B = real_B


    def compute_gt_dice(self):
        """
        calculate the dice score between the deformed mask and the ground truth if we have it
        Returns
        -------

        """
        if self.mask_B is not None:
            shape = list(self.mask_A_deformed.shape)
            n = self.opt.num_classes  # number of classes
            shape[1] = n
            one_hot_fixed = torch.zeros(shape, device=self.mask_B.device)
            one_hot_deformed = torch.zeros(shape, device=self.mask_B.device)
            one_hot_moving = torch.zeros(shape, device=self.mask_B.device)
            for i in range(n):
                one_hot_fixed[:, i, self.mask_B[0, 0, ...] == i] = 1
                one_hot_deformed[:, i, self.mask_A_deformed[0, 0, ...] == i] = 1
                one_hot_moving[:, i, self.mask_A[0, 0, ...] == i] = 1


            self.loss_warped_dice = compute_meandice(one_hot_deformed, one_hot_fixed, include_background=False)
            self.loss_moving_dice = compute_meandice(one_hot_moving, one_hot_fixed, include_background=False)
            self.loss_diff_dice = self.loss_warped_dice - self.loss_moving_dice

        else:
            self.loss_warped_dice = None
            self.loss_moving_dice = None
            self.loss_diff_dice = None

    def compute_landmark_loss(self):

        # Calc landmark difference from original landmarks and deformed

        self.loss_landmarks_beginning = distance_landmarks.get_distance_lmark(self.landmarks_A, self.deformed_LM_B,
                                                                              self.deformed_LM_B.device)

        # Add affine transform from G to the original landmarks A to match deformed B landmarks
        self.landmarks_rigid_A = affine_transform.transform_image(self.landmarks_A,
                                                                  affine_transform.tensor_vector_to_matrix(
                                                                      self.reg_A_params.detach()),

                                                                  device=self.landmarks_A.device)

        #  Calc landmark difference from deformed landmarks and the affine(Deformed)
        self.loss_landmarks_rigid = distance_landmarks.get_distance_lmark(self.deformed_LM_B, self.landmarks_rigid_A,
                                                                          self.deformed_LM_B.device)

        self.loss_landmarks_def = distance_landmarks.get_distance_lmark(self.Landmark_A_dvf, self.deformed_LM_B,
                                                                        self.deformed_LM_B.device)

        self.loss_landmarks_rigid_diff = self.loss_landmarks_beginning - self.loss_landmarks_rigid
        self.loss_landmarks_def_diff = self.loss_landmarks_beginning - self.loss_landmarks_def

    def mt_g_backward(self, fake_B):
        # rigid registration
        loss_G_Reg = self.criterionRigidReg(self.reg_A_params, self.gt_vector) * self.opt.lambda_Reg

        # Def registration:
        if self.opt.bidir:
            loss_DefReg_fake = self.criterionDefReg(self.deformed_B.detach(), fake_B)
        else:
            loss_DefReg_fake = 0

        # segmentation
        loss_G_Seg = self.criterionSeg(self.seg_fake_B, self.mask_A) * self.opt.lambda_Seg

        # combine loss and calculate gradients
        loss_G = loss_DefReg_fake * (1 - self.first_phase_coeff) + \
                  loss_G_Seg * (1 - self.first_phase_coeff)

        if self.opt.use_rigid_branch:
            loss_G += loss_G_Reg * (1 - self.first_phase_coeff)
        return loss_G

    def backward_RigidReg(self):
        """
        Calculate Segmentation loss to update the segmentation networks
        Returns
        -------
        """
        fake_B = self.fake_B
        reg_A_params = self.netRigidReg(torch.cat([fake_B.detach(), self.deformed_B], dim=1))
        self.loss_RigidReg_fake = self.criterionRigidReg(reg_A_params,
                                                         self.gt_vector) * self.opt.lambda_Reg  # to be of the same order as loss_G_Seg
        self.loss_RigidReg_real = self.criterionRigidReg(self.reg_B_params,
                                                         self.gt_vector) * self.opt.lambda_Reg  # to be of the same order as loss_G_Seg

        self.loss_RigidReg = self.loss_RigidReg_real + self.loss_RigidReg_fake * (1 - self.first_phase_coeff)
        if torch.is_grad_enabled():
            self.loss_RigidReg.backward()

    def backward_DefReg_Seg(self):
        fixed, fake_B, real_B = self.fixed, self.fake_B, self.real_B
        # fixed = self.idt_B.detach() if self.opt.reg_idt_B else self.real_B
        def_reg_output = self.netDefReg(fake_B.detach() * 0.5 + 0.5, fixed * 0.5 + 0.5)  # fake_B is the moving image here
        if self.opt.bidir:
            (deformed_fake_B, deformed_B, dvf) = def_reg_output
        else:
            (deformed_fake_B, dvf) = def_reg_output

        self.loss_DefReg_real = self.criterionDefReg(deformed_fake_B, real_B)  # TODO add weights same as vxm!
        self.loss_DefReg = self.loss_DefReg_real
        if self.opt.bidir:
            self.loss_DefReg_fake = self.criterionDefReg(deformed_B, fake_B.detach())
            self.loss_DefReg += self.loss_DefReg_fake
        else:
            self.loss_DefReg_fake = torch.tensor([0.0])
        self.loss_DefRgl = self.criterionDefRgl(dvf, None)
        self.loss_DefReg += self.loss_DefRgl
        self.loss_DefReg *= (1 - self.first_phase_coeff)

        if self.opt.augment_segmentation:
            seg_B = self.netSeg(self.augmented_real)
            mask = self.augmented_mask[self.opt.batch_size:, ...]
        else:
            seg_B = self.netSeg(fixed)
            mask = self.mask_A_deformed
        self.loss_Seg_real = self.criterionSeg(seg_B, mask)

        if self.opt.augment_segmentation:
            seg_fake_B = self.netSeg(self.augmented_fake.detach())
            mask = self.augmented_mask[:self.opt.batch_size, ...].detach()
        else:
            seg_fake_B = self.netSeg(fake_B.detach())
            mask = self.mask_A
        self.loss_Seg_fake = self.criterionSeg(seg_fake_B, mask)

        self.loss_Seg = (self.loss_Seg_real + self.loss_Seg_fake) * (1 - self.first_phase_coeff)
        # self.loss_DefReg.backward()
        if torch.is_grad_enabled():
            (self.loss_DefReg * self.opt.lambda_Def + self.loss_Seg * self.opt.lambda_Seg).backward()


    def get_transformed_images(self, real_B) -> Tuple[torch.Tensor, torch.Tensor]:
        reg_A = affine_transform.transform_image(real_B,
                                                 affine_transform.tensor_vector_to_matrix(self.reg_A_params.detach()),
                                                 device=real_B.device)

        reg_B = affine_transform.transform_image(real_B,
                                                 affine_transform.tensor_vector_to_matrix(self.reg_B_params.detach()),
                                                 device=real_B.device)
        return reg_A, reg_B

    def compute_mt_visuals(self, real_B, shape):
        reg_A, reg_B = self.get_transformed_images(real_B)

        self.diff_A = reg_A - self.deformed_B
        self.diff_B = reg_B - self.deformed_B
        self.diff_orig = real_B - self.deformed_B
        seg_fake_B_img = torch.argmax(self.seg_fake_B, dim=1, keepdim=True)
        seg_B_img = torch.argmax(self.seg_B, dim=1, keepdim=True)

        n_c = shape[2]

        self.reg_A_center_sag = reg_A[:, :, int(n_c / 2), ...]
        self.diff_A_center_sag = self.diff_A[:, :, int(n_c / 2), ...]
        self.reg_B_center_sag = reg_B[:, :, int(n_c / 2), ...]
        self.diff_B_center_sag = self.diff_B[:, :, int(n_c / 2), ...]
        self.diff_orig_center_sag = self.diff_orig[:, :, int(n_c / 2), ...]
        self.deformed_center_sag = self.deformed_B[:, :, int(n_c / 2), ...]

        n_c = shape[3]
        self.reg_A_center_cor = reg_A[:, :, :, int(n_c / 2), ...]
        self.diff_A_center_cor = self.diff_A[:, :, :, int(n_c / 2), ...]
        self.reg_B_center_cor = reg_B[:, :, :, int(n_c / 2), ...]
        self.diff_B_center_cor = self.diff_B[:, :, :, int(n_c / 2), ...]
        self.diff_orig_center_cor = self.diff_orig[:, :, :, int(n_c / 2), ...]
        self.deformed_center_cor = self.deformed_B[:, :, :, int(n_c / 2), ...]

        n_c = shape[4]
        self.reg_A_center_axi = reg_A[..., int(n_c / 2)]
        self.diff_A_center_axi = self.diff_A[..., int(n_c / 2)]
        self.reg_B_center_axi = reg_B[..., int(n_c / 2)]
        self.diff_B_center_axi = self.diff_B[..., int(n_c / 2)]
        self.diff_orig_center_axi = self.diff_orig[..., int(n_c / 2)]
        self.deformed_center_axi = self.deformed_B[..., int(n_c / 2)]

        n_c = shape[2]
        # average over channel to get the real and fake image
        self.mask_A_center_sag = self.mask_A[:, :, int(n_c / 2), ...]
        self.seg_A_center_sag = seg_fake_B_img[:, :, int(n_c / 2), ...]
        self.seg_B_center_sag = seg_B_img[:, :, int(n_c / 2), ...]

        n_c = shape[3]
        self.mask_A_center_cor = self.mask_A[:, :, :, int(n_c / 2), ...]
        self.seg_A_center_cor = seg_fake_B_img[:, :, :, int(n_c / 2), ...]
        self.seg_B_center_cor = seg_B_img[:, :, :, int(n_c / 2), ...]

        n_c = shape[4]
        self.mask_A_center_axi = self.mask_A[..., int(n_c / 2)]
        self.seg_A_center_axi = seg_fake_B_img[..., int(n_c / 2)]
        self.seg_B_center_axi = seg_B_img[..., int(n_c / 2)]

        n_c = int(shape[2] / 2)
        self.dvf_center_sag = self.dvf[:, :, n_c, ...]
        self.deformed_B_center_sag = self.deformed_A[:, :, n_c, ...]

        n_c = int(shape[3] / 2)
        self.dvf_center_cor = self.dvf[..., n_c, :]
        self.deformed_B_center_cor = self.deformed_A[..., n_c, :]

        n_c = int(shape[4] / 2)
        self.dvf_center_axi = self.dvf[:, :, ..., n_c]
        self.deformed_B_center_axi = self.deformed_A[..., n_c]


    def log_mt_tensorboard(self, real_A, real_B, fake_B, writer: SummaryWriter, global_step: int = 0,
                           use_image_name=False, mode=''):
        self.add_rigid_figures(mode, global_step, writer, use_image_name=use_image_name)

        self.add_segmentation_figures(mode, fake_B, real_B, global_step, writer, use_image_name=use_image_name)

        self.add_deformable_figures(mode, real_A, real_B, global_step, writer, use_image_name=use_image_name)

        self.add_landmark_losses(mode, global_step, writer, use_image_name=use_image_name)

    def add_deformable_figures(self, mode, real_A, real_B, global_step, writer, use_image_name=False):
        n = 8 if self.mask_B is None else 10
        axs, fig = tensorboard.init_figure(3, n)
        tensorboard.set_axs_attribute(axs)
        tensorboard.fill_subplots(self.dvf.cpu()[:, 0:1, ...], axs=axs[0, :], img_name='Def. X', cmap='RdBu',
                                      fig=fig, show_colorbar=True)
        tensorboard.fill_subplots(self.dvf.cpu()[:, 1:2, ...], axs=axs[1, :], img_name='Def. Y', cmap='RdBu',
                                      fig=fig, show_colorbar=True)
        tensorboard.fill_subplots(self.dvf.cpu()[:, 2:3, ...], axs=axs[2, :], img_name='Def. Z', cmap='RdBu',
                                      fig=fig, show_colorbar=True)
        tensorboard.fill_subplots(self.deformed_A.detach().cpu(), axs=axs[3, :], img_name='Deformed')
        tensorboard.fill_subplots((self.deformed_A.detach() - real_A).abs().cpu(),
                                      axs=axs[4, :], img_name='Deformed - moving')
        overlay = real_A.repeat(1, 3, 1, 1, 1) * 0.5 + 0.5
        overlay *= 0.8
        overlay[:, 0:1, ...] += (0.5 * real_B.detach() + 0.5) * 0.8
        overlay[overlay > 1] = 1
        tensorboard.fill_subplots(overlay.cpu(), axs=axs[5, :], img_name='Moving overlay', cmap=None)
        overlay = self.deformed_A.repeat(1, 3, 1, 1, 1) * 0.5 + 0.5
        overlay *= 0.8
        overlay[:, 0:1, ...] += (0.5 * real_B.detach() + 0.5) * 0.8
        overlay[overlay > 1] = 1
        tensorboard.fill_subplots(overlay.cpu(), axs=axs[6, :], img_name='Deformed overlay', cmap=None)
        overlay = self.mask_A.repeat(1, 3, 1, 1, 1)
        overlay[:, 0:1, ...] = self.mask_A_deformed.detach()
        overlay[:, 2, ...] = 0
        tensorboard.fill_subplots(overlay.cpu(), axs=axs[7, :], img_name='Def. mask overlay', cmap=None)
        if self.mask_B is not None:
            overlay = self.mask_B.repeat(1, 3, 1, 1, 1)
            overlay[:, 0:1, ...] = self.mask_A.detach()
            overlay[:, 2, ...] = 0
            dice = self.loss_moving_dice.item() if self.loss_moving_dice is not None else -1
            tensorboard.fill_subplots(overlay.cpu(), axs=axs[8, :],
                                      img_name=f'mask moving on US\nDice {dice:.3f}', cmap=None)

            overlay = self.mask_B.repeat(1, 3, 1, 1, 1)
            overlay[:, 0:1, ...] = self.mask_A_deformed.detach()
            overlay[:, 2, ...] = 0
            dice = self.loss_warped_dice.item() if self.loss_warped_dice is not None else -1
            tensorboard.fill_subplots(overlay.cpu(), axs=axs[9, :],
                                      img_name=f'mask warped on US\nDice {dice:.3f}', cmap=None)
        #
        if use_image_name:
            tag = mode + f'{self.patient}/Deformable'
        else:
            tag = mode + '/Deformable'
        writer.add_figure(tag=tag, figure=fig, global_step=global_step)

    def add_rigid_figures(self, mode, global_step, writer, use_image_name=False):
        axs, fig = tensorboard.init_figure(3, 4)
        tensorboard.set_axs_attribute(axs)
        tensorboard.fill_subplots(self.diff_A.cpu(), axs=axs[0, :], img_name='Diff A')
        tensorboard.fill_subplots(self.diff_B.cpu(), axs=axs[1, :], img_name='Diff B')
        tensorboard.fill_subplots(self.diff_orig.cpu(), axs=axs[2, :], img_name='Diff orig')
        tensorboard.fill_subplots(self.deformed_B.detach().cpu(), axs=axs[3, :], img_name='Transformed')

        if use_image_name:
            tag = mode + f'{self.patient}/Rigid'
        else:
            tag = mode + '/Rigid'
        writer.add_figure(tag=tag, figure=fig, global_step=global_step)

    def add_segmentation_figures(self, mode, fake_B, real_B, global_step, writer, use_image_name=False):
        axs, fig = tensorboard.init_figure(3, 7)
        tensorboard.set_axs_attribute(axs)
        tensorboard.fill_subplots(self.mask_A.cpu(), axs=axs[0, :], img_name='Mask MR')
        with torch.no_grad():
            seg_fake_B = self.netSeg(self.fake_B.detach())
            seg_fake_B = torch.argmax(seg_fake_B, dim=1, keepdim=True)
            tensorboard.fill_subplots(seg_fake_B.cpu(), axs=axs[1, :], img_name='Seg fake US')
        idx = 2
        if self.opt.augment_segmentation:
            img = self.augmented_fake.detach()
        else:
            img = fake_B.detach()
        overlay = img.repeat(1, 3, 1, 1, 1) * 0.5 + 0.5
        seg_fake_B_img = torch.argmax(self.seg_fake_B, dim=1, keepdim=True)
        overlay[:, 0:1, ...] += 0.5 * seg_fake_B_img.detach()
        overlay *= 0.8
        overlay[overlay > 1] = 1
        tensorboard.fill_subplots(overlay.cpu(), axs=axs[idx, :], img_name='Fake mask overlay', cmap=None)
        idx += 1
        tensorboard.fill_subplots(self.mask_A_deformed.detach().cpu(), axs=axs[idx, :], img_name='Deformed mask')
        idx += 1

        if self.opt.augment_segmentation:
            img = self.augmented_real.detach()
            mask = self.augmented_mask[self.opt.batch_size:, ...].detach()
        else:
            img = real_B.detach()
            mask = self.mask_A_deformed.detach()
        overlay = img.repeat(1, 3, 1, 1, 1) * 0.5 + 0.5
        overlay[:, 0:1, ...] += 0.5 * mask
        overlay *= 0.8
        overlay[overlay > 1] = 1
        tensorboard.fill_subplots(overlay.cpu(), axs=axs[idx, :], img_name='Def. mask overlay', cmap=None)
        idx += 1

        seg_B_img = torch.argmax(self.seg_B, dim=1, keepdim=True)
        tensorboard.fill_subplots(seg_B_img.detach().cpu(), axs=axs[idx, :], img_name='Seg. US')
        idx += 1

        overlay = img.repeat(1, 3, 1, 1, 1) * 0.5 + 0.5
        overlay[:, 0:1, ...] += 0.5 * seg_B_img.detach()
        overlay *= 0.8
        overlay[overlay > 1] = 1
        tensorboard.fill_subplots(overlay.cpu(), axs=axs[idx, :], img_name='Seg. US overlay', cmap=None)
        if use_image_name:
            tag = mode + f'{self.patient}/Segmentation'
        else:
            tag = mode + '/Segmentation'
        writer.add_figure(tag=tag, figure=fig, global_step=global_step)

    def add_landmark_losses(self, mode, global_step, writer, use_image_name=False):

        if self.loss_landmarks_rigid_diff is not None:
            writer.add_scalar(mode + '/landmarks/difference_rigid',
                              scalar_value=self.loss_landmarks_rigid_diff,
                              global_step=global_step)
        if self.loss_landmarks_rigid is not None:
            writer.add_scalar(mode + '/landmarks/rigid', scalar_value=self.loss_landmarks_rigid,
                              global_step=global_step)
        if self.loss_landmarks_def is not None:
            writer.add_scalar(mode + '/landmarks/def', scalar_value=self.loss_landmarks_def,
                              global_step=global_step)
        if self.loss_landmarks_def_diff is not None:
            writer.add_scalar(mode + '/landmarks/difference_def', scalar_value=self.loss_landmarks_def_diff,
                              global_step=global_step)
        if self.loss_diff_dice is not None:
            writer.add_scalar(mode + '/DICE/difference', scalar_value=self.loss_diff_dice, global_step=global_step)
            writer.add_scalar(mode + '/DICE/deformed', scalar_value=self.loss_warped_dice,
                              global_step=global_step)
            writer.add_scalar(mode + '/DICE/moving', scalar_value=self.loss_moving_dice,
                              global_step=global_step)

    def get_current_landmark_distances(self):
        return self.loss_landmarks_beginning , self.loss_landmarks_rigid, self.loss_landmarks_def

    def set_coeff_multitask_loss(self, epoch):
        if epoch >= self.opt.epochs_before_reg:
            self.first_phase_coeff = 1 / (epoch + 1 - self.opt.epochs_before_reg)