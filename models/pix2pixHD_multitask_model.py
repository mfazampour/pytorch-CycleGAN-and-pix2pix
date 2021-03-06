import argparse
import os
from collections import OrderedDict
from typing import Tuple
import util.util as util
from util.image_pool import ImagePool
from util import affine_transform

from util import distance_landmarks
from models.base_model import BaseModel
from torch.autograd import Variable
from monai.visualize import img2tensorboard
from monai.metrics import compute_meandice

import torch
from torch.utils.tensorboard import SummaryWriter
import sys
from models import networks3d
from models import networks
from torchgeometry import losses
from .pix2pixHD_model import pix2pixHDModel

os.environ['VXM_BACKEND'] = 'pytorch'
#sys.path.append('/home/kixcodes/Documents/python/Multitask/pytorch-CycleGAN-and-pix2pix/')
from voxelmorph import voxelmorph as vxm


class pix2pixHDMultitaskModel(pix2pixHDModel):

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
            parser.add_argument('--lambda_Seg', type=float, default=1, help='weight for the segmentation loss')
            parser.add_argument('--lr_Seg', type=float, default=0.0001, help='learning rate for the reg. network opt.')
            parser.add_argument('--lambda_Def', type=float, default=1.0, help='weight for the segmentation loss')
            parser.add_argument('--lr_Def', type=float, default=0.0001, help='learning rate for the reg. network opt.')

            parser.add_argument('--vxm_iteration_steps', type=int, default=5,
                                help='number of steps to train the registration network for each simulated US')
            parser.add_argument('--similarity', type=str, default='NCC', choices=['NCC', 'MIND'],
                                help='type of the similarity used for training voxelmorph')

            # loss hyperparameters
            parser.add_argument('--image-loss', default='mse',
                                help='image reconstruction loss - can be mse or ncc (default: mse)')
        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT


        return parser

    def init_loss_filter(self, use_gan_feat_loss):
        flags = (True, use_gan_feat_loss, True, True)

        def loss_filter(g_gan, g_gan_feat, d_real, d_fake):
            return [l for (l, f) in zip((g_gan, g_gan_feat, d_real, d_fake), flags) if f]

        return loss_filter

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


        # HD
        self.input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc
        # Generator network
        self.netG_input_nc = 36
        if self.isTrain:
            self.use_sigmoid = opt.no_lsgan
            self.netD_input_nc = self.netG_input_nc + 1
            if not opt.no_instance:
                self.netD_input_nc += 1

        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features

        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        self.set_networks(opt)




        if self.isTrain:
            self.criterionRigidReg = networks.RegistrationLoss()
            self.optimizer_RigidReg = torch.optim.Adam(self.netRigidReg.parameters(), lr=opt.lr_Reg,
                                                       betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_RigidReg)

            self.criterionDefReg = networks.NCCLoss()
            self.criterionDefRgl = networks.GradLoss('l2', loss_mult=opt.int_downsize)
            self.optimizer_DefReg = torch.optim.Adam(self.netDefReg.parameters(), lr=opt.lr_Def)
            self.optimizers.append(self.optimizer_DefReg)

            self.criterionSeg = networks.DiceLoss()
          #  self.criterionSeg = losses.dice()
            self.optimizer_Seg = torch.optim.Adam(self.netSeg.parameters(), lr=opt.lr_Seg, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_Seg)

            # pix2pixHD optimizers
            # optimizer G
            params = list(self.netG.parameters())
            if self.gen_features:
                params += list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # pix2pixHD
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss)
            self.criterionGAN = networks3d.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()

            # pix2pix HD
            if not opt.no_instance:
                self.netG_input_nc += 1
            if self.use_features:
                self.netG_input_nc += opt.feat_num


        self.first_phase_coeff = 0

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):
        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
        else:
            # create one-hot vector for label map
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3],size[4])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()

        # get edges from instance map
        if not self.opt.no_instance:
            inst_map = inst_map.data.cuda()
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)
        input_label = input_label

        # real images for training
        if real_image is not None:
            real_image = real_image

        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                feat_map = Variable(feat_map.data.cuda())
            if self.opt.label_feat:
                inst_map = label_map.cuda()

        return input_label, inst_map, real_image, feat_map


    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)


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
        if name == 'E':
            return self.netE

    def set_networks(self, opt):
        # specify the models you want to save to the disk. The training/test scripts will call
        # <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'E','D', 'RigidReg', 'DefReg', 'Seg']
        else:  # during test time, only load G
            self.model_names = ['G', 'E','RigidReg', 'DefReg', 'Seg']
        self.loss_functions = ['compute_D_loss', 'backward_G', 'backward_RigidReg', 'backward_DefReg_Seg']


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

        self.netSeg = networks3d.define_G(opt.input_nc, opt.num_classes, 64,
                                          opt.netSeg, opt.norm, use_dropout=not opt.no_dropout,
                                          gpu_ids=self.gpu_ids, is_seg_net=True)
        self.transformer_label = networks3d.init_net(vxm.layers.SpatialTransformer(size=opt.inshape, mode='nearest'),
                                                     gpu_ids=self.gpu_ids)
        self.transformer_intensity = networks3d.init_net(vxm.layers.SpatialTransformer(size=opt.inshape),
                                                         gpu_ids=self.gpu_ids)
        self.resizer = networks3d.init_net(vxm.layers.ResizeTransform(vel_resize=0.5, ndims=3), gpu_ids=self.gpu_ids)

        # pix2pixHD

        self.netG = networks3d.define_G(input_nc=self.netG_input_nc, output_nc=opt.output_nc, ngf=opt.ngf,
                                        netG=opt.netG,
                                        n_downsample_global=opt.n_downsample_global,
                                        n_blocks_global=opt.n_blocks_global, n_local_enhancers=opt.n_local_enhancers,
                                        n_blocks_local=opt.n_blocks_local, norm=opt.norm, gpu_ids=self.gpu_ids)

        self.netD = networks3d.define_D_HD(self.netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, self.use_sigmoid,
                                           opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
        self.netE = networks3d.define_G(input_nc=opt.output_nc, output_nc=opt.feat_num, ngf=opt.nef, netG='encoder',
                                        n_downsample_global=opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)

    def get_current_landmark_distances(self):
        return self.loss_landmarks_beginning , self.loss_landmarks_rigid, self.loss_landmarks_def

    def get_current_dice(self):
        return self.loss_warped_dice,self.loss_moving_dice, self.loss_diff_dice
    def set_visdom_names(self):
        # specify the training losses you want to print out. The training/test scripts will call
        # <BaseModel.get_current_losses>
        # Names so we can breakout loss
        self.loss_names = ['G_GAN', 'G_GAN_Feat', 'D_real', 'D_fake']
        self.loss_names += [ 'G','RigidReg_fake', 'RigidReg_real']
      #  self.loss_names += [ 'landmarks_def','landmarks_rigid','landmarks_def_diff','landmarks_rigid_diff']
        self.losses_pix2pix = ['G_GAN', 'G_GAN_Feat', 'D_real', 'D_fake']
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

        self.patient = input['Patient']
        self.landmarks_A = input['A_landmark'].to(self.device).unsqueeze(dim=0)
        self.landmarks_B = input['B_landmark'].to(self.device).unsqueeze(dim=0)

        affine, self.gt_vector = affine_transform.create_random_affine(self.real_B.shape[0],
                                                                       self.real_B.shape[-3:],
                                                                       self.real_B.dtype,
                                                                       device=self.real_B.device)

        self.deformed_B = affine_transform.transform_image(self.real_B, affine, self.real_B.device)
        self.deformed_LM_B = affine_transform.transform_image(self.landmarks_B, affine, self.landmarks_B.device)

        if self.opt.show_volumes:
            affine_transform.show_volumes([self.real_B, self.deformed_B])

        self.mask_A = input['A_mask'].to(self.device).type(self.real_A.dtype)
        if input['B_mask_available'][0]:  # TODO in this way it only works with batch size 1
            self.mask_B = input['B_mask'].to(self.device).type(self.real_A.dtype)
            self.deformed_mask_B = affine_transform.transform_image(self.mask_B, affine, self.mask_B.device)
        else:
            self.mask_B = None

        self.init_loss_tensors()



    def init_loss_tensors(self):

        self.loss_DefReg_real = torch.tensor([0.0])
        self.loss_DefReg = torch.tensor([0.0])
        self.loss_DefReg_fake = torch.tensor([0.0])

        self.loss_DefReg_fake = torch.tensor([0.0])
        self.loss_DefRgl = torch.tensor([0.0])
        self.loss_Seg_real = torch.tensor([0.0])
        self.loss_Seg_fake = torch.tensor([0.0])
        self.loss_Seg = torch.tensor([0.0])

        self.loss_RigidReg_fake = torch.tensor([0.0])
        self.loss_RigidReg_real = torch.tensor([0.0])
        self.loss_RigidReg = torch.tensor([0.0])

        self.loss_D = torch.tensor([0.0])

        self.loss_G_GAN =  torch.tensor([0.0])
        self.loss_G_GAN_Feat =  torch.tensor([0.0])
        self.loss_D_real =  torch.tensor([0.0])
        self.loss_D_fake =  torch.tensor([0.0])

        self.netG_input_nc = torch.tensor([0.0])

        #self.loss_landmarks_diff = None
        #self.loss_landmarks_deformed = None
        #self.loss_landmarks = None
        #self.loss_warped_dice = None
        #self.loss_beginning_dice = None





    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.opt.data_type==16:
            return edge.half()
        else:
            return edge.float()


    def forward(self,):
        super().forward()

        self.reg_A_params = self.netRigidReg(torch.cat([self.fake_B, self.deformed_B], dim=1))
        self.reg_B_params = self.netRigidReg(torch.cat([self.real_B, self.deformed_B], dim=1))

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
        self.Landmark_A_dvf = self.transformer_label(self.landmarks_A, self.dvf.detach())


    def backward_DefReg_Seg(self):
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
        ## ADDED: warped loss

        #if self.loss_warped_dice:
        #    self.loss_DefReg += self.loss_warped_dice.data[0][0]
        self.loss_DefReg *= (1 - self.first_phase_coeff)
        seg_B = self.netSeg(self.idt_B.detach() if self.opt.reg_idt_B else self.real_B)
        dvf_resized = self.resizer(dvf)
        mask_A_deformed = self.transformer_label(self.mask_A, dvf_resized)
        self.loss_Seg_real = self.criterionSeg(seg_B, mask_A_deformed)
      #  self.loss_Seg_real = self.compute_dice(seg_B, mask_A_deformed)

        seg_fake_B = self.netSeg(self.fake_B.detach())
        self.loss_Seg_fake = self.criterionSeg(seg_fake_B, self.mask_A)
       # self.loss_Seg_fake = self.compute_dice(seg_fake_B, self.mask_A)

        self.loss_Seg = (self.loss_Seg_real + self.loss_Seg_fake) * (1 - self.first_phase_coeff)
        if torch.is_grad_enabled():
            (self.loss_DefReg * self.opt.lambda_Def + self.loss_Seg * self.opt.lambda_Seg).backward()




    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G = 0.0
        ########   pix2pix HD    ########
        # Fake Detection and Loss

        pred_fake_pool = self.discriminate(self.input_cat, self.fake_B, use_pool=True)
        self.loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = self.discriminate(self.input_cat, self.real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        pred_fake = self.netD.forward(torch.cat((self.input_cat, self.fake_B), dim=1))
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # GAN feature matching loss
        self.loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    self.loss_G_GAN_Feat += D_weights * feat_weights * \
                                            self.criterionFeat(pred_fake[i][j],
                                                               pred_real[i][j].detach()) * self.opt.lambda_feat
        losses = self.loss_filter(self.loss_G_GAN, self.loss_G_GAN_Feat, self.loss_D_real, self.loss_D_fake)
        self.losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        self.loss_dict = dict(zip(self.losses_pix2pix, self.losses))

        self.loss_pix2pix = self.loss_dict['G_GAN'] + self.loss_dict.get('G_GAN_Feat', 0)

        ########   END pix2pix HD    ########

        # Third, rigid registration
        loss_G_Reg = self.criterionRigidReg(self.reg_A_params, self.gt_vector) * self.opt.lambda_Reg

        # fourth, Def registration:
        if self.opt.bidir:
            loss_DefReg_fake = self.criterionDefReg(self.deformed_fake_B, self.real_B)
        else:
            loss_DefReg_fake = 0

        # fifth, segmentation
      #  loss_G_Seg = self.criterionSeg(self.seg_fake_B, self.mask_A) * self.opt.lambda_Seg
        loss_G_Seg = self.compute_dice(self.seg_fake_B, self.mask_A) * self.opt.lambda_Seg


        # combine loss and calculate gradients
        self.loss_G = self.loss_pix2pix * self.first_phase_coeff + \
                      loss_DefReg_fake * (1 - self.first_phase_coeff) + \
                      loss_G_Seg * (1 - self.first_phase_coeff)


        if self.opt.use_rigid_branch:
            self.loss_G += loss_G_Reg * (1 - self.first_phase_coeff)

        ############### Backward Pass ####################
        # update generator weight

        self.compute_gt_dice()
        self.compute_landmark_loss()
        if torch.is_grad_enabled():
            self.loss_G.backward()



    def backward_D(self):
        self.loss_D.backward()
    #    print(f"{torch.cuda.memory_allocated()} backward D")


#
    def backward_RigidReg(self):
        """
        Calculate Segmentation loss to update the segmentation networks
        Returns
        -------
        """
        reg_A_params = self.netRigidReg(torch.cat([self.fake_B.detach(), self.deformed_B], dim=1))
        self.loss_RigidReg_fake = self.criterionRigidReg(reg_A_params,
                                                         self.gt_vector) * self.opt.lambda_Reg  # to be of the same order as loss_G_Seg
        self.loss_RigidReg_real = self.criterionRigidReg(self.reg_B_params,
                                                         self.gt_vector) * self.opt.lambda_Reg  # to be of the same order as loss_G_Seg

        self.loss_RigidReg = self.loss_RigidReg_real + self.loss_RigidReg_fake * (1 - self.first_phase_coeff)
        if torch.is_grad_enabled():
            self.loss_RigidReg.backward()



    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        pred_fake_pool = self.discriminate(self.input_cat, self.fake_B, use_pool=True)
        self.loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = self.discriminate(self.input_cat, self.real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A), rigid registration params, DVF and segmentation mask
        self.set_requires_grad(self.netRigidReg, True)
        self.set_requires_grad(self.netDefReg, False)
        self.set_requires_grad(self.netSeg, False)
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netG, False)


        # update G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # update G's weights

        # Update D
    #    self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.compute_D_loss()
        if torch.is_grad_enabled():
            self.loss_D.backward()
        self.optimizer_D.step()  # update D's weights
        self.set_requires_grad(self.netD, False)  # enable backprop for D

        # update G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate gradients for G
     #   self.optimizer_G.step()  # update G's weights

        # update rigid registration network
        if self.opt.use_rigid_branch:
            self.set_requires_grad(self.netRigidReg, True)
            self.optimizer_RigidReg.zero_grad()
            self.backward_RigidReg()
            self.optimizer_RigidReg.step()
        self.set_requires_grad(self.netRigidReg, False)

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
            one_hot_fixed = torch.zeros(shape, device=self.device)
            one_hot_deformed = torch.zeros(shape, device=self.device)
            one_hot_moving = torch.zeros(shape, device=self.device)
            for i in range(n):
                one_hot_fixed[:, i, self.mask_B[0, 0, ...] == i] = 1
                one_hot_deformed[:, i, self.mask_A_deformed[0, 0, ...] == i] = 1
                one_hot_moving[:, i, self.mask_A[0, 0, ...] == i] = 1

            self.loss_warped_dice = compute_meandice(one_hot_deformed, one_hot_fixed, include_background=False)
            self.loss_moving_dice = compute_meandice(one_hot_deformed, one_hot_moving, include_background=False)
            loss_beginning_dice = compute_meandice(one_hot_moving, one_hot_fixed, include_background=False)
            self.loss_diff_dice = loss_beginning_dice - self.loss_warped_dice

        else:
            self.loss_warped_dice = None
            self.loss_moving_dice = None
            self.loss_diff_dice = None




    def compute_dice(self,y_pred, y):
        """
        calculate the dice score between the y_pred and y
        -------

        """
        shape = list(y_pred.shape)
        n = self.opt.num_classes  # number of classes
        shape[1] = n
        one_hot_y_pred = torch.zeros(shape, device=self.device)
        one_hot_y = torch.zeros(shape, device=self.device)
        for i in range(n):
            one_hot_y_pred[:, i, y_pred[0, 0, ...] == i] = 1
            one_hot_y[:, i, y[0, 0, ...] == i] = 1

        return compute_meandice(one_hot_y_pred, one_hot_y, include_background=False)

    def get_transformed_images(self) -> Tuple[torch.Tensor, torch.Tensor]:
        reg_A = affine_transform.transform_image(self.real_B,
                                                 affine_transform.tensor_vector_to_matrix(self.reg_A_params.detach()),
                                                 device=self.real_B.device)

        reg_B = affine_transform.transform_image(self.real_B,
                                                 affine_transform.tensor_vector_to_matrix(self.reg_B_params.detach()),
                                                 device=self.real_B.device)
       # print(f"{torch.cuda.memory_allocated()} transformed images")

        return reg_A, reg_B

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




    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        super().compute_visuals()

        reg_A, reg_B = self.get_transformed_images()

        self.diff_A = reg_A - self.deformed_B
        self.diff_B = reg_B - self.deformed_B
        self.diff_orig = self.real_B - self.deformed_B
        self.seg_fake_B = torch.argmax(self.seg_fake_B, dim=1, keepdim=True)
        self.seg_B = torch.argmax(self.seg_B, dim=1, keepdim=True)

        n_c = self.real_A.shape[2]

        self.reg_A_center_sag = reg_A[:, :, int(n_c / 2), ...]
        self.diff_A_center_sag = self.diff_A[:, :, int(n_c / 2), ...]
        self.reg_B_center_sag = reg_B[:, :, int(n_c / 2), ...]
        self.diff_B_center_sag = self.diff_B[:, :, int(n_c / 2), ...]
        self.diff_orig_center_sag = self.diff_orig[:, :, int(n_c / 2), ...]
        self.deformed_center_sag = self.deformed_B[:, :, int(n_c / 2), ...]

        n_c = self.real_A.shape[3]
        self.reg_A_center_cor = reg_A[:, :, :, int(n_c / 2), ...]
        self.diff_A_center_cor = self.diff_A[:, :, :, int(n_c / 2), ...]
        self.reg_B_center_cor = reg_B[:, :, :, int(n_c / 2), ...]
        self.diff_B_center_cor = self.diff_B[:, :, :, int(n_c / 2), ...]
        self.diff_orig_center_cor = self.diff_orig[:, :, :, int(n_c / 2), ...]
        self.deformed_center_cor = self.deformed_B[:, :, :, int(n_c / 2), ...]

        n_c = self.real_A.shape[4]
        self.reg_A_center_axi = reg_A[..., int(n_c / 2)]
        self.diff_A_center_axi = self.diff_A[..., int(n_c / 2)]
        self.reg_B_center_axi = reg_B[..., int(n_c / 2)]
        self.diff_B_center_axi = self.diff_B[..., int(n_c / 2)]
        self.diff_orig_center_axi = self.diff_orig[..., int(n_c / 2)]
        self.deformed_center_axi = self.deformed_B[..., int(n_c / 2)]

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


    def log_tensorboard(self,status, writer: SummaryWriter, losses: OrderedDict = None, global_step: int = 0,
                        save_gif=True, use_image_name=False):
        super().log_tensorboard(writer=writer, losses=losses, global_step=global_step,
                                save_gif=save_gif, use_image_name=use_image_name)

        self.add_rigid_figures(global_step=global_step, writer=writer, use_image_name=use_image_name)

        self.add_segmentation_figures(global_step=global_step, writer=writer, use_image_name=use_image_name)

        self.add_deformable_figures(global_step=global_step, writer=writer, use_image_name=use_image_name)

        self.add_landmark_losses(status=status,global_step=global_step, writer=writer)




    def add_deformable_figures(self,global_step, writer, use_image_name=False):
        n = 8 if self.mask_B is None else 10
        axs, fig = vxm.torch.utils.init_figure(3, n)
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
        if self.mask_B is not None:
            overlay = self.mask_B.repeat(1, 3, 1, 1, 1)
            overlay[:, 0:1, ...] = self.mask_A.detach()
            overlay[:, 2, ...] = 0
            vxm.torch.utils.fill_subplots(overlay.cpu(), axs=axs[8, :], img_name='mask moving on US', cmap=None)

            overlay = self.mask_B.repeat(1, 3, 1, 1, 1)
            overlay[:, 0:1, ...] = self.mask_A_deformed.detach()
            overlay[:, 2, ...] = 0
            vxm.torch.utils.fill_subplots(overlay.cpu(), axs=axs[9, :], img_name='mask warped on US', cmap=None)
#
        if use_image_name:
            tag =  f'{self.patient}/Deformable'
        else:
            tag =  'Deformable'
        writer.add_figure(tag=tag, figure=fig, global_step=global_step)

    def add_rigid_figures(self, global_step, writer, use_image_name=False):
        axs, fig = vxm.torch.utils.init_figure(3, 4)
        vxm.torch.utils.set_axs_attribute(axs)
        vxm.torch.utils.fill_subplots(self.diff_A.cpu(), axs=axs[0, :], img_name='Diff A')
        vxm.torch.utils.fill_subplots(self.diff_B.cpu(), axs=axs[1, :], img_name='Diff B')
        vxm.torch.utils.fill_subplots(self.diff_orig.cpu(), axs=axs[2, :], img_name='Diff orig')
        vxm.torch.utils.fill_subplots(self.deformed_B.detach().cpu(), axs=axs[3, :], img_name='Transformed')

        if use_image_name:
            tag = f'{self.patient}/Rigid'
        else:
            tag = 'Rigid'
        writer.add_figure(tag=tag, figure=fig, global_step=global_step)

    def add_segmentation_figures(self, global_step, writer, use_image_name=False):
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
        if use_image_name:
            tag =  f'{self.patient}/Segmentation'
        else:
            tag =  'Segmentation'
        writer.add_figure(tag=tag, figure=fig, global_step=global_step)

    def add_landmark_losses(self, status,global_step, writer):

        if self.loss_landmarks_rigid_diff is not None:
            writer.add_scalar(status+'landmarks/difference_rigid', scalar_value=self.loss_landmarks_rigid_diff,
                              global_step=global_step)
        #      else:
        #         writer.add_scalar('landmarks/difference_rigid', scalar_value=torch.tensor([0.0]), global_step=global_step)

        if self.loss_landmarks_rigid is not None:
            writer.add_scalar(status+'landmarks/rigid', scalar_value=self.loss_landmarks_rigid, global_step=global_step)
        #     else:
        #        writer.add_scalar('landmarks/rigid', scalar_value=torch.tensor([0.0]), global_step=global_step)

        if self.loss_landmarks_def is not None:
            writer.add_scalar(status+'landmarks/def', scalar_value=self.loss_landmarks_def, global_step=global_step)
        # else:
        #    writer.add_scalar('landmarks/def', scalar_value=torch.tensor([0.0]), global_step=global_step)

        if self.loss_landmarks_def_diff is not None:
            writer.add_scalar(status+'landmarks/difference_def', scalar_value=self.loss_landmarks_def_diff,
                              global_step=global_step)
        # else:
        #    writer.add_scalar('landmarks/difference_def', scalar_value=torch.tensor([0.0]), global_step=global_step)

        if self.loss_diff_dice is not None:
            writer.add_scalar(status+'DICE/difference', scalar_value=self.loss_diff_dice, global_step=global_step)
        #   else:
        #      writer.add_scalar('DICE/difference', scalar_value=torch.tensor([0.0]), global_step=global_step)

        if self.loss_warped_dice is not None:
            writer.add_scalar(status+'DICE/deformed', scalar_value=self.loss_warped_dice, global_step=global_step)

        if self.loss_moving_dice is not None:
            writer.add_scalar(status+'DICE/moving', scalar_value=self.loss_moving_dice, global_step=global_step)
    # else:
    # else:
    #    writer.add_scalar('DICE/deformed', scalar_value=torch.tensor([0.0]), global_step=global_step)

                                                   #

