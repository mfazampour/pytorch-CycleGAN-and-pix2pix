import argparse
import os
from collections import OrderedDict
from typing import Tuple
import util.util as util
from util.image_pool import ImagePool
from util import affine_transform
from models.base_model import BaseModel
from torch.autograd import Variable
from monai.visualize import img2tensorboard

import torch
from torch.utils.tensorboard import SummaryWriter
import sys
from models import networks3d
from voxelmorph import voxelmorph as vxm

# from .cut3d_model import CUT3dModel

os.environ['VXM_BACKEND'] = 'pytorch'
# sys.path.append('/home/kixcodes/Documents/python/Multitask/pytorch-CycleGAN-and-pix2pix/')


class pix2pixHDModel(BaseModel):

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
        parser.add_argument('--show_volumes', type=bool, default=False, help='visualize transformed volumes w napari')
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
            parser.add_argument('--lambda_Def', type=float, default=1.0, help='weight for the segmentation loss')
            parser.add_argument('--lr_Def', type=float, default=0.0001, help='learning rate for the reg. network opt.')

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
        self.netG_input_nc = 35
        if self.isTrain:
            self.use_sigmoid = opt.no_lsgan
            self.netD_input_nc = self.netG_input_nc + 1
            if not opt.no_instance:
                self.netD_input_nc += 1

        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features

        self.set_networks(opt)
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if self.isTrain:

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

        self.first_phase_coeff = 1

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):
        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
        else:
            # create one-hot vector for label map
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3], size[4])
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
            self.model_names = ['G', 'E', 'D']
        else:  # during test time, only load G
            self.model_names = ['G', 'E']

        # We are using DenseNet for rigid registration -- actually DenseNet didn't provide any performance improvement
        # TODO change this to obelisk net
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

    def set_visdom_names(self):
        # specify the training losses you want to print out. The training/test scripts will call
        # <BaseModel.get_current_losses>
        # Names so we can breakout loss
        self.losses_pix2pix = ['G_GAN', 'G_GAN_Feat', 'D_real', 'D_fake']
        self.loss_names = ['G']
    def name(self):
        return 'Pix2PixHDModel'

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


        self.loss_G_GAN = torch.tensor([0.0])
        self.loss_G_GAN_Feat = torch.tensor([0.0])
        self.loss_D_real = torch.tensor([0.0])
        self.loss_D_fake = torch.tensor([0.0])

        self.netG_input_nc = torch.tensor([0.0])

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        if self.opt.data_type == 16:
            return edge.half()
        else:
            return edge.float()

    def forward(self, ):

        # pix2pixHD
        # Encode Inputs
        self.input_label, inst_map, real_image, feat_map = self.encode_input(label_map=self.mask_A, inst_map=None,
                                                                             real_image=self.real_B, feat_map=None)
        self.fake_B = self.netG.forward(self.input_label)


    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G = 0.0
        ########   pix2pix HD    ########
        # Fake Detection and Loss

        pred_fake_pool = self.discriminate(self.input_label, self.fake_B, use_pool=True)
        self.loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = self.discriminate(self.input_label, self.real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        pred_fake = self.netD.forward(torch.cat((self.input_label, self.fake_B), dim=1))
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

        self.loss_G = self.loss_pix2pix

        self.loss_G.backward()

    def backward_D(self):
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    #    print(f"{torch.cuda.memory_allocated()} backward D")

    #




    def optimize_parameters(self):
        self.forward()

        # update G
        # self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G

        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # update G's weights

        # Update D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def get_transformed_images(self) -> Tuple[torch.Tensor, torch.Tensor]:
        reg_A = affine_transform.transform_image(self.real_B,
                                                 affine_transform.tensor_vector_to_matrix(self.reg_A_params.detach()),
                                                 device=self.real_B.device)

        reg_B = affine_transform.transform_image(self.real_B,
                                                 affine_transform.tensor_vector_to_matrix(self.reg_B_params.detach()),
                                                 device=self.real_B.device)
        # print(f"{torch.cuda.memory_allocated()} transformed images")

        return reg_A, reg_B

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        super().compute_visuals()



    def update_learning_rate(self, epoch=0):
        super().update_learning_rate(epoch=epoch)


    def log_tensorboard(self, writer: SummaryWriter, losses: OrderedDict, global_step: int = 0):
        self.log_tensorboard_base(writer=writer, losses=losses, global_step=global_step)

    # print(f"{torch.cuda.memory_allocated()} log tensorboard")

    def log_tensorboard_base(self, writer: SummaryWriter, losses: OrderedDict, global_step: int):
        image = torch.add(torch.mul(self.real_A, 0.5), 0.5)
        image2 = torch.add(torch.mul(self.real_B, 0.5), 0.5)
        image3 = torch.add(torch.mul(self.fake_B, 0.5), 0.5)

        img2tensorboard.add_animated_gif(writer=writer, scale_factor=256, tag="GAN/Real A", max_out=85,
                                         image_tensor=image.squeeze(dim=0).cpu().detach().numpy(),
                                         global_step=global_step)
        img2tensorboard.add_animated_gif(writer=writer, scale_factor=256, tag="GAN/Real B", max_out=85,
                                         image_tensor=image2.squeeze(dim=0).cpu().detach().numpy(),
                                         global_step=global_step)
        img2tensorboard.add_animated_gif(writer=writer, scale_factor=256, tag="GAN/Fake B", max_out=85,
                                         image_tensor=image3.squeeze(dim=0).cpu().detach().numpy(),
                                         global_step=global_step)


        axs, fig = vxm.torch.utils.init_figure(3, 4)
        vxm.torch.utils.set_axs_attribute(axs)
        vxm.torch.utils.fill_subplots(self.real_A.cpu(), axs=axs[0, :], img_name='A')
        vxm.torch.utils.fill_subplots(self.fake_B.detach().cpu(), axs=axs[1, :], img_name='fake')
        vxm.torch.utils.fill_subplots(self.real_B.cpu(), axs=axs[2, :], img_name='B')
        writer.add_figure(tag='GAN', figure=fig, global_step=global_step)

        for key in losses:
            writer.add_scalar(f'losses/{key}', scalar_value=losses[key], global_step=global_step)
    #  print(f"{torch.cuda.memory_allocated()} log tensorboard base")
