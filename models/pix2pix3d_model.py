import numpy as np
import torch
from collections import OrderedDict
from .base_model import BaseModel
from . import networks3d
from voxelmorph import voxelmorph as vxm
from torch.utils.tensorboard import SummaryWriter
from monai.visualize import img2tensorboard
from util import affine_transform
from util import tensorboard

from . import networks


class Pix2Pix3dModel(BaseModel):

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
        parser.set_defaults(norm='batch', netG='unet_128', dataset_mode='volume')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
          #  parser.add_argument('--no_lsgan', type=bool, default=False)
            parser.add_argument('--lambda_MIND', type=float, default=10.0, help='weight for MIND loss')
            # parser.add_argument('--visualize_volume', type=bool, default=False)

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
        self.loss_names = ['G','G_GAN', 'G_L1', 'D_real', 'D_fake', 'MIND']
        # specify the images you want to save/display. The training/test scripts will call
        # <BaseModel.get_current_visuals>
        # the empty slice is added since the visualization would be 3 * 4
        self.visual_names = ['real_A_center_sag', 'fake_B_center_sag', 'real_B_center_sag']
        self.visual_names += ['real_A_center_cor', 'fake_B_center_cor', 'real_B_center_cor']
        self.visual_names += ['real_A_center_axi', 'fake_B_center_axi', 'real_B_center_axi']
        # specify the models you want to save to the disk. The training/test scripts will call
        # <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # load/define networks
        self.netG = networks3d.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                        opt.netG, opt.norm, use_dropout=not opt.no_dropout, gpu_ids=self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks3d.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                            opt.netD, opt.n_layers_D, norm=opt.norm,
                                            use_sigmoid=use_sigmoid, gpu_ids=self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionMIND = networks3d.MIND()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    # def name(self):
    #     return 'Pix2Pix3dModel'

    def set_input(self, input):

        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)

        self.patient = input['Patient']

        affine, self.gt_vector = affine_transform.create_random_affine(self.real_B.shape[0],
                                                                       self.real_B.shape[-3:],
                                                                       self.real_B.dtype,
                                                                       device=self.device)

        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5



    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # Third, MIND(G(A)) == MIND(A)
        self.loss_MIND = self.criterionMIND(self.fake_B, self.real_A) * self.opt.lambda_MIND
        # combine loss and calculate gradients
        self.loss_G_GAN = self.loss_G_GAN + self.loss_G_L1 + self.loss_MIND
        self.loss_G = self.loss_G_GAN
        if torch.is_grad_enabled():
            self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights


    def log_tensorboard(self, writer: SummaryWriter, losses: OrderedDict = None, global_step: int = 0,
                        save_gif=True, use_image_name=False, mode=''):
        image = torch.add(torch.mul(self.real_A, 0.5), 0.5)
        image2 = torch.add(torch.mul(self.real_B, 0.5), 0.5)
        image3 = torch.add(torch.mul(self.fake_B, 0.5), 0.5)

        if save_gif:
            img2tensorboard.add_animated_gif(writer=writer, scale_factor=256, tag="GAN/Real A", max_out=85,
                                             image_tensor=image.squeeze(dim=0).cpu().detach().numpy(),
                                             global_step=global_step)
            img2tensorboard.add_animated_gif(writer=writer, scale_factor=256, tag="GAN/Real B", max_out=85,
                                             image_tensor=image2.squeeze(dim=0).cpu().detach().numpy(),
                                             global_step=global_step)
            img2tensorboard.add_animated_gif(writer=writer, scale_factor=256, tag="GAN/Fake B", max_out=85,
                                             image_tensor=image3.squeeze(dim=0).cpu().detach().numpy(),
                                             global_step=global_step)


        axs, fig = tensorboard.init_figure(3, 3)
        tensorboard.set_axs_attribute(axs)
        tensorboard.fill_subplots(self.real_A.cpu(), axs=axs[0, :], img_name='A')
        tensorboard.fill_subplots(self.fake_B.detach().cpu(), axs=axs[1, :], img_name='fake')
        tensorboard.fill_subplots(self.real_B.cpu(), axs=axs[2, :], img_name='B')
        if use_image_name:
            tag = mode + f'{self.patient}/GAN'
        else:
            tag = mode + 'GAN'
        writer.add_figure(tag=tag, figure=fig, global_step=global_step)

        if losses is not None:
            for key in losses:
                writer.add_scalar(f'losses/{key}', scalar_value=losses[key], global_step=global_step)

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        n_c = self.real_A.shape[2]
        # average over channel to get the real and fake image
        self.real_A_center_sag = self.real_A[:, :, int(n_c / 2), ...]
        self.fake_B_center_sag = self.fake_B[:, :, int(n_c / 2), ...]
        self.real_B_center_sag = self.real_B[:, :, int(n_c / 2), ...]

        n_c = self.real_A.shape[3]
        self.real_A_center_cor = self.real_A[:, :, :, int(n_c / 2), ...]
        self.fake_B_center_cor = self.fake_B[:, :, :, int(n_c / 2), ...]
        self.real_B_center_cor = self.real_B[:, :, :, int(n_c / 2), ...]

        n_c = self.real_A.shape[4]
        self.real_A_center_axi = self.real_A[..., int(n_c / 2)]
        self.fake_B_center_axi = self.fake_B[..., int(n_c / 2)]
        self.real_B_center_axi = self.real_B[..., int(n_c / 2)]





