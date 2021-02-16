from collections import OrderedDict
import os

import numpy as np
import torch
from monai.visualize import img2tensorboard
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from .base_model import BaseModel
from . import networks
from . import networks3d
from .patchnce import PatchNCELoss
import util.util as util

os.environ['VXM_BACKEND'] = 'pytorch'
from voxelmorph import voxelmorph as vxm


#
class DRIT3dModel(BaseModel):
    """ This class implements DRIT described in the paper
    DRIT++: Diverse image to image translation using disentangled representations

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # common
        parser.add_argument('--pad_type', type=str, default='replicate', help='padding type')
        parser.add_argument('--activ', type=str, default='lrelu', help='activation layer type')
        # content
        parser.add_argument('--output_nc_cont', type=int, default=64, help='output ch of content encoder')
        parser.add_argument('--ndf_cont', type=int, default=16, help='number of filters of content encoder')
        parser.add_argument('--n_layers_cont', type=int, default=3, help='number of layers of content encoder')
        parser.add_argument('--norm_cont', type=str, default='none', help='norm layer of content encoder')
        parser.add_argument('--lr_content', type=float, default=0.0001, help='learning rate of the content encoder')
        # attribute
        parser.add_argument('--output_nc_attr', type=int, default=16, help='output ch of attribute encoder')
        parser.add_argument('--ndf_attr', type=int, default=16, help='number of filters of attribute encoder')
        parser.add_argument('--n_layers_attr', type=int, default=3, help='number of layers of attribute encoder')
        parser.add_argument('--norm_attr', type=str, default='in', help='norm layer of attribute encoder')
        # generator
        parser.add_argument('--mlp_nc', type=int, default=16, help='number of filters of mlp layer in generator')
        parser.add_argument('--norm_gen', type=str, default='in', help='norm layer of generator')
        # discriminator domain
        parser.add_argument('--n_layers_d', type=int, default=4, help='number of layers of the discriminator')
        parser.add_argument('--n_scale', type=int, default=3, help='number of scales of the discriminator')
        parser.add_argument('--norm_d', type=str, default='in', help='norm layer of discriminator')
        # discriminator content
        parser.add_argument('--ndcf', type=int, default=16, help='number of filters of disc content')
        parser.add_argument('--n_layers_d_cont', type=int, default=1, help='number of layers of disc content')
        parser.add_argument('--norm_d_cont', type=str, default='in', help='norm layer of disc content')
        parser.add_argument('--netD_cont', type=str, default='n_layers', help='type of the disc content')
        # losses
        parser.add_argument('--recon_loss', type=str, default='l1', choices=['l1', 'lcc'], help='type of the cycle consistency loss')
        parser.add_argument('--use_ms', action='store_true', help='use mode seeking loss')
        parser.add_argument('--cont_d_loss', type=str, default='minmax', choices=['minmax', 'lsgan', 'wgan-gp'], help='type content disc loss')
        parser.add_argument('--d_loss', type=str, default='minmax', choices=['minmax', 'lsgan', 'wgan-gp'], help='type domain disc loss')
        # lambdas
        parser.add_argument('--lambda_cont_adv', type=float, default=1.0, help='coeff of content adv loss')
        parser.add_argument('--lambda_domain_adv', type=float, default=1.0, help='coeff of domain adv loss')
        parser.add_argument('--lambda_domain_adv_random', type=float, default=1.0, help='coeff of domain adv loss')
        parser.add_argument('--lambda_attr_reg', type=float, default=0.01, help='coeff of attribute regularization loss')
        parser.add_argument('--lambda_cont_reg', type=float, default=0.01, help='coeff of attribute regularization loss')
        parser.add_argument('--lambda_cc', type=float, default=1.0, help='coeff of cylce consistency loss')
        parser.add_argument('--lambda_recon', type=float, default=1.0, help='coeff of reconstruction loss')
        parser.add_argument('--lambda_latent', type=float, default=1.0, help='coeff of latent reconstruction loss')


        parser.set_defaults(pool_size=0, dataset_mode='volume')  # no image pooling

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        if len(self.gpu_ids) > 0:
            self.gpu = self.gpu_ids[0]
        else:
            self.gpu = 'cpu'
        self.random_shape = None

        self.loss_names = ['G_GAN_A', 'G_GAN_B', 'G_GAN_Acontent', 'G_GAN_Bcontent', 'kl_za_a',
                           'kl_za_b', 'kl_zc_a', 'kl_zc_b', 'G_CC_A', 'G_CC_B', 'G_CC_AA', 'G_CC_BB',
                           'cc_za_a', 'cc_za_b', 'cc_zc_a', 'cc_zc_b', 'G', 'disA', 'disB', 'dis_cont']

        self.visual_names = []

        if self.isTrain:
            self.model_names = ['Enc_c', 'Enc_attr_A', 'Enc_attr_B', 'Gen_A', 'Gen_B']
            self.model_names += ['Dis_A', 'Dis_B', 'Dis_cont']
        else:  # during test time, only load G
            self.model_names = ['Enc_c', 'Enc_atr_A', 'Enc_atr_B', 'Gen_A', 'Gen_B']

        # encoders
        # input_nc, output_nc, ndf, netE, n_layers=3, n_res=3, active='rely', norm='batch', pad_type='replicate',
        #              init_type='normal', init_gain=0.02, gpu_ids=[]
        self.netEnc_c = networks3d.define_E(opt.input_nc, opt.output_nc_cont, opt.ndf_cont, 'content', opt.n_layers_cont,
                                            opt.activ, opt.norm_cont, opt.pad_type, opt.init_type,
                                            opt.init_gain, gpu_ids=self.gpu_ids)
        self.netEnc_attr_A = networks3d.define_E(opt.input_nc, opt.output_nc_attr, opt.ndf_attr, 'attribute',
                                                 opt.n_layers_attr, opt.activ, opt.norm_attr, opt.pad_type,
                                                 opt.init_type, opt.init_gain, gpu_ids=self.gpu_ids)
        self.netEnc_attr_B = networks3d.define_E(opt.input_nc, opt.output_nc_attr, opt.ndf_attr, 'attribute',
                                                 opt.n_layers_attr, opt.activ, opt.norm_attr, opt.pad_type,
                                                 opt.init_type, opt.init_gain, gpu_ids=self.gpu_ids)

        # generator
        self.netGen_A = networks3d.define_G(opt.output_nc_cont, opt.output_nc, ngf=opt.ngf, netG=opt.netG, opt=opt)
        self.netGen_B = networks3d.define_G(opt.output_nc_cont, opt.output_nc, ngf=opt.ngf, netG=opt.netG, opt=opt)

        if self.isTrain:
            # opt should have n_scale_d if multiscale
            self.netDis_A = networks3d.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_d, opt.norm_d,
                                                opt.init_type, opt.init_gain, gpu_ids=self.gpu_ids, opt=opt)
            self.netDis_B = networks3d.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_d, opt.norm_d,
                                                opt.init_type, opt.init_gain, gpu_ids=self.gpu_ids, opt=opt)
            self.netDis_cont = networks3d.define_D(opt.output_nc_cont, opt.ndcf, opt.netD_cont, opt.n_layers_d_cont,
                                                   opt.norm_d_cont, opt.init_type, opt.init_gain, gpu_ids=self.gpu_ids)

            # opt should contain lr, lr_content, weight_decay, beta1, beta2
            self.create_optimizers(opt)

            # Setup the loss function for training
            self.criterion_cc = vxm.losses.LCC(s=opt.lcc_s,
                                               device='cuda') if opt.recon_loss == 'lcc' else torch.nn.L1Loss().cuda()
            self.l1 = torch.nn.L1Loss().cuda()
            self.latent_l1 = torch.nn.L1Loss().cuda()
            self.bce = torch.nn.BCEWithLogitsLoss().cuda()

    def create_optimizers(self, opt):
        lr = opt.lr
        lr_dcontent = opt.lr_content
        weight_decay = 0
        beta1 = opt.beta1
        beta2 = opt.beta2
        # optimizers
        enc_a_params = list(self.netEnc_attr_A.parameters()) + list(self.netEnc_attr_B.parameters())
        gen_params = list(self.netGen_A.parameters()) + list(self.netGen_B.parameters())
        self.disA_opt = torch.optim.Adam(self.netDis_A.parameters(), lr=lr, betas=(beta1, beta2),
                                         weight_decay=weight_decay)
        self.disB_opt = torch.optim.Adam(self.netDis_B.parameters(), lr=lr, betas=(beta1, beta2),
                                         weight_decay=weight_decay)
        self.disContent_opt = torch.optim.Adam(self.netDis_cont.parameters(), lr=lr_dcontent, betas=(beta1, beta2),
                                               weight_decay=weight_decay)
        self.enc_c_opt = torch.optim.Adam(self.netEnc_c.parameters(), lr=lr, betas=(beta1, beta2),
                                          weight_decay=weight_decay)
        self.enc_a_opt = torch.optim.Adam([p for p in enc_a_params if p.requires_grad], lr=lr, betas=(beta1, beta2),
                                          weight_decay=weight_decay)
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad], lr=lr, betas=(beta1, beta2),
                                        weight_decay=weight_decay)
        self.optimizers = [self.disA_opt, self.disB_opt, self.disContent_opt,
                           self.enc_c_opt, self.enc_a_opt, self.gen_opt]

    def data_dependent_initialize(self, data):
        pass

    def optimize_parameters(self):
        # forward
        self.forward()

        self.update_D()
        self.update_EG()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        batch size should be dividable by two!
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.patient = input['Patient']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        data_split = self.opt.batch_size // 2

        self.real_a_encoded, self.real_b_encoded = self.real_A[0:data_split], self.real_B[0:data_split]
        self.real_a_random, self.real_b_random = self.real_A[data_split:], self.real_B[data_split:]

        self.zc_a, self.zc_b = self.netEnc_c(self.real_a_encoded, self.real_b_encoded)
        self.za_a = self.netEnc_attr_A(self.real_a_encoded)
        self.za_b = self.netEnc_attr_B(self.real_b_encoded)

        if self.random_shape is None:
            self.random_shape = self.za_a.shape
        self.z_random = self.get_z_random()

        if self.opt.use_ms:
            self.z_random2 = self.get_z_random()
            input_content_forA = torch.cat((self.zc_b, self.zc_a, self.zc_b, self.zc_b), 0)
            input_content_forB = torch.cat((self.zc_a, self.zc_b, self.zc_a, self.zc_a), 0)
            input_attr_forA = torch.cat((self.za_a, self.za_a, self.z_random, self.z_random2), 0)
            input_attr_forB = torch.cat((self.za_b, self.za_b, self.z_random, self.z_random2), 0)
            output_fakeA = self.netGen_A(input_content_forA, input_attr_forA)
            output_fakeB = self.netGen_B(input_content_forB, input_attr_forB)
            self.fake_a_encoded, self.fake_aa_encoded, self.fake_a_random, self.fake_a_random2 = torch.split(
                output_fakeA, self.zc_a.size(0), dim=0)
            self.fake_b_encoded, self.fake_bb_encoded, self.fake_b_random, self.fake_b_random2 = torch.split(
                output_fakeB, self.zc_a.size(0), dim=0)

        else:
            # Forward translation
            input_content_forA = torch.cat((self.zc_b, self.zc_a, self.zc_b), 0)
            input_content_forB = torch.cat((self.zc_a, self.zc_b, self.zc_a), 0)
            input_attr_forA = torch.cat((self.za_a, self.za_a, self.z_random,), 0)
            input_attr_forB = torch.cat((self.za_b, self.za_b, self.z_random), 0)
            output_fakeA = self.netGen_A(input_content_forA, input_attr_forA)
            output_fakeB = self.netGen_B(input_content_forB, input_attr_forB)

            self.fake_a_encoded, self.fake_aa_encoded, self.fake_a_random = torch.split(output_fakeA,
                                                                                        self.zc_a.size(0), dim=0)
            self.fake_b_encoded, self.fake_bb_encoded, self.fake_b_random = torch.split(output_fakeB,
                                                                                        self.zc_a.size(0), dim=0)

        # Backward translation
        self.z_content_recon_b, self.z_content_recon_a = self.netEnc_c(self.fake_a_encoded,
                                                                       self.fake_b_encoded)

        self.z_attr_recon_a = self.netEnc_attr_A(self.fake_a_encoded)
        self.z_attr_recon_b = self.netEnc_attr_B(self.fake_b_encoded)

        self.fake_a_recon = self.netGen_A(self.z_content_recon_a, self.z_attr_recon_a)
        self.fake_b_recon = self.netGen_B(self.z_content_recon_b, self.z_attr_recon_b)

        self.z_random_recon_b = self.netEnc_attr_B(self.fake_b_random)
        self.z_random_recon_a = self.netEnc_attr_A(self.fake_a_random)

    def get_z_random(self):
        z = torch.randn(self.random_shape)
        return z.detach().to(self.gpu)

    def compute_D_cont_loss(self):
        self.disContent_opt.zero_grad()
        loss_netD = self.discriminator_update_criterion(netD=self.netDis_cont, real=self.zc_a.detach(),
                                                        fake=self.zc_b.detach(), d_type=self.opt.cont_d_loss)
        loss_netD.backward()
        self.loss_dis_cont = loss_netD.item()
        # nn.utils.clip_grad_norm_(self.dis_cont.parameters(), 5)
        self.disContent_opt.step()

    def update_D(self):
        # update disA
        self.disA_opt.zero_grad()
        loss_D1_A = self.backward_D(self.netDis_A, self.real_a_encoded, self.fake_a_encoded)
        loss_D2_A = self.backward_D(self.netDis_A, self.real_a_random, self.fake_a_random)
        disA2_loss = loss_D2_A.item()
        if self.opt.use_ms:
            loss_D2_A2 = self.backward_D(self.netDis_A, self.real_a_random, self.fake_a_random2)
            disA2_loss += loss_D2_A2.item()

        self.loss_disA = (loss_D1_A.item() + disA2_loss) / 2
        self.disA_opt.step()

        # update disB
        self.disB_opt.zero_grad()
        loss_D1_B = self.backward_D(self.netDis_B, self.real_b_encoded, self.fake_b_encoded)
        loss_D2_B = self.backward_D(self.netDis_B, self.real_b_random, self.fake_b_random)
        disB2_loss = loss_D2_B.item()
        if self.opt.use_ms:
            loss_D2_B2 = self.backward_D(self.netDis_B, self.real_b_random, self.fake_b_random2)
            disB2_loss += loss_D2_B2.item()
        self.loss_disB = (loss_D1_B.item() + disB2_loss) / 2
        self.disB_opt.step()

        # update disContent
        self.compute_D_cont_loss()

    def backward_D(self, netD, real, fake):
        loss_netD = self.discriminator_update_criterion(netD=netD, real=real, fake=fake, d_type=self.opt.d_loss)
        loss_netD.backward()
        return loss_netD

    def discriminator_update_criterion(self, netD=None, real=None, fake=None, d_type=None):
        if d_type is None:
            raise NotImplementedError
        if d_type == 'wgan':
            return self.wgan_dis_update(netD=netD, real=real, fake=fake)
        elif d_type == 'minmax':
            return self.minmax_dis_loss(netD, real, fake, lsgan_mode=False)
        elif d_type == 'lsgan':
            return self.minmax_dis_loss(netD, real, fake, lsgan_mode=True)
        else:
            raise NotImplementedError

    def wgan_dis_update(self, netD=None, real=None, fake=None, g_lambda=None):
        disc_real = netD(real)
        disc_fake = -netD(fake)

        disc_real = disc_real.mean()
        disc_fake = disc_fake.mean()

        gradient_penalty = self.wasserstein_gradient_penalty(netD=netD, real=real, fake=fake, g_lambda=g_lambda)

        disc_cost = disc_real - disc_fake + gradient_penalty

        return disc_cost

    # Reference: https://github.com/jalola/improved-wgan-pytorch/blob/e664f47807105828c37258a74ee1508b6d9b667a/training_utils.py#L75
    def wasserstein_gradient_penalty(self, netD=None, real=None, fake=None, g_lambda=None):
        alpha = torch.rand(real.size(0), 1, 1, 1, 1)
        alpha = alpha.expand(real.size())
        alpha = alpha.to(self.gpu)

        interpolates = alpha * real.detach() + ((1 - alpha) * fake.detach())

        interpolates = interpolates.to(self.gpu)
        interpolates.requires_grad_(True)

        disc_interpolates = netD(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(self.gpu),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * g_lambda
        return gradient_penalty

    def minmax_dis_loss(self, netD=None, real=None, fake=None, lsgan_mode=False):
        out_a = netD(fake.detach())
        out_b = netD(real)

        all0 = torch.zeros_like(out_a).cuda(self.gpu)
        all1 = torch.ones_like(out_b).cuda(self.gpu)
        if lsgan_mode:
            ad_fake_loss = F.mse_loss(out_a, all0)
            ad_true_loss = F.mse_loss(out_b, all1)
        else:
            ad_fake_loss = F.binary_cross_entropy_with_logits(out_a, all0 + 0.1)
            ad_true_loss = F.binary_cross_entropy_with_logits(out_b, all1 * 0.9)
        loss_D = ad_true_loss + ad_fake_loss

        return loss_D

    def minmax_dis_cont_loss(self, netD=None, img_a=None, img_b=None):
        pred_fake = netD(img_a)
        pred_real = netD(img_b)

        all0 = torch.zeros_like(pred_real).cuda(self.gpu).detach()
        all1 = torch.ones_like(pred_fake).cuda(self.gpu).detach()
        ad_fake_loss = self.bce(pred_fake, all0)
        ad_true_loss = self.bce(pred_real, all1)
        loss_D = ad_true_loss + ad_fake_loss

        return loss_D

    def update_EG(self):
        self.set_requires_grad([self.netDis_A, self.netDis_B, self.netDis_cont], False)
        self.enc_c_opt.zero_grad()
        self.enc_a_opt.zero_grad()
        self.gen_opt.zero_grad()
        self.backward_EG()

        # update G, Ec
        self.set_requires_grad([self.netEnc_attr_A, self.netEnc_attr_B], False)
        self.backward_G_alone()
        self.set_requires_grad([self.netEnc_attr_A, self.netEnc_attr_B], True)

        self.enc_a_opt.step()
        self.enc_c_opt.step()
        self.gen_opt.step()
        self.set_requires_grad([self.netDis_A, self.netDis_B, self.netDis_cont], True)

    def backward_EG(self):
        """
        For definition see equation (6)
        """
        # 1. Content adversarial loss
        loss_G_GAN_Acontent = self.backward_G_GAN_content(self.zc_a) * self.opt.lambda_cont_adv
        loss_G_GAN_Bcontent = self.backward_G_GAN_content(self.zc_b) * self.opt.lambda_cont_adv

        # 2. Domain adversarial loss
        loss_G_GAN_A = self.backward_G_GAN(self.fake_a_encoded, self.netDis_A) * self.opt.lambda_domain_adv
        loss_G_GAN_B = self.backward_G_GAN(self.fake_b_encoded, self.netDis_B) * self.opt.lambda_domain_adv

        # Regularization on the attribute representation (KL loss)
        loss_kl_za_a = self._l2_regularize(self.za_a) * self.opt.lambda_attr_reg
        loss_kl_za_b = self._l2_regularize(self.za_b) * self.opt.lambda_attr_reg

        # Regularization on the content representation (KL loss)
        loss_kl_zc_a = self._l2_regularize(self.zc_a) * self.opt.lambda_cont_reg
        loss_kl_zc_b = self._l2_regularize(self.zc_b) * self.opt.lambda_cont_reg

        # 3. CC loss
        loss_cc_za_a = self.l1(self.za_a, self.z_attr_recon_a)
        loss_cc_za_b = self.l1(self.za_b, self.z_attr_recon_b)
        loss_cc_zc_a = self.l1(self.zc_a, self.z_content_recon_a)
        loss_cc_zc_b = self.l1(self.zc_b, self.z_content_recon_b)
        loss_G_CC_A = self.criterion_cc(self.fake_a_recon, self.real_a_encoded) * self.opt.lambda_cc
        loss_G_CC_B = self.criterion_cc(self.fake_b_recon, self.real_b_encoded) * self.opt.lambda_cc

        # 4. Reconstruction loss
        loss_G_CC_AA = self.criterion_cc(self.fake_aa_encoded, self.real_a_encoded) * self.opt.lambda_recon
        loss_G_CC_BB = self.criterion_cc(self.fake_bb_encoded, self.real_b_encoded) * self.opt.lambda_recon

        loss_G = loss_G_GAN_A + loss_G_GAN_B + \
                 loss_G_CC_AA + loss_G_CC_BB + \
                 loss_G_CC_A + loss_G_CC_B + \
                 loss_kl_za_a + loss_kl_za_b + \
                 loss_G_GAN_Acontent + loss_G_GAN_Bcontent + \
                 loss_kl_zc_a + loss_kl_zc_b

        loss_G.backward(retain_graph=True)

        self.loss_G_GAN_A = loss_G_GAN_A.item()
        self.loss_G_GAN_B = loss_G_GAN_B.item()
        self.loss_G_GAN_Acontent = loss_G_GAN_Acontent.item()
        self.loss_G_GAN_Bcontent = loss_G_GAN_Bcontent.item()
        self.loss_kl_za_a = loss_kl_za_a.item()
        self.loss_kl_za_b = loss_kl_za_b.item()
        self.loss_kl_zc_a = loss_kl_zc_a.item()
        self.loss_kl_zc_b = loss_kl_zc_b.item()
        self.loss_G_CC_A = loss_G_CC_A.item()
        self.loss_G_CC_B = loss_G_CC_B.item()
        self.loss_G_CC_AA = loss_G_CC_AA.item()
        self.loss_G_CC_BB = loss_G_CC_BB.item()
        self.loss_cc_za_a = loss_cc_za_a.item()
        self.loss_cc_za_b = loss_cc_za_b.item()
        self.loss_cc_zc_a = loss_cc_zc_a.item()
        self.loss_cc_zc_b = loss_cc_zc_b.item()
        self.loss_G = loss_G.item()

    def content_gan_update_criterion(self, content):
        if self.opt.cont_d_loss is None:
            raise NotImplementedError
        if self.opt.cont_d_loss == 'wgan':
            return 0.5 * self.wgan_gen_update(content, netD=self.netDis_cont)
        elif self.opt.cont_d_loss == 'minmax':
            return self.minmax_content_gan(content, lsgan_mode=False)
        elif self.opt.cont_d_loss == 'lsgan':
            return self.minmax_content_gan(content, lsgan_mode=True)
        else:
            raise NotImplementedError

    def minmax_content_gan(self, content, lsgan_mode=False):
        outs = self.netDis_cont(content)
        all_half = 0.5 * torch.ones_like(outs)
        if lsgan_mode:
            ad_loss = F.mse_loss(outs, all_half)
        else:
            ad_loss = F.binary_cross_entropy_with_logits(outs, all_half)
        return ad_loss

    def backward_G_GAN_content(self, data):

        return self.content_gan_update_criterion(data)

    def backward_G_GAN(self, fake, netD=None):

        return self.generator_update_criterion(fake, netD)

    def generator_update_criterion(self, fake=None, netD=None):
        if self.opt.d_loss is None:
            raise NotImplementedError
        if self.opt.d_loss == 'wgan':
            return self.wgan_gen_update(fake, netD=netD)
        elif self.opt.d_loss == 'minmax':
            return self.minmax_gan_update(fake, netD, lsgan_mode=False)
        elif self.opt.d_loss == 'lsgan':
            return self.minmax_gan_update(fake, netD, lsgan_mode=True)
        else:
            raise NotImplementedError

    def minmax_gan_update(self, fake, netD=None, lsgan_mode=False):
        outs_fake = netD(fake)
        all_ones = torch.ones_like(outs_fake).cuda(self.gpu)
        if lsgan_mode:
            loss_G = F.mse_loss(outs_fake, all_ones)
        else:
            loss_G = F.binary_cross_entropy_with_logits(outs_fake, 0.9 * all_ones)

        return loss_G

    def wgan_gen_update(self, fake, netD=None):
        gen_cost = netD(fake)
        gen_cost = -gen_cost
        gen_cost = gen_cost.mean()

        return gen_cost

    def backward_G_alone(self):

        loss_G_GAN2_A = self.backward_G_GAN(self.fake_a_random,
                                            self.netDis_A) * self.opt.lambda_domain_adv_random
        loss_G_GAN2_B = self.backward_G_GAN(self.fake_b_random,
                                            self.netDis_B) * self.opt.lambda_domain_adv_random

        if self.opt.use_ms:
            loss_G_GAN2_A2 = self.backward_G_GAN(self.fake_a_random2, self.netDis_A) * self.opt.lambda_domain_adv_random
            loss_G_GAN2_B2 = self.backward_G_GAN(self.fake_b_random2, self.netDis_B) * self.opt.lambda_domain_adv_random
            lz_AB = torch.mean(torch.abs(self.fake_b_random2 - self.fake_b_random)) / torch.mean(
                torch.abs(self.z_random2 - self.z_random))
            lz_BA = torch.mean(torch.abs(self.fake_a_random2 - self.fake_a_random)) / torch.mean(
                torch.abs(self.z_random2 - self.z_random))
            eps = 1 * 1e-5
            loss_lz_AB = 1 / (lz_AB + eps)
            loss_lz_BA = 1 / (lz_BA + eps)

        loss_z_L1_b = torch.mean(torch.abs(self.z_random_recon_b -
                                           self.z_random)) * self.opt.lambda_latent

        loss_z_L1_a = torch.mean(torch.abs(self.z_random_recon_a -
                                           self.z_random)) * self.opt.lambda_latent


        loss_z_L1 = loss_z_L1_b + loss_z_L1_a + loss_G_GAN2_A + loss_G_GAN2_B

        if self.opt.use_ms:
            loss_z_L1 += (loss_G_GAN2_A2 + loss_G_GAN2_B2)
            loss_z_L1 += (loss_lz_AB + loss_lz_BA)

        loss_z_L1.backward()

        self.l1_recon_z_loss_a = loss_z_L1_a.item()
        self.l1_recon_z_loss_b = loss_z_L1_b.item()
        self.gan2_loss_a = loss_G_GAN2_A.item()
        self.gan2_loss_b = loss_G_GAN2_B.item()

    def _l2_regularize(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def log_tensorboard(self, writer: SummaryWriter, losses: OrderedDict = None, global_step: int = 0,
                        save_gif=True, use_image_name=False):
        volumes = OrderedDict(mr_real=self.real_a_encoded[0:1].cpu(),
                              mr_content=self.fake_a_random[0:1].cpu(),
                              mr_fake=self.fake_a_encoded[0:1].cpu(),
                              mr_reconstructed=self.fake_aa_encoded[0:1].cpu(),
                              us_real=self.real_b_encoded[0:1].cpu(),
                              us_content=self.fake_b_random[0:1].cpu(),
                              us_fake=self.fake_b_encoded[0:1].cpu(),
                              us_reconstructed=self.fake_bb_encoded[0:1].cpu())
        axs, fig = vxm.torch.utils.init_figure(3, len(volumes.keys()))
        vxm.torch.utils.set_axs_attribute(axs)
        for i, (key) in enumerate(volumes):
            vxm.torch.utils.fill_subplots(volumes[key], axs=axs[i, :], img_name=key)
        if use_image_name:
            tag = f'{self.patient}/images'
        else:
            tag = 'images'
        writer.add_figure(tag=tag, figure=fig, global_step=global_step)

        if losses is not None:
            for key in losses:
                if 'gan' in key.lower() and 'cont' not in key.lower():
                    writer.add_scalar(f'GAN/{key}', scalar_value=losses[key], global_step=global_step)
                elif 'dis' in key.lower() and 'cont' not in key:
                    writer.add_scalar(f'DIS/{key}', scalar_value=losses[key], global_step=global_step)
                elif 'cont' in key.lower():
                    writer.add_scalar(f'CONT/{key}', scalar_value=losses[key], global_step=global_step)
                elif 'kl' in key.lower():
                    writer.add_scalar(f'KL/{key}', scalar_value=losses[key], global_step=global_step)
                elif 'recon' in key.lower():
                    writer.add_scalar(f'RECON/{key}', scalar_value=losses[key], global_step=global_step)
                else:
                    writer.add_scalar(f'other/{key}', scalar_value=losses[key], global_step=global_step)

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass
