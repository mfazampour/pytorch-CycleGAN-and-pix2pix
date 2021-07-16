from collections import OrderedDict
import os

import numpy as np
import torch
from monai.visualize import img2tensorboard
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from util import tensorboard
from .base_model import BaseModel
from . import networks
from . import networks3d
import util.util as util

os.environ['VXM_BACKEND'] = 'pytorch'
from voxelmorph import voxelmorph as vxm


class DRIT3dOrigModel(BaseModel):
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
        parser.add_argument('--n_layers_cont', type=int, default=2, help='number of layers of content encoder')
        parser.add_argument('--norm_cont', type=str, default='none', help='norm layer of content encoder')
        parser.add_argument('--lr_content', type=float, default=0.0001, help='learning rate of the content encoder')
        parser.add_argument('--gan_mode_cont', type=str, default='vanilla', choices=['vanilla', 'lsgan', 'wgan', 'wgan-gp'], help='content gan adversarial loss')
        # attribute
        parser.add_argument('--output_nc_attr', type=int, default=8, help='output ch of attribute encoder')
        parser.add_argument('--ndf_attr', type=int, default=16, help='number of filters of attribute encoder')
        parser.add_argument('--n_layers_attr', type=int, default=2, help='number of layers of attribute encoder')
        parser.add_argument('--norm_attr', type=str, default='none', help='norm layer of attribute encoder')
        # generator
        parser.add_argument('--mlp_nc', type=int, default=128, help='number of filters of mlp layer in generator')
        parser.add_argument('--norm_gen', type=str, default='in', help='norm layer of generator')

        if is_train:
            # discriminator domain
            # parser.add_argument('--n_layers_d', type=int, default=4, help='number of layers of the discriminator')
            parser.add_argument('--n_scale', type=int, default=3, help='number of scales of the discriminator')
            parser.add_argument('--norm_d', type=str, default='in', help='norm layer of discriminator')

            # discriminator content
            parser.add_argument('--ndcf', type=int, default=64, help='number of filters of disc content')
            parser.add_argument('--n_layers_d_cont', type=int, default=2, help='number of layers of disc content')
            parser.add_argument('--norm_d_cont', type=str, default='in', help='norm layer of disc content')
            parser.add_argument('--netD_cont', type=str, default='dis_cont', help='type of the disc content')
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
            parser.add_argument('--lambda_recon', type=float, default=10.0, help='coeff of reconstruction loss')
            parser.add_argument('--lambda_latent', type=float, default=10.0, help='coeff of latent reconstruction loss')
            parser.add_argument('--lambda_gp', type=float, default=10.0, help='coeff of gradient penalization')
            parser.add_argument('--lambda_cont_gp', type=float, default=10.0, help='coeff of gradient penalization of cont. disc.')


        parser.set_defaults(pool_size=0, dataset_mode='volume', netD='dis_drit')  # no image pooling

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.random_shape = None

        self.loss_names = ['G_GAN_A', 'G_GAN_B', 'G_GAN_A_content', 'G_GAN_B_content', 'kl_za_a',
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
        self.netEnc_c = networks3d.define_E(opt.input_nc, opt.output_nc_cont, opt.ndf_cont, 'content_orig', opt.n_layers_cont,
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
            self.netDis_A = networks3d.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm_d,
                                                opt.init_type, opt.init_gain, gpu_ids=self.gpu_ids, opt=opt)
            self.netDis_B = networks3d.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm_d,
                                                opt.init_type, opt.init_gain, gpu_ids=self.gpu_ids, opt=opt)
            self.netDis_A2 = networks3d.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm_d,
                                                opt.init_type, opt.init_gain, gpu_ids=self.gpu_ids, opt=opt)
            self.netDis_B2 = networks3d.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm_d,
                                                opt.init_type, opt.init_gain, gpu_ids=self.gpu_ids, opt=opt)
            self.netDis_cont = networks3d.define_D(opt.output_nc_cont, opt.ndcf, opt.netD_cont, opt.n_layers_d_cont,
                                                   opt.norm_d_cont, opt.init_type, opt.init_gain, gpu_ids=self.gpu_ids)

            # opt should contain lr, lr_content, weight_decay, beta1, beta2
            self.create_optimizers(opt)

            # Setup the loss function for training
            self.criterionCC = vxm.losses.LCC(s=opt.lcc_s,
                                              device='cuda') if opt.recon_loss == 'lcc' else torch.nn.L1Loss().cuda()
            self.criterionL1 = torch.nn.L1Loss().cuda()
            self.bce = torch.nn.BCEWithLogitsLoss().cuda()

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionGAN_cont = networks.GANLoss(opt.gan_mode_cont).to(self.device)
            # self.criterionGAN_cont_G = networks.GANLoss(opt.gan_mode_cont, target_fake_label=0.5, target_real_label=0.5).to(self.device)

    def create_optimizers(self, opt):
        lr = opt.lr
        lr_dcontent = opt.lr_content
        weight_decay = 0
        beta1 = opt.beta1
        beta2 = opt.beta2
        # optimizers
        enc_attr_params = list(self.netEnc_attr_A.parameters()) + list(self.netEnc_attr_B.parameters())
        gen_params = list(self.netGen_A.parameters()) + list(self.netGen_B.parameters())

        self.disA_opt = torch.optim.Adam(self.netDis_A.parameters(), lr=lr * 4, betas=(beta1, beta2), weight_decay=weight_decay)
        self.disB_opt = torch.optim.Adam(self.netDis_B.parameters(), lr=lr * 4, betas=(beta1, beta2), weight_decay=weight_decay)
        self.disContent_opt = torch.optim.Adam(self.netDis_cont.parameters(), lr=lr_dcontent * 4, betas=(beta1, beta2), weight_decay=weight_decay)
        self.enc_c_opt = torch.optim.Adam(self.netEnc_c.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
        self.enc_a_opt = torch.optim.Adam([p for p in enc_attr_params if p.requires_grad], lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad], lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
        self.optimizers = [self.disA_opt, self.disB_opt, self.disContent_opt, self.enc_c_opt, self.enc_a_opt, self.gen_opt]

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
        self.modality_A = input['modality_A']
        self.modality_B = input['modality_B']

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
        return z.detach().to(self.device)

    def compute_D_cont_loss(self):
        self.disContent_opt.zero_grad()

        a_is_real = True
        if np.random.rand() > 0.9:  # switch real/fake randomly when training
            a_is_real = False

        real_loss = self.criterionGAN_cont(self.netDis_cont(self.zc_a.detach()), target_is_real=a_is_real)
        fake_loss = self.criterionGAN_cont(self.netDis_cont(self.zc_b.detach()), target_is_real=not a_is_real)
        loss_netD = (real_loss + fake_loss) * 0.5
        if self.opt.gan_mode_cont == 'wgan-gp':
            loss_gp, _ = networks.cal_gradient_penalty(self.netDis_cont, self.zc_a.detach(), self.zc_b.detach(),
                                                       self.device, lambda_gp=self.opt.lambda_cont_gp)
            loss_netD += loss_gp
        loss_netD.backward()
        self.loss_dis_cont = loss_netD.item()
        # nn.utils.clip_grad_norm_(self.dis_cont.parameters(), 5)
        self.disContent_opt.step()

    def update_D(self):
        # update disA
        self.disA_opt.zero_grad()
        loss_D1_A = self.backward_D(self.netDis_A, self.real_a_encoded, self.fake_a_encoded)
        loss_D2_A = self.backward_D(self.netDis_A2, self.real_a_random, self.fake_a_random)
        disA2_loss = loss_D2_A
        if self.opt.use_ms:
            loss_D2_A2 = self.backward_D(self.netDis_A2, self.real_a_random, self.fake_a_random2)
            disA2_loss += loss_D2_A2

        loss_disA = (loss_D1_A + disA2_loss) / 2
        loss_disA.backward()
        self.loss_disA = loss_disA.item()
        self.disA_opt.step()

        # update disB
        self.disB_opt.zero_grad()
        loss_D1_B = self.backward_D(self.netDis_B, self.real_b_encoded, self.fake_b_encoded)
        loss_D2_B = self.backward_D(self.netDis_B2, self.real_b_random, self.fake_b_random)
        disB2_loss = loss_D2_B
        if self.opt.use_ms:
            loss_D2_B2 = self.backward_D(self.netDis_B2, self.real_b_random, self.fake_b_random2)
            disB2_loss += loss_D2_B2
        loss_disB = (loss_D1_B + disB2_loss) / 2
        loss_disB.backward()
        self.loss_disB = loss_disB.item()
        self.disB_opt.step()

        # update disContent
        self.compute_D_cont_loss()

    def backward_D(self, netD: torch.nn.Module, real: torch.Tensor, fake: torch.Tensor):
        sigma = 0.0
        mask = real != -1
        out_real = netD(real * mask + torch.randn_like(real) * sigma)
        out_fake = netD(fake.detach() * mask + torch.randn_like(fake.detach()) * sigma)
        real_loss = self.criterionGAN(out_real, target_is_real=True)
        fake_loss = self.criterionGAN(out_fake, target_is_real=False)
        loss_netD = (real_loss + fake_loss) * 0.5
        if self.opt.gan_mode == 'wgan-gp':
            loss_gp, _ = networks.cal_gradient_penalty(netD, real.detach(), fake.detach(),
                                                       self.device, lambda_gp=self.opt.lambda_gp)
            loss_netD += loss_gp
        return loss_netD

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

        loss_G_GAN_A_content = self.criterionGAN_cont(self.zc_a, target_is_real=False) * self.opt.lambda_cont_adv
        loss_G_GAN_B_content = self.criterionGAN_cont(self.zc_b, target_is_real=True) * self.opt.lambda_cont_adv

        # 2. Domain adversarial loss
        loss_G_GAN_A = self.backward_G_GAN(self.fake_a_encoded, self.netDis_A, self.real_a_encoded != -1) * self.opt.lambda_domain_adv
        loss_G_GAN_B = self.backward_G_GAN(self.fake_b_encoded, self.netDis_B, self.real_b_encoded != -1) * self.opt.lambda_domain_adv

        # Regularization on the attribute representation (KL loss)
        loss_kl_za_a = self._l2_regularize(self.za_a) * self.opt.lambda_attr_reg
        loss_kl_za_b = self._l2_regularize(self.za_b) * self.opt.lambda_attr_reg

        # Regularization on the content representation (KL loss)
        loss_kl_zc_a = self._l2_regularize(self.zc_a) * self.opt.lambda_cont_reg
        loss_kl_zc_b = self._l2_regularize(self.zc_b) * self.opt.lambda_cont_reg

        # 3. CC loss
        loss_cc_za_a = self.criterionL1(self.za_a, self.z_attr_recon_a)
        loss_cc_za_b = self.criterionL1(self.za_b, self.z_attr_recon_b)
        loss_cc_zc_a = self.criterionL1(self.zc_a, self.z_content_recon_a)
        loss_cc_zc_b = self.criterionL1(self.zc_b, self.z_content_recon_b)
        loss_G_CC_A = self.criterionCC(self.fake_a_recon, self.real_a_encoded) * self.opt.lambda_cc
        loss_G_CC_B = self.criterionCC(self.fake_b_recon, self.real_b_encoded) * self.opt.lambda_cc

        # 4. Reconstruction loss
        loss_G_CC_AA = self.criterionCC(self.fake_aa_encoded, self.real_a_encoded) * self.opt.lambda_recon
        loss_G_CC_BB = self.criterionCC(self.fake_bb_encoded, self.real_b_encoded) * self.opt.lambda_recon

        loss_G = loss_G_GAN_A + loss_G_GAN_B + \
                 loss_G_CC_AA + loss_G_CC_BB + \
                 loss_G_CC_A + loss_G_CC_B + \
                 loss_kl_za_a + loss_kl_za_b + \
                 loss_G_GAN_A_content + loss_G_GAN_B_content + \
                 loss_kl_zc_a + loss_kl_zc_b

        loss_G.backward(retain_graph=True)

        self.loss_G_GAN_A = loss_G_GAN_A.item()
        self.loss_G_GAN_B = loss_G_GAN_B.item()
        self.loss_G_GAN_A_content = loss_G_GAN_A_content.item()
        self.loss_G_GAN_B_content = loss_G_GAN_B_content.item()
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

    def backward_G_GAN(self, fake: torch.Tensor, netD: torch.nn.Module, mask: torch.Tensor):
        loss = self.criterionGAN(netD(fake), target_is_real=True)
        return loss

    def backward_G_alone(self):
        loss_G_GAN2_A = self.backward_G_GAN(self.fake_a_random, self.netDis_A, self.real_a_encoded != -1) * self.opt.lambda_domain_adv_random
        loss_G_GAN2_B = self.backward_G_GAN(self.fake_b_random, self.netDis_B, self.real_b_encoded != -1) * self.opt.lambda_domain_adv_random

        if self.opt.use_ms:
            loss_G_GAN2_A2 = self.backward_G_GAN(self.fake_a_random2, self.netDis_A, self.real_a_random != -1) * self.opt.lambda_domain_adv_random
            loss_G_GAN2_B2 = self.backward_G_GAN(self.fake_b_random2, self.netDis_B, self.real_b_random != -1) * self.opt.lambda_domain_adv_random
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
                        save_gif=True, use_image_name=False, mode=''):
        volumes = [(f'{self.modality_A} real', self.real_a_encoded[0:1].cpu()),
                   (f'{self.modality_A} content random', self.fake_a_random[0:1].cpu()),
                   (f'{self.modality_A} fake', self.fake_a_encoded[0:1].cpu()),
                   (f'{self.modality_A} reconstructed', self.fake_aa_encoded[0:1].cpu()),
                   (f'{self.modality_B} real', self.real_b_encoded[0:1].cpu()),
                   (f'{self.modality_B} content random', self.fake_b_random[0:1].cpu()),
                   (f'{self.modality_B} fake', self.fake_b_encoded[0:1].cpu()),
                   (f'{self.modality_B} reconstructed', self.fake_bb_encoded[0:1].cpu())]

        axs, fig = vxm.torch.utils.init_figure(3, len(volumes))
        vxm.torch.utils.set_axs_attribute(axs)
        for i, (key, img) in enumerate(volumes):
            tensorboard.fill_subplots(img, axs=axs[i, :], img_name=key)
        fig.suptitle(f'ID {self.patient}')
        if use_image_name:
            tag = mode + f'{self.patient}/GAN'
        else:
            tag = mode + '/GAN'
        writer.add_figure(tag=tag, figure=fig, global_step=global_step, close=False)
        fig.clf()
        plt.close(fig)

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
                elif 'cc' in key.lower():
                    writer.add_scalar(f'RECON/{key}', scalar_value=losses[key], global_step=global_step)
                else:
                    writer.add_scalar(f'other/{key}', scalar_value=losses[key], global_step=global_step)

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass
