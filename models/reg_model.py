import torch
from .base_model import BaseModel
from . import networks
from scipy import linalg
from torchio.transforms import (
    RescaleIntensity,
    RandomAffine,
    RandomElasticDeformation,
    Compose,
)
import torch
import math
import torchvision
import transforms3d
import torchio
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def set_border_value(img: torch.Tensor, value= None):
    if value is None:
        value = img.min()
    img[:, :, 0, :, :] = value
    img[:, :, -1, :, :] = value
    img[:, :, :, 0, :] = value
    img[:, :, :, -1, :] = value
    img[:, :, :, :, 0] = value
    img[:, :, :, :, -1] = value
    return img

def transform_image(img: torch.Tensor, transform, device):
    # img = torch.tensor(img.view(1,1,9,256,256)).to(self.device)
    img = img.unsqueeze(dim=0)
    # img = set_border_value(img)
    grid = F.affine_grid(transform, img.shape).to(device)
    x_trans = F.grid_sample(img, grid, padding_mode='border')
    # x_trans = torch.tensor(x_trans.view(1,9,256,256))
    return x_trans.squeeze(dim=0)


class RegModel(BaseModel):
    """ This class implements the GAN registration model, for learning 6 parameters from moving images for registration
    with fixed images given paired data.

    GAN paper: https://arxiv.org/pdf/1804.11024.pdf
    """

    @staticmethod
    def modify_commandline_oppiptions(parser, is_train=True):
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
        # changing the default values to match the GAN paper (https://arxiv.org/pdf/1804.11024.pdf)
        parser.set_defaults(norm='batch', netG='reg_net', dataset_mode='multichannel', output_nc=1, netD='reg',
                            input_nc=9)
        print(parser)
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        return parser

    def __init__(self, opt):
        """Initialize the reg class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # default `log_dir` is "runs" - we'll be more specific here
        self.writer = SummaryWriter()
        self.device = torch.device("cpu")
        self.train_counter = 0
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # square of difference between the transformation parameters as part of the generator loss
        self.loss_names = ['loss_G', 'D_real', 'D_fake','loss_D']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'real_A_inverted', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            # define loss functions
            # WGAN values from paper
            self.learning_rate = 0.00005
            self.batch_size = 64
            self.weight_cliping_limit = 0.01

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            D_params = list(self.netD.parameters())
            G_params = list(self.netG.parameters())
            self.optimizer_D = torch.optim.RMSprop(D_params, lr=self.learning_rate)
            self.optimizer_G = torch.optim.RMSprop(G_params, lr=self.learning_rate)

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)




    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        # AtoB = self.opt.direction == 'AtoB'
        self.original_mri = input['A']
        self.original_us = input['B']
        # Transforms can be composed as in torchvision.transforms
        affine_transform = RandomAffine(degrees=(-25, 25), translation=(-5, 5), scales=(1,1), center="image")

        image = torchio.ScalarImage(tensor=self.original_mri)
        subject = torchio.Subject(image_1=image)
        transformed = affine_transform(subject)
        # Get the image daya
        real_A_trans = subject['image_1'].data
        print(transformed.history[0][1].get('rotation'))

        # Compose translations, rotations, zooms, [shears]  to affine  Parameters
        rot_matr = transforms3d.euler.euler2mat(math.radians(transformed.history[0][1].get('rotation')[0]),
                                                math.radians(transformed.history[0][1].get('rotation')[1]),
                                                math.radians(transformed.history[0][1].get('rotation')[2]), axes='sxyz')
        affine_matrix = transforms3d.affines.compose(transformed.history[0][1].get('translation'), rot_matr,
                                                     transformed.history[0][1].get('scaling'))
        # invert the matrix to fit to the Gnet output
        affine_matrix_inv = linalg.inv(affine_matrix)

        self.mri_subject = transformed
        self.original_transf_matrix_inv = torch.tensor(affine_matrix_inv, dtype=torch.float).to(self.device)
        self.mri_random_deformed = real_A_trans.to(self.device)
        self.mri_us_concat = torch.cat((self.original_us, real_A_trans), 1).to(self.device)
        self.image_paths = input['A_paths']
        self.original_mri = self.original_mri.to(self.device)
        self.original_us = self.original_us.to(self.device)
      #  print(type(real_A_trans))
        img_grid = torchvision.utils.make_grid(self.original_mri[:, 0:2, :, :])
        self.writer.add_image('train_img_correct', img_grid)
        img_grid = torchvision.utils.make_grid(self.mri_random_deformed[:,0:2,:,:])
        self.writer.add_image('train_img_deformed', img_grid)



    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        output_g = self.netG(self.mri_us_concat)  # G(A)
        output_matrix = self.create_matrix(output_g)
        print("shape")
        print(output_matrix.shape)
        output_matrix = output_matrix.view(1, 3, 4)
        self.mri_inv_output_transf = transform_image(self.mri_random_deformed, output_matrix, self.device)
        img_grid = torchvision.utils.make_grid(self.mri_inv_output_transf[:, 0:2, :, :])
        self.writer.add_image('train_img_correct', img_grid)

    # def create_matrix(self, output):
    #     print(output[0][3], output[0][4], output[0][5])
    #     rot_matr = torchgeometry.angle_axis_to_rotation_matrix((output[0][0],
    #                                             output[0][1],
    #                                             output[0][2]))
    #     new_rot_matr = torch.cat(((rot_matr), ([output[0][3]], [output[0][4]], [output[0][5]])),1)
    #     return new_rot_matr

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""

        # Randomize if the input to D is the real or fake tranform

        pred_fake = self.netD(torch.cat((self.original_us, self.mri_inv_output_transf.detach()),1))  # TODO add ultrasound image as the input to the network
        # this should be pretty low (since the output of the G is used to transform the image)

       # self.mri_inv_original_transf = self.transform_image(self.mri_random_deformed, self.original_transf_matrix_inv[:3, :].view(1, 3, 4))
        pred_real = self.netD(torch.cat((self.original_us,self.original_mri),1))
        # this should be a high value since we are using the GT transform

        self.loss_D_fake = self.criterionGAN(pred_fake, target_is_real=False)
        self.loss_D_real = self.criterionGAN(pred_real, target_is_real=True)

        # Critic Loss = [average critic score on real images] – [average critic score on fake images]
        self.loss_D = self.loss_D_fake + self.loss_D_real
        self.writer.add_scalar("Loss/loss_D", self.loss_D)
        self.loss_D.backward()



    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        # Generator Loss = -[average critic score on fake images]
        #https: // machinelearningmastery.com / how - to - implement - wasserstein - loss - for -generative - adversarial - networks /
        pred_fake = self.netD(torch.cat((self.original_us,self.mri_inv_output_transf),1))
        self.loss_D_fake = self.criterionGAN(pred_fake, target_is_real=False)
        self.writer.add_scalar("Loss/loss_G",  self.loss_D_fake)
        self.loss_G = self.loss_D_fake
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)

        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        for p in self.netD.parameters():
            p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

        self.train_counter += 1
        if self.train_counter == 2:
            # update G
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()  # set G's gradients to zero
            self.backward_G()  # calculate graidents for G
            self.optimizer_G.step()  # udpate G's weights
            # reset the counter
            self.train_counter = 0
