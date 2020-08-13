from .pix2pix_model import Pix2PixModel
import torch
from skimage import color  # used for lab2rgb
import numpy as np


class MultiChannelModel(Pix2PixModel):
    """This is a subclass of Pix2PixModel for image colorization (black & white image -> colorful images).

    The model training requires '-dataset_model colorization' dataset.
    It trains a pix2pix model, mapping from L channel to ab channels in Lab color space.
    By default, the colorization dataset will automatically set '--input_nc 1' and '--output_nc 2'.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        By default, we use 'colorization' dataset for this model.
        See the original pix2pix paper (https://arxiv.org/pdf/1611.07004.pdf) and colorization results (Figure 9 in the paper)
        """
        Pix2PixModel.modify_commandline_options(parser, is_train)
        parser.set_defaults(dataset_mode='multichannel')
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
        # reuse the pix2pix model
        Pix2PixModel.__init__(self, opt)
        # specify the images to be visualized.
        self.visual_names = ['real_A_avg', 'real_B_avg', 'fake_B_avg',
                             'real_A_center', 'real_B_center', 'fake_B_center']


    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        n_c = self.real_A.shape[1]
        r = np.random.random_integers(0, self.real_A.shape[0] - 1, 1)
        # average over channel to get the real and fake image
        self.real_A_center = self.real_A[:, int(n_c/2), ...].unsqueeze(1)
        self.real_B_center = self.real_B[:, int(n_c/2), ...].unsqueeze(1)
        self.fake_B_center = self.fake_B[:, int(n_c/2), ...].unsqueeze(1)
        self.real_A_avg = torch.mean(self.real_A, dim=1, keepdim=True)
        self.real_B_avg = torch.mean(self.real_B, dim=1, keepdim=True)
        self.fake_B_avg = torch.mean(self.fake_B, dim=1, keepdim=True)