import os.path

os.environ["TORCHIO_HIDE_CITATION_PROMPT"] = str(1)

import random
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from data.base_dataset import BaseDataset
from util import create_landmarks
# from data.image_folder import make_dataset😍
import pickle
import numpy as np
import SimpleITK as sitk
import scipy
from scipy.ndimage import median_filter
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

import torchio
from torchio.transforms import (
    RescaleIntensity,
    RandomAffine,
    RandomElasticDeformation,
    Compose,
    OneOf,
    Crop,
    Resample,
    Pad,
    RandomFlip,
    CropOrPad,
    ZNormalization,
    Lambda,
)

try:
    import napari
except:
    print("failed to import napari")


def load_image_file(path: str) -> np.ndarray:
    img = sitk.ReadImage(path)
    return img


class ThoraxDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        """
        parser.add_argument('--visualize_volume', type=bool, default=False, help='Set visualize to False. it\'s only '
                                                                                 'used for debugging.')
        parser.add_argument('--load_mask', type=bool, default=True, help='load prostate mask for seg. loss')
        parser.add_argument('--inshape', type=int, nargs='+', default=[192, 160, 192],
                            help='after cropping shape of input. '
                                 'default is equal to image size. specify if the input can\'t path through UNet')
        parser.add_argument('--origshape', type=int, nargs='+', default=[192, 160, 192],
                            help='original shape of input images')
        parser.add_argument('--min_size', type=int, default=80, help='minimum length of the axes')
        parser.add_argument('--transforms', nargs='+', default=[],
                            help='list of possible augmentations, currently [flip, affine]')
        return parser

    def __init__(self, opt, mode=None):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt, mode)
        self.opt = opt

        print(f'dataroot: {self.root}')
        self.load_mask = opt.load_mask

        self.ct_list = self.read_list_of_patients(keyword='CT')
        self.mr_list = self.read_list_of_patients(keyword='MR')
        random.shuffle(self.ct_list)
        random.shuffle(self.mr_list)
        self.subjects = {}
        # self.mr = {}
        # self.trus = {}

        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        self.input_size = opt.inshape
        self.min_size = opt.min_size

        self.transform = self.create_transforms()

        self.means = []
        self.std = []

    def create_transforms(self):
        transforms = []

        # clipping to remove outliers (if any)
        # clip_intensity = Lambda(VolumeDataset.clip_image, types_to_apply=[torchio.INTENSITY])
        # transforms.append(clip_intensity)

        rescale = RescaleIntensity((-1, 1))
        # normalize with mu = 0 and sigma = 1/3 to have data in -1...1 almost
        # ZNormalization()

        transforms.append(rescale)

        if self.mode == 'train':
            # transforms = [rescale]
            if 'affine' in self.opt.transforms:
                transforms.append(RandomAffine(translation=5, p=0.8))

            if 'flip' in self.opt.transforms:
                transforms.append(RandomFlip(axes=(0, 2), p=0.8))

        # # As RandomAffine is faster then RandomElasticDeformation, we choose to
        # # apply RandomAffine 80% of the times and RandomElasticDeformation the rest
        # # Also, there is a 25% chance that none of them will be applied
        # if self.opt.isTrain:
        #     spatial = OneOf(
        #         {RandomAffine(translation=5): 0.8, RandomElasticDeformation(): 0.2},
        #         p=0.75,
        #     )
        #     transforms += [RandomFlip(axes=(0, 2), p=0.8), spatial]

        # self.ratio = self.min_size / np.max(self.input_size)
        # transforms.append(Resample(self.ratio))
        transforms.append(CropOrPad(self.input_size))
        transform = Compose(transforms)
        return transform

    def reverse_resample(self, min_value=-1):
        return Compose([CropOrPad(self.opt.origshape, padding_mode=min_value)])

    def read_list_of_patients(self, keyword: str):
        patients = []
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if keyword in file and 'nii.gz' in file and 'seg' not in file:
                    patients.append(os.path.join(root, file))
        return patients

    def __getitem__(self, index):
        index1 = index % len(self.mr_list)
        index2 = index // len(self.mr_list)
        sample_ct, sample_mr, subject = self.load_subject_(index1, index2)
        transformed_ = self.transform(subject)
        if self.opt.visualize_volume:
            try:
                with napari.gui_qt():
                    napari.view_image(np.stack([transformed_['mr'].data.squeeze().numpy(),
                                                transformed_['ct'].data.squeeze().numpy()]))
            except:
                pass

        dict_ = {
            'A': transformed_['ct'].data[:, :self.input_size[0], :self.input_size[1], :self.input_size[2]],
            'B': transformed_['mr'].data[:, :self.input_size[0], :self.input_size[1], :self.input_size[2]],
            'Patient': f"{sample_ct.split('/')[-1].replace('_CT.nii.gz', '')}-"
                       f"{sample_mr.split('/')[-1].replace('_MR.nii.gz', '')}",
            'modality_A': 'CT',
            'modality_B': 'MR',
            'A_paths': sample_ct,
            'B_paths': sample_mr
        }
        if self.load_mask:
            dict_['A_mask'] = transformed_['ct_tree'].data[:, :self.input_size[0], :self.input_size[1],
                              :self.input_size[2]].type(torch.uint8)
            dict_['B_mask'] = transformed_['mr_tree'].data[:, :self.input_size[0], :self.input_size[1],
                                  :self.input_size[2]].type(torch.uint8)
        return dict_

    def load_subject_(self, index1, index2):
        sample_ct = self.ct_list[index1 % len(self.ct_list)]
        sample_mr = self.mr_list[index2 % len(self.mr_list)]

        # load ct and mr file
        if self.load_mask:
            subject = torchio.Subject(mr=torchio.ScalarImage(sample_mr),
                                      ct=torchio.ScalarImage(sample_ct),
                                      mr_tree=torchio.LabelMap(sample_mr.replace('img', 'seg')),
                                      ct_tree=torchio.LabelMap(sample_ct.replace('img', 'seg')))
        else:
            subject = torchio.Subject(mr=torchio.ScalarImage(sample_mr),
                                      ct=torchio.Image(sample_ct))
        return sample_ct, sample_mr, subject

    def __len__(self):
        return len(self.ct_list) * len(self.mr_list)

    def name(self):
        return 'VolumeDataset'
