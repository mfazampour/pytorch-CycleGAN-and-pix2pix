import os.path

os.environ["TORCHIO_HIDE_CITATION_PROMPT"] = str(1)

import random
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from data.base_dataset import BaseDataset
from util import create_landmarks
# from data.image_folder import make_dataset
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
    Lambda
)

try:
    import napari
except:
    print("failed to import napari")


def load_image_file(path: str) -> np.ndarray:
    img = sitk.ReadImage(path)
    return img


class VolumeDataset(BaseDataset):

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
        parser.add_argument('--inshape', type=int, nargs='+', default=[80] * 3,
                            help='after cropping shape of input. '
                                 'default is equal to image size. specify if the input can\'t path through UNet')
        parser.add_argument('--origshape', type=int, nargs='+', default=[80] * 3,
                            help='original shape of input images')
        parser.add_argument('--min_size', type=int, default=80, help='minimum length of the axes')
        parser.add_argument('--transforms', nargs='+', default=[],
                            help='list of possible augmentations, currently [flip, affine]')
        parser.add_argument('--denoising', nargs='+', default=[],
                            help='list of possible denoising, currently [median, lee_filter]')
        parser.add_argument('--denoising_size', type=int, default=4, help='size of the denoising filter kernel')
        parser.add_argument('--load_uncropped', action='store_true', help='load the original uncropped TRUS')
        # parser.add_argument('--replaced_denoised', action='store_true', help='replace B with the denoised version')
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

        self.patients = self.read_list_of_patients()
        random.shuffle(self.patients)
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

    @staticmethod
    def clip_image(x: torch.Tensor):
        [l, h] = np.quantile(x.cpu().numpy(), [0.02, 0.98])
        x[x < l] = l
        x[x > h] = h
        return x

    @staticmethod
    def median_filter_creator(size: int):
        def median_image(x: torch.Tensor):
            im = median_filter(x.squeeze().cpu().numpy(), size=size)
            im = torch.tensor(im, device=x.device, dtype=x.dtype)
            im = torch.reshape(im, x.shape)
            return im
        return median_image

    @staticmethod
    def lee_filter_creator(size: int):
        # def lee_filter(img: torch.Tensor):
        #     img_mean = F.conv3d(img, weight=torch.ones(size=(size, size, size)), bias=torch.tensor([0]))
        #     img_sqr_mean = F.conv3d(img ** 2, weight=torch.ones(size=(size, size, size)), bias=torch.tensor([0]))
        #     img_variance = img_sqr_mean - img_mean ** 2
        #     overall_variance = torch.var(img)
        #
        #     img_weights = img_variance / (img_variance + overall_variance)
        #     img_output = img_mean + img_weights * (img - img_mean)
        #     return img_output

        def lee_filter(x: torch.Tensor):
            img = x.squeeze().cpu().numpy()
            img_mean = uniform_filter(img, [size]*len(img.shape))
            img_sqr_mean = uniform_filter(img ** 2, [size] * len(img.shape))
            img_variance = img_sqr_mean - img_mean ** 2

            overall_variance = variance(img)

            img_weights = img_variance / (img_variance + overall_variance)
            img_output = img_mean + img_weights * (img - img_mean)
            img = torch.tensor(img_output, device=x.device, dtype=x.dtype)
            img = torch.reshape(img, x.shape)
            return img
        return lee_filter

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

        self.denoising_transform = None
        if len(self.opt.denoising) > 0:
            if 'median' in self.opt.denoising:
                self.denoising_transform = Lambda(VolumeDataset.median_filter_creator(self.opt.denoising_size),
                                                  types_to_apply=[torchio.INTENSITY])
            if 'lee_filter' in self.opt.denoising:
                self.denoising_transform = Lambda(VolumeDataset.lee_filter_creator(self.opt.denoising_size),
                                                  types_to_apply=[torchio.INTENSITY])

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
        transforms = [Resample(1 / self.ratio)]
        return Compose(transforms + [CropOrPad(self.opt.origshape, padding_mode=min_value)])

    def read_list_of_patients(self):
        patients = []
        for root, dirs, files in os.walk(self.root):
            if ('nonrigid' in root) or ('cropped' not in root) or ('5992C6' in root) or ('D5656C' in root):
                continue
            if 'trus_cut.mhd' not in files:
                continue
            patients.append(root)
        return patients

    def __getitem__(self, index):
        sample, subject = self.load_subject_(index)
        landmarks_a = create_landmarks.getLandmarks(sample + "/mr.mhd", sample[:-8] + "/mr_pcd.txt")
        landmarks_b = create_landmarks.getLandmarks(sample + "/trus.mhd", sample[:-8] + "/trus_pcd.txt")
        transformed_ = self.transform(subject)
        if self.opt.visualize_volume:
            try:
                with napari.gui_qt():
                    napari.view_image(np.stack([transformed_['mr'].data.squeeze().numpy(),
                                                transformed_['trus'].data.squeeze().numpy()]))
            except:
                pass

        dict_ = {
            'A': transformed_['mr'].data[:, :self.input_size[0], :self.input_size[1], :self.input_size[2]],
            'B': transformed_['trus'].data[:, :self.input_size[0], :self.input_size[1], :self.input_size[2]],
            'Patient': sample.split('/')[-4].replace(' ', ''),
            'A_paths': sample + "/mr.mhd",
            'B_paths': sample + "/trus_cut.mhd",
            'A_landmark': landmarks_a,
            'B_landmark': landmarks_b,
            'modality_A': 'MR',
            'modality_B': 'US'
        }
        if self.load_mask:
            dict_['A_mask'] = transformed_['mr_tree'].data[:, :self.input_size[0], :self.input_size[1],
                              :self.input_size[2]].type(torch.uint8)
            if 'trus_tree' in transformed_.keys():
                dict_['B_mask'] = transformed_['trus_tree'].data[:, :self.input_size[0], :self.input_size[1],
                                  :self.input_size[2]].type(torch.uint8)
                dict_['B_mask_available'] = True
            else:
                dict_['B_mask'] = torch.zeros_like(dict_['A_mask'])
                dict_['B_mask_available'] = False
        return dict_

    def load_subject_(self, index):
        sample = self.patients[index % len(self.patients)]

        # load mr and turs file if it hasn't already been loaded
        if sample not in self.subjects:
            # print(f'loading patient {sample}')
            trus_path = sample + "/trus.mhd" if self.opt.load_uncropped else sample + "/trus_cut.mhd"
            if self.load_mask:
                if os.path.isfile(sample + "/trus_tree.mhd"):
                    subject = torchio.Subject(mr=torchio.ScalarImage(sample + "/mr.mhd"),
                                              trus=torchio.ScalarImage(trus_path),
                                              mr_tree=torchio.LabelMap(sample + "/mr_tree.mhd"),
                                              trus_tree=torchio.LabelMap(sample + "/trus_tree.mhd"))
                else:
                    subject = torchio.Subject(mr=torchio.ScalarImage(sample + "/mr.mhd"),
                                              trus=torchio.ScalarImage(trus_path),
                                              mr_tree=torchio.LabelMap(sample + "/mr_tree.mhd"))
            else:
                subject = torchio.Subject(mr=torchio.ScalarImage(sample + "/mr.mhd"),
                                          trus=torchio.Image(trus_path))
            if self.denoising_transform is not None:
                subject['trus'] = self.denoising_transform(subject['trus'])
            self.subjects[sample] = subject
        subject = self.subjects[sample]
        return sample, subject

    def __len__(self):
        if self.opt.isTrain:
            return len(self.patients)
        else:
            return len(self.patients)

    def name(self):
        return 'VolumeDataset'
