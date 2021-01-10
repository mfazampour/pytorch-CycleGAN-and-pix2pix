import os.path

os.environ["TORCHIO_HIDE_CITATION_PROMPT"] = str(1)

import random
from data.base_dataset import BaseDataset
from util import create_landmarks
import numpy as np
import SimpleITK as sitk
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
    RandomFlip
)


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
        parser.add_argument('--inshape', type=int, nargs='+', default=[80] * 3,
                            help='after cropping shape of input. '
                                 'default is equal to image size. specify if the input can\'t path through UNet')
        parser.add_argument('--origshape', type=int, nargs='+', default=[80] * 3,
                            help='original shape of input images')
        parser.add_argument('--load_mask', type=bool, default=False, help='load prostate mask for seg. loss')
        parser.add_argument('--min_size', type=int, default=80, help='minimum length of the axes')

        return parser

    def __init__(self, opt, to_validate=False):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.opt = opt
        if to_validate:
            self.root = os.path.join(self.opt.dataroot, self.opt.val_fold)
        else:
            self.root = os.path.join(self.opt.dataroot, self.opt.train_fold)

        print(f'dataroot: {self.root}')
        self.load_mask = opt.load_mask
        self.patient = self.read_list_of_patients()

        random.shuffle(self.patient)
        self.subjects = {}
        self.subjects_val = {}

        # self.mr = {}
        # self.trus = {}

        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        self.input_size = [80] * 3
        self.min_size = 80

        self.transform = self.create_transforms()

    def create_transforms(self):
        # Let's use one preprocessing transform and one augmentation transform
        # This transform will be applied only to scalar images:
        rescale = RescaleIntensity((-1, 1))
        transforms = [rescale]
        # As RandomAffine is faster then RandomElasticDeformation, we choose to
        # apply RandomAffine 80% of the times and RandomElasticDeformation the rest
        # Also, there is a 25% chance that none of them will be applied
        if self.opt.isTrain:
            spatial = OneOf(
                {RandomAffine(translation=5): 0.8, RandomElasticDeformation(): 0.2},
                p=0.75,
            )
            transforms += [RandomFlip(axes=(0, 2), p=0.8), spatial]

        self.ratio = self.min_size / np.max(self.input_size)
        transforms.append(Resample(self.ratio))
        crop_size = list(((np.array([80, 80, 80]) / self.ratio - self.input_size) / 2).astype(np.int))
        transforms.append(Crop(crop_size))
        transform = Compose(transforms)
        return transform

    def reverse_resample(self, min_value=-1):
        transforms = [Resample(1 / self.ratio)]
        pad_size = list(np.ceil((np.array([80, 80, 80]) - np.asarray(self.input_size) * self.ratio) / 2).astype(np.int))
        return Compose(transforms + [Pad(pad_size, padding_mode=min_value)])

    def read_list_of_patients(self):
        patients = []
        for root, dirs, files in os.walk(self.root):
            if ('nonrigid' in root) or ('cropped' not in root) or ('@eaDir' in root):
                continue
            if os.path.exists(root + "/trus.mhd") and os.path.exists(root + "/mr.mhd"):
                patients.append(root)
        return patients

    def __getitem__(self, index):
        sample, subject = self.load_subject_(index)
        landmarks_a = create_landmarks.getLandmarks(sample + "/mr.mhd", sample[:-8] + "/mr_pcd.txt")
        landmarks_b = create_landmarks.getLandmarks(sample + "/trus.mhd", sample[:-8] + "/trus_pcd.txt")

        rescale = RescaleIntensity((-1, 1))
        dict_ = {
            'A': rescale(subject['mr'].data[:, :self.input_size[0], :self.input_size[1], :self.input_size[2]]),
            'B': rescale(subject['trus'].data[:, :self.input_size[0], :self.input_size[1], :self.input_size[2]]),
            'Patient': sample.split('/')[-4],
            'A_paths': sample + "/mr.mhd",
            'A_landmark': landmarks_a,
            'B_landmark': landmarks_b,
            'B_paths': sample + "/trus.mhd",
        }

        if self.load_mask:
            dict_['A_mask'] = subject['mr_tree'].data[:, :self.input_size[0], :self.input_size[1], :self.input_size[2]]

        return dict_

    def load_subject_(self, index):
        sample = self.patient[index % len(self.patient)]
        # load mr and turs file if it hasn't already been loaded
        if sample not in self.subjects:
            # print(f'loading patient {sample}')
            if self.load_mask:
                subject = torchio.Subject(mr=torchio.ScalarImage(sample + "/mr.mhd"),
                                          trus=torchio.ScalarImage(sample + "/trus.mhd"),
                                          mr_tree=torchio.LabelMap(sample + "/mr_tree.mhd"))
            else:
                subject = torchio.Subject(mr=torchio.ScalarImage(sample + "/mr.mhd"),
                                          trus=torchio.Image(sample + "/trus.mhd"))

            self.subjects[sample] = subject

        subject = self.subjects[sample]

        return sample, subject

    def __len__(self):
        return 5 * len(self.patient)

    def name(self):
        return 'VolumeDataset'


class UnalignedVolumeDataset(VolumeDataset):
    # TODO: check if actually it makes sense to define this unaligned class
    pass
