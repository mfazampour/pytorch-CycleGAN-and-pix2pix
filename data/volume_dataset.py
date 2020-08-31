import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
# from data.image_folder import make_dataset
import pickle
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
    Resample
)


def load_image_file(path: str) -> np.ndarray:
    img = sitk.ReadImage(path)
    return img


class VolumeDataset(BaseDataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.opt = opt
        self.root = opt.dataroot

        self.patients = self.read_list_of_patients()
        random.shuffle(self.patients)
        self.mr = {}
        self.trus = {}

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        self.transform = self.create_transforms()

    def create_transforms(self):
        # Let's use one preprocessing transform and one augmentation transform
        # This transform will be applied only to scalar images:
        rescale = RescaleIntensity((0, 1))

        # As RandomAffine is faster then RandomElasticDeformation, we choose to
        # apply RandomAffine 80% of the times and RandomElasticDeformation the rest
        # Also, there is a 25% chance that none of them will be applied
        spatial = OneOf(
            {RandomAffine(): 0.8, RandomElasticDeformation(): 0.2},
            p=0.75,
        )

        # Transforms can be composed as in torchvision.transforms
        transforms = [rescale, spatial]
        transforms.append(Resample(0.25))
        crop_size = list(((np.array([85, 66, 79]) * 4 - 256)/2).astype(np.int))
        transforms.append(Crop(crop_size))
        transform = Compose(transforms)
        return transform

    def read_list_of_patients(self):
        patients = []
        for root, dirs, files in os.walk(self.root):
            if ('nonrigid' in root) or ('cropped' not in root):
                continue
            patients.append(root)
        return patients

    def __getitem__(self, index):
        # returns samples of dimension [channels, z, x, y]

        sample = self.patients[index]

        # data_folder = '/preprocessed/cropped/'
        # load mr and turs file if it hasn't already been loaded
        if sample not in self.mr:
            print(f'loading patient {sample}')
            self.mr[sample] = load_image_file(sample + "/mr.mhd")
            self.trus[sample] = load_image_file(sample + "/trus.mhd")
        mr = self.mr[sample]
        trus = self.trus[sample]

        mr = self.transform(mr)
        trus = self.transform(trus)

        mr, mr_affine = torchio.utils.sitk_to_nib(mr)
        mr = torch.tensor(mr)[:, :256, :256, :256]
        trus, trus_affine = torchio.utils.sitk_to_nib(trus)
        trus = torch.tensor(trus)[:, :256, :256, :256]

        # # convert to torch tensors with dimension [channel, z, x, y]
        # mr = torch.from_numpy(mr[None, :])
        # trus = torch.from_numpy(trus[None, :])
        return {
            'A': mr,
            'B': trus
        }

    def __len__(self):
        return len(self.patients)

    def name(self):
        return 'VolumeDataset'
