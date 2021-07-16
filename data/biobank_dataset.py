import os.path
from pathlib import Path

os.environ["TORCHIO_HIDE_CITATION_PROMPT"] = str(1)

import random
import torch
import torch.nn.functional as F
from data.base_dataset import BaseDataset
from data.volume_dataset import VolumeDataset
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


def load_vol_pathes(patient_list_src: str, source_folder: str, t1_pattern, t2_pattern, seg_pattern: str, mode: str):
    t1_names = []
    t2_names = []
    seg_names = []
    if mode is 'train':
        patient_list_file = os.path.join(patient_list_src, 'train.txt')
    elif mode is 'validation':
        patient_list_file = os.path.join(patient_list_src, 'validation.txt')
    elif mode is 'test':
        patient_list_file = os.path.join(patient_list_src, 'test.txt')
    else:
        raise Exception(f'mode is should be one of [train, validation, test], it is {mode}')
    if not os.path.isfile(patient_list_file):
        create_patient_list(t1_pattern, patient_list_src, source_folder)

    # read folder names from file
    file = open(patient_list_file, 'r')
    lines = file.read().splitlines()
    for line in lines:
        t1_names.append(os.path.join(line, t1_pattern))
        t2_names.append(os.path.join(line, t2_pattern))
        seg_names.append(os.path.join(line, seg_pattern))
    return t1_names, t2_names, seg_names


def create_patient_list(img_pattern, patient_list_src, source_folder):
    patient_list = []
    for path in Path(source_folder).rglob(f'*{img_pattern}*'):
        patient_list.append(str(path.parent) + "\n")
    random.shuffle(patient_list)
    index_train = int(len(patient_list) * 0.7)
    validation_train = int(len(patient_list) * 0.85)
    file = open(patient_list_src + 'train.txt', 'w')
    file.writelines(patient_list[:index_train])
    file.close()
    file = open(patient_list_src + 'validation.txt', 'w')
    file.writelines(patient_list[index_train:validation_train])
    file.close()
    file = open(patient_list_src + 'test.txt', 'w')
    file.writelines(patient_list[validation_train:])
    file.close()


class BiobankDataset(VolumeDataset):

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
        parser.add_argument('--inshape', type=int, nargs='+', default=[128] * 3,
                            help='after cropping shape of input. '
                                 'default is equal to image size. specify if the input can\'t path through UNet')
        # parser.add_argument('--origshape', type=int, nargs='+', default=[80] * 3,
        #                     help='original shape of input images')
        # parser.add_argument('--transforms', nargs='+', default=[],
        #                     help='list of possible augmentations, currently [flip, affine]')

        parser.add_argument('--seg_pattern', type=str, default='T1_first_all_fast_firstseg_affine_to_mni.nii.gz')
        parser.add_argument('--t1_pattern', type=str, default='T1_unbiased_brain_affine_to_mni.nii.gz')
        parser.add_argument('--t2_pattern', type=str, default='T2_FLAIR_unbiased_brain_affine_to_mni.nii.gz')
        parser.add_argument('--patient_list_src', type=str, default='/tmp/', help='source folder containing the list of patients names')
        parser.add_argument('--inspacing', type=float, default=1.823, help='voxel spacing to resample to')
        return parser

    def __init__(self, opt, mode):
        BaseDataset.__init__(self, opt, None)
        print(f'dataroot: {self.root}')
        assert os.path.isdir(self.root), f'{self.root} is not a folder '

        self.t1_names, self.t2_names, self.seg_names = load_vol_pathes(opt.patient_list_src, self.root,
                                                                       t1_pattern=opt.t1_pattern,
                                                                       t2_pattern=opt.t2_pattern,
                                                                       seg_pattern=opt.seg_pattern, mode=mode)

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        self.input_size = opt.inshape

        self.transform = self.create_transforms(opt.inshape, target_spacing=opt.inspacing, min_value=-1)
        self.load_mask = opt.load_mask
        self.subjects = {}

    def create_transforms(self, target_shape=None, min_value=0, max_value=1, target_spacing=None):
        transforms = []
        if target_spacing is not None:
            transforms.append(Resample(target_spacing))
        if target_shape is not None:
            transforms.append(CropOrPad(target_shape=target_shape))
        if min_value is not None:
            rescale = RescaleIntensity((min_value, max_value))
            transforms.append(rescale)
        return Compose(transforms)

    def __getitem__(self, index):
        index_t2 = np.random.randint(0, len(self.t2_names), 1)[0]

        if self.opt.load_mask:
            subject = torchio.Subject(t1=torchio.ScalarImage(self.t1_names[index]),
                                      t2=torchio.ScalarImage(self.t2_names[index_t2]),
                                      seg_t1=torchio.LabelMap(self.seg_names[index]),
                                      seg_t2=torchio.LabelMap(self.seg_names[index_t2]))
        else:
            subject = torchio.Subject(t1=torchio.ScalarImage(self.t1_names[index]),
                                      t2=torchio.ScalarImage(self.t2_names[index_t2]))
        subject = self.transform(subject)
        transformed_ = self.transform(subject)
        if self.opt.visualize_volume:
            try:
                with napari.gui_qt():
                    napari.view_image(np.stack([transformed_['mr'].data.squeeze().numpy(),
                                                transformed_['trus'].data.squeeze().numpy()]))
            except:
                pass

        dict_ = {
            'A': transformed_['t1'].data[:, :self.input_size[0], :self.input_size[1], :self.input_size[2]],
            'B': transformed_['t2'].data[:, :self.input_size[0], :self.input_size[1], :self.input_size[2]],
            'Patient': f'{self.t1_names[index].split("/")[-2]}, {self.t2_names[index_t2].split("/")[-2]}',
            'A_paths': self.t1_names[index],
            'B_paths': self.t2_names[index_t2],
            'modality_A': 'T1',
            'modality_B': 'FLAIR'
        }
        if self.load_mask:
            dict_['A_mask'] = transformed_['seg_t1'].data[:, :self.input_size[0], :self.input_size[1],
                              :self.input_size[2]].type(torch.uint8)
            dict_['B_mask'] = transformed_['seg_t2'].data[:, :self.input_size[0], :self.input_size[1],
                              :self.input_size[2]].type(torch.uint8)
        return dict_

    def __len__(self):
        if self.opt.isTrain:
            return len(self.t1_names)
        else:
            return len(self.t1_names)

    def name(self):
        return 'BiobankDataset'
