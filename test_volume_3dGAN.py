import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import SimpleITK as sitk
from PIL import Image
import torch.nn.functional as F

from options.base_options import BaseOptions
from data import create_dataset
from models import create_model
from models.pix2pix3d_model import Pix2Pix3dModel
from models.pix2pix3d_seg_model import Pix2Pix3dSegModel
from models.pix2pix3d_reg_model import Pix2Pix3dRegModel
from models.segmentation_model import SegmentationModel
from models.cut3d_model import CUT3dModel
from util.visualizer import save_images
from util import html
from data.volume_dataset import VolumeDataset
from util.se3 import loss


class VolumeTestOptions(BaseOptions):
    """This class includes volumetric test options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--res', nargs=3, metavar=('res_x', 'res_y', 'res_z'), type=float,
                            default=[1.0, 1.0, 1.0], help='spacing of result images')
        parser.add_argument('--orig_size', nargs=2, metavar=('size x', 'size y'), type=int,
                            default=None, help='original slice size')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=100, help='how many test images to run, set -1 to run all')
        parser.add_argument('--plot_result', type=bool, default=False, help='flag to plot the images when running')
        parser.add_argument('--num_of_channels', type=int, default=1,
                            help='number of channels expected in the output image. '
                                 'if the output image of network has more than one channel, in compounding,'
                                 ' average over the channels in z direction should be calculated.'
                                 'should be an odd number')
        parser.add_argument('--patients_dir', type=str, default='./datasets/patients', help='source folder of patients')
        parser.add_argument('--output_name', type=str, default='fake_US', help='name of output image')
        parser.add_argument('--store_with_original', action='store_true', help='save in the original image folder')

        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser


def save_image(vol, file_name, transform):
    vol = transform(vol)
    vol = np.squeeze(vol)
    vol = np.transpose(vol, axes=[2, 1, 0])
    vol = sitk.GetImageFromArray(vol)
    vol.SetSpacing(opt.res)
    path = os.path.join(opt.results_dir, file_name)
    dir_ = os.path.dirname(path)
    if not os.path.isdir(dir_):
        os.makedirs(dir_)
    sitk.WriteImage(vol, path)
    return path


def copy_transformation(src, dst):
    with open(src, 'r') as f:
        lines = f.readlines()
        pos = [line for line in lines if 'Position' in line]
        ori = [line for line in lines if 'Orientation' in line]
    if len(pos) == 0 or len(ori) == 0:
        print(f'warning: can not find tags for pos/ori for image {src}')
        return
    pos = pos[0]
    ori = ori[0]
    with open(dst, 'r+') as f:
        lines = f.readlines()
        reduced = [line for line in lines if
                   'Position' not in line and
                   'Orientation' not in line and
                   'Transform' not in line and
                   'Offset' not in line]
        reduced.append(pos)
        reduced.append(ori)
    with open(dst, 'w') as f:
        f.writelines(reduced)


if __name__ == '__main__':
    opt = VolumeTestOptions().parse()  # get test options
    opt.isTrain = False
    opt.visualize_volume = False
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    dataset = create_dataset(opt)

    print(len(dataset))
    transform_img = VolumeDataset(opt).reverse_resample(min_value=-1)
    transform_label = VolumeDataset(opt).reverse_resample(min_value=0)
    # assert isinstance(model, Pix2Pix3dModel), 'not the right type of model'
    for i, data in enumerate(dataset):
        if opt.num_test != -1 and i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = data['A_paths'][0]  # get image paths
        patient_name = data['Patient'][0]
        if opt.store_with_original:
            opt.results_dir = os.path.dirname(img_path)
            file_name = f'{opt.output_name}.mhd'
        else:
            file_name = f'volumes/{patient_name}_{opt.output_name}.mhd'

        # vol = model.seg_A if isinstance(model, SegmentationModel) else model.fake_B
        # vol = vol.cpu().squeeze(dim=0).numpy()
        if isinstance(model, SegmentationModel):
            vol = model.seg_A.cpu().squeeze(dim=0).numpy()
            dst = save_image(vol, file_name, transform_label)
            copy_transformation(img_path, dst)

        if isinstance(model, Pix2Pix3dModel) or isinstance(model, CUT3dModel):
            vol = model.fake_B.cpu().squeeze(dim=0).numpy()
            dst = save_image(vol, file_name, transform_img)
            copy_transformation(img_path, dst)

            if isinstance(model, Pix2Pix3dSegModel):
                # store the segmentations too
                vol = F.softmax(model.seg_B, dim=1)[:, 1, ...].cpu().numpy()
                vol = vol >= 0.5
                file_name = f'volumes/{patient_name}_fake_seg.mhd'
                save_image(vol, file_name, transform_label)

                vol = F.softmax(model.seg_A, dim=1)[:, 1, ...].cpu().numpy()
                vol = vol >= 0.5
                file_name = f'volumes/{patient_name}_real_seg.mhd'
                save_image(vol, file_name, transform_label)

            if isinstance(model, Pix2Pix3dRegModel):
                # check the values for registration parameters
                gt_vector = model.gt_vector.cpu()
                est_vector = model.reg_B_params.detach().cpu()
                loss_ = loss(est_vector, gt_vector)
                print(f'for patient {patient_name} registration loss is {loss_}')
                reg_A, reg_B = model.get_transformed_images()
                file_name = f'volumes/{patient_name}_original_registered.mhd'
                save_image(reg_B.cpu().squeeze(dim=0).numpy(), file_name, transform_img)
                file_name = f'volumes/{patient_name}_transformed.mhd'
                save_image(model.transformed_B.cpu().squeeze(dim=0).numpy(), file_name, transform_img)
