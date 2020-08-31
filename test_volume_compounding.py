import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import SimpleITK as sitk
from PIL import Image


from options.base_options import BaseOptions
from data import create_dataset
from models import create_model
from models.pix2pix_model import Pix2PixModel
from util.visualizer import save_images
from util import html


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
        parser.add_argument('--num_test', type=int, default=100, help='how many test images to run')
        parser.add_argument('--plot_result', type=bool, default=False, help='flag to plot the images when running')
        parser.add_argument('--num_of_channels', type=int, default=1,
                            help='number of channels expected in the output image. '
                                 'if the output image of network has more than one channel, in compounding,'
                                 ' average over the channels in z direction should be calculated.'
                                 'should be an odd number')
        parser.add_argument('--patients_dir', type=str, default='./datasets/patients', help='source folder of patients')

        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser


def compound_volume(slices, opt, patient_name:str):
    resized = []
    for slice_ in slices:
        resized.append(Image.fromarray(slice_.squeeze()).resize(opt.orig_size, Image.ANTIALIAS))
    resized = np.stack(resized, axis=0)
    vol = sitk.GetImageFromArray(resized)
    vol.SetSpacing(opt.res)
    path = opt.results_dir + f'volumes/fake_{patient_name}.mhd'
    dir_ = os.path.dirname(path)
    if not os.path.isdir(dir_):
        os.makedirs(dir_)
    sitk.WriteImage(vol, opt.results_dir + f'volumes/fake_{patient_name}.mhd')


def slices_from_multichannel(fake_slices, opt):
    averaged_slices = []
    for i in range(len(fake_slices)):
        fake_single_slice = []
        start_idx = int(np.max([0, i - (opt.num_of_channels - 1) / 2]))
        end_idx = int(np.min([len(fake_slices), i + (opt.num_of_channels + 1) / 2]))
        for idx in range(start_idx, end_idx):
            fake_single_slice.append(fake_slices[idx][:, :, i - start_idx])
        # averaged_slices.append(np.mean(np.stack(fake_single_slice, axis=2), axis=2))
        averaged_slices.append(np.median(np.stack(fake_single_slice, axis=2), axis=2))
    return averaged_slices

def save_network_output(fake_volume, opt, idx, patient_name):
    fake_volume = np.split(fake_volume, fake_volume.shape[2], axis=2)
    compound_volume(fake_volume, opt, f'{patient_name}/chunk_{idx}')

def patient_fake_compound(dataset, patient_name: str):

    fake_slices = []
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        try:
            fake = visuals['fake_B'].cpu().squeeze().numpy()
            fake_volume = fake
        except KeyError:
            fake = visuals['fake_B_center'].cpu().squeeze(dim=0).numpy()
            if isinstance(model, Pix2PixModel):
                fake_volume = model.fake_B.cpu().squeeze(dim=0).numpy()
                fake_volume = np.transpose(fake_volume, axes=(1, 2, 0))
        fake = np.transpose(fake, axes=[1, 2, 0])
        if opt.num_of_channels == 1:
            if opt.plot_result:
                plt.imshow(fake, cmap='gray')
                plt.title(f'show image at index {i}')
                plt.show()
            fake = cv2.cvtColor(fake, cv2.COLOR_RGB2GRAY)
            fake_slices.append(fake)
        else:
            save_network_output(fake_volume, opt, patient_name=patient_name, idx=i)
            fake_slices.append(fake_volume)
        img_path = model.get_image_paths()  # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    if opt.num_of_channels:
        fake_slices = slices_from_multichannel(fake_slices, opt)
    compound_volume(fake_slices, opt, patient_name)
    webpage.save()  # save the HTML


if __name__ == '__main__':
    opt = VolumeTestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name,
                           '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    assert np.mod(opt.num_of_channels, 2) == 1, "num of channels should be an odd number, (for simplicity)"

    for root, dirs, files in os.walk(opt.patients_dir):
        if 'test' in dirs:
            patient_name = root.split('/')[-1]
            opt.dataroot = root
            dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
            patient_fake_compound(dataset, patient_name)
