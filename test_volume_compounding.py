import os
from options.base_options import BaseOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np
import cv2
import matplotlib.pyplot as plt
import SimpleITK as sitk
from PIL import Image


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
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.add_argument('--plot_result', type=bool, default=False, help='flag to plot the images when running')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser


def compound_volume(slices, opt):
    resized = []
    for slice_ in slices:
        resized.append(Image.fromarray(slice_).resize(opt.orig_size, Image.ANTIALIAS))
    resized = np.stack(resized, axis=0)
    vol = sitk.GetImageFromArray(resized)
    vol.SetSpacing(opt.res)
    sitk.WriteImage(vol, opt.results_dir + 'fake.mhd')


if __name__ == '__main__':
    opt = VolumeTestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
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

    fake_slices = []
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        fake = visuals['fake_B'].cpu().squeeze().numpy()
        fake = np.transpose(fake, axes=[1, 2, 0])
        fake = cv2.cvtColor(fake, cv2.COLOR_RGB2GRAY)
        if opt.plot_result:
            plt.imshow(fake, cmap='gray')
            plt.title(f'show image at index {i}')
            plt.show()
        fake_slices.append(fake)
        img_path = model.get_image_paths()  # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    compound_volume(fake_slices, opt)
    webpage.save()  # save the HTML
