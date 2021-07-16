"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os

import torchio

from options.test_options import TestOptions
from util.visualizer import save_images
import torch
import numpy as np
import csv
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from torch.utils.tensorboard import SummaryWriter
from models.segmentation_model import SegmentationModel

def main():
  #  opt = TestOptions().parse()  # get test options
    parser = TestOptions()
    opt = parser.parse()  # get training options


    try:
        from polyaxon_helper import (
            get_outputs_path,
            get_data_paths,
        )

        base_path = get_data_paths()
        print("You are running on the cluster :)")
        opt.dataroot = base_path['data1'] + opt.dataroot
        opt.checkpoints_dir = get_outputs_path()
        opt.display_id = -1  # no visdom available on the cluster
        parser.print_options(opt)
    except Exception as e:
        print(e)
        print("You are Running on the local Machine")

    dataset = create_dataset(opt, mode='train')  # create a dataset given opt.dataset_mode and other options
    dataset_val = create_dataset(opt, mode='validation')  # validation dataset
    dataset_test = create_dataset(opt, mode='test')

    dataset_size = len(dataset_test)  # get the number of images in the dataset.

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    print('The number of test images = %d' % dataset_size)


    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.

    times = []
    opt.tensorboard_path = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(opt.tensorboard_path, exist_ok=True)
    writer = SummaryWriter(opt.tensorboard_path)
    # create a website

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    print('evaluating model on labeled data')

    model.eval()  # change networks to eval mode
    for j, (data) in enumerate(dataset):
        test(model, opt, data, writer)
    for j, (val_data) in enumerate(dataset_val):
        test(model, opt, val_data, writer)
    for j, (test_data) in enumerate(dataset_test):
        test(model, opt, test_data, writer)


def test(model, opt, test_data, writer):
    with torch.no_grad():
        model.set_input(test_data)  # unpack data from data loader
        model.forward()  # run inference
        # model.calculate_loss_values()  # get the loss values
        model.compute_visuals()
        model.get_current_visuals()
        if opt.save_volume:
            if isinstance(model, SegmentationModel):
                os.makedirs(os.path.join(opt.checkpoints_dir, f'{opt.name}/vol'), exist_ok=True)
                img = torchio.ScalarImage(tensor=model.seg_A[0, ...].to(torch.float).detach().cpu())
                img.save(os.path.join(opt.checkpoints_dir, f'{opt.name}/vol/{model.patient[0]}_seg.nii'))
                img = torchio.ScalarImage(tensor=model.data_A[0, ...].detach().cpu())
                img.save(os.path.join(opt.checkpoints_dir, f'{opt.name}/vol/{model.patient[0]}_img.nii'))
        model.log_tensorboard(writer=writer, losses=None, global_step=0, save_gif=False,
                              use_image_name=True, mode=f'val-')


if __name__ == '__main__':
    main()