import time
import os

import torch
import numpy as np
import csv
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from torch.utils.tensorboard import SummaryWriter
from models.multitask_parent import Multitask

def main():
    parser = TrainOptions()
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
    dataset_size = len(dataset)  # get the number of images in the dataset.

    model = create_model(opt)  # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    total_iters = 0  # the total number of training iterations

    print('visualizer started')

    optimize_time = -1

    times = []
    opt.tensorboard_path = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(opt.tensorboard_path, exist_ok=True)
    writer = SummaryWriter(opt.tensorboard_path)

    print('tensorboard started')

    for epoch in range(opt.epoch_count,
                       opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        dataset.set_epoch(epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch

            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                print("initializing data dependent parts of the network")
                model.data_dependent_initialize(data)
                model.setup(opt)  # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            new_time = (time.time() - optimize_start_time) / opt.batch_size
            optimize_time = new_time * 0.5 if optimize_time == -1 else new_time * 0.02 + 0.98 * optimize_time

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)

            # display images on visdom and save images to a HTML file
            if total_iters % opt.display_freq == 0:
                display_results(data, epoch, model, opt, total_iters, visualizer, writer)

            # cache our latest model every <save_latest_freq> iterations
            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            # evaluate model performance every evaluation_freq iterations
            if total_iters % opt.evaluation_freq == 0:
                evaulate_model(dataset_val, model, total_iters, writer, opt.num_validation_samples)
            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate(epoch=epoch)  # update learning rates at the end of every epoch.


def display_results(data, epoch, model, opt, total_iters, visualizer, writer):
    losses = model.get_current_losses()  # read losses before setting to no_grad for validation
    data = data
    model.eval()  # change networks to eval mode
    with torch.no_grad():
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
    save_result = total_iters % opt.update_html_freq == 0
    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
    if isinstance(model, Multitask):
        model.compute_landmark_loss()
        model.compute_gt_dice()
    model.train()  # change networks back to train mode
    model.log_tensorboard(writer, losses, total_iters, save_gif=False, mode='train')
    if isinstance(model, Multitask):
        model.log_mt_tensorboard(model.real_A, model.real_B, model.fake_B, writer, global_step=total_iters, mode='train')


def evaulate_model(dataset_val, model, total_iters, writer, num_validation_samples):
    print('evaluating model on labeled data')
    losses_total = []
    keys = []
    loss_aggregate = {}
    land_rig = []
    land_def = []
    land_beg = []
    for j, (val_data) in enumerate(dataset_val):
        model.eval()  # change networks to eval mode
        if j > num_validation_samples:
            break
        with torch.no_grad():
            model.set_input(val_data)  # unpack data from data loader
            model.forward()  # run inference
            model.calculate_loss_values()  # get the loss values
            model.compute_visuals()
            losses = model.get_current_losses()
            losses_total.append(losses)
            model.get_current_visuals()
            if isinstance(model, Multitask):
                landmarks_beg, landmarks_rig, landmarks_def = model.get_current_landmark_distances()
                land_beg.append(landmarks_beg.item())
                land_rig.append(landmarks_rig.item())
                land_def.append(landmarks_def.item())
                model.compute_landmark_loss()
                model.compute_gt_dice()
                model.log_mt_tensorboard(model.real_A, model.real_B, model.fake_B, writer=writer,
                                         global_step=total_iters, use_image_name=False, mode=f'val-{model.patient}')
            model.log_tensorboard(writer=writer, losses=None, global_step=total_iters, save_gif=False,
                                  use_image_name=True, mode=f'val-')
        keys = losses.keys()
    for key in keys:
        loss_aggregate[key] = np.nanmean([losses.get(key, np.NaN) for losses in losses_total])
    for key in loss_aggregate:
        writer.add_scalar(f'val-losses/{key}', scalar_value=loss_aggregate[key], global_step=total_iters)


if __name__ == '__main__':
    main()