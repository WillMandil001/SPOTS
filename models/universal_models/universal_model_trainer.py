# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import sys
import csv
import cv2
import numpy as np
import click
import random

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torchvision

# standard video and tactile prediction models:
from universal_networks.SVG import Model as SVG
from universal_networks.SVTG_SE import Model as SVTG_SE
from universal_networks.SPOTS_SVG_ACTP import Model as SPOTS_SVG_ACTP
from universal_networks.SPOTS_SVG_PTI_ACTP import Model as SPOTS_SVG_PTI_ACTP
from universal_networks.SPOTS_SVG_ACTP_STP import Model as SPOTS_SVG_ACTP_STP

# tactile conditioned models:
from universal_networks.SVG_TC import Model as SVG_TC
from universal_networks.SVG_TC_TE import Model as SVG_TC_TE

# non-stochastic models:
from universal_networks.VG import Model as VG
from universal_networks.SPOTS_VG_ACTP import Model as SPOTS_VG_ACTP
from universal_networks.VG_MMMM import Model as VG_MMMM

# artificial occlusion training:
from universal_networks.SVG_occ import Model as SVG_occ
from universal_networks.SVTG_SE_occ import Model as SVTG_SE_occ
from universal_networks.SPOTS_SVG_ACTP_occ import Model as SPOTS_SVG_ACTP_occ
from universal_networks.SVG_TC_occ import Model as SVG_TC_occ
from universal_networks.SVG_TC_TE_occ import Model as SVG_TC_TE_occ


class BatchGenerator:
    def __init__(self, train_percentage, train_data_dir, batch_size, image_size, num_workers, occlusion_test=False, occlusion_size=0):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.train_data_dir = train_data_dir
        self.occlusion_test = occlusion_test
        self.occlusion_size = occlusion_size
        self.train_percentage = train_percentage

        self.data_map = []
        with open(train_data_dir + 'map.csv', 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                self.data_map.append(row)
        print("self.data_map: ", len(self.data_map))

    def load_full_data(self):
        dataset_train = FullDataSet(self.data_map, self.train_data_dir, train=True, train_percentage=self.train_percentage, image_size=self.image_size, occlusion_test=self.occlusion_test, occlusion_size=self.occlusion_size)
        dataset_validate = FullDataSet(self.data_map, self.train_data_dir, validation=True, train_percentage=self.train_percentage, image_size=self.image_size, occlusion_test=self.occlusion_test, occlusion_size=self.occlusion_size)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        validation_loader = torch.utils.data.DataLoader(dataset_validate, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.data_map = []
        return train_loader, validation_loader


class FullDataSet:
    def __init__(self, data_map, train_data_dir, train=False, validation=False, train_percentage=1.0, image_size=64, occlusion_test=False, occlusion_size=0):
        self.image_size = image_size
        self.train_data_dir = train_data_dir
        self.occlusion_test = occlusion_test
        self.occlusion_size = occlusion_size
        if train:
            self.samples = data_map[1:int((len(data_map) * train_percentage))]
        if validation:
            self.samples = data_map[int((len(data_map) * train_percentage)): -1]
        data_map = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        value = self.samples[idx]
        robot_data = np.load(self.train_data_dir + value[0])

        tactile_data = np.load(self.train_data_dir + value[1])
        tactile_images = []
        for tactile_data_sample in tactile_data:
            tactile_images.append(create_image(tactile_data_sample, image_size=self.image_size))

        images = []
        occ_images = []
        # random start point:
        min_pixel = int(((self.occlusion_size / 2) + 1))
        max_pixel = int(self.image_size - ((self.occlusion_size / 2) + 1))
        if min_pixel >= max_pixel:
            rand_x = None
            rand_y = None
        else:
            rand_x = random.randint(min_pixel, max_pixel)
            rand_y = random.randint(min_pixel, max_pixel)
        for image_name in np.load(self.train_data_dir + value[2]):
            images.append(np.load(self.train_data_dir + image_name))
            occ_images.append(self.add_occlusion(np.load(self.train_data_dir + image_name), min_pixel, max_pixel, rand_x, rand_y))

        experiment_number = np.load(self.train_data_dir + value[3])
        time_steps = np.load(self.train_data_dir + value[4])
        return [robot_data.astype(np.float32), np.array(images).astype(np.float32), np.array(tactile_images).astype(np.float32), np.array(tactile_data).astype(np.float32), experiment_number, time_steps, np.array(occ_images).astype(np.float32)]

    def add_occlusion(self, image, min_pixel, max_pixel, rand_x, rand_y):
        if min_pixel >= max_pixel:
            image[:,:,:] = 0.0
        else:
            for i in range(rand_x - int(self.occlusion_size / 2), rand_x + int(self.occlusion_size / 2)):
                for j in range(rand_y - int(self.occlusion_size / 2), rand_y + int(self.occlusion_size / 2)):
                    try:
                        image[i,j,:] = 0.0
                    except:
                        pass  # out of bounds
        return image


def create_image(tactile, image_size):
    # convert tactile data into an image:
    image = np.zeros((4, 4, 3), np.float32)
    index = 0
    for x in range(4):
        for y in range(4):
            image[x][y] = [tactile[0][index],
                           tactile[1][index],
                           tactile[2][index]]
            index += 1
    reshaped_image = cv2.resize(image.astype(np.float32), dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)
    return reshaped_image


class UniversalModelTrainer:
    def __init__(self, features):
        self.lr = features["lr"]
        self.seed = features["seed"]
        self.beta = features["beta"]
        self.z_dim = features["z_dim"]
        self.g_dim = features["g_dim"]
        self.beta1 = features["beta1"]
        self.niter = features["niter"]
        self.n_past = features["n_past"]
        self.epochs = features["epochs"]
        self.n_eval = features["n_eval"]
        self.device = features["device"]
        self.log_dir = features["log_dir"]
        self.dataset = features["dataset"]
        self.rnn_size = features["rnn_size"]
        self.channels = features["channels"]
        self.n_future = features["n_future"]
        self.model_dir = features["model_dir"]
        self.data_root = features["data_root"]
        self.optimizer = features["optimizer"]
        self.criterion = features["criterion"]
        self.model_name = features["model_name"]
        self.scaler_dir = features["scaler_dir"]
        self.batch_size = features["batch_size"]
        self.num_digits = features["num_digits"]
        self.num_workers = features["num_workers"]
        self.image_width = features["image_width"]
        self.out_channels = features["out_channels"]
        self.data_threads = features["data_threads"]
        self.tactile_size = features["tactile_size"]
        self.train_data_dir = features["train_data_dir"]
        self.model_save_path = features["model_save_path"]
        self.last_frame_skip = features["last_frame_skip"]
        self.training_stages = features["training_stages"]
        self.prior_rnn_layers = features["prior_rnn_layers"]
        self.train_percentage = features["train_percentage"]
        self.state_action_size = features["state_action_size"]
        self.posterior_rnn_layers = features["posterior_rnn_layers"]
        self.predictor_rnn_layers = features["predictor_rnn_layers"]
        self.validation_percentage = features["validation_percentage"]
        self.training_stages_epochs = features["training_stages_epochs"]
        self.model_name_save_appendix = features["model_name_save_appendix"]
        self.tactile_encoder_hidden_size = features["tactile_encoder_hidden_size"]
        self.tactile_encoder_output_size = features["tactile_encoder_output_size"]

        self.occlusion_test = features["occlusion_test"]
        self.occlusion_max_size = features["occlusion_max_size"]
        self.occlusion_start_epoch = features["occlusion_start_epoch"]
        self.occlusion_gain_per_epoch = features["occlusion_gain_per_epoch"]

        self.gain = 0.0
        self.stage = self.training_stages[0]

        if self.model_name == "SVG":
            self.model = SVG(features)
        elif self.model_name == "SVG_occ":
            self.model = SVG_occ(features)
        elif self.model_name == "SVTG_SE":
            self.model = SVTG_SE(features)
        elif self.model_name == "SVTG_SE_occ":
            self.model = SVTG_SE_occ(features)
        elif self.model_name == "SPOTS_SVG_ACTP":
            self.model = SPOTS_SVG_ACTP(features)
        elif self.model_name == "SPOTS_SVG_PTI_ACTP":
            self.model = SPOTS_SVG_PTI_ACTP(features)
        elif self.model_name == "SVG_TC":
            self.model = SVG_TC(features)
        elif self.model_name == "SVG_TC_TE":
            self.model = SVG_TC_TE(features)
        elif self.model_name == "VG":
            self.model = VG(features)
        elif self.model_name == "VG_MMMM":
            self.model = VG_MMMM(features)
        elif self.model_name == "SPOTS_VG_ACTP":
            self.model = SPOTS_VG_ACTP(features)
        elif self.model_name == "SPOTS_SVG_ACTP_occ":
            self.model = SPOTS_SVG_ACTP_occ(features)
        elif self.model_name == "SVG_TC_occ":
            self.model = SVG_TC_occ(features)
        elif self.model_name == "SVG_TC_TE_occ":
            self.model = SVG_TC_TE_occ(features)
        elif self.model_name == "SPOTS_SVG_ACTP_STP":
            self.model = SPOTS_SVG_ACTP_STP(features)

        print(features)

        self.model.initialise_model()

        BG = BatchGenerator(self.train_percentage, self.train_data_dir, self.batch_size, self.image_width, self.num_workers)
        self.train_full_loader, self.valid_full_loader = BG.load_full_data()

        if self.criterion == "L1":
            self.criterion = nn.L1Loss()
        if self.criterion == "L2":
            self.criterion = nn.MSELoss()

    def train_full_model(self):
        plot_training_loss = []
        plot_validation_loss = []
        plot_training_save_points = []
        previous_val_mean_loss = 100.0
        best_val_loss = 100.0
        early_stop_clock = 0
        if self.occlusion_test:
            self.start_saving_now = True
            occlusion_gain = 0.0
        progress_bar = tqdm(range(0, self.epochs), total=(self.epochs*len(self.train_full_loader)))
        for epoch in progress_bar:
            # set the stages:
            if epoch <= self.training_stages_epochs[0]:
                self.stage = self.training_stages[0]
            elif epoch <= self.training_stages_epochs[1]:
                self.stage = self.training_stages[1]
            elif epoch <= self.training_stages_epochs[2]:
                self.stage = self.training_stages[2]

            # occlusion edit
            if self.occlusion_test:
                if epoch >= self.occlusion_start_epoch:
                    occlusion_gain += self.occlusion_gain_per_epoch
                    if self.occlusion_max_size < occlusion_gain:
                        if self.start_saving_now:  # reset the validation scores, so that the models will be saved wrt the occlusion at its highest setting.
                            previous_val_mean_loss = 100.0
                            best_val_loss = 100.0
                        self.start_saving_now = False
                        occlusion_gain = self.occlusion_max_size
                    box_size = self.image_width * occlusion_gain
                    BG = BatchGenerator(self.train_percentage, self.train_data_dir, self.batch_size, self.image_width, self.num_workers, self.occlusion_test, box_size)
                    self.train_full_loader, self.valid_full_loader = BG.load_full_data()

            self.model.set_train()

            epoch_mae_losses = 0.0
            epoch_kld_losses = 0.0
            for index, batch_features in enumerate(self.train_full_loader):

                if self.stage == "scene_loss_plus_tactile_gradual_increase":
                    self.gain += 0.00001
                    if self.gain > 0.01:
                        self.gain = 0.01

                if batch_features[1].shape[0] == self.batch_size:
                    mae, kld, mae_tactile, predictions = self.format_and_run_batch(batch_features, test=False)
                    epoch_mae_losses += mae.item()
                    if kld != None:
                        epoch_kld_losses += kld.item()
                        kld = float(kld.item())
                    else:
                        kld = 100
                    if index:
                        mean_mae = epoch_mae_losses / index
                        if kld != None:
                            mean_kld = epoch_kld_losses / index
                    else:
                        mean_kld = 0.0
                        mean_mae = 0.0

                    progress_bar.set_description("epoch: {}, ".format(epoch) + "MAE: {:.4f}, ".format(float(mae.item())) + "kld: {:.4f}, ".format(kld) + "mean MAE: {:.4f}, ".format(mean_mae) + "mean kld: {:.4f}, ".format(mean_kld))
                    progress_bar.update()

            plot_training_loss.append([mean_mae, mean_kld])

            # Validation checking:
            self.model.set_test()
            val_mae_losses = 0.0
            with torch.no_grad():
                for index__, batch_features in enumerate(self.valid_full_loader):
                    if batch_features[1].shape[0] == self.batch_size:
                        val_mae, val_kld, mae_tactile, predictions = self.format_and_run_batch(batch_features, test=True)
                        val_mae_losses += val_mae.item()

            plot_validation_loss.append(val_mae_losses / index__)
            print("Validation mae: {:.4f}, ".format(val_mae_losses / index__))

            # save the train/validation performance data
            np.save(self.model_save_path + "plot_validation_loss", np.array(plot_validation_loss))
            np.save(self.model_save_path + "plot_training_loss", np.array(plot_training_loss))

            # save at the end of every training stage:
            if self.stage != "":
                if epoch-1 in self.training_stages_epochs:
                    self.model.save_model(self.stage)

            # Early stopping:
            if previous_val_mean_loss < val_mae_losses / index__:
                early_stop_clock += 1
                previous_val_mean_loss = val_mae_losses / index__
                # if early_stop_clock == 4 and self.training_stages[0] == "":
                #     print("Early stopping")
                #     break
                # if early_stop_clock == 4 and self.training_stages[0] != "": # skip to the next training phase:
                #     epoch = self.training_stages_epochs[self.training_stages.index(self.stage)]
            else:
                if best_val_loss > val_mae_losses / index__:
                    print("saving model")
                    plot_training_save_points.append(epoch+1)
                    # save the model
                    if self.stage != "":
                        self.model.save_model("best")
                    else:
                        self.model.save_model()
                    best_val_loss = val_mae_losses / index__
                early_stop_clock = 0
                previous_val_mean_loss = val_mae_losses / index__

            np.save(self.model_save_path + "plot_training_loss", np.array(plot_training_save_points))
            lines = list(plot_training_save_points)
            with open (self.model_save_path + "plot_training_loss.txt", 'w') as f:
                for line in lines:
                    f.write("saved after epoch: " + str(line))
                    f.write('\n')

    def format_and_run_batch(self, batch_features, test):
        mae, kld, mae_tactile, predictions = None, None, None, None
        if self.model_name == "SVG":
            images = batch_features[1].permute(1, 0, 4, 3, 2).to(self.device)
            action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.device)
            mae, kld, predictions = self.model.run(scene=images, actions=action, test=test)

        elif self.model_name == "SVG_occ":
            images = batch_features[1].permute(1, 0, 4, 3, 2).to(self.device)
            images_occ = batch_features[6].permute(1, 0, 4, 3, 2).to(self.device)
            action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.device)
            mae, kld, predictions = self.model.run(scene=images_occ, actions=action, scene_gt=images, test=test)

        elif self.model_name == "VG" or self.model_name == "VG_MMMM":
            images = batch_features[1].permute(1, 0, 4, 3, 2).to(self.device)
            action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.device)
            mae, predictions = self.model.run(scene=images, actions=action, test=test)

        elif self.model_name == "SVTG_SE":
            images  = batch_features[1].permute(1, 0, 4, 3, 2).to(self.device)
            tactile = batch_features[2].permute(1, 0, 4, 3, 2).to(self.device)
            action  = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.device)
            scene_and_touch = torch.cat((tactile, images), 2)
            mae, kld, predictions = self.model.run(scene_and_touch=scene_and_touch, actions=action, test=test)

        elif self.model_name == "SVTG_SE_occ":
            images  = batch_features[1].permute(1, 0, 4, 3, 2).to(self.device)
            images_occ = batch_features[6].permute(1, 0, 4, 3, 2).to(self.device)
            tactile = batch_features[2].permute(1, 0, 4, 3, 2).to(self.device)
            action  = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.device)
            scene_and_touch = torch.cat((tactile, images_occ), 2)
            scene_and_touch_gt = torch.cat((tactile, images), 2)
            mae, kld, predictions = self.model.run(scene_and_touch=scene_and_touch, scene_and_touch_gt=scene_and_touch_gt, actions=action, test=test)

        elif self.model_name == "SPOTS_SVG_ACTP" or self.model_name == "SPOTS_SVG_PTI_ACTP" or self.model_name == "SPOTS_SVG_ACTP_STP":
            action  = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.device)
            images  = batch_features[1].permute(1, 0, 4, 3, 2).to(self.device)
            tactile = torch.flatten(batch_features[3].permute(1, 0, 2, 3).to(self.device), start_dim=2)
            mae, kld, mae_tactile, predictions, tactile = self.model.run(scene=images, tactile=tactile, actions=action, gain=self.gain, test=test, stage=self.stage)

        elif self.model_name == "SPOTS_SVG_ACTP_occ":
            action  = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.device)
            images  = batch_features[1].permute(1, 0, 4, 3, 2).to(self.device)
            images_occ  = batch_features[6].permute(1, 0, 4, 3, 2).to(self.device)
            tactile = torch.flatten(batch_features[3].permute(1, 0, 2, 3).to(self.device), start_dim=2)
            mae, kld, mae_tactile, predictions, tactile = self.model.run(scene=images_occ, tactile=tactile, actions=action, scene_gt=images, gain=self.gain, test=test, stage=self.stage)

        if self.model_name == "SPOTS_VG_ACTP":
            action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.device)
            images = batch_features[1].permute(1, 0, 4, 3, 2).to (self.device)
            tactile = torch.flatten (batch_features[3].permute(1, 0, 2, 3).to(self.device), start_dim=2)
            mae, mae_tactile, predictions, tactile = self.model.run(scene=images, tactile=tactile, actions=action,gain=self.gain, test=test, stage=self.stage)

        elif self.model_name == "SVG_TC" or self.model_name == "SVG_TC_TE":
            images  = batch_features[1].permute(1, 0, 4, 3, 2).to(self.device)
            tactile = torch.flatten(batch_features[3].permute(1, 0, 2, 3).to(self.device), start_dim=2)
            action  = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.device)
            mae, kld, predictions = self.model.run(scene=images, tactile=tactile, actions=action, gain=self.gain, test=test, stage=self.stage)

        elif self.model_name == "SVG_TC_occ" or self.model_name == "SVG_TC_TE_occ":
            images  = batch_features[1].permute(1, 0, 4, 3, 2).to(self.device)
            images_occ = batch_features[6].permute(1, 0, 4, 3, 2).to(self.device)
            tactile = torch.flatten(batch_features[3].permute(1, 0, 2, 3).to(self.device), start_dim=2)
            action  = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.device)
            mae, kld, predictions = self.model.run(scene=images_occ, tactile=tactile, actions=action, scene_gt=images, gain=self.gain, test=test, stage=self.stage)

        return mae, kld, mae_tactile, predictions


@click.command()
@click.option('--model_name', type=click.Path(), default="SVG_occ", help='Set name for prediction model, SVG, SVTG_SE, SPOTS_SVG_ACTP, SVG_TC')
@click.option('--batch_size', type=click.INT, default=64, help='Batch size for training.')
@click.option('--lr', type=click.FLOAT, default = 0.0001, help = "learning rate")
@click.option('--beta1', type=click.FLOAT, default = 0.9, help = "Beta gain")
@click.option('--log_dir', type=click.Path(), default = 'logs/lp', help = "Not sure :D")
@click.option('--optimizer', type=click.Path(), default = 'adam', help = "what optimiser to use - only adam available currently")
@click.option('--niter', type=click.INT, default = 300, help = "")
@click.option('--seed', type=click.INT, default = 1, help = "")
@click.option('--image_width', type=click.INT, default = 64, help = "Size of scene image data")
@click.option('--dataset', type=click.Path(), default = 'object1_motion1_position1', help = "name of the dataset")
@click.option('--n_past', type=click.INT, default = 2, help = "context sequence length")
@click.option('--n_future', type=click.INT, default = 5, help = "time horizon sequence length")
@click.option('--n_eval', type=click.INT, default = 7, help = "sum of context and time horizon")
@click.option('--prior_rnn_layers', type=click.INT, default = 3, help = "number of LSTMs in the prior model")
@click.option('--posterior_rnn_layers', type=click.INT, default = 3, help = "number of LSTMs in the posterior model")
@click.option('--predictor_rnn_layers', type=click.INT, default = 4, help = "number of LSTMs in the frame predictor model")
@click.option('--state_action_size', type=click.INT, default = 12, help = "size of action conditioning data")
@click.option('--z_dim', type=click.INT, default = 10, help = "number of latent variables to estimate")
@click.option('--beta', type=click.FLOAT, default = 0.0001, help = "beta gain")
@click.option('--data_threads', type=click.INT, default = 5, help = "")
@click.option('--num_digits', type=click.INT, default = 2, help = "")
@click.option('--last_frame_skip', type=click.Path(), default = 'store_true', help = "")
@click.option('--epochs', type=click.INT, default = 100, help = "number of epochs to run for ")
@click.option('--train_percentage', type=click.FLOAT, default = 0.9, help = "")
@click.option('--validation_percentage', type=click.FLOAT, default = 0.1, help = "")
@click.option('--criterion', type=click.Path(), default = "L1", help = "")
@click.option('--tactile_size', type=click.INT, default = 0, help = "size of tacitle frame - 48, if no tacitle data set to 0")
@click.option('--g_dim', type=click.INT, default = 256, help = "size of encoded data for input to prior")
@click.option('--rnn_size', type=click.INT, default = 256, help = "size of encoded data for input to frame predictor (g_dim = rnn-size)")
@click.option('--channels', type=click.INT, default = 3, help = "input channels")
@click.option('--out_channels', type=click.INT, default = 3, help = "output channels")
@click.option('--training_stages', type=click.Path(), default = "", help = "define the training stages - if none leave blank - available: 3part")
@click.option('--training_stages_epochs', type=click.Path(), default = "50,75,125", help = "define the end point of each training stage")
@click.option('--num_workers', type=click.INT, default = 12, help = "number of workers used by the data loader")
@click.option('--model_save_path', type=click.Path(), default = "/home/willmandil/Robotics/SPOTS/models/universal_models/saved_models/", help = "")
@click.option('--train_data_dir', type=click.Path(), default = "/home/willmandil/Robotics/Data_sets/PRI/object1_motion1_position1/train_formatted/", help = "")
@click.option('--scaler_dir', type=click.Path(), default = "/home/willmandil/Robotics/Data_sets/PRI/object1_motion1_position1/scalars/", help = "")
@click.option('--model_name_save_appendix', type=click.Path(), default = "", help = "What to add to the save file to identify the model as a specific subset (_1c= 1 conditional frame, GTT=groundtruth tactile data)")
@click.option('--tactile_encoder_hidden_size', type=click.INT, default = 0, help = "Size of hidden layer in tactile encoder, 200")
@click.option('--tactile_encoder_output_size', type=click.INT, default = 0, help = "size of output layer from tactile encoder, 100")
@click.option('--occlusion_test', type=click.BOOL, default = False, help = "if you would like to train for occlusion")
@click.option('--occlusion_gain_per_epoch', type=click.FLOAT, default = 0.05, help = "increasing size of the occlusion block per epoch 0.1=(0.1 x MAX) each epoch")
@click.option('--occlusion_start_epoch', type=click.INT, default = 35, help = "size of output layer from tactile encoder, 100")
@click.option('--occlusion_max_size', type=click.FLOAT, default = 0.4, help = "max size of the window as a % of total size (0.5 = 50% of frame (32x32 squares in ))")
def main(model_name, batch_size, lr, beta1, log_dir, optimizer, niter, seed, image_width, dataset,
         n_past, n_future, n_eval, prior_rnn_layers, posterior_rnn_layers, predictor_rnn_layers, state_action_size,
         z_dim, beta, data_threads, num_digits, last_frame_skip, epochs, train_percentage, validation_percentage,
         criterion, tactile_size, g_dim, rnn_size, channels, out_channels, training_stages, training_stages_epochs,
         num_workers, model_save_path, train_data_dir, scaler_dir, model_name_save_appendix, tactile_encoder_hidden_size,
         tactile_encoder_output_size, occlusion_test, occlusion_gain_per_epoch, occlusion_start_epoch, occlusion_max_size):

    # Home PC:
    # /home/user/Robotics/SPOTS/models/universal_models/saved_models/
    # /home/user/Robotics/Data_sets/PRI/object1_motion1/train_formatted/
    # /home/user/Robotics/Data_sets/PRI/object1_motion1/scalars/

    # Lab PC
    # /home/willmandil/Robotics/SPOTS/models/universal_models/saved_models/
    # /home/willmandil/Robotics/Data_sets/PRI/object1_motion1_position1/train_formatted/
    # /home/willmandil/Robotics/Data_sets/PRI/object1_motion1_position1/scalars/

    # unique save title:
    model_save_path = model_save_path + model_name
    try:
        os.mkdir(model_save_path)
    except FileExistsError or FileNotFoundError:
        pass
    try:
        model_save_path = model_save_path + "/model_" + datetime.now().strftime("%d_%m_%Y_%H_%M/")
        os.mkdir(model_save_path)
    except FileExistsError or FileNotFoundError:
        pass

    model_dir = model_save_path
    data_root = train_data_dir

    if model_name == "SVG" or model_name == "SVG_TC" or model_name == "VG" or model_name == "VG_MMMM" or model_name == "SVG_occ" or model_name == "SVG_TC_occ":
        g_dim = 256  # 128
        rnn_size = 256
        channels = 3
        out_channels = 3
        training_stages = [""]
        training_stages_epochs = [epochs]
    elif model_name == "SVTG_SE" or model_name == "SVTG_SE_occ":
        g_dim = 256 * 2
        rnn_size = 256 * 2
        channels = 6
        out_channels = 6
        training_stages = [""]
        training_stages_epochs = [epochs]
        tactile_size = [image_width, image_width]
    elif model_name == "SPOTS_SVG_ACTP" or model_name == "SPOTS_VG_ACTP" or model_name == "SPOTS_SVG_ACTP_occ" or model_name == "SPOTS_SVG_PTI_ACTP"  or model_name == "SPOTS_SVG_ACTP_STP":
        g_dim = 256
        rnn_size = 256
        channels = 3
        out_channels = 3
        training_stages = ["scene_only", "tactile_loss_plus_scene_fixed", "scene_loss_plus_tactile_gradual_increase"]
        training_stages_epochs = [35, 65, 150]
        # training_stages_epochs = [50, 75, 125]
        tactile_size = 48
        epochs = training_stages_epochs[-1] + 1
    elif model_name == "SVG_TC_TE" or model_name == "SVG_TC_TE_occ":
        g_dim = 256
        rnn_size = 256
        channels = 3
        out_channels = 3
        training_stages = [""]
        training_stages_epochs = [epochs]
        tactile_size = 48
        tactile_encoder_hidden_size = 200
        tactile_encoder_output_size = 100

    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available

    features = {"lr": lr, "beta1": beta1, "batch_size": batch_size, "log_dir": log_dir, "model_dir": model_dir, "data_root": data_root, "optimizer": optimizer, "niter": niter, "seed": seed,
                "image_width": image_width, "channels": channels, "out_channels": out_channels, "dataset": dataset, "n_past": n_past, "n_future": n_future, "n_eval": n_eval, "rnn_size": rnn_size, "prior_rnn_layers": prior_rnn_layers,
                "posterior_rnn_layers": posterior_rnn_layers, "predictor_rnn_layers": predictor_rnn_layers, "state_action_size": state_action_size, "z_dim": z_dim, "g_dim": g_dim, "beta": beta, "data_threads": data_threads, "num_digits": num_digits,
                "last_frame_skip": last_frame_skip, "epochs": epochs, "train_percentage": train_percentage, "validation_percentage": validation_percentage, "criterion": criterion, "model_name": model_name,
                "train_data_dir": train_data_dir, "scaler_dir": scaler_dir, "device": device, "training_stages":training_stages, "training_stages_epochs": training_stages_epochs, "tactile_size":tactile_size, "num_workers":num_workers,
                "model_save_path":model_save_path, "model_name_save_appendix":model_name_save_appendix, "tactile_encoder_hidden_size":tactile_encoder_hidden_size, "tactile_encoder_output_size":tactile_encoder_output_size,
                "occlusion_test": occlusion_test, "occlusion_gain_per_epoch":occlusion_gain_per_epoch, "occlusion_start_epoch":occlusion_start_epoch, "occlusion_max_size":occlusion_max_size}

    # save features
    w = csv.writer(open(model_save_path + "/features.csv", "w"))
    for key, val in features.items():
        w.writerow([key, val])

    UMT = UniversalModelTrainer(features)
    UMT.train_full_model()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    main ()
