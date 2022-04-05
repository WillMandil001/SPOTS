# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import cv2
import copy
import math
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from matplotlib.animation import FuncAnimation

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from tqdm import tqdm
from pickle import load
from datetime import datetime
from torch.utils.data import Dataset

import os
import csv
import copy
import utils
import numpy as np

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

model_path      = "/home/wmandil/Robotics/SPOTS/SPOTS/models/SVG_ACTP_SPOTS/saved_models/PRI_single_object_purple_SPOTS_SVG_ACTP_model_03_03_2022_14_22/SVG_SPOTS_3S_SOP_stage1"
data_save_path  = "/home/wmandil/Robotics/SPOTS/SPOTS/models/SVG_ACTP_SPOTS/saved_models/PRI_single_object_purple_SPOTS_SVG_ACTP_model_03_03_2022_14_22/"
test_data_dir   = "/home/wmandil/Robotics/Data_sets/PRI/single_object_purple/single_object_purple_formatted/test_no_new_formatted/"
scaler_dir      = "/home/wmandil/Robotics/Data_sets/PRI/single_object_purple/single_object_purple_formatted/scalars/"

context_frames = 10
sequence_length = 20

(lr, beta1, batch_size, log_dir, model_dir, name, data_root, optimizer, niter, seed, epoch_size,
            image_width, channels, out_channels, dataset, n_past, n_future, n_eval, rnn_size, prior_rnn_layers,
            posterior_rnn_layers, predictor_rnn_layers, z_dim, g_dim, beta, data_threads, num_digits,
            last_frame_skip, epochs, train_percentage, validation_percentage, loss_function) = torch.load(model_path)["features"]

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #  use gpu if available


class BatchGenerator:
    def __init__(self, test_data_dir, batch_size, trial_number):
        self.batch_size = batch_size
        self.trial_number = trial_number
        self.test_data_dir = test_data_dir + 'test_trial_' + str(self.trial_number) + '/'
        self.data_map = []
        print(self.test_data_dir + 'map_' + str(self.trial_number) + '.csv')
        with open(self.test_data_dir + 'map_' + str(self.trial_number) + '.csv', 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                self.data_map.append(row)

    def load_full_data(self):
        dataset_test = FullDataSet(self.data_map, self.test_data_dir)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=1)
        self.data_map = []
        return test_loader


class FullDataSet:
    def __init__(self, data_map, test_data_dir):
        self.test_data_dir = test_data_dir
        self.samples = data_map[1:]
        data_map = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        value = self.samples[idx]
        robot_data = np.load(self.test_data_dir + value[0])

        tactile_data = np.load(self.test_data_dir + value[1])

        images = []
        for image_name in np.load(self.test_data_dir + value[2]):
            images.append(np.load(self.test_data_dir + image_name))

        experiment_number = np.load(self.test_data_dir + value[3])
        time_steps = np.load(self.test_data_dir + value[4])
        return [robot_data.astype(np.float32), np.array(images).astype(np.float32), np.array(tactile_data).astype(np.float32), experiment_number, time_steps]



class ModelTester:
    def __init__(self):
        self.full_model = torch.load(model_path)

        (lr, beta1, batch_size, log_dir, model_dir, name, data_root, optimizer, niter, seed, epoch_size,
            image_width, channels, out_channels, dataset, n_past, n_future, n_eval, rnn_size, prior_rnn_layers,
            posterior_rnn_layers, predictor_rnn_layers, z_dim, g_dim, beta, data_threads, num_digits,
            last_frame_skip, epochs, train_percentage, validation_percentage, loss_function) = self.full_model["features"]

        batch_size = 8

        self.optimizer = optim.Adam
        self.frame_predictor_scene = self.full_model["frame_predictor_scene"]
        self.frame_predictor_tactile = self.full_model["frame_predictor_tactile"]
        self.encoder_scene = self.full_model["encoder_scene"]
        self.decoder_scene = self.full_model["decoder_scene"]
        self.prior = self.full_model["prior"]
        self.posterior = self.full_model["posterior"]
        self.MMFM_scene = self.full_model["MMFM_scene"]
        self.MMFM_tactile = self.full_model["MMFM_tactile"]

        self.mae_criterion_tactile = nn.L1Loss()
        self.mae_criterion_scene = nn.L1Loss()
        self.mae_criterion = nn.L1Loss()

        self.frame_predictor_scene.cuda()
        self.frame_predictor_tactile.cuda()
        self.encoder_scene.cuda()
        self.decoder_scene.cuda()
        self.prior.cuda()
        self.posterior.cuda()
        self.mae_criterion_tactile.cuda()
        self.mae_criterion_scene.cuda()
        self.mae_criterion.cuda()

        self.criterion = nn.L1Loss()

    def run(self, scene, tactile, actions, test=False, stage=False):
        mae_tactile = 0
        kld_tactile = 0
        mae_scene = 0
        kld_scene = 0
        outputs = []

        # scene
        self.frame_predictor_scene.zero_grad()
        self.encoder_scene.zero_grad()
        self.decoder_scene.zero_grad()
        self.frame_predictor_scene.hidden = self.frame_predictor_scene.init_hidden()

        # tactile
        self.frame_predictor_tactile.zero_grad()
        self.frame_predictor_tactile.init_hidden(scene.shape[1])

        self.MMFM_scene.zero_grad()
        self.MMFM_tactile.zero_grad()

        # prior
        self.posterior.zero_grad()
        self.prior.zero_grad()
        self.posterior.hidden = self.posterior.init_hidden()
        self.prior.hidden = self.prior.init_hidden()

        state = actions[0].to(device)
        for index, (sample_scene, sample_tactile, sample_action) in enumerate(zip(scene[:-1], tactile[:-1], actions[1:])):
            state_action = torch.cat((state, actions[index]), 1)

            if index > n_past - 1:  # horizon
                # Scene Encoding
                h_scene, skip_scene = self.encoder_scene(x_pred_scene)
                h_target_scene      = self.encoder_scene(scene[index + 1])[0]

                # cat scene and tactile together for crossover input to pipelines
                h_scene_and_tactile = torch.cat([x_pred_tactile, h_scene], 1)

                # Learned Prior - Z_t calculation
                if test:
                    _, mu, logvar = self.posterior(h_target_scene)  # learned prior
                    z_t, mu_p, logvar_p = self.prior(h_scene)  # learned prior
                else:
                    z_t, mu, logvar = self.posterior(h_target_scene)  # learned prior
                    _, mu_p, logvar_p = self.prior(h_scene)  # learned prior

                # Multi-modal feature model:
                MM_rep_scene = self.MMFM_scene(h_scene_and_tactile)
                MM_rep_tactile = self.MMFM_tactile(h_scene_and_tactile)

                # Tactile Prediction
                x_pred_tactile= self.frame_predictor_tactile(MM_rep_tactile, state_action, x_pred_tactile)  # prediction model

                # Scene Prediction
                h_pred_scene = self.frame_predictor_scene(torch.cat([MM_rep_scene, z_t, state_action], 1))  # prediction model
                x_pred_scene = self.decoder_scene([h_pred_scene, skip_scene])  # prediction model

                # loss calulations for tactile and scene:
                mae_tactile += self.mae_criterion_tactile(x_pred_tactile, tactile[index + 1])  # prediction model

                mae_scene += self.mae_criterion_scene(x_pred_scene, scene[index + 1])  # prediction model
                kld_scene += self.kl_criterion_scene(mu, logvar, mu_p, logvar_p)  # learned prior

                outputs.append(x_pred_scene)

            else:  # context
                # Scene Encoding
                h_scene, skip_scene = self.encoder_scene(scene[index])
                h_target_scene      = self.encoder_scene(scene[index + 1])[0]

                # cat scene and tactile together for crossover input to pipelines
                h_scene_and_tactile = torch.cat([tactile[index], h_scene], 1)

                # Learned Prior - Z_t calculation
                if test:
                    _, mu, logvar = self.posterior(h_target_scene)  # learned prior
                    z_t, mu_p, logvar_p = self.prior(h_scene)  # learned prior
                else:
                    z_t, mu, logvar = self.posterior(h_target_scene)  # learned prior
                    _, mu_p, logvar_p = self.prior(h_scene)  # learned prior

                # Multi-modal feature model:
                MM_rep_scene = self.MMFM_scene(h_scene_and_tactile)
                MM_rep_tactile = self.MMFM_tactile(h_scene_and_tactile)

                # Tactile Prediction
                x_pred_tactile = self.frame_predictor_tactile(MM_rep_tactile, state_action, tactile[index])  # prediction model

                # Scene Prediction
                h_pred_scene = self.frame_predictor_scene(torch.cat([MM_rep_scene, z_t, state_action], 1))  # prediction model
                x_pred_scene = self.decoder_scene([h_pred_scene, skip_scene])  # prediction model

                # loss calulations for tactile and scene:
                mae_tactile += self.mae_criterion_tactile(x_pred_tactile, tactile[index + 1])  # prediction model

                mae_scene += self.mae_criterion_scene(x_pred_scene, scene[index + 1])  # prediction model
                kld_scene += self.kl_criterion_scene(mu, logvar, mu_p, logvar_p)  # learned prior

                last_output = x_pred_scene

        outputs = [last_output] + outputs

        return torch.stack(outputs)

    def kl_criterion_scene(self, mu1, logvar1, mu2, logvar2):
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2)) - 1 / 2
        return kld.sum() / batch_size

    def kl_criterion_tactile(self, mu1, logvar1, mu2, logvar2):
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2)) - 1 / 2
        return kld.sum() / batch_size

    def test_full_model(self):
        self.objects = []
        self.performance_data = []
        self.prediction_data = []
        self.tg_back_scaled = []
        self.tp1_back_scaled = []
        self.tp5_back_scaled = []
        self.tp10_back_scaled = []
        self.current_exp = 0

        self.number_of_trials = [i for i in range(30)]
        self.performance_data = []
        self.prediction_data = []
        self.tg_back_scaled = []
        self.tp1_back_scaled = []
        self.tp5_back_scaled = []
        self.tp10_back_scaled = []
        self.current_exp = 0
        self.objects = []

        self.total_losses = []

        for trial in self.number_of_trials:
            BG = BatchGenerator(test_data_dir, batch_size, trial)
            self.test_full_loader = BG.load_full_data()

            for index, batch_features in enumerate(self.test_full_loader):
                if batch_features[1].shape[0] == batch_size:
                    images = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
                    tactile = torch.flatten(batch_features[2].permute(1, 0, 2, 3).to(device), start_dim=2)
                    action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
                    predictions = self.run(scene=images, tactile=tactile, actions=action, test=True)

            #         for i in range(10):
            #             plt.figure(1)
            #             f, axarr = plt.subplots(1, 2)
            #             axarr[0].set_title("predictions: t_" + str(i))
            #             axarr[0].imshow(np.array(predictions[i][7].permute(1, 2, 0).cpu().detach()))
            #             axarr[1].set_title("ground truth: t_" + str(i))
            #             axarr[1].imshow(np.array(images[context_frames + i][7].permute(1, 2, 0).cpu().detach()))
            #             plt.savefig(data_save_path + str(i) + ".png")
            #         break
            #     break
            # break

            mae = self.mae_criterion(predictions, images[context_frames:]).data
        
            # calculate trial loss:
            self.total_losses.append(["trial number: " + str(trial), np.array(mae.cpu().detach())])

        np.save(data_save_path + "stage_1_test_no_new_losses_per_trial", np.array(self.total_losses))
        
        print(self.total_losses)
        
        final_loss = sum([i[1] for i in self.total_losses]) / len(self.total_losses)
        
        np.save(data_save_path + "stage_1_test_no_new_loss", np.array(final_loss))
        print(np.array(self.total_losses))
        print(final_loss)

if __name__ == "__main__":
    MT = ModelTester()
    MT.test_full_model()

