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

model_path      = "/home/user/Robotics/SPOTS/models/SVG/saved_models/PRI_single_object_purple_model_09_02_2022_14_47/SVG_model.zip"
data_save_path  = "/home/user/Robotics/SPOTS/models/SVG/saved_models/PRI_single_object_purple_model_09_02_2022_14_47/"
test_data_dir   = "/home/user/Robotics/Data_sets/PRI/single_object_purple/test_novel_formatted/"
scaler_dir      = "/home/user/Robotics/Data_sets/PRI/single_object_purple/scalars/"

context_frames = 10
sequence_length = 20

(lr, beta1, batch_size, log_dir, model_dir, name, data_root, optimizer, niter, seed, epoch_size,
image_width, channels, out_channels, dataset, n_past, n_future, n_eval, rnn_size, prior_rnn_layers,
posterior_rnn_layers, predictor_rnn_layers, z_dim, g_dim, beta, data_threads, num_digits,
last_frame_skip, epochs, _, __) = torch.load(model_path)["features"]

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#  use gpu if available


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
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=6)
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

        images = []
        for image_name in np.load(self.test_data_dir + value[2]):
            images.append(np.load(self.test_data_dir + image_name))

        experiment_number = np.load(self.test_data_dir + value[3])
        time_steps = np.load(self.test_data_dir + value[4])
        return [robot_data.astype(np.float32), np.array(images).astype(np.float32), experiment_number, time_steps]


class ModelTester:
    def __init__(self):
        self.full_model = torch.load(model_path)

        (lr, beta1, batch_size, log_dir, model_dir, name, data_root, optimizer, niter, seed, epoch_size,
        image_width, channels, out_channels, dataset, n_past, n_future, n_eval, rnn_size, prior_rnn_layers,
        posterior_rnn_layers, predictor_rnn_layers, z_dim, g_dim, beta, data_threads, num_digits,
        last_frame_skip, epochs, _, __) = self.full_model["features"]


        self.optimizer = optim.Adam
        self.frame_predictor = self.full_model["frame_predictor"]
        self.posterior = self.full_model["posterior"]
        self.prior = self.full_model["prior"]
        self.encoder = self.full_model["encoder"]
        self.decoder = self.full_model["decoder"]

        self.mae_criterion = nn.L1Loss()

        self.frame_predictor.cuda()
        self.posterior.cuda()
        self.prior.cuda()
        self.encoder.cuda()
        self.decoder.cuda()
        self.mae_criterion.cuda()

        self.criterion = nn.L1Loss()

    def run(self, scene, actions, test=False):
        mae, kld = 0, 0
        outputs = []

        self.frame_predictor.zero_grad()
        self.posterior.zero_grad()
        self.prior.zero_grad()
        self.encoder.zero_grad()
        self.decoder.zero_grad()

        self.frame_predictor.hidden = self.frame_predictor.init_hidden()
        self.posterior.hidden = self.posterior.init_hidden()
        self.prior.hidden = self.prior.init_hidden()

        state = actions[0].to(device)
        for index, (sample_sscene, sample_action) in enumerate(zip(scene[:-1], actions[1:])):
            state_action = torch.cat((state, actions[index]), 1)

            if index > n_past - 1:  # horizon
                h, skip = self.encoder(x_pred)
                h_target = self.encoder(scene[index + 1])[0]

                _, mu, logvar = self.posterior(h_target)  # learned prior
                z_t, mu_p, logvar_p = self.prior(h)  # learned prior

                h_pred = self.frame_predictor(torch.cat([h, z_t, state_action], 1))  # prediction model
                x_pred = self.decoder([h_pred, skip])  # prediction model
                outputs.append(x_pred)
            else:  # context
                h, skip = self.encoder(scene[index])
                h_target = self.encoder(scene[index + 1])[0]

                _, mu, logvar = self.posterior(h_target)  # learned prior
                z_t, mu_p, logvar_p = self.prior(h)  # learned prior

                h_pred = self.frame_predictor(torch.cat([h, z_t, state_action], 1))  # prediction model
                x_pred = self.decoder([h_pred, skip])  # prediction model
                last_output = x_pred

        outputs = [last_output] + outputs

        return torch.stack(outputs)

    def kl_criterion(self, mu1, logvar1, mu2, logvar2):
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
        self.prediction_data  = []
        self.tg_back_scaled   = []
        self.tp1_back_scaled = []
        self.tp5_back_scaled = []
        self.tp10_back_scaled = []
        self.current_exp      = 0
        self.objects = []
        self.total_losses = []

        for trial in self.number_of_trials:
            BG = BatchGenerator(test_data_dir, batch_size, trial)
            self.test_full_loader = BG.load_full_data()

            for index, batch_features in enumerate(self.test_full_loader):
                if batch_features[1].shape[0] == batch_size:
                    images = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
                    action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
                    scene_prediction = self.run(scene=images, actions=action, test=True)

            #         for i in range(10):
            #             plt.figure(1)
            #             f, axarr = plt.subplots(1, 2)
            #             axarr[0].set_title("predictions: t_" + str(i))
            #             axarr[0].imshow(np.array(scene_prediction[i][7].permute(1, 2, 0).cpu().detach()))
            #             axarr[1].set_title("ground truth: t_" + str(i))
            #             axarr[1].imshow(np.array(images[context_frames+i][7].permute(1, 2, 0).cpu().detach()))
            #             plt.savefig(data_save_path + str(i) + ".png")
            #         break
            #     break
            # break

            mae = self.mae_criterion(scene_prediction, images[context_frames:]).data

            # calculate trial loss:
            self.total_losses.append(["trial number JUST T10: " + str(trial), np.array(mae.cpu().detach())])

        np.save(data_save_path + "test_novel_losses_per_trial", np.array(self.total_losses))

        print(self.total_losses)

        final_loss = sum([i[1] for i in self.total_losses]) / len(self.total_losses)

        np.save(data_save_path + "test_novel_loss", np.array(final_loss))
        print(np.array(self.total_losses))
        print(final_loss)


if __name__ == "__main__":
    MT = ModelTester()
    MT.test_full_model()

