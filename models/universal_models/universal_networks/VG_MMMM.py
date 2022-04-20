# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import cv2
import copy
# from utils import init_weights
import universal_networks.utils as utility_prog
import numpy as np

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

class Model:
    def __init__(self, features):
        self.features = features
        self.lr = features["lr"]
        self.beta1 = features["beta1"]
        self.batch_size = features["batch_size"]
        self.log_dir = features["log_dir"]
        self.model_dir = features["model_dir"]
        self.data_root = features["data_root"]
        self.optimizer = features["optimizer"]
        self.niter = features["niter"]
        self.seed = features["seed"]
        self.image_width = features["image_width"]
        self.channels = features["channels"]
        self.out_channels = features["out_channels"]
        self.dataset = features["dataset"]
        self.n_past = features["n_past"]
        self.n_future = features["n_future"]
        self.n_eval = features["n_eval"]
        self.rnn_size = features["rnn_size"]
        self.prior_rnn_layers = features["prior_rnn_layers"]
        self.posterior_rnn_layers = features["posterior_rnn_layers"]
        self.predictor_rnn_layers = features["predictor_rnn_layers"]
        self.state_action_size = features["state_action_size"]
        self.z_dim = features["z_dim"]
        self.g_dim = features["g_dim"]
        self.beta = features["beta"]
        self.data_threads = features["data_threads"]
        self.num_digits = features["num_digits"]
        self.last_frame_skip = features["last_frame_skip"]
        self.epochs = features["epochs"]
        self.train_percentage = features["train_percentage"]
        self.validation_percentage = features["validation_percentage"]
        self.criterion = features["criterion"]
        self.model_name = features["model_name"]
        self.device = features["device"]
        self.model_name_save_appendix = features["model_name_save_appendix"]

        if self.optimizer == "adam" or self.optimizer == "Adam":
            self.optimizer = optim.Adam

        if self.criterion == "L1":
            self.mae_criterion = nn.L1Loss()
        if self.criterion == "L2":
            self.mae_criterion = nn.MSELoss()


    def load_model(self, full_model):
        self.frame_predictor = full_model["frame_predictor"]
        self.MMFM_scene = full_model["MMFM_scene"]
        self.encoder = full_model["encoder"]
        self.decoder = full_model["decoder"]

        self.frame_predictor.cuda()
        self.MMFM_scene.cuda()
        self.encoder.cuda()
        self.decoder.cuda()
        self.mae_criterion.cuda()

    def initialise_model(self):
        from universal_networks.lstm import lstm
        from universal_networks.lstm import gaussian_lstm
        import universal_networks.dcgan_64 as model

        self.frame_predictor = lstm(self.g_dim + self.state_action_size, self.g_dim, self.rnn_size, self.predictor_rnn_layers, self.batch_size)
        self.frame_predictor.apply(utility_prog.init_weights)

        self.MMFM_scene = model.MMFM_scene(self.g_dim, self.g_dim, self.channels)
        self.MMFM_scene.apply(utility_prog.init_weights)

        self.encoder = model.encoder(self.g_dim, self.channels)
        self.decoder = model.decoder(self.g_dim, self.channels)
        self.encoder.apply(utility_prog.init_weights)
        self.decoder.apply(utility_prog.init_weights)

        self.frame_predictor_optimizer = self.optimizer(self.frame_predictor.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.MMFM_scene_optimizer = self.optimizer(self.MMFM_scene.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.encoder_optimizer = self.optimizer(self.encoder.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.decoder_optimizer = self.optimizer(self.decoder.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.frame_predictor.cuda()
        self.MMFM_scene.cuda()
        self.encoder.cuda()
        self.decoder.cuda()
        self.mae_criterion.cuda()

    def save_model(self):
        torch.save({'encoder': self.encoder, 'decoder': self.decoder, 'frame_predictor': self.frame_predictor, "MMFM_scene": self.MMFM_scene, 'features': self.features}, self.model_dir + "VG_MMMM_model" + self.model_name_save_appendix)

    def set_train(self):
        self.frame_predictor.train()
        self.MMFM_scene.train()
        self.encoder.train()
        self.decoder.train()

    def set_test(self):
        self.frame_predictor.eval()
        self.MMFM_scene.eval()
        self.encoder.eval()
        self.decoder.eval()

    def run(self, scene, actions, test=False):
        mae, kld = 0, 0
        outputs = []

        self.frame_predictor.zero_grad()
        self.MMFM_scene.zero_grad()
        self.encoder.zero_grad()
        self.decoder.zero_grad()

        self.frame_predictor.hidden = self.frame_predictor.init_hidden()

        state = actions[0].to(self.device)
        for index, (sample_sscene, sample_action) in enumerate(zip(scene[:-1], actions[1:])):
            state_action = torch.cat((state, sample_action), 1)

            if index > self.n_past - 1:  # horizon
                h, skip = self.encoder(x_pred)
                h_target = self.encoder(scene[index + 1])[0]

                h_m = self.MMFM_scene(h)

                h_pred = self.frame_predictor(torch.cat([h_m, state_action], 1))  # prediction model
                x_pred = self.decoder([h_pred, skip])  # prediction model

                mae += self.mae_criterion(x_pred, scene[index + 1])  # prediction model

                outputs.append(x_pred)
            else:  # context
                h, skip = self.encoder(scene[index])
                h_target = self.encoder(scene[index + 1])[0]

                h_m = self.MMFM_scene(h)

                h_pred = self.frame_predictor(torch.cat([h_m, state_action], 1))  # prediction model
                x_pred = self.decoder([h_pred, skip])  # prediction model

                mae += self.mae_criterion(x_pred, scene[index + 1])  # prediction model

                last_output = x_pred

        outputs = [last_output] + outputs

        if test is False:
            loss = mae
            loss.backward()
            self.frame_predictor_optimizer.step()
            self.MMFM_scene_optimizer.step()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

        return mae.data.cpu().numpy() / (self.n_past + self.n_future), torch.stack(outputs)

    def kl_criterion(self, mu1, logvar1, mu2, logvar2):
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2)) - 1 / 2
        return kld.sum() / self.batch_size
