# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import cv2
import copy
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

# lr=0.0001
# beta1=0.9
# batch_size=8
# log_dir='logs/lp'
# model_dir=''
# name=''
# data_root='data'
# optimizer='adam'
# niter=300
# seed=1
# epoch_size=600
# image_width=64
# channels=6
# out_channels = 6
# dataset='smmnist'
# n_past=10
# n_future=10
# n_eval=20
# rnn_size=256*2
# prior_rnn_layers=3
# posterior_rnn_layers=3
# predictor_rnn_layers=4
# state_action_size = 12
# z_dim=10  # number of latent variables
# g_dim=256*2  # 128
# beta=0.0001  # was 0.0001
# data_threads=5
# num_digits=2
# last_frame_skip = 'store_true'
# epochs = 100

# train_percentage = 0.9
# validation_percentage = 0.1


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

        if self.optimizer == "adam" or self.optimizer == "Adam":
            self.optimizer = optim.Adam

        import universal_networks.lstm as lstm_models
        self.frame_predictor = lstm_models.lstm(self.g_dim + self.z_dim + self.state_action_size, self.g_dim, self.rnn_size, self.predictor_rnn_layers, self.batch_size)
        self.posterior = lstm_models.gaussian_lstm(self.g_dim, self.z_dim, self.rnn_size, self.posterior_rnn_layers, self.batch_size)
        self.prior = lstm_models.gaussian_lstm(self.g_dim, self.z_dim, self.rnn_size, self.prior_rnn_layers, self.batch_size)
        self.frame_predictor.apply(utility_prog.init_weights)
        self.posterior.apply(utility_prog.init_weights)
        self.prior.apply(utility_prog.init_weights)

        import universal_networks.dcgan_64 as model
        self.encoder = model.encoder(self.g_dim, self.channels)
        self.decoder = model.decoder(self.g_dim, self.channels)
        self.encoder.apply(utility_prog.init_weights)
        self.decoder.apply(utility_prog.init_weights)

        self.frame_predictor_optimizer = self.optimizer(self.frame_predictor.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.posterior_optimizer = self.optimizer(self.posterior.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.prior_optimizer = self.optimizer(self.prior.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.encoder_optimizer = self.optimizer(self.encoder.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.decoder_optimizer = self.optimizer(self.decoder.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.mae_criterion = nn.L1Loss()

        self.frame_predictor.cuda()
        self.posterior.cuda()
        self.prior.cuda()
        self.encoder.cuda()
        self.decoder.cuda()
        self.mae_criterion.cuda()

    def save_model(self):
        torch.save({'encoder': self.encoder, 'decoder': self.decoder, 'frame_predictor': self.frame_predictor,
                    'posterior': self.posterior, 'prior': self.prior, 'features': self.features},
                    self.model_dir + "SVG_model")

    def set_train(self):
        self.frame_predictor.train()
        self.posterior.train()
        self.prior.train()
        self.encoder.train()
        self.decoder.train()

    def set_test(self):
        self.frame_predictor.eval()
        self.posterior.eval()
        self.prior.eval()
        self.encoder.eval()
        self.decoder.eval()

    def run(self, scene_and_touch, actions, test=False):
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

        state = actions[0].to(self.device)
        for index, (sample_scene_and_touch, sample_action) in enumerate(zip(scene_and_touch[:-1], actions[1:])):
            state_action = torch.cat((state, actions[index]), 1)

            if index > self.n_past - 1:  # horizon
                h, skip = self.encoder(x_pred)
                h_target = self.encoder(scene_and_touch[index + 1])[0]

                if test:
                    _, mu, logvar = self.posterior(h_target)  # learned prior
                    z_t, mu_p, logvar_p = self.prior(h)  # learned prior
                else:
                    z_t, mu, logvar = self.posterior(h_target)  # learned prior
                    _, mu_p, logvar_p = self.prior(h)  # learned prior

                h_pred = self.frame_predictor(torch.cat([h, z_t, state_action], 1))  # prediction model
                x_pred = self.decoder([h_pred, skip])  # prediction model

                mae += self.mae_criterion(x_pred, scene_and_touch[index + 1])  # prediction model
                kld += self.kl_criterion(mu, logvar, mu_p, logvar_p)  # learned prior

                outputs.append(x_pred)
            else:  # context
                h, skip = self.encoder(scene_and_touch[index])
                h_target = self.encoder(scene_and_touch[index + 1])[0]   # should this [0] be here????

                if test:
                    _, mu, logvar = self.posterior(h_target)  # learned prior
                    z_t, mu_p, logvar_p = self.prior(h)  # learned prior
                else:
                    z_t, mu, logvar = self.posterior(h_target)  # learned prior
                    _, mu_p, logvar_p = self.prior(h)  # learned prior

                h_pred = self.frame_predictor(torch.cat([h, z_t, state_action], 1))  # prediction model
                x_pred = self.decoder([h_pred, skip])  # prediction model

                mae += self.mae_criterion(x_pred, scene_and_touch[index + 1])  # prediction model
                kld += self.kl_criterion(mu, logvar, mu_p, logvar_p)  # learned prior

                last_output = x_pred

        outputs = [last_output] + outputs

        if test is False:
            loss = mae + (kld * self.beta)
            loss.backward()

            self.frame_predictor_optimizer.step()
            self.posterior_optimizer.step()
            self.prior_optimizer.step()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

        return mae.data.cpu().numpy() / (self.n_past + self.n_future), kld.data.cpu().numpy() / (self.n_future + self.n_past), torch.stack(outputs)

    def kl_criterion(self, mu1, logvar1, mu2, logvar2):
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2)) - 1 / 2
        return kld.sum() / self.batch_size