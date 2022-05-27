# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import copy
import numpy as np

from datetime import datetime
from torch.utils.data import Dataset
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import universal_networks.utils as utility_prog


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
        self.train_data_dir = features["train_data_dir"]
        self.scaler_dir = features["scaler_dir"]
        self.device = features["device"]
        self.training_stages = features["training_stages"]
        self.training_stages_epochs = features["training_stages_epochs"]
        self.tactile_size = features["tactile_size"]
        self.model_name_save_appendix = features["model_name_save_appendix"]

        if self.optimizer == "adam" or self.optimizer == "Adam":
            self.optimizer = optim.Adam

        if self.criterion == "L1":
            self.mae_criterion = nn.L1Loss()
            self.mae_criterion_scene = nn.L1Loss()
            self.mae_criterion_tactile = nn.L1Loss()
        if self.criterion == "L2":
            self.mae_criterion = nn.MSELoss()
            self.mae_criterion_scene = nn.MSELoss()
            self.mae_criterion_tactile = nn.MSELoss()

    def load_model(self, full_model):
        self.frame_predictor_tactile = full_model["frame_predictor_tactile"]
        self.frame_predictor_scene = full_model["frame_predictor_scene"]
        self.encoder_scene = full_model["encoder_scene"]
        self.decoder_scene = full_model["decoder_scene"]
        self.MMFM_scene = full_model["MMFM_scene"]
        self.MMFM_tactile = full_model["MMFM_tactile"]

        self.frame_predictor_tactile.cuda()
        self.frame_predictor_scene.cuda()
        self.encoder_scene.cuda()
        self.decoder_scene.cuda()
        self.MMFM_scene.cuda()
        self.MMFM_tactile.cuda()
        self.mae_criterion_scene.cuda()
        self.mae_criterion_tactile.cuda()

    def initialise_model(self):
        import universal_networks.dcgan_64 as model
        import universal_networks.lstm as lstm_models
        import universal_networks.ACTP as ACTP_model
        import universal_networks.lstm as lstm_models

        # SCENE:
        self.frame_predictor_scene = lstm_models.lstm(self.g_dim + self.tactile_size + self.state_action_size, self.g_dim, self.rnn_size, self.predictor_rnn_layers, self.batch_size)
        self.frame_predictor_scene.apply(utility_prog.init_weights)

        self.MMFM_scene = model.MMFM_scene(self.g_dim + self.tactile_size, self.g_dim + self.tactile_size, self.channels)
        self.MMFM_scene.apply(utility_prog.init_weights)
        self.MMFM_scene_optimizer = self.optimizer(self.MMFM_scene.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.MMFM_scene.cuda()

        self.encoder_scene = model.encoder(self.g_dim, self.channels)
        self.decoder_scene = model.decoder(self.g_dim, self.channels)
        self.encoder_scene.apply(utility_prog.init_weights)
        self.decoder_scene.apply(utility_prog.init_weights)

        self.frame_predictor_optimizer_scene = self.optimizer(self.frame_predictor_scene.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.encoder_optimizer_scene = self.optimizer(self.encoder_scene.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.decoder_optimizer_scene = self.optimizer(self.decoder_scene.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.frame_predictor_scene.cuda()
        self.encoder_scene.cuda()
        self.decoder_scene.cuda()
        self.mae_criterion_scene.cuda()

        # TACTILE:
        self.frame_predictor_tactile = ACTP_model.ACTP(device=self.device, input_size=(self.g_dim + self.tactile_size), tactile_size=self.tactile_size)
        self.frame_predictor_tactile.apply(utility_prog.init_weights)

        self.MMFM_tactile = model.MMFM_tactile(self.g_dim + self.tactile_size, self.g_dim + self.tactile_size, self.channels)
        self.MMFM_tactile.apply(utility_prog.init_weights)
        self.MMFM_tactile_optimizer = self.optimizer(self.MMFM_tactile.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.MMFM_tactile.cuda()

        self.frame_predictor_optimizer_tactile = self.optimizer(self.frame_predictor_tactile.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.frame_predictor_tactile.cuda()
        self.mae_criterion_tactile.cuda()


    def run(self, scene, tactile, actions, gain, test=False, stage=False):
        mae_tactile = 0
        kld_tactile = 0
        mae_scene = 0
        kld_scene = 0
        outputs_scene = []
        outputs_tactile = []

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

        state = actions[0].to(self.device)
        for index, (sample_scene, sample_tactile, sample_action) in enumerate(zip(scene[:-1], tactile[:-1], actions[1:])):
            state_action = torch.cat((state, sample_action), 1)

            if index > self.n_past - 1:  # horizon
                # Scene Encoding
                h_scene, skip_scene = self.encoder_scene(x_pred_scene)
                h_target_scene      = self.encoder_scene(scene[index + 1])[0]

                # cat scene and tactile together for crossover input to pipelines
                h_scene_and_tactile = torch.cat([x_pred_tactile, h_scene], 1)

                # Multi-modal feature model:
                MM_rep_scene = self.MMFM_scene(h_scene_and_tactile)
                MM_rep_tactile = self.MMFM_tactile(h_scene_and_tactile)

                # Tactile Prediction
                x_pred_tactile= self.frame_predictor_tactile(MM_rep_tactile, state_action, x_pred_tactile)  # prediction model

                # Scene Prediction
                h_pred_scene = self.frame_predictor_scene(torch.cat([MM_rep_scene, state_action], 1))  # prediction model
                x_pred_scene = self.decoder_scene([h_pred_scene, skip_scene])  # prediction model

                # loss calulations for tactile and scene:
                mae_tactile += self.mae_criterion_tactile(x_pred_tactile, tactile[index + 1])  # prediction model

                mae_scene += self.mae_criterion_scene(x_pred_scene, scene[index + 1])  # prediction model

                outputs_tactile.append(x_pred_tactile)
                outputs_scene.append(x_pred_scene)

            else:  # context
                # Scene Encoding
                h_scene, skip_scene = self.encoder_scene(scene[index])
                h_target_scene      = self.encoder_scene(scene[index + 1])[0]

                # cat scene and tactile together for crossover input to pipelines
                h_scene_and_tactile = torch.cat([tactile[index], h_scene], 1)

                # Multi-modal feature model:
                MM_rep_scene = self.MMFM_scene(h_scene_and_tactile)
                MM_rep_tactile = self.MMFM_tactile(h_scene_and_tactile)

                # Tactile Prediction
                x_pred_tactile = self.frame_predictor_tactile(MM_rep_tactile, state_action, tactile[index])  # prediction model

                # Scene Prediction
                h_pred_scene = self.frame_predictor_scene(torch.cat([MM_rep_scene, state_action], 1))  # prediction model
                x_pred_scene = self.decoder_scene([h_pred_scene, skip_scene])  # prediction model

                # loss calulations for tactile and scene:
                mae_tactile += self.mae_criterion_tactile(x_pred_tactile, tactile[index + 1])  # prediction model

                mae_scene += self.mae_criterion_scene(x_pred_scene, scene[index + 1])  # prediction model

                last_output_scene = x_pred_scene
                last_output_tactile = x_pred_tactile

        outputs_scene = [last_output_scene] + outputs_scene
        outputs_tactile = [last_output_tactile] + outputs_tactile

        if test is False:
            if stage == "scene_only":
                loss_scene = mae_scene + (kld_scene * self.beta)
                loss_scene.backward()

                self.frame_predictor_optimizer_scene.step()
                self.encoder_optimizer_scene.step()
                self.decoder_optimizer_scene.step()

                self.frame_predictor_optimizer_tactile.step()

                self.MMFM_scene_optimizer.step()

            elif stage == "tactile_loss_plus_scene_fixed":
                loss_tactile = mae_tactile
                loss_tactile.backward()

                self.frame_predictor_optimizer_tactile.step()
                self.MMFM_tactile_optimizer.step()

            elif stage == "scene_loss_plus_tactile_gradual_increase":
                loss_scene = mae_scene + (kld_scene * self.beta)
                loss_tactile = mae_tactile + (kld_tactile * self.beta)
                combined_loss = loss_scene + (loss_tactile * gain)
                combined_loss.backward()

                self.frame_predictor_optimizer_scene.step()
                self.encoder_optimizer_scene.step()
                self.decoder_optimizer_scene.step()

                self.frame_predictor_optimizer_tactile.step()

                self.MMFM_scene_optimizer.step()
                self.MMFM_tactile_optimizer.step()

        return mae_scene.data.cpu().numpy() / (self.n_past + self.n_future), mae_tactile.cpu().data.numpy() / (self.n_past + self.n_future), torch.stack(outputs_scene), torch.stack(outputs_tactile)

    def kl_criterion_scene(self, mu1, logvar1, mu2, logvar2):
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2)) - 1 / 2
        return kld.sum() / self.batch_size

    def kl_criterion_tactile(self, mu1, logvar1, mu2, logvar2):
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2)) - 1 / 2
        return kld.sum() / self.batch_size

    def set_train(self):
        self.frame_predictor_tactile.train()
        self.frame_predictor_scene.train()
        self.encoder_scene.train()
        self.decoder_scene.train()

        self.MMFM_scene.train()
        self.MMFM_tactile.train()

    def set_test(self):
        self.frame_predictor_tactile.eval()
        self.frame_predictor_scene.eval()
        self.encoder_scene.eval()
        self.decoder_scene.eval()

        self.MMFM_scene.eval()
        self.MMFM_tactile.eval()

    def save_model(self, stage="best"):
        if stage == "best":
            save_name = "SPOTS_VG_ACTP_BEST"
        elif stage == "scene_only":
            save_name = "SPOTS_VG_ACTP_stage1"
        elif stage == "tactile_loss_plus_scene_fixed":
            save_name = "SPOTS_VG_ACTP_stage2"
        elif stage == "scene_loss_plus_tactile_gradual_increase":
            save_name = "SPOTS_VG_ACTP_stage3"

        torch.save({"frame_predictor_tactile": self.frame_predictor_tactile,
                    "frame_predictor_scene": self.frame_predictor_scene, "encoder_scene": self.encoder_scene,
                    "decoder_scene": self.decoder_scene, 'features': self.features,
                    "MMFM_scene": self.MMFM_scene, "MMFM_tactile": self.MMFM_tactile}, self.model_dir + save_name + self.model_name_save_appendix)

