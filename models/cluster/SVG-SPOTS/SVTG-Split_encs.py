# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import cv2
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

warm_start_model_path = ""
warm_start_plot_path = ""
model_save_path = "/home/wmandil/Robotics/SPOTS/SPOTS/models/SVG-SPOTS/saved_models/PRI_single_object_purple_SVTG_SE"
train_data_dir  = "/home/wmandil/Robotics/Data_sets/PRI/single_object_purple/single_object_purple_formatted/train_formatted/"
scaler_dir      = "/home/wmandil/Robotics/Data_sets/PRI/single_object_purple/single_object_purple_formatted/scalars/"

# unique save title:
model_save_path = model_save_path + "model_" + datetime.now().strftime("%d_%m_%Y_%H_%M/")
os.mkdir(model_save_path)

warm_start = False

lr=0.0001
beta1=0.9
batch_size=32
log_dir='logs/lp'
model_dir=''
name=''
data_root='data'
optimizer='adam'
niter=300
seed=1
epoch_size=600
image_width=64
channels=3
out_channels = 3
dataset='smmnist'
n_past=10
n_future=10
n_eval=20
rnn_size=256*2
prior_rnn_layers=3
posterior_rnn_layers=3
predictor_rnn_layers=4
state_action_size = 12
z_dim=10  # number of latent variables
g_dim=256  # 128
# g_dim_scene = 256
# g_dim_touch = 256
beta=0.0001  # was 0.0001
data_threads=5
num_digits=2
last_frame_skip = 'store_true'
epochs = 100

train_percentage = 0.9
validation_percentage = 0.1

features = [lr, beta1, batch_size, log_dir, model_dir, name, data_root, optimizer, niter, seed, epoch_size,
            image_width, channels, out_channels, dataset, n_past, n_future, n_eval, rnn_size, prior_rnn_layers,
            posterior_rnn_layers, predictor_rnn_layers, z_dim, g_dim, beta, data_threads, num_digits,
            last_frame_skip, epochs, train_percentage, validation_percentage]

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available


class BatchGenerator:
    def __init__(self):
        self.data_map = []
        with open(train_data_dir + 'map.csv', 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                self.data_map.append(row)

    def load_full_data(self):
        dataset_train = FullDataSet(self.data_map, train=True)
        dataset_validate = FullDataSet(self.data_map, validation=True)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=6)
        validation_loader = torch.utils.data.DataLoader(dataset_validate, batch_size=batch_size, shuffle=True, num_workers=6)
        self.data_map = []
        return train_loader, validation_loader


class FullDataSet:
    def __init__(self, data_map, train=False, validation=False):
        if train:
            self.samples = data_map[1:int((len(data_map) * train_percentage))]
        if validation:
            self.samples = data_map[int((len(data_map) * train_percentage)): -1]
        data_map = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        value = self.samples[idx]
        robot_data = np.load(train_data_dir + value[0])

        tactile_data = np.load(train_data_dir + value[1])
        tactile_images = []
        for tactile_data_sample in tactile_data:
            tactile_images.append(create_image(tactile_data_sample))

        images = []
        for image_name in np.load(train_data_dir + value[2]):
            images.append(np.load(train_data_dir + image_name))

        experiment_number = np.load(train_data_dir + value[3])
        time_steps = np.load(train_data_dir + value[4])
        return [robot_data.astype(np.float32), np.array(images).astype(np.float32), np.array(tactile_images).astype(np.float32), experiment_number, time_steps]


def create_image(tactile):
    # convert tactile data into an image:
    return cv2.resize(tactile.reshape(3, 4, 4).astype(np.float32), dsize=(64, 64), interpolation=cv2.INTER_CUBIC)


class ModelTrainer:
    def __init__(self):

        import models.lstm as lstm_models
        self.frame_predictor = lstm_models.lstm((g_dim*2) + z_dim + state_action_size, (g_dim*2), rnn_size, predictor_rnn_layers, batch_size)
        self.posterior = lstm_models.gaussian_lstm((g_dim*2), z_dim, rnn_size, posterior_rnn_layers, batch_size)
        self.prior = lstm_models.gaussian_lstm((g_dim*2), z_dim, rnn_size, prior_rnn_layers, batch_size)
        self.frame_predictor.apply(utils.init_weights)
        self.posterior.apply(utils.init_weights)
        self.prior.apply(utils.init_weights)

        import models.dcgan_64 as model
        self.encoder_tactile = model.encoder(g_dim, channels)
        self.encoder_scene = model.encoder(g_dim, channels)
        self.decoder_tactile = model.decoder(g_dim*2, channels)
        self.decoder_scene = model.decoder(g_dim*2, channels)
        self.encoder_tactile.apply(utils.init_weights)
        self.encoder_scene.apply(utils.init_weights)
        self.decoder_tactile.apply(utils.init_weights)
        self.decoder_scene.apply(utils.init_weights)

        self.train_full_loader, self.valid_full_loader = BG.load_full_data ()
        self.optimizer = optim.Adam

        self.frame_predictor_optimizer = self.optimizer(self.frame_predictor.parameters(), lr=lr, betas=(beta1, 0.999))
        self.posterior_optimizer = self.optimizer(self.posterior.parameters(), lr=lr, betas=(beta1, 0.999))
        self.prior_optimizer = self.optimizer(self.prior.parameters(), lr=lr, betas=(beta1, 0.999))
        self.encoder_tactile_optimizer = self.optimizer(self.encoder_tactile.parameters(), lr=lr, betas=(beta1, 0.999))
        self.encoder_scene_optimizer = self.optimizer(self.encoder_scene.parameters(), lr=lr, betas=(beta1, 0.999))
        self.decoder_tactile_optimizer = self.optimizer(self.decoder_tactile.parameters(), lr=lr, betas=(beta1, 0.999))
        self.decoder_scene_optimizer = self.optimizer(self.decoder_scene.parameters(), lr=lr, betas=(beta1, 0.999))

        self.mae_criterion = nn.L1Loss()

        self.frame_predictor.cuda()
        self.posterior.cuda()
        self.prior.cuda()
        self.encoder_tactile.cuda()
        self.encoder_scene.cuda()
        self.decoder_tactile.cuda()
        self.decoder_scene.cuda()
        self.mae_criterion.cuda()

    def run(self, scene, touch, actions, test=False):
        mae_scene, mae_tactile, kld = 0, 0, 0
        outputs = []

        self.frame_predictor.zero_grad()
        self.posterior.zero_grad()
        self.prior.zero_grad()
        self.encoder_scene.zero_grad()
        self.decoder_scene.zero_grad()
        self.encoder_tactile.zero_grad()
        self.decoder_tactile.zero_grad()

        self.frame_predictor.hidden = self.frame_predictor.init_hidden()
        self.posterior.hidden = self.posterior.init_hidden()
        self.prior.hidden = self.prior.init_hidden()

        state = actions[0].to(device)
        for index, (sample_scene, sample_touch, sample_action) in enumerate(zip(scene[:-1], touch[:-1], actions[1:])):
            state_action = torch.cat((state, actions[index]), 1)

            if index > n_past - 1:  # horizon
                h_scene, skip_scene = self.encoder_scene(x_pred_scene)
                h_target_scene = self.encoder_scene(scene[index + 1])[0]

                h_tactile, skip_tactile = self.encoder_tactile(x_pred_tactile)
                h_target_tactile = self.encoder_tactile(touch[index + 1])[0]

                if test:
                    _, mu, logvar = self.posterior(torch.cat([h_target_scene, h_target_tactile], 1))
                    z_t, mu_p, logvar_p = self.prior(torch.cat([h_scene, h_tactile], 1))
                else:
                    z_t, mu, logvar = self.posterior(torch.cat([h_target_scene, h_target_tactile], 1))
                    _, mu_p, logvar_p = self.prior(torch.cat([h_scene, h_tactile], 1))

                h_pred = self.frame_predictor(torch.cat([h_scene, h_tactile, z_t, state_action], 1))  # prediction model

                x_pred_scene = self.decoder_scene([h_pred, skip_scene])  # prediction model
                x_pred_tactile = self.decoder_tactile([h_pred, skip_tactile])  # prediction model

                mae_scene += self.mae_criterion(x_pred_scene, scene[index + 1])  # prediction model
                mae_tactile += self.mae_criterion(x_pred_tactile, touch[index + 1])  # prediction model
                kld += self.kl_criterion(mu, logvar, mu_p, logvar_p)  # learned prior

                outputs.append(x_pred_scene)
            else:  # context
                h_scene, skip_scene = self.encoder_scene(scene[index])
                h_target_scene = self.encoder_scene(scene[index + 1])[0]

                h_tactile, skip_tactile = self.encoder_tactile(touch[index])
                h_target_tactile = self.encoder_tactile(touch[index + 1])[0]

                if test:
                    _, mu, logvar = self.posterior(torch.cat([h_target_scene, h_target_tactile], 1))
                    z_t, mu_p, logvar_p = self.prior(torch.cat([h_scene, h_tactile], 1))
                else:
                    z_t, mu, logvar = self.posterior(torch.cat([h_target_scene, h_target_tactile], 1))
                    _, mu_p, logvar_p = self.prior(torch.cat([h_scene, h_tactile], 1))

                h_pred = self.frame_predictor(torch.cat([h_scene, h_tactile, z_t, state_action], 1))  # prediction model

                x_pred_scene = self.decoder_scene([h_pred, skip_scene])  # prediction model
                x_pred_tactile = self.decoder_tactile([h_pred, skip_tactile])  # prediction model

                mae_scene += self.mae_criterion(x_pred_scene, scene[index + 1])  # prediction model
                mae_tactile += self.mae_criterion(x_pred_tactile, touch[index + 1])  # prediction model
                kld += self.kl_criterion(mu, logvar, mu_p, logvar_p)  # learned prior

                last_output = x_pred_scene

        outputs = [last_output] + outputs

        if test is False:
            loss = mae_scene + mae_tactile + (kld * beta)
            loss.backward()
            self.frame_predictor_optimizer.step()
            self.posterior_optimizer.step()
            self.prior_optimizer.step()
            self.encoder_tactile_optimizer.step()
            self.decoder_tactile_optimizer.step()
            self.encoder_scene_optimizer.step()
            self.decoder_scene_optimizer.step()

        return mae_scene.data.cpu().numpy() / (n_past + n_future), mae_tactile.data.cpu().numpy() / (n_past + n_future), kld.data.cpu().numpy() / (n_future + n_past), torch.stack(outputs)

    def kl_criterion(self, mu1, logvar1, mu2, logvar2):
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2)) - 1 / 2
        return kld.sum() / batch_size

    def train_full_model(self):
        self.frame_predictor.train()
        self.posterior.train()
        self.prior.train()
        self.encoder_scene.train()
        self.encoder_tactile.train()
        self.decoder_scene.train()
        self.decoder_tactile.train()

        plot_training_loss = []
        plot_validation_loss = []
        best_val_loss = 100.0
        previous_val_mean_loss = 100.0

        early_stop_clock = 0
        print("beginning Training")
        for epoch in range(epochs):
            epoch_mae_losses = 0.0
            epoch_kld_losses = 0.0
            for index, batch_features in enumerate(self.train_full_loader):
                if batch_features[1].shape[0] == batch_size:
                    images = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
                    tactile = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
                    action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
                    mae_scene, mae_tactile, kld, predictions = self.run(scene=images, touch=tactile, actions=action, test=False)
                    epoch_mae_losses += mae_scene.item()
                    epoch_kld_losses += kld.item()
                    if index:
                        mean_kld = epoch_kld_losses / index
                        mean_mae = epoch_mae_losses / index
                    else:
                        mean_kld = 0.0
                        mean_mae = 0.0

            print("Epoch:", epoch, "mean_mae_scene: ", mean_mae, "mean_kld_scene: ", mean_kld)
            plot_training_loss.append([mean_mae, mean_kld])

            self.frame_predictor.eval()
            self.posterior.eval()
            self.prior.eval()
            self.encoder_scene.eval()
            self.encoder_tactile.eval()
            self.decoder_scene.eval()
            self.decoder_tactile.eval()

            # Validation checking:
            val_mae_losses = 0.0
            val_kld_losses = 0.0
            with torch.no_grad():
                for index__, batch_features in enumerate(self.valid_full_loader):
                    if batch_features[1].shape[0] == batch_size:
                        images = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
                        tactile = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
                        action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
                        val_mae_scene, val_mae_tact, val_kld, predictions = self.run(scene=images, touch=tactile, actions=action, test=True)
                        val_mae_losses += val_mae_scene.item()

            plot_validation_loss.append(val_mae_losses / index__)
            print("Validation mae: {:.4f}, ".format(val_mae_losses / index__))

            # save the train/validation performance data
            np.save(model_save_path + "plot_validation_loss", np.array(plot_validation_loss))
            np.save(model_save_path + "plot_training_loss", np.array(plot_training_loss))

            # Early stopping:
            if previous_val_mean_loss < val_mae_losses / index__:
                early_stop_clock += 1
                previous_val_mean_loss = val_mae_losses / index__
                if early_stop_clock == 4:
                    print("Early stopping")
                    break
            else:
                if best_val_loss > val_mae_losses / index__:
                    print("saving model")
                    # save the model
                    torch.save({'encoder_tactile': self.encoder_tactile, 'decoder_tactile': self.decoder_tactile, 'encoder_scene': self.encoder_scene, 'decoder_scene': self.decoder_scene,
                                'frame_predictor': self.frame_predictor, 'posterior': self.posterior, 'prior': self.prior, 'features': features}, model_save_path + "SVTG_SE_model")

                    best_val_loss = val_mae_losses / index__
                early_stop_clock = 0
                previous_val_mean_loss = val_mae_losses / index__


if __name__ == "__main__":
    BG = BatchGenerator()
    MT = ModelTrainer()
    MT.train_full_model()
