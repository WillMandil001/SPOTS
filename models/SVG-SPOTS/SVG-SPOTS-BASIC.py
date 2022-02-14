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

model_save_path = "/home/willmandil/Robotics/tactile_prediction/tactile_prediction/PRI/SVG-SPOTS/saved_models/PRI_single_object_purple_"
train_data_dir  = "/home/willmandil/Robotics/Data_sets/PRI/single_object_purple/train_formatted/"
scaler_dir      = "/home/willmandil/Robotics/Data_sets/PRI/single_object_purple/scalars/"

# unique save title:
model_save_path = model_save_path + "model_" + datetime.now().strftime("%d_%m_%Y_%H_%M/")
os.mkdir(model_save_path)


loss_function = "Basic"  # "basic", "split"
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
rnn_size=256
prior_rnn_layers=3
posterior_rnn_layers=3
predictor_rnn_layers=4
state_action_size = 12
z_dim=10  # number of latent variables
g_dim=256  # 128
beta=0.0001  # was 0.0001
data_threads=5
num_digits=2
last_frame_skip='store_true'
epochs = 100

train_percentage = 0.9
validation_percentage = 0.1

features = [lr, beta1, batch_size, log_dir, model_dir, name, data_root, optimizer, niter, seed, epoch_size,
            image_width, channels, out_channels, dataset, n_past, n_future, n_eval, rnn_size, prior_rnn_layers,
            posterior_rnn_layers, predictor_rnn_layers, z_dim, g_dim, beta, data_threads, num_digits,
            last_frame_skip, epochs, train_percentage, validation_percentage, loss_function]

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#  use gpu if available


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
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=12)
        validation_loader = torch.utils.data.DataLoader(dataset_validate, batch_size=batch_size, shuffle=True, num_workers=12)
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
            tactile_images.append(create_image(tactile_data_sample[0], tactile_data_sample[1], tactile_data_sample[2]))

        images = []
        for image_name in np.load(train_data_dir + value[2]):
            images.append(np.load(train_data_dir + image_name))

        experiment_number = np.load(train_data_dir + value[3])
        time_steps = np.load(train_data_dir + value[4])
        return [robot_data.astype(np.float32), np.array(images).astype(np.float32), np.array(tactile_images).astype(np.float32), experiment_number, time_steps]


def create_image(tactile_x, tactile_y, tactile_z):
    # convert tactile data into an image:
    image = np.zeros((4, 4, 3), np.float32)
    index = 0
    for x in range(4):
        for y in range(4):
            image[x][y] =  [tactile_x[index],
                            tactile_y[index],
                            tactile_z[index]]
            index += 1
    reshaped_image = np.rot90(cv2.resize(image.astype(np.float32), dsize=(64, 64), interpolation=cv2.INTER_CUBIC), k=1, axes=(0, 1))
    return reshaped_image


class ModelTrainer:
    def __init__(self):
        self.train_full_loader, self.valid_full_loader = BG.load_full_data()

        self.optimizer = optim.Adam

        # SCENE:
        import models.lstm as lstm_models
        self.frame_predictor_scene = lstm_models.lstm((g_dim*2) + z_dim + state_action_size, g_dim, rnn_size, predictor_rnn_layers, batch_size)
        self.frame_predictor_scene.apply(utils.init_weights)
        import models.dcgan_64 as model
        self.encoder_scene = model.encoder(g_dim, channels)
        self.decoder_scene = model.decoder(g_dim, channels)
        self.encoder_scene.apply(utils.init_weights)
        self.decoder_scene.apply(utils.init_weights)

        self.frame_predictor_optimizer_scene = self.optimizer(self.frame_predictor_scene.parameters(), lr=lr, betas=(beta1, 0.999))
        self.encoder_optimizer_scene = self.optimizer(self.encoder_scene.parameters(), lr=lr, betas=(beta1, 0.999))
        self.decoder_optimizer_scene = self.optimizer(self.decoder_scene.parameters(), lr=lr, betas=(beta1, 0.999))

        self.mae_criterion_scene = nn.L1Loss()

        self.frame_predictor_scene.cuda()
        self.encoder_scene.cuda()
        self.decoder_scene.cuda()
        self.mae_criterion_scene.cuda()

        # TACTILE:
        import models.lstm as lstm_models
        self.frame_predictor_tactile = lstm_models.lstm((g_dim*2) + z_dim + state_action_size, g_dim, rnn_size, predictor_rnn_layers, batch_size)
        self.frame_predictor_tactile.apply(utils.init_weights)
        import models.dcgan_64 as model
        self.encoder_tactile = model.encoder(g_dim, channels)
        self.decoder_tactile = model.decoder(g_dim, channels)
        self.encoder_tactile.apply(utils.init_weights)
        self.decoder_tactile.apply(utils.init_weights)

        self.frame_predictor_optimizer_tactile = self.optimizer(self.frame_predictor_tactile.parameters(), lr=lr, betas=(beta1, 0.999))
        self.encoder_optimizer_tactile = self.optimizer(self.encoder_tactile.parameters(), lr=lr, betas=(beta1, 0.999))
        self.decoder_optimizer_tactile = self.optimizer(self.decoder_tactile.parameters(), lr=lr, betas=(beta1, 0.999))

        self.mae_criterion_tactile = nn.L1Loss()

        self.frame_predictor_tactile.cuda()
        self.encoder_tactile.cuda()
        self.decoder_tactile.cuda()
        self.mae_criterion_tactile.cuda()

        # PRIOR:
        self.posterior = lstm_models.gaussian_lstm(g_dim*2, z_dim, rnn_size, posterior_rnn_layers, batch_size)
        self.prior = lstm_models.gaussian_lstm(g_dim*2, z_dim, rnn_size, prior_rnn_layers, batch_size)
        self.posterior.apply(utils.init_weights)
        self.prior.apply(utils.init_weights)
        self.posterior_optimizer = self.optimizer(self.posterior.parameters(), lr=lr, betas=(beta1, 0.999))
        self.prior_optimizer = self.optimizer(self.prior.parameters(), lr=lr, betas=(beta1, 0.999))
        self.posterior.cuda()
        self.prior.cuda()

    def run(self, scene, tactile, actions, test=False):
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
        self.encoder_tactile.zero_grad()
        self.decoder_tactile.zero_grad()
        self.frame_predictor_tactile.hidden = self.frame_predictor_tactile.init_hidden()

        # prior
        self.posterior.zero_grad()
        self.prior.zero_grad()
        self.posterior.hidden = self.posterior.init_hidden()
        self.prior.hidden = self.prior.init_hidden()


        state = actions[0].to(device)
        for index, (sample_scene, sample_tactile, sample_action) in enumerate(zip(scene[:-1], tactile[:-1], actions[1:])):
            state_action = torch.cat((state, actions[index]), 1)

            if index > n_past - 1:  # horizon
                # Tactile Encoding
                h_tactile, skip_tactile = self.encoder_tactile(x_pred_tactile)
                h_target_tactile        = self.encoder_tactile(tactile[index + 1])[0]

                # Scene Encoding
                h_scene, skip_scene = self.encoder_scene(x_pred_scene)
                h_target_scene      = self.encoder_scene(scene[index + 1])[0]

                # cat scene and tactile together for crossover input to pipelines
                h_scene_and_tactile = torch.cat([h_tactile, h_scene], 1)
                h_target_scene_and_tactile = torch.cat([h_target_tactile, h_target_scene], 1)

                # Learned Prior - Z_t calculation
                if test:
                    _, mu, logvar = self.posterior(h_target_scene_and_tactile)  # learned prior
                    z_t, mu_p, logvar_p = self.prior(h_scene_and_tactile)  # learned prior
                else:
                    z_t, mu, logvar = self.posterior(h_target_scene_and_tactile)  # learned prior
                    _, mu_p, logvar_p = self.prior(h_scene_and_tactile)  # learned prior

                # Tactile Prediction
                h_pred_tactile = self.frame_predictor_tactile(torch.cat([h_scene_and_tactile, z_t, state_action], 1))  # prediction model
                x_pred_tactile = self.decoder_tactile([h_pred_tactile, skip_tactile])  # prediction model

                # Scene Prediction
                h_pred_scene = self.frame_predictor_scene(torch.cat([h_scene_and_tactile, z_t, state_action], 1))  # prediction model
                x_pred_scene = self.decoder_scene([h_pred_scene, skip_scene])  # prediction model

                # loss calulations for tactile and scene:
                mae_tactile += self.mae_criterion_tactile(x_pred_tactile, tactile[index + 1])  # prediction model
                kld_tactile += self.kl_criterion_tactile(mu, logvar, mu_p, logvar_p)  # learned prior

                mae_scene += self.mae_criterion_scene(x_pred_scene, scene[index + 1])  # prediction model
                kld_scene += self.kl_criterion_scene(mu, logvar, mu_p, logvar_p)  # learned prior

                outputs.append(torch.cat([x_pred_scene, x_pred_tactile], 1))

            else:  # context
                # Tactile Encoding
                h_tactile, skip_tactile = self.encoder_tactile(tactile[index])
                h_target_tactile        = self.encoder_tactile(tactile[index + 1])[0]

                # Scene Encoding
                h_scene, skip_scene = self.encoder_scene(scene[index])
                h_target_scene      = self.encoder_scene(scene[index + 1])[0]

                # cat scene and tactile together for crossover input to pipelines
                h_scene_and_tactile = torch.cat([h_tactile, h_scene], 1)
                h_target_scene_and_tactile = torch.cat([h_target_tactile, h_target_scene], 1)

                # Learned Prior - Z_t calculation
                if test:
                    _, mu, logvar = self.posterior(h_target_scene_and_tactile)  # learned prior
                    z_t, mu_p, logvar_p = self.prior(h_scene_and_tactile)  # learned prior
                else:
                    z_t, mu, logvar = self.posterior(h_target_scene_and_tactile)  # learned prior
                    _, mu_p, logvar_p = self.prior(h_scene_and_tactile)  # learned prior

                # Tactile Prediction
                h_pred_tactile = self.frame_predictor_tactile(torch.cat([h_scene_and_tactile, z_t, state_action], 1))  # prediction model
                x_pred_tactile = self.decoder_tactile([h_pred_tactile, skip_tactile])  # prediction model

                # Scene Prediction
                h_pred_scene = self.frame_predictor_scene(torch.cat([h_scene_and_tactile, z_t, state_action], 1))  # prediction model
                x_pred_scene = self.decoder_scene([h_pred_scene, skip_scene])  # prediction model

                # loss calulations for tactile and scene:
                mae_tactile += self.mae_criterion_tactile(x_pred_tactile, tactile[index + 1])  # prediction model
                kld_tactile += self.kl_criterion_tactile(mu, logvar, mu_p, logvar_p)  # learned prior

                mae_scene += self.mae_criterion_scene(x_pred_scene, scene[index + 1])  # prediction model
                kld_scene += self.kl_criterion_scene(mu, logvar, mu_p, logvar_p)  # learned prior

                last_output = torch.cat([x_pred_scene, x_pred_tactile], 1)

        outputs = [last_output] + outputs

        if test is False:
            if loss_function == "Basic":
                loss_scene = mae_scene + (kld_scene * beta)
                loss_tactile = mae_tactile + (kld_tactile * beta)
                combined_loss = loss_scene + loss_tactile
                combined_loss.backward()

                self.frame_predictor_optimizer_scene.step()
                self.encoder_optimizer_scene.step()
                self.decoder_optimizer_scene.step()

                self.frame_predictor_optimizer_tactile.step()
                self.encoder_optimizer_tactile.step()
                self.decoder_optimizer_tactile.step()

                self.posterior_optimizer.step()
                self.prior_optimizer.step()
            elif loss_function == "Split":
                loss_scene = mae_scene + (kld_scene * beta)
                loss_scene.backward(retain_graph=True)

                self.frame_predictor_optimizer_scene.step()
                self.encoder_optimizer_scene.step()
                self.decoder_optimizer_scene.step()

                loss_tactile = mae_tactile + (kld_tactile * beta)
                loss_tactile.backward(retain_graph=True)
                self.frame_predictor_optimizer_tactile.step()
                self.encoder_optimizer_tactile.step()
                self.decoder_optimizer_tactile.step()

                combined_loss = loss_scene + loss_tactile
                combined_loss.backward()
                self.posterior_optimizer.step()
                self.prior_optimizer.step()

        return mae_scene.data.cpu().numpy() / (n_past + n_future), kld_scene.data.cpu().numpy() / (n_future + n_past), \
               mae_tactile.data.cpu().numpy() / (n_past + n_future), kld_tactile.data.cpu().numpy() / (n_future + n_past), \
               torch.stack(outputs)

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

    def train_full_model(self):
        self.frame_predictor_tactile.train()
        self.encoder_tactile.train()
        self.decoder_tactile.train()

        self.frame_predictor_scene.train()
        self.encoder_scene.train()
        self.decoder_scene.train()

        self.posterior.train()
        self.prior.train()

        plot_training_loss = []
        plot_validation_loss = []
        previous_val_mean_loss = 100.0
        best_val_loss = 100.0
        early_stop_clock = 0
        progress_bar = tqdm(range(0, epochs), total=(epochs*len(self.train_full_loader)))
        for epoch in progress_bar:
            epoch_mae_losses_scene = 0.0
            epoch_kld_losses_scene = 0.0
            epoch_mae_losses_tactile = 0.0
            epoch_kld_losses_tactile = 0.0

            for index, batch_features in enumerate(self.train_full_loader):
                if batch_features[1].shape[0] == batch_size:
                    images = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
                    tactile = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
                    action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
                    mae_scene, kld_scene, mae_tactile, kld_tactile, predictions = self.run(scene=images, tactile=tactile, actions=action, test=False)
                    epoch_mae_losses_scene += mae_scene.item()
                    epoch_kld_losses_scene += kld_scene.item()
                    epoch_mae_losses_tactile += mae_tactile.item()
                    epoch_kld_losses_tactile += kld_tactile.item()
                    if index:
                        mean_kld_tactile = epoch_kld_losses_tactile / index
                        mean_mae_tactile = epoch_mae_losses_tactile / index
                        mean_kld_scene = epoch_kld_losses_scene / index
                        mean_mae_scene = epoch_mae_losses_scene / index
                    else:
                        mean_kld_scene = 0.0
                        mean_mae_scene = 0.0
                        mean_kld_tactile = 0.0
                        mean_mae_tactile = 0.0

                    progress_bar.set_description("epoch: {}, ".format(epoch) + "MAE: {:.4f}, ".format(float(mae_scene.item())) + "kld: {:.4f}, ".format(float(kld_scene.item())) + "mean MAE: {:.4f}, ".format(mean_mae_scene) + "mean kld: {:.4f}, ".format(mean_kld_scene))
                    progress_bar.update()

            plot_training_loss.append([mean_mae_scene, mean_kld_scene, mean_mae_tactile, mean_kld_tactile])

            self.frame_predictor_tactile.eval()
            self.encoder_tactile.eval()
            self.decoder_tactile.eval()

            self.frame_predictor_scene.eval()
            self.encoder_scene.eval()
            self.decoder_scene.eval()

            self.posterior.eval()
            self.prior.eval()

            # Validation checking:
            val_mae_losses_scene = 0.0
            val_kld_losses_scene = 0.0
            with torch.no_grad():
                for index__, batch_features in enumerate(self.valid_full_loader):
                    if batch_features[1].shape[0] == batch_size:
                        images = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
                        tactile = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
                        action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
                        val_mae_scene, val_kld_scene, val_mae_tactile, val_kld_tactile, predictions = self.run(scene=images, tactile=tactile, actions=action, test=True)
                        val_mae_losses_scene += val_mae_scene.item()

            plot_validation_loss.append(val_mae_losses_scene / index__)
            print("Validation mae: {:.4f}, ".format(val_mae_losses_scene / index__))

            # save the train/validation performance data
            np.save(model_save_path + "plot_validation_loss", np.array(plot_validation_loss))
            np.save(model_save_path + "plot_training_loss", np.array(plot_training_loss))

            # Early stopping:
            if previous_val_mean_loss < val_mae_losses_scene / index__:
                early_stop_clock += 1
                previous_val_mean_loss = val_mae_losses_scene / index__
                if early_stop_clock == 4:
                    print("Early stopping")
                    break
            else:
                if best_val_loss > val_mae_losses_scene / index__:
                    print("saving model")
                    # save the model

                    torch.save({"frame_predictor_tactile": self.frame_predictor_tactile, "encoder_tactile": self.encoder_tactile, "decoder_tactile": self.decoder_tactile,
                    "frame_predictor_scene": self.frame_predictor_scene, "encoder_scene": self.encoder_scene, "decoder_scene": self.decoder_scene,
                    "posterior": self.posterior, "prior": self.prior, 'features': features}, model_save_path + "SVG_SPOTS_BASIC")

                    best_val_loss = val_mae_losses_scene / index__
                early_stop_clock = 0
                previous_val_mean_loss = val_mae_losses_scene / index__


if __name__ == "__main__":
    BG = BatchGenerator()
    MT = ModelTrainer()
    MT.train_full_model()