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

model_path      = "/home/user/Robotics/SPOTS/models/SVG-SPOTS/saved_models/PRI_single_object_purple_model_11_02_2022_10_10/SVG_model.zip"
data_save_path  = "/home/user/Robotics/SPOTS/models/SVG-SPOTS/saved_models/PRI_single_object_purple_model_11_02_2022_10_10/"
test_data_dir   = "/home/user/Robotics/Data_sets/PRI/single_object_purple/test_no_new_formatted/"
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
        tactile_images = []
        for tactile_data_sample in tactile_data:
            tactile_images.append(create_image(tactile_data_sample[0], tactile_data_sample[1], tactile_data_sample[2]))

        images = []
        for image_name in np.load(self.test_data_dir + value[2]):
            images.append(np.load(self.test_data_dir + image_name))

        experiment_number = np.load(self.test_data_dir + value[3])
        time_steps = np.load(self.test_data_dir + value[4])
        return [robot_data.astype(np.float32), np.array(images).astype(np.float32), np.array(tactile_images).astype(np.float32), experiment_number, time_steps]


def create_image(tactile_x, tactile_y, tactile_z):
    # convert tactile data into an image:
    image = np.zeros((4, 4, 3), np.float32)
    index = 0
    for x in range(4):
        for y in range(4):
            image[x][y] = [tactile_x[index],
                           tactile_y[index],
                           tactile_z[index]]
            index += 1
    reshaped_image = np.rot90(cv2.resize (image.astype (np.float32), dsize=(64, 64), interpolation=cv2.INTER_CUBIC), k=1, axes=(0, 1))
    return reshaped_image


class ModelTester:
    def __init__(self):
        self.full_model = torch.load(model_path)

        (lr, beta1, batch_size, log_dir, model_dir, name, data_root, optimizer, niter, seed, epoch_size,
        image_width, channels, out_channels, dataset, n_past, n_future, n_eval, rnn_size, prior_rnn_layers,
        posterior_rnn_layers, predictor_rnn_layers, z_dim, g_dim, beta, data_threads, num_digits,
        last_frame_skip, epochs, _, __) = self.full_model["features"]

        batch_size = 8

        self.optimizer = optim.Adam
        self.frame_predictor_scene = self.full_model["frame_predictor_scene"]
        self.frame_predictor_tactile = self.full_model["frame_predictor_tactile"]
        self.encoder_scene = self.full_model["encoder_scene"]
        self.decoder_scene = self.full_model["decoder_scene"]
        self.encoder_tactile = self.full_model["encoder_tactile"]
        self.decoder_tactile = self.full_model["decoder_tactile"]
        self.prior = self.full_model["prior"]
        self.posterior = self.full_model["posterior"]

        self.mae_criterion_tactile = nn.L1Loss()
        self.mae_criterion_scene = nn.L1Loss()

        self.frame_predictor_scene.cuda()
        self.frame_predictor_tactile.cuda()
        self.encoder_scene.cuda()
        self.decoder_scene.cuda()
        self.encoder_tactile.cuda()
        self.decoder_tactile.cuda()
        self.prior.cuda()
        self.posterior.cuda()
        self.mae_criterion_tactile.cuda()
        self.mae_criterion_scene.cuda()

        self.criterion = nn.L1Loss()
        self.load_scalars()

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

        return mae_scene.data.cpu().numpy() / (n_past + n_future), kld_scene.data.cpu().numpy() / (n_future + n_past), \
               mae_tactile.data.cpu().numpy() / (n_past + n_future), kld_tactile.data.cpu().numpy() / (n_future + n_past),\
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

    def test_full_model(self):
        self.objects = []
        self.performance_data = []
        self.prediction_data = []
        self.tg_back_scaled = []
        self.tp1_back_scaled = []
        self.tp5_back_scaled = []
        self.tp10_back_scaled = []
        self.current_exp = 0

        self.number_of_trials = [2]  # i for i in range(30)]
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
                    tactile = batch_features[2].permute(1, 0, 4, 3, 2).to(device)
                    action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
                    mae, kld, predictions = self.run(scene=images, tactile=tactile, actions=action, test=True)

                    for i in range(10):
                        plt.figure(1)
                        f, axarr = plt.subplots(1, 2)
                        axarr[0].set_title("predictions: t_" + str(i))
                        axarr[0].imshow(np.array(predictions[i][7].permute(1, 2, 0).cpu().detach())[:, :, 3:])
                        axarr[1].set_title("ground truth: t_" + str(i))
                        axarr[1].imshow(np.array(images[context_frames + i][7].permute(1, 2, 0).cpu().detach()))
                        plt.savefig(data_save_path + str(i) + ".png")
                    break
                break
            break

        #     mae = self.mae_criterion(predictions[:, :, 3:, :, :], images[context_frames:]).data
        #
        #     # calculate trial loss:
        #     self.total_losses.append(["trial number: " + str(trial), np.array(mae.cpu().detach())])
        #
        # np.save(data_save_path + "test_no_new_losses_per_trial", np.array(self.total_losses))
        #
        # print(self.total_losses)
        #
        # final_loss = sum([i[1] for i in self.total_losses]) / len(self.total_losses)
        #
        # np.save(data_save_path + "test_no_new_loss", np.array(final_loss))
        # print(np.array(self.total_losses))
        # print(final_loss)


    def calc_train_trial_performance(self):
        mae_loss, mae_loss_1, mae_loss_5, mae_loss_10, mae_loss_x, mae_loss_y, mae_loss_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        meta_data_file_name = str(np.load(self.meta)[0]) + "/meta_data.csv"
        meta_data = []
        with open(meta_data_file_name, 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                meta_data.append(row)
        seen = meta_data[1][2]
        object = meta_data[1][0]

        index = 0
        with torch.no_grad():
            for batch_set in self.prediction_data:
                index += 1

                mae_loss_check = self.criterion (batch_set[0], batch_set[1]).item ()
                if math.isnan(mae_loss_check):
                    index -= 1
                else:
                    ## MAE:
                    mae_loss    += mae_loss_check
                    mae_loss_1  += self.criterion(batch_set[0][0], batch_set[1][0]).item()
                    mae_loss_5  += self.criterion(batch_set[0][4], batch_set[1][4]).item()
                    mae_loss_10 += self.criterion(batch_set[0][9], batch_set[1][9]).item()
                    mae_loss_x  += self.criterion(batch_set[0][:,:,0], batch_set[1][:,:,0]).item()
                    mae_loss_y  += self.criterion(batch_set[0][:,:,1], batch_set[1][:,:,1]).item()
                    mae_loss_z  += self.criterion(batch_set[0][:,:,2], batch_set[1][:,:,2]).item()

        self.performance_data.append([mae_loss/index, mae_loss_1/index, mae_loss_5/index, mae_loss_10/index, mae_loss_x/index,
                                mae_loss_y/index, mae_loss_z/index, seen, object])

        print("object: ", object)
        self.objects.append(object)

    def calc_train_performance(self):
        '''
        - Calculates PSNR, SSIM, MAE for ts1, 5, 10 and x,y,z forces
        - Save Plots for qualitative analysis
        - Slip classification test
        '''
        performance_data_full = []
        performance_data_full.append(["test loss MAE(L1): ", (sum([i[0] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 1: ", (sum([i[1] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 5: ", (sum([i[2] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 10: ", (sum([i[3] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred force x: ", (sum([i[4] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred force y: ", (sum([i[5] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred force z: ", (sum([i[6] for i in self.performance_data]) / len(self.performance_data))])

        self.objects = list(set(self.objects))
        for object in self.objects:
            performance_data_full.append(["test loss MAE(L1) Trained object " + str(object) + ": ", (sum([i[0] for i in self.performance_data if i[-1] == str(object)]) / len([i[0] for i in self.performance_data if i[-1] == str(object)]))])

        [print (i) for i in performance_data_full]
        np.save(data_save_path + 'TRAIN_model_performance_loss_data', np.asarray(performance_data_full))

    def save_predictions(self, experiment_to_test):
        '''
        - Plot the descaled 48 feature tactile vector for qualitative analysis
        - Save plots in a folder with name being the trial number.
        '''
        trial_groundtruth_data = []
        trial_predicted_data_t1 = []
        trial_predicted_data_t5 = []
        trial_predicted_data_t10 = []
        for index in range(len(self.tg_back_scaled)):
            for batch_number in range(len(self.tp10_back_scaled[index])):
                if experiment_to_test == self.prediction_data[index][2].T[batch_number][0]:
                    trial_predicted_data_t1.append(self.tp1_back_scaled[index][batch_number])
                    trial_predicted_data_t5.append(self.tp5_back_scaled[index][batch_number])
                    trial_predicted_data_t10.append(self.tp10_back_scaled[index][batch_number])
                    trial_groundtruth_data.append(self.tg_back_scaled[index][batch_number])

        plot_save_dir = data_save_path + "test_plots_" + str(experiment_to_test)
        try:
            os.mkdir(plot_save_dir)
        except:
            "directory already exists"

        np.save(plot_save_dir + '/trial_groundtruth_data', np.array(trial_groundtruth_data))
        np.save(plot_save_dir + '/prediction_data_t1', np.array(trial_predicted_data_t1))
        np.save(plot_save_dir + '/prediction_data_t5', np.array(trial_predicted_data_t5))
        np.save(plot_save_dir + '/prediction_data_t10', np.array(trial_predicted_data_t10))

        meta_data_file_name = str(np.load(self.meta)[0]) + "/meta_data.csv"
        meta_data = []
        with open(meta_data_file_name, 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                meta_data.append(row)
        np.save(plot_save_dir + '/meta_data', np.array(meta_data))

    def calc_trial_performance(self):
        mae_loss, mae_loss_1, mae_loss_5, mae_loss_10, mae_loss_x, mae_loss_y, mae_loss_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ssim_loss, ssim_loss_1, ssim_loss_5, ssim_loss_10, ssim_loss_x, ssim_loss_y, ssim_loss_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        psnr_loss, psnr_loss_1, psnr_loss_5, psnr_loss_10, psnr_loss_x, psnr_loss_y, psnr_loss_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        ssim_calc = SSIM(window_size=32)
        psnr_calc = PSNR()

        meta_data_file_name = str(np.load(self.meta)[0]) + "/meta_data.csv"
        meta_data = []
        with open(meta_data_file_name, 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                meta_data.append(row)
        seen = meta_data[1][2]
        object = meta_data[1][0]

        index = 0
        index_ssim = 0
        with torch.no_grad():
            for batch_set in self.prediction_data:
                index += 1

                mae_loss_check = self.criterion (batch_set[0], batch_set[1]).item ()
                if math.isnan(mae_loss_check):
                    index -= 1
                else:
                    ## MAE:
                    mae_loss    += mae_loss_check
                    mae_loss_1  += self.criterion(batch_set[0][0], batch_set[1][0]).item()
                    mae_loss_5  += self.criterion(batch_set[0][4], batch_set[1][4]).item()
                    mae_loss_10 += self.criterion(batch_set[0][9], batch_set[1][9]).item()
                    mae_loss_x  += self.criterion(batch_set[0][:,:,0], batch_set[1][:,:,0]).item()
                    mae_loss_y  += self.criterion(batch_set[0][:,:,1], batch_set[1][:,:,1]).item()
                    mae_loss_z  += self.criterion(batch_set[0][:,:,2], batch_set[1][:,:,2]).item()
                    ## SSIM:
                    for i in range(len(batch_set)):
                        index_ssim += 1
                        ssim_loss += ssim_calc(batch_set[0][i], batch_set[1][i])
                    ssim_loss_1  += ssim_calc(batch_set[0][0], batch_set[1][0])
                    ssim_loss_5  += ssim_calc(batch_set[0][4], batch_set[1][4])
                    ssim_loss_10 += ssim_calc(batch_set[0][9], batch_set[1][9])
                    ssim_loss_x  += ssim_calc(batch_set[0][:,:,0], batch_set[1][:,:,0])
                    ssim_loss_y  += ssim_calc(batch_set[0][:,:,1], batch_set[1][:,:,1])
                    ssim_loss_z  += ssim_calc(batch_set[0][:,:,2], batch_set[1][:,:,2])
                    ## PSNR:
                    psnr_loss    += psnr_calc(batch_set[0], batch_set[1])
                    psnr_loss_1  += psnr_calc(batch_set[0][0], batch_set[1][0])
                    psnr_loss_5  += psnr_calc(batch_set[0][4], batch_set[1][4])
                    psnr_loss_10 += psnr_calc(batch_set[0][9], batch_set[1][9])
                    psnr_loss_x  += psnr_calc(batch_set[0][:,:,0], batch_set[1][:,:,0])
                    psnr_loss_y  += psnr_calc(batch_set[0][:,:,1], batch_set[1][:,:,1])
                    psnr_loss_z  += psnr_calc(batch_set[0][:,:,2], batch_set[1][:,:,2])

        self.performance_data.append([mae_loss/index, mae_loss_1/index, mae_loss_5/index, mae_loss_10/index, mae_loss_x/index,
                                mae_loss_y/index, mae_loss_z/index, ssim_loss/index_ssim, ssim_loss_1/index,
                                ssim_loss_5/index, ssim_loss_10/index, ssim_loss_x/index, ssim_loss_y/index,
                                ssim_loss_z/index, psnr_loss/index, psnr_loss_1/index, psnr_loss_5/index,
                                psnr_loss_10/index, psnr_loss_x/index, psnr_loss_y/index, psnr_loss_z/index, seen, object])


        print("object: ", object)
        self.objects.append(object)

    def calc_test_performance(self):
        '''
        - Calculates PSNR, SSIM, MAE for ts1, 5, 10 and x,y,z forces
        - Save Plots for qualitative analysis
        - Slip classification test
        '''
        performance_data_full = []
        performance_data_full.append(["test loss MAE(L1): ", (sum([i[0] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 1: ", (sum([i[1] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 5: ", (sum([i[2] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 10: ", (sum([i[3] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred force x: ", (sum([i[4] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred force y: ", (sum([i[5] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred force z: ", (sum([i[6] for i in self.performance_data]) / len(self.performance_data))])

        performance_data_full.append (["test loss MAE(L1) Seen objects: ", (sum([i[0] for i in self.performance_data if i[-2] == '1']) / len([i[0] for i in self.performance_data if i[-2] == '1']))])
        performance_data_full.append (["test loss MAE(L1) Novel objects: ", (sum([i[0] for i in self.performance_data if i[-2] == '0']) / len([i[0] for i in self.performance_data if i[-2] == '0']))])

        self.objects = list(set(self.objects))
        for object in self.objects:
            performance_data_full.append(["test loss MAE(L1) Trained object " + str(object) + ": ", (sum([i[0] for i in self.performance_data if i[-1] == str(object)]) / len([i[0] for i in self.performance_data if i[-1] == str(object)]))])


        performance_data_full.append(["test loss SSIM: ", (sum([i[7] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss SSIM pred ts 1: ", (sum([i[8] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss SSIM pred ts 5: ", (sum([i[9] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss SSIM pred ts 10: ", (sum([i[10] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss SSIM pred force x: ", (sum([i[11] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss SSIM pred force y: ", (sum([i[12] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss SSIM pred force z: ", (sum([i[13] for i in self.performance_data]) / len(self.performance_data))])

        performance_data_full.append (["test loss SSIM Seen objects: ", (sum([i[7] for i in self.performance_data if i[-2] == '1']) / len([i[7] for i in self.performance_data if i[-2] == '1']))])
        performance_data_full.append (["test loss SSIM Novel objects: ", (sum([i[7] for i in self.performance_data if i[-2] == '0']) / len([i[7] for i in self.performance_data if i[-2] == '0']))])

        performance_data_full.append(["test loss PSNR: ", (sum([i[14] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss PSNR pred ts 1: ", (sum([i[15] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss PSNR pred ts 5: ", (sum([i[16] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss PSNR pred ts 10: ", (sum([i[17] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss PSNR pred force x: ", (sum([i[18] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss PSNR pred force y: ", (sum([i[19] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss PSNR pred force z: ", (sum([i[20] for i in self.performance_data]) / len(self.performance_data))])

        performance_data_full.append (["test loss PSNR Seen objects: ", (sum([i[14] for i in self.performance_data if i[-2] == '1']) / len([i[14] for i in self.performance_data if i[-2] == '1']))])
        performance_data_full.append (["test loss PSNR Novel objects: ", (sum([i[14] for i in self.performance_data if i[-2] == '0']) / len([i[14] for i in self.performance_data if i[-2] == '0']))])

        [print (i) for i in performance_data_full]
        np.save(data_save_path + 'model_performance_loss_data', np.asarray(performance_data_full))

    def load_scalars(self):
        self.scaler_tx = load(open(scaler_dir + "tactile_standard_scaler_x.pkl", 'rb'))
        self.scaler_ty = load(open(scaler_dir + "tactile_standard_scaler_y.pkl", 'rb'))
        self.scaler_tz = load(open(scaler_dir + "tactile_standard_scaler_z.pkl", 'rb'))
        self.min_max_scalerx_full_data = load(open(scaler_dir + "tactile_min_max_scalar_x.pkl", 'rb'))
        self.min_max_scalery_full_data = load(open(scaler_dir + "tactile_min_max_scalar_y.pkl", 'rb'))
        self.min_max_scalerz_full_data = load(open(scaler_dir + "tactile_min_max_scalar.pkl", 'rb'))


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze (0).unsqueeze (0)
    window = Variable (_2D_window.expand (channel, 1, window_size, window_size).contiguous ())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d (img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d (img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow (2)
    mu2_sq = mu2.pow (2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d (img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d (img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d (img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean ()
    else:
        return ssim_map.mean (1).mean (1).mean (1)


class SSIM (torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super (SSIM, self).__init__ ()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window (window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size ()

        if channel == self.channel and self.window.data.type () == img1.data.type ():
            window = self.window
        else:
            window = create_window (self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda (img1.get_device ())
            window = window.type_as (img1)

            self.window = window
            self.channel = channel

        return _ssim (img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size ()
    window = create_window (window_size, channel)

    if img1.is_cuda:
        window = window.cuda (img1.get_device ())
    window = window.type_as (img1)

    return _ssim (img1, img2, window, window_size, channel, size_average)

if __name__ == "__main__":
    MT = ModelTester()
    MT.test_full_model()

