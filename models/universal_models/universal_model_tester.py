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
from universal_networks.SVG import Model as SVG
from universal_networks.SVG_tactile_enhanced import Model as SVG_TE
from universal_networks.SPOTS_SVG_ACTP import Model as SPOTS_SVG_ACTP


class BatchGenerator:
    def __init__(self, test_data_dir, batch_size, image_size):
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.data_map = []
        with open(test_data_dir + 'map.csv', 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                self.data_map.append(row)

    def load_data(self):
        dataset_test = FullDataSet(self.data_map, image_size=self.image_size)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=1)
        self.data_map = []
        return test_loader


class FullDataSet:
    def __init__(self, data_map, image_size=64):
        self.image_size = image_size
        self.samples = data_map[1:]
        data_map = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        value = self.samples[idx]
        robot_data = np.load(test_data_dir + value[0])

        tactile_data = np.load(test_data_dir + value[1])
        tactile_images = []
        for tactile_data_sample in tactile_data:
            tactile_images.append(create_image(tactile_data_sample, image_size=self.image_size))

        images = []
        for image_name in np.load(test_data_dir + value[2]):
            images.append(np.load(test_data_dir + image_name))

        experiment_number = np.load(test_data_dir + value[3])
        time_steps = np.load(test_data_dir + value[4])
        return [robot_data.astype(np.float32), np.array(images).astype(np.float32), np.array(tactile_images).astype(np.float32), np.array(tactile_data).astype(np.float32), experiment_number, time_steps]


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


class UniversalTester():
    def __init__(self, data_save_path, model_save_path, test_data_dir, scaler_dir, model_save_name):
        self.data_save_path = data_save_path
        self.model_save_path = model_save_path
        self.test_data_dir = test_data_dir
        self.scaler_dir = scaler_dir

        saved_model = torch.load(model_save_path + model_save_name)

        # load features
        features = saved_model["features"]
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

        # load model
        print(features["model_name"])
        if self.model_name == "SVG":
            self.model = SVG(features)
        if self.model_name == "SVG_TE":
            self.model = SVG_TE(features)
        if self.model_name == "SPOTS_SVG_ACTP":
            self.model = SPOTS_SVG_ACTP(features)

        self.model.load_model(full_model = saved_model)
        # [saved_model[name].to("cpu") for name in saved_model if name != "features"]
        saved_model = []

        # load test set:
        BG = BatchGenerator(self.test_data_dir, self.batch_size, self.image_width)
        self.test_full_loader = BG.load_data()

        # test dataset
        self.test_model()

    def test_model(self):
        self.gain = None
        self.stage = None
        self.objects = []
        self.performance_data = []
        self.prediction_data = []
        self.current_exp = 0
        self.objects = []
        self.total_losses = []

        self.model.set_test()

        for index, batch_features in enumerate(self.test_full_loader):
            if batch_features[1].shape[0] == self.batch_size:
                mae, kld, mae_tactile, predictions = self.format_and_run_batch(batch_features, test=True)

        np.save(data_save_path + "test_no_new_losses_per_trial", np.array(self.total_losses))
        print(self.total_losses)
        final_loss = sum([i[1] for i in self.total_losses]) / len(self.total_losses)
        np.save(data_save_path + "test_no_new_loss", np.array(final_loss))
        print(np.array(self.total_losses))
        print(final_loss)

    # def test_qualitative(self):
    #     for i in range(10):
    #         plt.figure(1)
    #         f, axarr = plt.subplots(1, 2)
    #         axarr[0].set_title("predictions: t_" + str(i))
    #         axarr[0].imshow(np.array(predictions[i][7].permute(1, 2, 0).cpu().detach()))
    #         axarr[1].set_title("ground truth: t_" + str(i))
    #         axarr[1].imshow(np.array(images[context_frames + i][7].permute(1, 2, 0).cpu().detach()))
    #         plt.savefig(data_save_path + str(i) + ".png")

    def format_and_run_batch(self, batch_features, test):
        mae, kld, mae_tactile, predictions = None, None, None, None
        if self.model_name == "SVG":
            images = batch_features[1].permute(1, 0, 4, 3, 2).to(self.device)
            action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.device)
            mae, kld, predictions = self.model.run(scene=images, actions=action, test=test)

        elif self.model_name == "SVG_TE":
            images = batch_features[1].permute(1, 0, 4, 3, 2).to(self.device)
            tactile = batch_features[2].permute(1, 0, 4, 3, 2).to(self.device)
            scene_and_touch = torch.cat((tactile, images), 2)
            action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.device)
            mae, kld, predictions = self.model.run(scene_and_touch=scene_and_touch, actions=action, test=test)

        elif self.model_name == "SPOTS_SVG_ACTP":
            action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.device)
            images = batch_features[1].permute(1, 0, 4, 3, 2).to(self.device)
            tactile = torch.flatten(batch_features[3].permute(1, 0, 2, 3).to(self.device), start_dim=2)
            mae, kld, mae_tactile, predictions = self.model.run(scene=images, tactile=tactile, actions=action, gain=self.gain, test=test, stage=self.stage)

        return mae, kld, mae_tactile, predictions


if __name__ == "__main__":
    # model names: SVG, SVG_TE, SPOTS_SVG_ACTP
    model_name = "SVG"
    model_folder_name = "model_05_04_2022_16_16"

    model_save_path = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/" + model_name + "/" + model_folder_name + "/"
    test_data_dir  = "/home/user/Robotics/Data_sets/PRI/object1_motion1/test_no_new_formatted/"
    scaler_dir      = "/home/user/Robotics/Data_sets/PRI/object1_motion1/scalars/"

    data_save_path = model_save_path + "performance_data/"
    try:
        os.mkdir(data_save_path)
    except FileExistsError or FileNotFoundError:
        pass

    model_save_name = model_name + "_model"

    MT = UniversalTester(data_save_path, model_save_path, test_data_dir, scaler_dir, model_save_name)