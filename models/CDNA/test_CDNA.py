# -*- coding: utf-8 -*-
# RUN IN PYTHON 3

import os
import csv
import cv2
import math

from pickle import load
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

import sv2p.cdna as cdna
from sv2p.ssim import DSSIM
from sv2p.criteria import RotationInvarianceLoss

model_path      = "/home/user/Robotics/SPOTS/models/CDNA/saved_models/single_object_purple_L2_model_19_02_2022_13_27/CDNA_L2_PRI_single_object_purple"
data_save_path  = "/home/user/Robotics/SPOTS/models/CDNA/saved_models/single_object_purple_L2_model_19_02_2022_13_27/"
test_data_dir   = "/home/user/Robotics/Data_sets/PRI/single_object_purple/test_no_new_formatted/"
scaler_dir      = "/home/user/Robotics/Data_sets/PRI/single_object_purple/scalars/"

lr = 0.001
seed = 42
mfreg = 0.1
seqlen = 20
krireg = 0.1
n_masks = 10
indices =(0.9, 0.1)
max_epoch = 100
batch_size = 8
in_channels = 3
cond_channels = 0
context_frames = 10
train_percentage = 0.9
validation_percentage = 0.1

device = "cuda"
warm_start = False
dataset_name = "PRI_single_object_purple"
criterion_name = "L2"
scheduled_sampling_k = False

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available


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


class ModelTester():
    def __init__(self):
        # load model:
        self.full_model = torch.load(model_path).to(device)
        self.full_model.eval()

        self.stat_names = 'predloss', 'kernloss', 'maskloss', 'loss'
        self.optimizer = optim.Adam(self.full_model.parameters(), lr=lr)
        self.criterion = {'L1': nn.L1Loss(), 'L2': nn.MSELoss(),
                          'DSSIM': DSSIM(self.full_model.in_channels),
                         }[criterion_name].to(device)
        self.kernel_criterion = RotationInvarianceLoss().to(device)

    def test_full_model(self):
        self.number_of_trials = [2]  # i for i in range(30)]
        self.performance_data = []
        self.prediction_data  = []
        self.tg_back_scaled   = []
        self.tp1_back_scaled = []
        self.tp5_back_scaled = []
        self.tp10_back_scaled = []
        self.current_exp      = 0
        self.objects = []

        for trial in self.number_of_trials:
            BG = BatchGenerator(test_data_dir, batch_size, trial)
            self.test_full_loader = BG.load_full_data()

            for index, batch_features in enumerate(self.test_full_loader):
                images = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
                action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
                image_predictions = self.CDNA_pass_through(images, action)
                for i in range(10):
                    plt.figure(1)
                    f, axarr = plt.subplots(1, 2)
                    axarr[0].set_title("predictions: t_" + str(i))
                    axarr[0].imshow(np.array(image_predictions[i][7].permute(1, 2, 0).cpu().detach()))
                    axarr[1].set_title("ground truth: t_" + str(i))
                    axarr[1].imshow(np.array(images[context_frames:][i][7].permute(1, 2, 0).cpu().detach()))
                    plt.savefig(data_save_path + str (i) + ".png")
                break
            break

    def CDNA_pass_through(self, images, actions):
        self.full_model.zero_grad()

        hidden = None
        outputs = []
        state = actions[0].to(device)
        with torch.no_grad():
            for index, (sample_tactile, sample_action) in enumerate(zip(images[0:-1].squeeze(), actions[1:].squeeze())):
                state_action = torch.cat((state, sample_action), 1)
                tsa = torch.cat(8*[torch.cat(8*[state_action.unsqueeze(2)], axis=2).unsqueeze(3)], axis=3)
                if index > context_frames-1:
                    predictions_t, hidden, cdna_kerns_t, masks_t = self.full_model.forward(predictions_t, conditions=tsa, hidden_states=hidden)
                    outputs.append(predictions_t)
                else:
                    predictions_t, hidden, cdna_kerns_t, masks_t = self.full_model.forward(sample_tactile, conditions=tsa, hidden_states=hidden)
                    last_output = predictions_t

        outputs = [last_output] + outputs

        return torch.stack(outputs)


if __name__ == "__main__":
    MT = ModelTester()
    MT.test_full_model()