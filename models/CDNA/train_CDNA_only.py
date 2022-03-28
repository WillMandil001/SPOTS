# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import numpy as np
from tqdm import tqdm

from datetime import datetime
from torch.utils.data import Dataset

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

import sv2p.cdna as cdna
from sv2p.ssim import DSSIM
from sv2p.criteria import RotationInvarianceLoss

model_save_path = "/home/user/Robotics/SPOTS/models/CDNA/saved_models/"
train_data_dir = "/home/user/Robotics/Data_sets/PRI/single_object_purple/train_formatted/"
model_name = "CDNA_L2_PRI_single_object_purple"

# unique save title:
model_save_path = model_save_path + "single_object_purple_L2_model_" + datetime.now().strftime("%d_%m_%Y_%H_%M/")
os.mkdir(model_save_path)

lr = 0.001
seed = 42
mfreg = 0.01
seqlen = 20
krireg = 0.01
n_masks = 10
indices = (0.9, 0.1)
epochs = 100
batch_size = 16
in_channels = 3
cond_channels = 12
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

        images = []
        for image_name in np.load(train_data_dir + value[2]):
            images.append(np.load(train_data_dir + image_name))

        experiment_number = np.load(train_data_dir + value[3])
        time_steps = np.load(train_data_dir + value[4])
        return [robot_data.astype(np.float32), np.array(images).astype(np.float32), experiment_number, time_steps]


class CDNATrainer():
    def __init__(self):
        self.train_full_loader, self.valid_full_loader = BG.load_full_data()

        self.net = cdna.CDNA(in_channels, cond_channels, n_masks, with_generator=False).to(device)

        self.stat_names = 'predloss', 'kernloss', 'maskloss', 'loss'
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = {'L1': nn.L1Loss(), 'L2': nn.MSELoss(),
                          'DSSIM': DSSIM(self.net.in_channels),
                         }[criterion_name].to(device)
        self.kernel_criterion = RotationInvarianceLoss().to(device)

    def run(self):
        best_training_loss = 100.0
        training_val_losses = []
        progress_bar = tqdm(range(0, epochs), total=(epochs*(len(self.train_full_loader) + len(self.valid_full_loader))))
        for epoch in progress_bar:
            model_save = ""
            self.train_loss, self.val_loss = 0.0, 0.0
            train_max_index, val_max_index = 0.0, 0.0

            # training:
            self.net.train()
            for index, batch_features in enumerate(self.train_full_loader):
                train_max_index += 1
                image = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
                action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
                loss, kernloss, maskloss, total_loss = self.CDNA_pass_through(image, action)
                train_max_index = index
                self.train_loss += loss
                progress_bar.set_description("epoch: {}, ".format(epoch) + "loss: {:.4f}, ".format(float(loss)) + "mean loss: {:.4f}, ".format(self.train_loss/(index+1)))

            # validation:
            self.net.eval()
            with torch.no_grad():
                for index, batch_features in enumerate(self.valid_full_loader):
                    val_max_index += 1
                    image = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
                    action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
                    loss, kernloss, maskloss, total_loss = self.CDNA_pass_through(image, action, validation=True)
                    val_max_index = index
                    self.val_loss += loss
                    progress_bar.set_description("epoch: {}, ".format(epoch) + "VAL loss: {:.4f}, ".format(float(loss)) + "VAL mean loss: {:.4f}, ".format(self.val_loss / (index+1)))

            training_val_losses.append([self.train_loss/(train_max_index+1), self.val_loss/(val_max_index+1)])
            np.save(model_save_path + "train_val_losses", np.array(training_val_losses))

            # early stopping and saving:
            if best_training_loss > self.val_loss/(val_max_index+1):
                best_training_loss = self.val_loss/(val_max_index+1)
                torch.save(self.net, model_save_path + model_name)
                model_save = "saved model"

            print("Training mean loss: {:.4f} || Validation mean loss: {:.4f} || {}".format(self.train_loss/(train_max_index+1), self.val_loss/(val_max_index+1), model_save))

    def CDNA_pass_through(self, images, actions, validation=False):
        self.net.zero_grad()

        hidden = None
        outputs = []
        state = actions[0].to(device)
        if validation:
            with torch.no_grad():
                for index, (sample_tactile, sample_action) in enumerate(zip(images[0:-1].squeeze(), actions[1:].squeeze())):
                    state_action = torch.cat((state, sample_action), 1)
                    tsa = torch.cat(8*[torch.cat(8*[state_action.unsqueeze(2)], axis=2).unsqueeze(3)], axis=3)
                    if index > context_frames-1:
                        predictions_t, hidden, cdna_kerns_t, masks_t = self.net.forward(predictions_t, conditions=tsa, hidden_states=hidden)
                        outputs.append(predictions_t)
                    else:
                        predictions_t, hidden, cdna_kerns_t, masks_t = self.net.forward(sample_tactile, conditions=tsa, hidden_states=hidden)
                        last_output = predictions_t
        else:
            for index, (sample_tactile, sample_action) in enumerate(zip(images[0:-1].squeeze(), actions[1:].squeeze())):
                state_action = torch.cat((state, sample_action), 1)
                tsa = torch.cat(8*[torch.cat(8*[state_action.unsqueeze(2)], axis=2).unsqueeze(3)], axis=3)
                if index > context_frames - 1:
                    predictions_t, hidden, cdna_kerns_t, masks_t = self.net.forward(predictions_t, conditions=tsa, hidden_states=hidden)
                    outputs.append(predictions_t)
                else:
                    predictions_t, hidden, cdna_kerns_t, masks_t = self.net.forward(sample_tactile, conditions=tsa, hidden_states=hidden)
                    last_output = predictions_t
        outputs = [last_output] + outputs

        loss = 0.0
        kernloss = 0.0
        maskloss = 0.0
        for prediction_t, target_t in zip(outputs, images[context_frames:]):
            loss_t, kernloss_t, maskloss_t = self.__compute_loss(prediction_t, cdna_kerns_t, masks_t, target_t)
            loss += loss_t
            kernloss += kernloss_t
            maskloss += maskloss_t
        total_loss = (loss + kernloss + maskloss) / context_frames

        if not validation:
            total_loss.backward()
            self.optimizer.step()

        return  (loss.detach().cpu().item() / context_frames,
                 kernloss.detach().cpu().item() / context_frames,
                 maskloss.detach().cpu().item() / context_frames,
                 total_loss.detach().cpu().item())

    def __compute_loss(self, predictions_t, cdna_kerns_t, masks_t, targets_t):
        loss_t = self.criterion(predictions_t, targets_t)
        kernloss_t = krireg * self.kernel_criterion(cdna_kerns_t)
        maskloss_t = mfreg * masks_t[:, 1:].reshape(-1, masks_t.size(-2) * masks_t.size(-1)).abs().sum(1).mean()
        return loss_t, kernloss_t, maskloss_t


if __name__ == '__main__':
    BG = BatchGenerator()
    Trainer = CDNATrainer()
    Trainer.run()

