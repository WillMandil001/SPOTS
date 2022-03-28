# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import cv2
import numpy as np

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from universal_models.SVG import Model as SVG


class BatchGenerator:
    def __init__(self, train_percentage, train_data_dir, batch_size, image_size):
        self.train_percentage = train_percentage
        self.train_data_dir = train_data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.data_map = []
        with open(train_data_dir + 'map.csv', 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                self.data_map.append(row)

    def load_full_data(self):
        dataset_train = FullDataSet(self.data_map, train=True, train_percentage=self.train_percentage, image_size=self.image_size)
        dataset_validate = FullDataSet(self.data_map, validation=True, train_percentage=self.train_percentage, image_size=self.image_size)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=12)
        validation_loader = torch.utils.data.DataLoader(dataset_validate, batch_size=batch_size, shuffle=True, num_workers=12)
        self.data_map = []
        return train_loader, validation_loader


class FullDataSet:
    def __init__(self, data_map, train=False, validation=False, train_percentage=1.0, image_size=[64,64]):
        self.image_size = image_size
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
            tactile_images.append(create_image(tactile_data_sample[0], tactile_data_sample[1], tactile_data_sample[2], self.image_size))

        images = []
        for image_name in np.load(train_data_dir + value[2]):
            images.append(np.load(train_data_dir + image_name))

        experiment_number = np.load(train_data_dir + value[3])
        time_steps = np.load(train_data_dir + value[4])
        return [robot_data.astype(np.float32), np.array(images).astype(np.float32), np.array(tactile_images).astype(np.float32), experiment_number, time_steps]


def create_image(tactile_x, tactile_y, tactile_z, image_size):
    # convert tactile data into an image:
    image = np.zeros((4, 4, 3), np.float32)
    index = 0
    for x in range(4):
        for y in range(4):
            image[x][y] = [tactile_x[index], tactile_y[index], tactile_z[index]]
            index += 1
    reshaped_image = cv2.resize(image.astype(np.float32), dsize=(image_size[0], image_size[1]), interpolation=cv2.INTER_CUBIC)
    return reshaped_image


class UniversalModelTrainer:
    def __init__(self, features):
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

        if model_name == "SVG":
            self.model = SVG(features)

        BG = BatchGenerator(self.train_percentage, self.train_data_dir, self.batch_size, self.image_width)
        self.train_full_loader, self.valid_full_loader = BG.load_full_data()

        if criterion == "L1":
            self.criterion = nn.L1Loss()
        if criterion == "L2":
            self.criterion = nn.MSELoss()

    def train_full_model(self):
        plot_training_loss = []
        plot_validation_loss = []
        previous_val_mean_loss = 100.0
        best_val_loss = 100.0
        early_stop_clock = 0
        progress_bar = tqdm(range(0, epochs), total=(epochs*len(self.train_full_loader)))
        for epoch in progress_bar:
            self.model.set_train()

            epoch_mae_losses = 0.0
            epoch_kld_losses = 0.0
            for index, batch_features in enumerate(self.train_full_loader):
                if batch_features[1].shape[0] == batch_size:
                    images = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
                    action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
                    mae, kld, predictions = self.model.run(scene=images, actions=action, test=False)
                    epoch_mae_losses += mae.item()
                    epoch_kld_losses += kld.item()
                    if index:
                        mean_kld = epoch_kld_losses / index
                        mean_mae = epoch_mae_losses / index
                    else:
                        mean_kld = 0.0
                        mean_mae = 0.0

                    progress_bar.set_description("epoch: {}, ".format(epoch) + "MAE: {:.4f}, ".format(float(mae.item())) + "kld: {:.4f}, ".format(float(kld.item())) + "mean MAE: {:.4f}, ".format(mean_mae) + "mean kld: {:.4f}, ".format(mean_kld))
                    progress_bar.update()

            plot_training_loss.append([mean_mae, mean_kld])

            # Validation checking:
            self.model.set_test()
            val_mae_losses = 0.0
            val_kld_losses = 0.0
            with torch.no_grad():
                for index__, batch_features in enumerate(self.valid_full_loader):
                    if batch_features[1].shape[0] == batch_size:
                        images = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
                        action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
                        val_mae, val_kld, predictions = self.model.run(scene=images, actions=action, test=True)
                        val_mae_losses += val_mae.item()

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
                    self.model.save_model()
                    best_val_loss = val_mae_losses / index__
                early_stop_clock = 0
                previous_val_mean_loss = val_mae_losses / index__


if __name__ == "__main__":
    model_save_path = "/home/wmandil/Robotics/SPOTS/SPOTS/models/SVG/saved_models/PRI_object1_motion1_SVG_"
    train_data_dir = "/home/wmandil/Robotics/Data_sets/PRI/object1_motion1/train_formatted/"
    scaler_dir = "/home/wmandil/Robotics/Data_sets/PRI/object1_motion1/scalars/"

    # unique save title:
    model_save_path = model_save_path + "model_" + datetime.now().strftime("%d_%m_%Y_%H_%M/")
    os.mkdir(model_save_path)

    lr = 0.0001
    beta1 = 0.9
    batch_size = 32
    log_dir = 'logs/lp'
    model_dir = model_save_path
    data_root = 'data'
    optimizer = 'adam'
    niter = 300
    seed = 1
    image_width = 64
    channels = 3
    out_channels = 3
    dataset = 'smmnist'
    n_past = 10
    n_future = 10
    n_eval = 20
    rnn_size = 256
    prior_rnn_layers = 3
    posterior_rnn_layers = 3
    predictor_rnn_layers = 4
    state_action_size = 12
    z_dim = 10  # number of latent variables
    g_dim = 256  # 128
    beta = 0.0001  # was 0.0001
    data_threads = 5
    num_digits = 2
    last_frame_skip = 'store_true'
    epochs = 100
    train_percentage = 0.9
    validation_percentage = 0.1
    criterion = "L1"
    model_name = "SVG"

    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available

    features = {"lr": lr, "beta1": beta1, "batch_size": batch_size, "log_dir": log_dir, "model_dir": model_dir, "data_root": data_root, "optimizer": optimizer, "niter": niter, "seed": seed,
                "image_width": image_width, "channels": channels, "out_channels": out_channels, "dataset": dataset, "n_past": n_past, "n_future": n_future, "n_eval": n_eval, "rnn_size": rnn_size, "prior_rnn_layers": prior_rnn_layers,
                "posterior_rnn_layers": posterior_rnn_layers, "predictor_rnn_layers": predictor_rnn_layers, "z_dim": z_dim, "g_dim": g_dim, "beta": beta, "data_threads": data_threads, "num_digits": num_digits,
                "last_frame_skip": last_frame_skip, "epochs": epochs, "train_percentage": train_percentage, "validation_percentage": validation_percentage, "criterion": criterion, "model_name": model_name,
                "train_data_dir": train_data_dir, "scaler_dir": scaler_dir, "device": device}

    # save features
    w = csv.writer(open(model_save_path + "/features.csv", "w"))
    for key, val in features.items():
        w.writerow([key, val])

    UMT = UniversalModelTrainer(model_name, features)
    UMT.train_full_model()
