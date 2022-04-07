# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import sys
import csv
import cv2
import numpy as np
import click

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torchvision

from universal_networks.SVG import Model as SVG
from universal_networks.SVG_tactile_enhanced import Model as SVG_TE
from universal_networks.SPOTS_SVG_ACTP import Model as SPOTS_SVG_ACTP


class BatchGenerator:
    def __init__(self, train_percentage, train_data_dir, batch_size, image_size, num_workers):
        self.train_percentage = train_percentage
        self.train_data_dir = train_data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.data_map = []
        with open(train_data_dir + 'map.csv', 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                self.data_map.append(row)

    def load_full_data(self):
        dataset_train = FullDataSet(self.data_map, self.train_data_dir, train=True, train_percentage=self.train_percentage, image_size=self.image_size)
        dataset_validate = FullDataSet(self.data_map, self.train_data_dir, validation=True, train_percentage=self.train_percentage, image_size=self.image_size)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        validation_loader = torch.utils.data.DataLoader(dataset_validate, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.data_map = []
        return train_loader, validation_loader


class FullDataSet:
    def __init__(self, data_map, train_data_dir, train=False, validation=False, train_percentage=1.0, image_size=64):
        self.image_size = image_size
        self.train_data_dir = train_data_dir
        if train:
            self.samples = data_map[1:int((len(data_map) * train_percentage))]
        if validation:
            self.samples = data_map[int((len(data_map) * train_percentage)): -1]
        data_map = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        value = self.samples[idx]
        robot_data = np.load(self.train_data_dir + value[0])

        tactile_data = np.load(self.train_data_dir + value[1])
        tactile_images = []
        for tactile_data_sample in tactile_data:
            tactile_images.append(create_image(tactile_data_sample, image_size=self.image_size))

        images = []
        for image_name in np.load(self.train_data_dir + value[2]):
            images.append(np.load(self.train_data_dir + image_name))

        experiment_number = np.load(self.train_data_dir + value[3])
        time_steps = np.load(self.train_data_dir + value[4])
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
        self.training_stages = features["training_stages"]
        self.training_stages_epochs = features["training_stages_epochs"]
        self.tactile_size = features["tactile_size"]
        self.device = features["device"]
        self.num_workers = features["num_workers"]
        self.model_save_path = features["model_save_path"]
        self.stage = self.training_stages[0]

        self.gain = 0.0

        if self.model_name == "SVG":
            self.model = SVG(features)
        elif self.model_name == "SVG_TE":
            self.model = SVG_TE(features)
        elif self.model_name == "SPOTS_SVG_ACTP":
            self.model = SPOTS_SVG_ACTP(features)

        self.model.initialise_model()

        BG = BatchGenerator(self.train_percentage, self.train_data_dir, self.batch_size, self.image_width, self.num_workers)
        self.train_full_loader, self.valid_full_loader = BG.load_full_data()

        if self.criterion == "L1":
            self.criterion = nn.L1Loss()
        if self.criterion == "L2":
            self.criterion = nn.MSELoss()

    def train_full_model(self):
        plot_training_loss = []
        plot_validation_loss = []
        previous_val_mean_loss = 100.0
        best_val_loss = 100.0
        early_stop_clock = 0
        progress_bar = tqdm(range(0, self.epochs), total=(self.epochs*len(self.train_full_loader)))
        for epoch in progress_bar:
            if epoch <= self.training_stages_epochs[0]:
                self.stage = self.training_stages[0]
            elif epoch <= self.training_stages_epochs[1]:
                self.stage = self.training_stages[1]
            elif epoch <= self.training_stages_epochs[2]:
                self.stage = self.training_stages[2]

            self.model.set_train()

            epoch_mae_losses = 0.0
            epoch_kld_losses = 0.0
            for index, batch_features in enumerate(self.train_full_loader):

                if self.stage == "scene_loss_plus_tactile_gradual_increase":
                    gain += 0.00001
                    if gain > 0.01:
                        gain = 0.01

                if batch_features[1].shape[0] == self.batch_size:
                    mae, kld, mae_tactile, predictions = self.format_and_run_batch(batch_features, test=False)
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
                    if batch_features[1].shape[0] == self.batch_size:
                        val_mae, val_kld, mae_tactile, predictions = self.format_and_run_batch(batch_features, test=True)
                        val_mae_losses += val_mae.item()

            plot_validation_loss.append(val_mae_losses / index__)
            print("Validation mae: {:.4f}, ".format(val_mae_losses / index__))

            # save the train/validation performance data
            np.save(self.model_save_path + "plot_validation_loss", np.array(plot_validation_loss))
            np.save(self.model_save_path + "plot_training_loss", np.array(plot_training_loss))

            # save at the end of every training stage:
            if self.stage != "":
                if epoch in self.training_stages_epochs:
                    self.model.save_model(self.stage)

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
                    if self.stage != "":
                        self.model.save_model("best")
                    else:
                        self.model.save_model()
                    best_val_loss = val_mae_losses / index__
                early_stop_clock = 0
                previous_val_mean_loss = val_mae_losses / index__

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


@click.command()
@click.option('--model_name', type=click.Path(), default="SVG_TE", help='Set name for prediction model, SVG, SVG_TE, SPOTS_SVG_ACTP')
@click.option('--batch_size', type=click.INT, default=8, help='Batch size for training.')
@click.option('--lr', type=click.FLOAT, default = 0.0001, help = "learning rate")
@click.option('--beta1', type=click.FLOAT, default = 0.9, help = "Beta gain")
@click.option('--log_dir', type=click.Path(), default = 'logs/lp', help = "Not sure :D")
@click.option('--optimizer', type=click.Path(), default = 'adam', help = "what optimiser to use - only adam available currently")
@click.option('--niter', type=click.INT, default = 300, help = "")
@click.option('--seed', type=click.INT, default = 1, help = "")
@click.option('--image_width', type=click.INT, default = 64, help = "Size of scene image data")
@click.option('--dataset', type=click.Path(), default = 'object1_motion1', help = "name of the dataset")
@click.option('--n_past', type=click.INT, default = 2, help = "context sequence length")
@click.option('--n_future', type=click.INT, default = 5, help = "time horizon sequence length")
@click.option('--n_eval', type=click.INT, default = 7, help = "sum of context and time horizon")
@click.option('--prior_rnn_layers', type=click.INT, default = 3, help = "number of LSTMs in the prior model")
@click.option('--posterior_rnn_layers', type=click.INT, default = 3, help = "number of LSTMs in the posterior model")
@click.option('--predictor_rnn_layers', type=click.INT, default = 4, help = "number of LSTMs in the frame predictor model")
@click.option('--state_action_size', type=click.INT, default = 12, help = "size of action conditioning data")
@click.option('--z_dim', type=click.INT, default = 10, help = "number of latent variables to estimate")
@click.option('--beta', type=click.FLOAT, default = 0.0001, help = "beta gain")
@click.option('--data_threads', type=click.INT, default = 5, help = "")
@click.option('--num_digits', type=click.INT, default = 2, help = "")
@click.option('--last_frame_skip', type=click.Path(), default = 'store_true', help = "")
@click.option('--epochs', type=click.INT, default = 125, help = "number of epochs to run for ")
@click.option('--train_percentage', type=click.FLOAT, default = 0.9, help = "")
@click.option('--validation_percentage', type=click.FLOAT, default = 0.1, help = "")
@click.option('--criterion', type=click.Path(), default = "L1", help = "")
@click.option('--tactile_size', type=click.INT, default = 0, help = "size of tacitle frame - 48, if no tacitle data set to 0")
@click.option('--g_dim', type=click.INT, default = 256, help = "size of encoded data for input to prior")
@click.option('--rnn_size', type=click.INT, default = 256, help = "size of encoded data for input to frame predictor (g_dim = rnn-size)")
@click.option('--channels', type=click.INT, default = 3, help = "input channels")
@click.option('--out_channels', type=click.INT, default = 3, help = "output channels")
@click.option('--training_stages', type=click.Path(), default = "", help = "define the training stages - if none leave blank - available: 3part")
@click.option('--training_stages_epochs', type=click.Path(), default = "50,75,125", help = "define the end point of each training stage")
@click.option('--num_workers', type=click.INT, default = 12, help = "number of workers used by the data loader")
@click.option('--model_save_path', type=click.Path(), default = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/", help = "")
@click.option('--train_data_dir', type=click.Path(), default = "/home/user/Robotics/Data_sets/PRI/object1_motion1/train_formatted/", help = "")
@click.option('--scaler_dir', type=click.Path(), default = "/home/user/Robotics/Data_sets/PRI/object1_motion1/scalars/", help = "")
def main(model_name, batch_size, lr, beta1, log_dir, optimizer, niter, seed, image_width, dataset,
         n_past, n_future, n_eval, prior_rnn_layers, posterior_rnn_layers, predictor_rnn_layers, state_action_size,
         z_dim, beta, data_threads, num_digits, last_frame_skip, epochs, train_percentage, validation_percentage,
         criterion, tactile_size, g_dim, rnn_size, channels, out_channels, training_stages, training_stages_epochs,
         num_workers, model_save_path, train_data_dir, scaler_dir):

    # unique save title:
    model_save_path = model_save_path + model_name
    try:
        os.mkdir(model_save_path)
    except FileExistsError or FileNotFoundError:
        pass
    try:
        model_save_path = model_save_path + "/model_" + datetime.now().strftime("%d_%m_%Y_%H_%M/")
        os.mkdir(model_save_path)
    except FileExistsError or FileNotFoundError:
        pass

    model_dir = model_save_path
    data_root = train_data_dir

    if model_name == "SVG":
        g_dim = 256  # 128
        rnn_size = 256
        channels = 3
        out_channels = 3
        training_stages = [""]
        training_stages_epochs = [epochs]
    elif model_name == "SVG_TE":
        g_dim = 256 * 2
        rnn_size = 256 * 2
        channels = 6
        out_channels = 6
        training_stages = [""]
        training_stages_epochs = [epochs]
        tactile_size = [image_width, image_width]
    elif model_name == "SPOTS_SVG_ACTP":
        g_dim = 256
        rnn_size = 256
        channels = 3
        out_channels = 3
        training_stages = ["scene_only", "tactile_loss_plus_scene_fixed", "scene_loss_plus_tactile_gradual_increase"]
        training_stages_epochs = [50, 75, 125]
        tactile_size = 48

    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available

    features = {"lr": lr, "beta1": beta1, "batch_size": batch_size, "log_dir": log_dir, "model_dir": model_dir, "data_root": data_root, "optimizer": optimizer, "niter": niter, "seed": seed,
                "image_width": image_width, "channels": channels, "out_channels": out_channels, "dataset": dataset, "n_past": n_past, "n_future": n_future, "n_eval": n_eval, "rnn_size": rnn_size, "prior_rnn_layers": prior_rnn_layers,
                "posterior_rnn_layers": posterior_rnn_layers, "predictor_rnn_layers": predictor_rnn_layers, "state_action_size": state_action_size, "z_dim": z_dim, "g_dim": g_dim, "beta": beta, "data_threads": data_threads, "num_digits": num_digits,
                "last_frame_skip": last_frame_skip, "epochs": epochs, "train_percentage": train_percentage, "validation_percentage": validation_percentage, "criterion": criterion, "model_name": model_name,
                "train_data_dir": train_data_dir, "scaler_dir": scaler_dir, "device": device, "training_stages":training_stages, "training_stages_epochs": training_stages_epochs, "tactile_size":tactile_size, "num_workers":num_workers,
                "model_save_path":model_save_path}

    # save features
    w = csv.writer(open(model_save_path + "/features.csv", "w"))
    for key, val in features.items():
        w.writerow([key, val])

    UMT = UniversalModelTrainer(features)
    UMT.train_full_model()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    main ()
