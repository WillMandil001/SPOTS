# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import cv2
import numpy as np
from tqdm import tqdm

from pickle import load
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torchvision

seed = 42

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available


class BatchGenerator:
    def __init__(self, test_data_dir, batch_size, image_size, trial_number):
        self.batch_size = batch_size
        self.image_size = image_size
        self.trial_number = trial_number
        self.test_data_dir = test_data_dir + 'test_trial_' + str(self.trial_number) + '/'
        self.data_map = []
        print(self.test_data_dir + 'map_' + str(self.trial_number) + '.csv')
        with open(self.test_data_dir + 'map_' + str(self.trial_number) + '.csv', 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                self.data_map.append(row)

    def load_full_data(self):
        dataset_test = FullDataSet(self.data_map, self.test_data_dir, self.image_size)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False)
        self.data_map = []
        return test_loader


class FullDataSet:
    def __init__(self, data_map, test_data_dir, image_size):
        self.image_size = image_size
        self.test_data_dir = test_data_dir
        self.samples = data_map[1:]
        data_map = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        value = self.samples[idx]
        robot_data = np.load(self.test_data_dir + value[0])

        if self.image_size == 0:
            tactile_data = np.load(self.test_data_dir + value[1])
            experiment_number = np.load(self.test_data_dir + value[2])
            time_steps = np.load(self.test_data_dir + value[3])
        else:
            tactile_data = []
            for image_name in np.load(self.test_data_dir + value[2]):
                tactile_data.append(np.load(self.test_data_dir + image_name))
            tactile_data = np.array(tactile_data)
            experiment_number = np.load(self.test_data_dir + value[3])
            time_steps = np.load(self.test_data_dir + value[4])

        return [robot_data.astype(np.float32), tactile_data.astype(np.float32), experiment_number, time_steps]


class UniversalModelTester:
    def __init__(self, model, number_of_trials, test_data_dir, image_size, criterion, model_path, data_save_path,
                 scaler_dir, model_name, batch_size, context_frames, sequence_length, latent_channels):
        self.number_of_trials = number_of_trials
        self.batch_size = batch_size
        self.context_frames = context_frames
        self.sequence_length = sequence_length
        self.model_name = model_name
        self.model_save_path = model_path
        self.data_save_path = data_save_path
        self.test_data_dir = test_data_dir
        self.scaler_dir = scaler_dir
        self.model = model
        self.image_size = image_size
        self.latent_channels = latent_channels
        if criterion == "L1":
            self.criterion = nn.L1Loss()
        if criterion == "L2":
            self.criterion = nn.MSELoss()
        self.load_scalars()

        if self.model_name == "SVG":
            (self.lr, self.beta1, self.batch_size, self.log_dir, self.model_dir, self.name, self.data_root, self.optimizer, self.niter, self.seed, self.epoch_size,
             self.image_width, self.channels, self.out_channels, self.dataset, self.n_past, self.n_future, self.n_eval, self.rnn_size, self.prior_rnn_layers,
             self.posterior_rnn_layers, self.predictor_rnn_layers, self.z_dim, self.g_dim, self.beta, self.data_threads, self.num_digits,
             self.last_frame_skip, self.epochs, _, __) = self.model["features"]
            self.frame_predictor = self.model["frame_predictor"].to(device)
            self.posterior = self.model["posterior"].to(device)
            self.prior = self.model["prior"].to(device)
            self.encoder = self.model["encoder"].to(device)
            self.decoder = self.model["decoder"].to(device)

    def test_model(self):
        for trial in self.number_of_trials:
            print(trial)
            BG = BatchGenerator(self.test_data_dir, self.batch_size, self.image_size, trial)
            self.test_full_loader = BG.load_full_data()

            for index, batch_features in tqdm(enumerate(self.test_full_loader)):
                tactile_predictions, tactile_groundtruth, loss = self.run_batch(batch_features[1], batch_features[0])
                if index == 0:
                    if self.image_size == 0:
                        prediction_data = np.array(tactile_predictions.permute(1, 0, 2).cpu().detach())
                        groundtruth_data = np.array(tactile_groundtruth.permute(1, 0, 2).cpu().detach())
                    else:
                        prediction_data = np.array (tactile_predictions.permute(1, 0, 2, 3, 4).cpu().detach())
                        groundtruth_data = np.array (tactile_groundtruth.permute(1, 0, 2, 3, 4).cpu().detach())
                else:
                    if self.image_size == 0:
                        prediction_data = np.concatenate((prediction_data, np.array(tactile_predictions.permute(1, 0, 2).cpu().detach())), axis=0)
                        groundtruth_data = np.concatenate((groundtruth_data, np.array(tactile_groundtruth.permute(1, 0, 2).cpu().detach())), axis=0)
                    else:
                        prediction_data = np.concatenate((prediction_data, np.array(tactile_predictions.permute(1, 0, 2, 3, 4).cpu().detach())), axis=0)
                        groundtruth_data = np.concatenate((groundtruth_data, np.array(tactile_groundtruth.permute(1, 0, 2, 3, 4).cpu().detach())), axis=0)

            prediction_data_descaled, groundtruth_data_descaled = self.scale_back(np.array(prediction_data), np.array(groundtruth_data))
            self.save_trial(prediction_data_descaled, groundtruth_data_descaled, trial_number=trial)

    def scale_back(self, tactile_data, groundtruth_data):
        pt_descalled_data = []
        gt_descalled_data = []
        if self.image_size == 0:
            (ptx, pty, ptz) = np.split(tactile_data, 3, axis=2)
            (gtx, gty, gtz) = np.split(groundtruth_data, 3, axis=2)
            for time_step in range(tactile_data.shape[0]):
                xela_ptx_inverse_minmax = self.min_max_scalerx_full_data.inverse_transform(ptx[time_step])
                xela_pty_inverse_minmax = self.min_max_scalery_full_data.inverse_transform(pty[time_step])
                xela_ptz_inverse_minmax = self.min_max_scalerz_full_data.inverse_transform(ptz[time_step])
                xela_ptx_inverse_full = self.scaler_tx.inverse_transform(xela_ptx_inverse_minmax)
                xela_pty_inverse_full = self.scaler_ty.inverse_transform(xela_pty_inverse_minmax)
                xela_ptz_inverse_full = self.scaler_tz.inverse_transform(xela_ptz_inverse_minmax)
                pt_descalled_data.append(np.concatenate((xela_ptx_inverse_full, xela_pty_inverse_full, xela_ptz_inverse_full), axis=1))

                xela_gtx_inverse_minmax = self.min_max_scalerx_full_data.inverse_transform(gtx[time_step])
                xela_gty_inverse_minmax = self.min_max_scalery_full_data.inverse_transform(gty[time_step])
                xela_gtz_inverse_minmax = self.min_max_scalerz_full_data.inverse_transform(gtz[time_step])
                xela_gtx_inverse_full = self.scaler_tx.inverse_transform(xela_gtx_inverse_minmax)
                xela_gty_inverse_full = self.scaler_ty.inverse_transform(xela_gty_inverse_minmax)
                xela_gtz_inverse_full = self.scaler_tz.inverse_transform(xela_gtz_inverse_minmax)
                gt_descalled_data.append(np.concatenate((xela_gtx_inverse_full, xela_gty_inverse_full, xela_gtz_inverse_full), axis=1))
        else:
            for time_step in range(tactile_data.shape[0]):
                # convert the image back to the 48 taxel features:
                sequence_p = []
                sequence_g = []
                for ps in range(tactile_data.shape[1]):
                    sequence_p.append(cv2.resize(torch.tensor(tactile_data)[time_step][ps].permute(1, 2, 0).numpy(), dsize=(4, 4), interpolation=cv2.INTER_CUBIC).flatten())
                    sequence_g.append(cv2.resize(torch.tensor(groundtruth_data)[time_step][ps].permute(1, 2, 0).numpy(), dsize=(4, 4), interpolation=cv2.INTER_CUBIC).flatten())

                (ptx, pty, ptz) = np.split(np.array(sequence_p), 3, axis=1)
                (gtx, gty, gtz) = np.split(np.array(sequence_g), 3, axis=1)

                xela_ptx_inverse_minmax = self.min_max_scalerx_full_data.inverse_transform(ptx)
                xela_pty_inverse_minmax = self.min_max_scalery_full_data.inverse_transform(pty)
                xela_ptz_inverse_minmax = self.min_max_scalerz_full_data.inverse_transform(ptz)
                xela_ptx_inverse_full = self.scaler_tx.inverse_transform(xela_ptx_inverse_minmax)
                xela_pty_inverse_full = self.scaler_ty.inverse_transform(xela_pty_inverse_minmax)
                xela_ptz_inverse_full = self.scaler_tz.inverse_transform(xela_ptz_inverse_minmax)
                pt_descalled_data.append(np.concatenate((xela_ptx_inverse_full, xela_pty_inverse_full, xela_ptz_inverse_full), axis=1))

                xela_gtx_inverse_minmax = self.min_max_scalerx_full_data.inverse_transform(gtx)
                xela_gty_inverse_minmax = self.min_max_scalery_full_data.inverse_transform(gty)
                xela_gtz_inverse_minmax = self.min_max_scalerz_full_data.inverse_transform(gtz)
                xela_gtx_inverse_full = self.scaler_tx.inverse_transform(xela_gtx_inverse_minmax)
                xela_gty_inverse_full = self.scaler_ty.inverse_transform(xela_gty_inverse_minmax)
                xela_gtz_inverse_full = self.scaler_tz.inverse_transform(xela_gtz_inverse_minmax)
                gt_descalled_data.append(np.concatenate((xela_gtx_inverse_full, xela_gty_inverse_full, xela_gtz_inverse_full), axis=1))

        return np.array(pt_descalled_data), np.array(gt_descalled_data)

    def SVG_pass_through(self, tactiles, actions):
        # if not batch size == add a bunch of zeros and then remove them from the predictions:
        appended = False
        if tactiles.shape[1] != self.batch_size:
            cut_point = tactiles.shape[1]
            tactiles = torch.cat((tactiles, torch.zeros([20, (self.batch_size - tactiles.shape[1]), 3, self.image_size,self.image_size]).to(device)), axis=1)
            actions = torch.cat((actions, torch.zeros([20, (self.batch_size - actions.shape[1]), 6]).to(device)), axis=1)
            appended = True


        states = actions[0, :, :]
        state_action = torch.cat((torch.cat(20*[states.unsqueeze(0)], 0), actions), 2)
        state_action_image = torch.cat(32*[torch.cat(32*[state_action.unsqueeze(3)], axis=3).unsqueeze(4)], axis=4)
        x = torch.cat((state_action_image, tactiles), 2)

        self.frame_predictor.zero_grad()
        self.posterior.zero_grad()
        self.prior.zero_grad()
        self.encoder.zero_grad()
        self.decoder.zero_grad()

        self.frame_predictor.hidden = self.frame_predictor.init_hidden()
        self.posterior.hidden = self.posterior.init_hidden()
        self.prior.hidden = self.prior.init_hidden()

        gen_seq = []
        gt_seq = []
        x_in = x[0]
        for i in range(1, self.n_eval):
            h = self.encoder(x_in)
            if self.last_frame_skip or i < self.n_past:
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < self.n_past:
                h_target = self.encoder(x[i])[0].detach()
                z_t, _, _ = self.posterior(h_target)
                self.prior(h)
                self.frame_predictor(torch.cat([h, z_t], 1))
                x_in = x[i]
            else:
                z_t, _, _ = self.prior(h)
                h = self.frame_predictor(torch.cat([h, z_t], 1)).detach()
                x_in = self.decoder([h, skip]).detach()
                x_in = torch.cat((state_action_image[i], x_in), 1)  # add the next actions back into the set
                gen_seq.append(x_in[:, 12:, :, :].data.cpu().numpy())
                gt_seq.append(x[i][:, 12:, :, :].data.cpu().numpy())

        if appended:
            gen_seq = torch.from_numpy(np.array(gen_seq))[:, 0:cut_point, :, :, :]

        return torch.from_numpy(np.array(gen_seq))


    def CDNA_pass_through(self, tactiles, actions):
        hidden = None
        outputs = []
        state = actions[0].to(device)
        with torch.no_grad():
            for index, (sample_tactile, sample_action) in enumerate(zip(tactiles[0:-1].squeeze(), actions[1:].squeeze())):
                state_action = torch.cat((state, sample_action), 1)
                tsa = torch.cat(8*[torch.cat(8*[state_action.unsqueeze(2)], axis=2).unsqueeze(3)], axis=3)
                if index > self.context_frames-1:
                    predictions_t, hidden, cdna_kerns_t, masks_t = self.model(predictions_t, conditions=tsa, hidden_states=hidden)
                    outputs.append(predictions_t)
                else:
                    predictions_t, hidden, cdna_kerns_t, masks_t = self.model(sample_tactile, conditions=tsa, hidden_states=hidden)
                    last_output = predictions_t
        outputs = [last_output] + outputs

        return torch.stack(outputs)

    def SV2P_pass_through(self, tactiles, actions):
        hidden = None
        outputs = []
        state = actions[0].to(device)

        samples = torch.normal(0.0, 1.0, size=(tactiles.shape[1], self.latent_channels, 8, 8)).to(device)
        prior_sample = samples
        pMean, pSTD = None, None

        with torch.no_grad():
            for index, (sample_tactile, sample_action) in enumerate(zip(tactiles[0:-1].squeeze(), actions[1:].squeeze())):
                state_action = torch.cat((state, sample_action), 1)
                tsa = torch.cat(8*[torch.cat(8*[state_action.unsqueeze(2)], axis=2).unsqueeze(3)], axis=3)
                tsap = torch.cat([prior_sample, tsa], axis=1)
                if index > self.context_frames-1:
                    predictions_t, hidden, cdna_kerns_t, masks_t = self.model(predictions_t, conditions=tsap, hidden_states=hidden)
                    outputs.append(predictions_t)
                else:
                    predictions_t, hidden, cdna_kerns_t, masks_t = self.model(sample_tactile, conditions=tsap, hidden_states=hidden)
                    last_output = predictions_t

        outputs = [last_output] + outputs

        return torch.stack(outputs)

    def run_batch(self, tactile, action):
        if self.image_size == 0:
            action = action.squeeze(-1).permute(1, 0, 2).to(device)
            tactile = torch.flatten(tactile, start_dim=2).permute(1, 0, 2).to(device)
        else:
            tactile = tactile.permute(1, 0, 4, 3, 2).to(device)
            action = action.squeeze(-1).permute(1, 0, 2).to(device)

        if self.model_name == "CDNA":
            tactile_predictions = self.CDNA_pass_through(tactiles=tactile, actions=action)
        elif self.model_name == "SV2P":
            tactile_predictions = self.SV2P_pass_through(tactiles=tactile, actions=action)
        elif self.model_name == "SVG":
            tactile_predictions = self.SVG_pass_through(tactiles=tactile, actions=action).to(device)
        else:
            tactile_predictions = self.model.forward(tactiles=tactile, actions=action)  # Step 3. Run our forward pass.

        loss = self.criterion(tactile_predictions, tactile[self.context_frames:])
        return tactile_predictions, tactile[self.context_frames:], loss.item()

    def save_trial(self, data_prediction, data_gt, trial_number):
        meta_data = np.load(self.test_data_dir + 'test_trial_' + str(trial_number) + '/trial_meta_0.npy')
        path_save = self.data_save_path + "test_trial_" + str(trial_number) + '/'
        try:
            os.mkdir(path_save)
        except:
            pass
        np.save(path_save + "prediction_data", data_prediction)
        np.save(path_save + "groundtruth_data", data_gt)
        np.save(path_save + "meta_data", meta_data)

    def load_scalars(self):
        self.scaler_tx = load(open(self.scaler_dir + "tactile_standard_scaler_x.pkl", 'rb'))
        self.scaler_ty = load(open(self.scaler_dir + "tactile_standard_scaler_y.pkl", 'rb'))
        self.scaler_tz = load(open(self.scaler_dir + "tactile_standard_scaler_z.pkl", 'rb'))
        self.min_max_scalerx_full_data = load(open(self.scaler_dir + "tactile_min_max_scalar_x.pkl", 'rb'))
        self.min_max_scalery_full_data = load(open(self.scaler_dir + "tactile_min_max_scalar_y.pkl", 'rb'))
        self.min_max_scalerz_full_data = load(open(self.scaler_dir + "tactile_min_max_scalar.pkl", 'rb'))


def main():
    model_path = "/home/user/Robotics/tactile_prediction/tactile_prediction/models/SVG/saved_models/box_only_3layers_model_13_12_2021_13_23/SVG_model"
    data_save_path = "/home/user/Robotics/tactile_prediction/tactile_prediction/models/SVG/saved_models/box_only_3layers_model_13_12_2021_13_23/"
    test_data_dir = "/home/user/Robotics/Data_sets/box_only_dataset/test_image_dataset_10c_10h_32_universal/"
    scaler_dir = "/home/user/Robotics/Data_sets/box_only_dataset/scalar_info_universal/"

    number_of_trials = [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22]  # 0-22
    batch_size = 16
    context_frames = 10
    sequence_length = 20
    image_size = 32
    latent_channels = 10
    criterion = "L1"

    model = torch.load(model_path)
    model_name = "SVG"

    UMT = UniversalModelTester(model, number_of_trials, test_data_dir, image_size, criterion, model_path,
                         data_save_path, scaler_dir, model_name, batch_size,
                         context_frames, sequence_length, latent_channels)  # if not an image set image size to 0
    UMT.test_model()


if __name__ == "__main__":
    main()
