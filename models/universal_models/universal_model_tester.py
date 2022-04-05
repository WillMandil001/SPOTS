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
import torchvision

from universal_networks.SVG import Model as SVG
from universal_networks.SVG_tactile_enhanced import Model as SVG_TE
from universal_networks.SPOTS_SVG_ACTP import Model as SPOTS_SVG_ACTP


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
    def __init__(self, data_map, train=False, validation=False, train_percentage=1.0, image_size=64):
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
            tactile_images.append(create_image(tactile_data_sample, image_size=self.image_size))

        images = []
        for image_name in np.load(train_data_dir + value[2]):
            images.append(np.load(train_data_dir + image_name))

        experiment_number = np.load(train_data_dir + value[3])
        time_steps = np.load(train_data_dir + value[4])
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
