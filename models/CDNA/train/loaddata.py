import os
import gzip
import tempfile
import shutil
import typing
from datetime import datetime
import csv

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as trans
import train.more_trans as more_trans


__all__ = [
    'MovingMNIST',
    'KTH',
]
# PYTORCH_DATA_HOME = os.path.normpath(os.environ['/home/user/Robotics/Data_sets/slip_detection/formatted_dataset/train_image_dataset_10c_10h/'])

model_save_path = "/home/user/Robotics/tactile_prediction/tactile_prediction/models/SV2P/saved_models/"
train_data_dir = '/home/user/Robotics/Data_sets/slip_detection/formatted_dataset/train_image_dataset_10c_10h/'
scaler_dir = "/home/user/Robotics/Data_sets/slip_detection/formatted_dataset/"

# unique save title:
model_save_path = model_save_path + "model_" + datetime.now().strftime("%d_%m_%Y_%H_%M/")
# os.mkdir(model_save_path)


class TactileSingle:
    def __init__(self, batch_size, train_percentage, validation_percentage):
        self.train_percentage = train_percentage
        self.validation_percentage = validation_percentage

        self.data_map = []
        self.batch_size = batch_size
        with open(train_data_dir + 'map.csv', 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                self.data_map.append(row)

    def load_full_data(self):
        dataset_train = FullDataSet(self.data_map, self.train_percentage, self.validation_percentage, train=True)
        dataset_validate = FullDataSet(self.data_map, self.train_percentage, self.validation_percentage, validation=True)
        transform = trans.Compose([trans.ToTensor()])
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(dataset_validate, batch_size=self.batch_size, shuffle=True)
        self.data_map = []
        return train_loader, validation_loader


class FullDataSet():
    def __init__(self, data_map, train_percentage, validation_percentage, train=False, validation=False):
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

        tactile_images = []
        for image_name in np.load(train_data_dir + value[2]):
            tactile_images.append(np.load(train_data_dir + image_name))

        experiment_number = np.load(train_data_dir + value[3])
        time_steps = np.load(train_data_dir + value[4])
        return [robot_data.astype(np.float32), np.array(tactile_images).astype(np.float32), experiment_number,
                time_steps]


class MovingMNIST(torch.utils.data.Dataset):
    """
    Returns single image in MovingMNIST dataset. The index
    """

    seqlen = 20

    # obtained by ./compute-nmlstats.py
    normalize = trans.Normalize(mean=(0.049270592390472503,),
                                std=(0.2002874575763297,))
    denormalize = more_trans.DeNormalize(mean=(0.049270592390472503,),
                                         std=(0.2002874575763297,))

    def __init__(self, transform: typing.Callable = None):
        datafile = os.path.join(PYTORCH_DATA_HOME,
                                'MovingMNIST',
                                'mnist_test_seq.npy')
        data = np.load(datafile)  # shape:(T, N, H, W), dtype: uint8
        self.videos = np.transpose(data,(1, 0, 2, 3))
        self.transform = more_trans.VideoTransform(transform)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index: int):
        v = self.videos[index]
        if self.transform:
            v = self.transform(v)
        return v


def extract_gz(filename, tofile):
    with gzip.open(filename) as infile:
        with open(tofile, 'wb') as outfile:
            shutil.copyfileobj(infile, outfile)


class KTH(torch.utils.data.Dataset):
    def __init__(self, transform=None, wd=os.getcwd()):
        """
        :param wd: the temporary working directory used as cache to expand
               the dataset; default to current working directory
        """
        self.transform = transform

        self.datadir = os.path.join(os.path.normpath(
            os.environ['PYTORCH_DATA_HOME']), 'KTH')
        with open(os.path.join(self.datadir, 'kth.lst')) as infile:
            self.videolist = list(map(str.strip, infile))
        self._tempdir = tempfile.TemporaryDirectory(dir=wd)
        self._tempdir_name = self._tempdir.name

    def __len__(self):
        return len(self.videolist)

    def __getitem__(self, index):
        filename = os.path.join(self.datadir, self.videolist[index])
        npyfile = os.path.join(self._tempdir_name, os.path.basename(filename))
        try:
            v = np.load(npyfile)
        except FileNotFoundError:
            extract_gz(filename + '.gz', npyfile)
            v = np.load(npyfile)
        if self.transform:
            v = self.transform(v)
        return v

    def __enter__(self):
        return self

    def __exit__(self, _a, _b, _c):
        self.teardown()

    def teardown(self):
        self._tempdir.cleanup()
        self._tempdir_name = None
