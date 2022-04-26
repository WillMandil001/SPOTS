# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import cv2
import copy

import matplotlib.pyplot as plt
from matplotlib.ticker import(AutoMinorLocator, MultipleLocator)
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from sklearn import preprocessing

import numpy as np
import pandas as pd
import matplotlib
from tqdm import tqdm
from pickle import load
from datetime import datetime
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from pykalman import KalmanFilter

# asdf
class plot_player():
    def __init__(self, scene, tactile, save_name):
        self.save_name = save_name
        self.file_save_name = self.save_name + '.mp4'
        print(self.file_save_name)
        self.taxel_to_test = 8
        self.run_the_tape(scene, tactile)

        font = {'family': 'normal',
                'weight': 'bold',
                'size': 22}
        matplotlib.rcParams.update({'font.size': 22})

    def init_plot(self):
        a = [i for i in range(100)], [i for i in range(100)]
        # a = [i for i in range(len(self.tactile))], [i for i in range(len(self.tactile))]
        return a

    def update(self, i):
        plt.title(i)
        self.tact1.set_data(list([value for value in range(0, self.indexyyy)]), list([self.tactile[i][self.taxel_to_test] for i in range(0, self.indexyyy)]))
        self.im1.set_data(self.scene[self.indexyyy])

        # self.im1.set_data(list([value for value in range(len(self.tactile))]), list([value for value in self.tactile[:self.indexyyy,0,0]] + [None for i in range(len(self.tactile) - self.indexyyy)]))
        # self.tact1.set_data([t for t in range(100)], [t for t in range(100)])
        self.indexyyy += 1
        if self.indexyyy == len(self.scene):
            self.indexyyy = 0

    def run_the_tape(self, scene, tactile):
        fig = plt.figure()
        self.ax1 = fig.add_subplot(1, 2, 1)
        self.ax2 = fig.add_subplot(2, 2, 2)

        self.indexyyy = 0
        self.tactile = tactile
        self.scene = scene

        self.ax1.set_title("scene_image")
        self.ax1.set_xlim((0, len(self.tactile)))
        self.ax1.set_ylim((min(self.tactile[:, self.taxel_to_test]), max(self.tactile[:, self.taxel_to_test])))
        self.ax1.grid(True)
        self.ax1.set_title("Taxel sequence")
        # ax1[1].xlabel('time step')
        # ax1[1].ylabel('tactile reading')
        # ax1[1].legend(loc="lower left")
        a, b = self.init_plot()
        print(len(a), len(b))
        self.tact1 = self.ax1.plot(a, b, "-o", alpha=0.5, c="b", label="t1")[0]

        self.im1 = self.ax2.imshow(self.scene[0])

        ani = FuncAnimation(plt.gcf(), self.update, interval=20.8, save_count=len(self.scene), repeat=False)
        # ani.save(self.file_save_name)
        ani.save(self.file_save_name, fps=10, extra_args=['-vcodec', 'libx264'], dpi=400)


def visualise_data_sequence():
    data_dir = "/home/user/Robotics/Data_sets/PRI/object1_motion1/Train/data_sample_2022-03-25-14-19-39/"

    scene = np.load(data_dir + "color_images.npy")
    tactile = np.load(data_dir + "tactile_states.npy")

    print(scene.shape)
    print(tactile.shape)

    plot_player(scene, tactile, save_name=data_dir + "sequence")

if __name__ == "__main__":
    visualise_data_sequence()


    # class image_player ():
    #     def __init__(self, images, save_name, feature, experiment_to_test, data_save_path):
    #         self.feature = feature
    #         self.save_name = save_name
    #         self.experiment_to_test = experiment_to_test
    #         self.file_save_name = data_save_path + '/' + self.save_name + '_feature_' + str (self.feature) + '.mp4'
    #         print (self.file_save_name)
    #         self.run_the_tape (images)
    #
    #     def grab_frame(self):
    #         frame = self.images[self.indexyyy][:, :, self.feature] * 255
    #         return frame
    #
    #     def update(self, i):
    #         plt.title (i)
    #         self.im1.set_data (self.grab_frame ())
    #         self.indexyyy += 1
    #         if self.indexyyy == len (self.images):
    #             self.indexyyy = 0
    #
    #     def run_the_tape(self, images):
    #         self.indexyyy = 0
    #         self.images = images
    #         ax1 = plt.subplot (1, 2, 1)
    #         self.im1 = ax1.imshow (self.grab_frame (), cmap='gray', vmin=0, vmax=255)
    #         ani = FuncAnimation (plt.gcf (), self.update, interval=20.8, save_count=len (images), repeat=False)
    #         ani.save (self.file_save_name, fps=48, extra_args=['-vcodec', 'libx264'], dpi=400)
