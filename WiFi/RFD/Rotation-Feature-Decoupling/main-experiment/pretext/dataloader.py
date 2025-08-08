import pdb

import os
import sys
import io
import random

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import csv
import math
import json
import torch
import torch.utils.data as data

import torchnet as tnt
from torch.utils.data.dataloader import default_collate

import config_env_example as env

def normalize_dataX(x):
    max_value = x.max(axis=2)
    min_value = x.min(axis=2)
    max_value = np.expand_dims(max_value, 2)
    min_value = np.expand_dims(min_value, 2)
    max_value = max_value.repeat(x.shape[2], axis=2)
    min_value = min_value.repeat(x.shape[2], axis=2)
    x = (x - min_value) / (max_value - min_value)
    return x

def WiFi_Dataset_slice(ft, classi):
    devicename = ['3123D7B', '3123D7D', '3123D7E', '3123D52', '3123D54', '3123D58', '3123D64', '3123D65',
                  '3123D70', '3123D76', '3123D78', '3123D79', '3123D80', '3123D89', '3123EFE', '3124E4A']
    data_IQ_wifi_all = np.zeros((1,2,6000))
    data_target_all = np.zeros((1,))
    target = 0
    for classes in classi:
        for recoder in range(1):
            inputFilename = f'/data/dataset/WiFi/{ft}ft/WiFi_air_X310_{devicename[classes]}_{ft}ft_run{recoder+1}'
            with open("{}.sigmf-meta".format(inputFilename),'rb') as read_file:
                meta_dict = json.load(read_file)
            with open("{}.sigmf-data".format(inputFilename),'rb') as read_file:
                binary_data = read_file.read()
            fullVect = np.frombuffer(binary_data, dtype=np.complex128)
            even = np.real(fullVect)
            odd = np.imag(fullVect)
            length = 6000
            num = 0
            data_IQ_wifi = np.zeros((math.floor(len(even)/length), 2, 6000))
            data_target = np.zeros((math.floor(len(even)/length),))
            for begin in range(0,len(even)-(len(even)-math.floor(len(even)/length)*length),length):
                data_IQ_wifi[num,0,:] = even[begin:begin+length]
                data_IQ_wifi[num,1,:] = odd[begin:begin+length]
                data_target[num,] = target
                num = num + 1
            data_IQ_wifi_all = np.concatenate((data_IQ_wifi_all,data_IQ_wifi),axis=0)
            data_target_all = np.concatenate((data_target_all, data_target), axis=0)
        target = target + 1
    return data_IQ_wifi_all[1:,], data_target_all[1:,]

def PreTrainDataset_prepared(ft, classi):
    x, y = WiFi_Dataset_slice(ft, classi)
    train_index_shot = []
    for i in classi:
        index_classi = [index for index, value in enumerate(y) if value == i]
        train_index_shot += index_classi[0:2000]
    X_train = x[train_index_shot]
    Y_train = y[train_index_shot].astype(np.uint8)

    max_value = X_train.max()
    min_value = X_train.min()
    X_train = (X_train - min_value) / (max_value - min_value)
    return X_train, Y_train

def get_database(root, n_class=10, test_size=0.2, random_state=30):
    X_train, Y_train = PreTrainDataset_prepared(62, range(0,10))

    # X_train = normalize_dataX(X_train)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=test_size, random_state=random_state)
    return {
        "train": {
            "x": X_train,
            "y": Y_train
        },
        "val": {
            "x": X_val,
            "y": Y_val
        }
    }


class GenericDataset(data.Dataset):
    def __init__(self, database, dataset_name, split):
        self.dataset_name = dataset_name.lower()
        self.split = split.lower()
        self.name = self.dataset_name + '_' + self.split
        assert (self.split == 'train' or self.split == 'val')
        self.data = database[self.split]["x"]
        self.labels = database[self.split]["y"]

    def __getitem__(self, index):
        data = self.data[index]
        data = torch.tensor(data)
        label = self.labels[index]
        return data, int(label), index

    def __len__(self):
        return len(self.labels)


class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class GenericDataLoader(object):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 unsupervised=True,
                 epoch_size=None,
                 num_workers=0,
                 shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.unsupervised = unsupervised
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.rand_seed = 0
        self.data_loader_ = self.get_iterator()

    def get_iterator(self):
        random.seed(self.rand_seed)
        if self.unsupervised:
            # 如果在无监督模式下定义一个加载器函数，给定信号的索引，它返回原始信号及其在数据集中的索引以及旋转的标签，即 0 表示 0 度旋转，1 表示 90 度，2 表示 180 度，3 表示 270 度。在神经网络转发期间将创建信号的 4 个旋转副本。
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img, _, index = self.dataset[idx]
                rotation_labels = torch.LongTensor([0, 1, 2, 3])
                image_indices = torch.LongTensor([index, index, index, index])
                return img, rotation_labels, image_indices

            def _collate_fun(batch):
                batch = default_collate(batch)
                assert (len(batch) == 3)
                batch_size, rotations = batch[1].size()
                batch[1] = batch[1].view([batch_size * rotations])
                batch[2] = batch[2].view([batch_size * rotations])
                return batch
        else:  # supervised mode
            # 如果在监督模式下定义一个加载器函数，该函数给定图像的索引，它将返回图像及其在数据集中的分类标签和索引。
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img, categorical_label, index = self.dataset[idx]
                return img, categorical_label, index

            _collate_fun = default_collate

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
                                              load=_load_function)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
                                           collate_fn=_collate_fun,
                                           num_workers=self.num_workers,
                                           shuffle=self.shuffle)

        return data_loader

    def __call__(self, epoch=0):
        self.rand_seed = epoch * self.epoch_size
        random.seed(self.rand_seed)
        return self.data_loader_

    def __len__(self):
        return len(self.data_loader_)
