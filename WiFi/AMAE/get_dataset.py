import numpy as np
import math
import json
import random

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
    return X_train, Y_train

def add_noise_with_snr(signal, snr_db):
    # 计算信号的功率
    signal_power = np.mean(signal ** 2)

    # 计算应有的噪声功率
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # 计算噪声的标准差
    noise_std = np.sqrt(noise_power)

    # 生成相同形状的噪声
    noise = np.random.normal(0, noise_std, signal.shape)

    # 加噪声
    noisy_signal = signal + noise

    return noisy_signal

def FineTuneDataset_prepared(ft, classi, k, snr, SEED):
    x, y = WiFi_Dataset_slice(ft, classi)
    x = add_noise_with_snr(x, snr)
    test_index_shot = []
    finetune_index_shot = []
    random.seed(SEED)
    for i in classi:
        i -= classi[0]
        index_classi = [index for index, value in enumerate(y) if value == i]
        finetune_index_shot += random.sample(index_classi[2000:3000], k)
        test_index_shot += index_classi[3000:4000]
    X_train = x[finetune_index_shot]
    Y_train = y[finetune_index_shot]
    X_test = x[test_index_shot]
    Y_test = y[test_index_shot]
    return X_train, X_test, Y_train, Y_test

def Max_Min_PT(ft, classi):
    x, y = WiFi_Dataset_slice(ft, classi)
    train_index_shot = []
    for i in classi:
        index_classi = [index for index, value in enumerate(y) if value == i]
        train_index_shot += index_classi[0:2000]
    X_train = x[train_index_shot]

    max_value = X_train.max()
    min_value = X_train.min()
    return max_value, min_value

def TestDataset_prepared(ft, classi):
    x, y = WiFi_Dataset_slice(ft, classi)
    test_index_shot = []
    for i in classi:
        i -= classi[0]
        index_classi = [index for index, value in enumerate(y) if value == i]
        test_index_shot += index_classi[3000:4000]
    X_test = x[test_index_shot]
    Y_test = y[test_index_shot]
    return X_test, Y_test

if __name__ == "__main__":
    X_train, X_test1, Y_train, Y_test1 = FineTuneDataset_prepared(62, range(0,16), 10)
    # x, y = WiFi_Dataset_slice(62, range(0,16))
    # for i in range(0,16):
    #     index_classi_len = len([index for index, value in enumerate(y) if value == i])
    #     print(f'class{i},size{index_classi_len}')