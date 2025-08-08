import numpy as np
import h5py
import yaml
import random

def convert_to_I_Q_complex(data):
    '''Convert the loaded data to complex I and Q samples.'''
    num_row = data.shape[0]
    num_col = data.shape[1]
    data_complex = np.zeros([num_row, 2, round(num_col/2)])
    data_complex[:,0,:] = data[:,:round(num_col/2)]
    data_complex[:,1,:] = data[:,round(num_col/2):]

    return data_complex


def LoadDataset(file_path, dev_range, pkt_range):
    '''
    Load IQ sample from a dataset
    Input:
    file_path is the dataset path
    dev_range specifies the loaded device range
    pkt_range specifies the loaded packets range

    Return:
    data is the loaded complex IQ samples
    label is the true label of each received packet
    '''

    dataset_name = 'data'
    labelset_name = 'label'

    f = h5py.File(file_path, 'r')
    label = f[labelset_name][:]
    label = label.astype(int)
    label = np.transpose(label)
    label = label - 1

    label_start = int(label[0]) + 1
    label_end = int(label[-1]) + 1
    num_dev = label_end - label_start + 1
    num_pkt = len(label)
    num_pkt_per_dev = int(num_pkt/num_dev)

    print('Dataset information: Dev ' + str(label_start) + ' to Dev ' +
          str(label_end) + ',' + str(num_pkt_per_dev) + ' packets per device.')

    sample_index_list = []

    for dev_idx in dev_range:
        sample_index_dev = np.where(label==dev_idx)[0][pkt_range].tolist()
        sample_index_list.extend(sample_index_dev)

    data = f[dataset_name][sample_index_list]
    data = convert_to_I_Q_complex(data)
    label = label[sample_index_list]

    f.close()
    return data, np.squeeze(label)

def PreTrainDataset_prepared():
    config = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)
    params = config['trainer']

    file_path = '/data/dataset/LoRa_dataset/Train/dataset_training_no_aug.h5'
    dev_range = np.arange(params['class_start'], params['class_end']+1, dtype=int)
    pkt_range = np.arange(0, 500, dtype=int)
    X_train, Y_train = LoadDataset(file_path, dev_range, pkt_range)
    Y_train = Y_train.astype(np.uint8)

    # max_value = X_train.max()
    # min_value = X_train.min()
    # X_train = (X_train - min_value) / (max_value - min_value)
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

def FineTuneDataset_prepared(classi, k, snr, seed):
    file_path = '/data/dataset/LoRa_dataset/Test/dataset_seen_devices.h5'
    dev_range = np.arange(classi[0], classi[-1]+1, dtype=int)
    pkt_range = np.arange(0, 400, dtype=int)
    x, y = LoadDataset(file_path, dev_range, pkt_range)
    x = add_noise_with_snr(x, snr)
    y = y.astype(np.uint8)-classi[0]

    test_index_shot = []
    finetune_index_shot = []
    random.seed(seed)
    for i in classi:
        i -= classi[0]
        index_classi = [index for index, value in enumerate(y) if value == i]
        finetune_index_shot += random.sample(index_classi[0:300], k)
        test_index_shot += index_classi[300:400]
    X_train = x[finetune_index_shot]
    Y_train = y[finetune_index_shot]
    X_test = x[test_index_shot]
    Y_test = y[test_index_shot]

    # max_value = X_train.max()
    # min_value = X_train.min()
    # X_train = (X_train - min_value) / (max_value - min_value)
    # X_test = (X_test - min_value) / (max_value - min_value)
    return X_train, X_test, Y_train, Y_test

def TestDataset_prepared(classi):
    file_path = '/data/dataset/LoRa_dataset/Test/dataset_seen_devices.h5'
    dev_range = np.arange(classi[0], classi[-1] + 1, dtype=int)
    pkt_range = np.arange(0, 400, dtype=int)
    x, y = LoadDataset(file_path, dev_range, pkt_range)
    y = y.astype(np.uint8) - classi[0]

    test_index_shot = []
    for i in classi:
        i -= classi[0]
        index_classi = [index for index, value in enumerate(y) if value == i]
        test_index_shot += index_classi[300:400]
    X_test = x[test_index_shot]
    Y_test = y[test_index_shot]
    return X_test, Y_test

if __name__ == "__main__":
    x1, y1 = WiFi_Dataset_slice(62, range(0, 16))
    # for i in range(0,16):
    #     index_classi_len = len([index for index, value in enumerate(y) if value == i])
    #     print(f'class{i},size{index_classi_len}')