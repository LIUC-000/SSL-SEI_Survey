import torch
import sys
from torch.utils.data import TensorDataset, DataLoader
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from get_dataset_local import *
import os
import sklearn.metrics as sm
from sklearn import manifold
import yaml
import random
from model_complexcnn import CVCNN as Encoder
from utils import set_log_file_handler
from config.config import get_config

def visualize_data(data, labels, title, num_clusters):  # feature visualization
    labels = labels.astype(int)
    tsne = manifold.TSNE(n_components=2)  # init='pca'
    data_tsne = tsne.fit_transform(data)
    fig = plt.figure(figsize=(6.3, 5))
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], lw=0, s=10, c=labels, cmap=plt.cm.get_cmap("jet", num_clusters))
    plt.colorbar(ticks=range(num_clusters))
    fig.savefig(title, dpi=600)

def obtain_embedding_feature_map(model, test_dataloader):
    model.eval()
    device = torch.device("cuda:0")
    with torch.no_grad():
        feature_map = []
        target_output = []
        for data, target in test_dataloader:
            #target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                #target = target.to(device)
            output = model(data)
            feature_map[len(feature_map):len(output)-1] = output.tolist()
            target_output[len(target_output):len(target)-1] = target.tolist()
        feature_map = torch.Tensor(feature_map)
        target_output = np.array(target_output)
    return feature_map, target_output

def get_pretrain_encoder(network, pretrained_path, process_unit="cuda"):
    assert (os.path.isfile(pretrained_path))
    if "cpu" == process_unit:
        pretrained_model = torch.load(pretrained_path, map_location='cpu')
    elif "gpu" == process_unit or "cuda" == process_unit:
        pretrained_model = torch.load(pretrained_path)
    else:
        raise ValueError('Process unit platform is not specified')

    if 'module' in list(pretrained_model['network'].keys())[0]:
        from collections import OrderedDict
        pretrained_model_nomodule = OrderedDict()
        for key, value in pretrained_model['network'].items():
            key_nomodule = key[7:]  # remove module
            pretrained_model_nomodule[key_nomodule] = value
    else:
        pretrained_model_nomodule = pretrained_model['network']

    if pretrained_model_nomodule.keys() == network.state_dict().keys():
        network.load_state_dict(pretrained_model_nomodule)
    else:
        network.load_state_dict(pretrained_model_nomodule, strict=False)
    return network

def main():
    X_train, X_test, Y_train, Y_test = FineTuneDataset_prepared(62, range(10, 16), 1000)
    config = get_config("./config/config.yaml")
    params = config['trainer']
    model = Encoder()
    model = get_pretrain_encoder(model, params['pretrained_path'], "cpu")

    test_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    X_test_embedding_feature_map, target = obtain_embedding_feature_map(model, test_dataloader)
    print(X_test_embedding_feature_map.shape)

    visualize_data(X_test_embedding_feature_map, target.astype('int64'), "feature_visual", 6)
    print(sm.silhouette_score(X_test_embedding_feature_map, target, sample_size=len(X_test_embedding_feature_map), metric='euclidean'))

if __name__ == "__main__":
    main()
