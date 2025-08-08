import torch
import sys
print(sys.path)
sys.path.insert(0, './models')
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset, DataLoader
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from get_dataset_local import *
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sklearn.metrics as sm
from sklearn import manifold
from models.encoder_and_projection import Encoder_and_projection
import yaml

def scatter(features, targets, subtitle = None, n_classes = 5):
    palette = np.array(sns.color_palette("hls", n_classes))  # "hls",
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    #sc = ax.scatter(features[:, 0], features[:, 1], lw=0, s=40, c=palette[targets, :])  #
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    txts = []
    targets = sum(targets, [])  # [[1],[2]] to [1, 2]
    for i in range(n_classes):
        xtext, ytext = np.median(features[targets == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.savefig(f"Visualization/{n_classes}classes_{subtitle}.png", dpi=600)

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
            feature_map[len(feature_map):len(output[0])-1] = output[0].tolist()
            target_output[len(target_output):len(target)-1] = target.tolist()
        feature_map = torch.Tensor(feature_map)
        target_output = np.array(target_output)
    return feature_map, target_output

def transfer_model(pretrained_file, model):

    pretrained_model = torch.load(pretrained_file, map_location='cpu')  # get pretrained model
    pretrained_dict = pretrained_model.state_dict()
    model_dict = model.state_dict()  # get model dict
    # 在合并前(update),需要去除pretrained_dict一些不需要的参数
    pretrained_dict = transfer_state_dict(pretrained_dict, model_dict)
    model_dict.update(pretrained_dict)  # 更新(合并)模型的参数
    model.load_state_dict(model_dict)
    return model

def transfer_state_dict(pretrained_dict, model_dict):
    # state_dict2 = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            # state_dict.setdefault(k, v)
            state_dict[k] = v
        else:
            print("Missing key(s) in state_dict :{}".format(k))
    return state_dict

def main():
    X_train, X_test, Y_train, Y_test = FineTuneDataset_prepared(62, range(10, 16), 100)
    #X_test, Y_test = PreTrainDataset_prepared(62, range(0, 9 + 1))

    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

    # online network
    model = Encoder_and_projection(**config['network'])

    checkpoints_folder = os.path.join('./runs',
                                      f"seed300_pretrain_62ft_class_in_0-9",
                                      'checkpoints')

    # load pre-trained parameters
    load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                             map_location='cpu')

    model.load_state_dict(load_params['state_dict'])

    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    # train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
    # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    X_test_embedding_feature_map, target = obtain_embedding_feature_map(model, test_dataloader)
    # print(X_train_embedding_feature_map.shape)
    #
    # tsne = TSNE(n_components=2)
    # eval_tsne_embeds = tsne.fit_transform(torch.Tensor.cpu(X_train_embedding_feature_map))
    # print(eval_tsne_embeds.shape)
    #scatter(eval_tsne_embeds, target.astype('int64'), "SCNN_snr=2", 6)
    visualize_data(X_test_embedding_feature_map, target.astype('int64'), "downstream_test_visual", 6)
    print(sm.silhouette_score(X_test_embedding_feature_map, target, sample_size=len(X_test_embedding_feature_map), metric='euclidean'))

if __name__ == "__main__":
    main()