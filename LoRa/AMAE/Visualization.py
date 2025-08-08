import torch
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset, DataLoader
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from get_dataset import *
from sklearn import metrics
from sklearn.cluster import KMeans
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def scatter(features, targets, subtitle = None, n_classes = 30):
    targets = targets.reshape(targets.shape[0], )
    palette = np.array(sns.color_palette("hls", n_classes))
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(features[:, 0], features[:, 1], lw=0, s=40, c=palette[targets, :])  #
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    ax.axis('off')
    ax.axis('tight')

    txts = []
    for i in range(n_classes):
        xtext, ytext = np.median(features[targets == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.savefig(f"Visualization/{n_classes}classes_{subtitle}.png", dpi=600)
    plt.show()

def obtain_embedding_feature_map(model, test_dataloader):
    model.eval()
    device = torch.device("cuda:0")
    with torch.no_grad():
        feature_map = []
        target_output = []
        for data, target in test_dataloader:
            if torch.cuda.is_available():
                data = data.to(device)
            output = model(data)
            feature_map[len(feature_map):len(output)-1] = output.tolist()
            target_output[len(target_output):len(target)-1] = target.tolist()
        feature_map = torch.Tensor(feature_map)
        target_output = np.array(target_output)
    return feature_map, target_output

def get_accuracy_score(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype = np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    print('ACC = ', acc)
    return acc

def get_silhouette_score(x_test_feature, y_test):
    sc = metrics.silhouette_score(x_test_feature, y_test)
    print('SC = ', sc)

def get_classification_report(x_test_feature, y_test):
    classification_report = metrics.classification_report(x_test_feature, y_test)
    print('classification_report = ', classification_report)


def main():
    num = [0,30]

    X_test, Y_test = get_num_class_Sourcetestdata(num)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    model = torch.load('model_weight/pretrain_MAE_encoder_IQ.pth')
    X_test_embedding_feature_map, real_target= obtain_embedding_feature_map(model, test_dataloader)

    tsne = TSNE(n_components=2)
    eval_tsne_embeds = tsne.fit_transform(torch.Tensor.cpu(X_test_embedding_feature_map))
    scatter(eval_tsne_embeds, real_target.astype('int64'), "AMAE_yzs_wrongdata", 30)

    km = KMeans(n_clusters=30, n_init=30)
    km.fit(eval_tsne_embeds)
    cluster_target = km.predict(eval_tsne_embeds)
    real_target = np.squeeze(real_target, axis=1)

    get_silhouette_score(X_test_embedding_feature_map, real_target)
    get_accuracy_score(real_target, cluster_target)
    print('NMI = ', metrics.normalized_mutual_info_score(real_target, cluster_target))
    print('AMI = ', metrics.adjusted_rand_score(real_target, cluster_target))

if __name__ == "__main__":
    main()