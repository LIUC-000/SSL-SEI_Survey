import sys
print(sys.path)
sys.path.insert(0, 'models')
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import torch
import yaml
import numpy as np
from get_dataset import FineTuneDataset_prepared
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from algorithms import *

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

def classify(k, SEED, ALL_acc, snr):
    # 获取微调和测试数据集
    X_train, X_test, Y_train, Y_test = FineTuneDataset_prepared(62, range(10, 16), k, snr, SEED)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 加载预训练模型
    online_network = torch.load('model_weight/pretrain_MAE_encoder_IQ.pth')
    if torch.cuda.is_available():
        online_network = online_network.to(device)

    # 获取特征图
    X_train_embedding_feature_map, target_train = obtain_embedding_feature_map(online_network, train_dataloader)
    X_test_embedding_feature_map, target_test = obtain_embedding_feature_map(online_network, test_dataloader)

    # 创建 StandardScaler 对象
    scaler = StandardScaler()
    # 拟合并转换训练数据
    X_train_embedding_feature_map = scaler.fit_transform(X_train_embedding_feature_map)
    # 只转换测试数据
    X_test_embedding_feature_map = scaler.transform(X_test_embedding_feature_map)

    # knn
    print("KNN:")
    KNN_acc = KNN_Classifier(X_train_embedding_feature_map, target_train, X_test_embedding_feature_map, target_test)
    ALL_acc['KNN_acc'].append(KNN_acc)

    # tree
    print("DecisionTree:")
    DT_acc = DecisionTree_Classifier(X_train_embedding_feature_map, target_train, X_test_embedding_feature_map, target_test)
    ALL_acc['DT_acc'].append(DT_acc)

    # forest
    print("RandomForest:")
    RF_acc = RandomForest_Classifier(X_train_embedding_feature_map, target_train, X_test_embedding_feature_map, target_test)
    ALL_acc['RF_acc'].append(RF_acc)

    # Bayes
    print("Bayes:")
    Bays_acc = Bayes_Classifier(X_train_embedding_feature_map, target_train, X_test_embedding_feature_map, target_test)
    ALL_acc['Bays_acc'].append(Bays_acc)

    # SVM
    print("SVM:")
    SVM_acc = SVM_Classifier(X_train_embedding_feature_map, target_train, X_test_embedding_feature_map, target_test)
    ALL_acc['SVM-linear_acc'].append(SVM_acc[0])
    ALL_acc['SVM-poly_acc'].append(SVM_acc[1])
    ALL_acc['SVM-rbf_acc'].append(SVM_acc[2])
    ALL_acc['SVM-signmoid_acc'].append(SVM_acc[3])


def clustering_FT():
    from sklearn.cluster import KMeans
    from get_dataset import TestDataset_prepared

    X_test, Y_test = TestDataset_prepared(62, range(10, 16))

    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 加载预训练模型
    online_network = torch.load('model_weight/pretrain_MAE_encoder_IQ.pth')
    if torch.cuda.is_available():
        online_network = online_network.to(device)

    # 获取特征图
    X_test_embedding_feature_map, target_test = obtain_embedding_feature_map(online_network, test_dataloader)

    k_means = KMeans(n_clusters=6, random_state=10)
    k_means.fit(X_test_embedding_feature_map)
    predict_test = k_means.predict(X_test_embedding_feature_map)

    # 外部指标
    purity_ = Purity(target_test.astype(np.int64), predict_test.astype(np.int64))
    print(f"purity={purity_}")

    ARI_ = ARI(target_test.astype(np.int64), predict_test.astype(np.int64))
    print(f"ARI={ARI_}")

    NMI_ = NMI(target_test.astype(np.int64), predict_test.astype(np.int64))
    print(f"NMI={NMI_}")

    # ACC_ = ACC(target_test.astype(np.int64), predict_test.astype(np.int64))
    # print(f"ACC={ACC_}")
    #
    # # 内部指标
    # Entropy_ = Entropy(X_test_embedding_feature_map, predict_test)
    # print(f"Entropy={Entropy_}")
    #
    # compactness_ = Compactness(X_test_embedding_feature_map, predict_test)
    # print(f"compactness={compactness_}")

    silhouette_score_ = Silhouette_score(X_test_embedding_feature_map, predict_test)
    print(f"silhouette_score={silhouette_score_}")

    calinski_harabasz_score = Calinski_harabasz_score(X_test_embedding_feature_map, predict_test)
    print(f"calinski_harabasz_score={calinski_harabasz_score}")

    davies_bouldin_score = Davies_bouldin_score(X_test_embedding_feature_map, predict_test)
    print(f"davies_bouldin_score={davies_bouldin_score}")

def clustering_PT():
    from sklearn.cluster import KMeans
    from get_dataset import TestDataset_prepared

    X_test, Y_test = TestDataset_prepared(62, range(0, 10))

    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 加载预训练模型
    online_network = torch.load('model_weight/pretrain_MAE_encoder_IQ.pth')
    if torch.cuda.is_available():
        online_network = online_network.to(device)

    # 获取特征图
    X_test_embedding_feature_map, target_test = obtain_embedding_feature_map(online_network, test_dataloader)

    k_means = KMeans(n_clusters=10, random_state=10)
    k_means.fit(X_test_embedding_feature_map)
    predict_test = k_means.predict(X_test_embedding_feature_map)

    # 外部指标
    purity_ = Purity(target_test.astype(np.int64), predict_test.astype(np.int64))
    print(f"purity={purity_}")

    ARI_ = ARI(target_test.astype(np.int64), predict_test.astype(np.int64))
    print(f"ARI={ARI_}")

    NMI_ = NMI(target_test.astype(np.int64), predict_test.astype(np.int64))
    print(f"NMI={NMI_}")

    # ACC_ = ACC(target_test.astype(np.int64), predict_test.astype(np.int64))
    # print(f"ACC={ACC_}")

    # 内部指标
    # Entropy_ = Entropy(X_test_embedding_feature_map, predict_test)
    # print(f"Entropy={Entropy_}")

    # compactness_ = Compactness(X_test_embedding_feature_map, predict_test)
    # print(f"compactness={compactness_}")

    silhouette_score_ = Silhouette_score(X_test_embedding_feature_map, predict_test)
    print(f"silhouette_score={silhouette_score_}")

    calinski_harabasz_score = Calinski_harabasz_score(X_test_embedding_feature_map, predict_test)
    print(f"calinski_harabasz_score={calinski_harabasz_score}")

    davies_bouldin_score = Davies_bouldin_score(X_test_embedding_feature_map, predict_test)
    print(f"davies_bouldin_score={davies_bouldin_score}")

if __name__ == '__main__':
    import pandas as pd
    for snr in [-10,-5,0,5,10]:
        ALL_acc = {'KNN_acc':[], 'DT_acc':[], 'RF_acc':[], 'Bays_acc':[], 'SVM-linear_acc':[], 'SVM-poly_acc':[], 'SVM-rbf_acc':[], 'SVM-signmoid_acc':[]}
        for seed in range(30):
            print(f'-------------------------k={20}------------------------------')
            classify(20, 2024+seed, ALL_acc, snr)
        df = pd.DataFrame(ALL_acc)
        df.to_excel(f'./test_result/downstream_classification_acc_SNR={snr}.xlsx', index=False, sheet_name='Sheet1')

    print('Clustering on pretext dataset:')
    clustering_PT()
    print('Clustering on downstream dataset:')
    clustering_FT()
