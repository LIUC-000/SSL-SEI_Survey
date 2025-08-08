import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import sklearn.svm as svm
from sklearn import metrics

def KNN_Classifier(X_train_embedding_feature_map, target_train, X_test_embedding_feature_map, target_test):
    # 定义 k 折交叉验证
    k_fold = KFold(n_splits=5)  # 例如，5 折交叉验证
    # 考虑的 n_neighbors 候选值
    neighbors_to_try = range(1,int(X_train_embedding_feature_map.shape[0]*0.8))

    # 用于存储每个 n_neighbors 的平均得分
    average_scores = []

    for n_neighbors in neighbors_to_try:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)

        # 计算当前 n_neighbors 的交叉验证得分
        scores = cross_val_score(knn, X_train_embedding_feature_map, target_train, cv=k_fold, scoring='accuracy')  # 例如，使用准确率作为评分标准

        # 计算平均得分
        average_score = np.mean(scores)
        average_scores.append(average_score)

    # 找到最优的 n_neighbors
    # print(average_scores)
    optimal_n_neighbors = neighbors_to_try[np.argmax(average_scores)]

    # print("Optimal number of neighbors:", optimal_n_neighbors)

    knn_best = KNeighborsClassifier(n_neighbors=optimal_n_neighbors)
    knn_best.fit(X_train_embedding_feature_map, target_train)

    #test
    y_pred = knn_best.predict(X_test_embedding_feature_map)
    accuracy = accuracy_score(target_test, y_pred)
    print("Accuracy:", accuracy)
    return accuracy

def DecisionTree_Classifier(X_train_embedding_feature_map, target_train, X_test_embedding_feature_map, target_test):
    model = DecisionTreeClassifier()

    # 训练模型
    model.fit(X_train_embedding_feature_map, target_train)

    # 预测测试集
    y_pred = model.predict(X_test_embedding_feature_map)

    # 评估模型
    accuracy = accuracy_score(target_test, y_pred)
    print(f'Accuracy: {accuracy}')
    return accuracy

def RandomForest_Classifier(X_train_embedding_feature_map, target_train, X_test_embedding_feature_map, target_test):
    # 创建随机森林分类器实例
    random_forest_model = RandomForestClassifier(n_estimators=100, random_state=30)
    # 训练模型
    random_forest_model.fit(X_train_embedding_feature_map, target_train)

    # 预测测试集
    y_pred = random_forest_model.predict(X_test_embedding_feature_map)

    # 评估模型
    accuracy = accuracy_score(target_test, y_pred)
    print(f'Accuracy: {accuracy}')
    return accuracy

def Bayes_Classifier(X_train_embedding_feature_map, target_train, X_test_embedding_feature_map, target_test):
    nb_model = GaussianNB()
    # 训练模型
    nb_model.fit(X_train_embedding_feature_map, target_train)

    # 预测测试集
    y_pred = nb_model.predict(X_test_embedding_feature_map)

    # 评估模型
    accuracy = accuracy_score(target_test, y_pred)
    print(f'Accuracy: {accuracy}')
    return accuracy

def svc(kernel):
    return svm.SVC(kernel=kernel, decision_function_shape="ovo")

def nusvc():
    return svm.NuSVC(decision_function_shape="ovo")

def linearsvc():
    return svm.LinearSVC(multi_class="ovr")

def SVM_Classifier(X_train_embedding_feature_map, target_train, X_test_embedding_feature_map, target_test):
    modelist = []
    kernalist = {"linear", "poly", "rbf", "sigmoid"}
    acc_all = []
    for each in kernalist:
        modelist.append(svc(each))
    modelist.append(nusvc())
    modelist.append(linearsvc())
    for model in modelist:
        model.fit(X_train_embedding_feature_map, target_train)
        accuracy = model.score(X_test_embedding_feature_map, target_test)
        print(f'Accuracy: {accuracy}')
        acc_all.append(accuracy)
    return acc_all

def Purity(labels_true, labels_pred):
    clusters = np.unique(labels_pred)
    labels_true = np.reshape(labels_true, (-1, 1))
    labels_pred = np.reshape(labels_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(labels_pred == c)[0]
        labels_tmp = labels_true[idx, :].reshape(-1)
        count.append(np.bincount(labels_tmp).max())
    return np.sum(count) / labels_true.shape[0]


def ARI(labels_true, labels_pred, beta=1.):
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(labels_true, labels_pred)
    return ari

def NMI(label, predict):
    from sklearn import metrics
    # return metrics.adjusted_mutual_info_score(predict, label)
    return metrics.normalized_mutual_info_score(label, predict)

def ACC(labels_true, labels_pre):
    acc = accuracy_score(labels_true, labels_pre)
    # acc = np.sum(labels_true==labels_pre) / np.size(labels_true)
    return acc

def NCC(label, x):
    m = x.shape[0]
    n = x.shape[1]
    Y = np.zeros((m, m))
    for r in range(m):
        for s in range(m):
            if label[r] == label[s]:
                Y[r, s] = 1
    drs = np.zeros((m, m))
    for r in range(m):
        for s in range(m):
            for att in range(n):
                if x[r, att] != x[s, att]:
                    drs[r, s] += 1
    ncc = 0.0
    for r in range(m):
        for s in range(m):
            if r != s:
                ncc += (n - 2 * drs[r, s]) * Y[r, s] + drs[r, s]
    return ncc

def Entropy(x, label):
    label = label.astype(np.int64)
    m = x.shape[0]
    n = x.shape[1]
    k = len(np.unique(label))
    # 每一个属性可能出现的值
    no_values = []
    for i in range(n):
        no_values.append(len(np.unique(x[:, i])))
    # cluster 成员数
    num_in_cluster = np.ones(k)
    for i in range(m):
        num_in_cluster[label[i]] += 1
    # p_u_lt
    P = []
    for t in range(k):
        # mt
        tp = np.where(label == t)[0]
        p_u_l = []
        for l in range(n):
            p_u_lt = []
            for u in range(no_values[l]):
                belong_lt = np.where(x[tp][:, l] == u)[0]
                p_u_lt.append(len(belong_lt) / len(tp))
            p_u_l.append(p_u_lt)
        P.append(p_u_l)
    # H
    H = np.zeros(k)
    for t in range(k):
        H_lt = np.zeros(n)
        for l in range(n):
            H_lt_u = np.zeros(no_values[l])
            for u in range(no_values[l]):
                if P[t][l][u] != 0:
                    H_lt_u[u] = - P[t][l][u] * np.log(P[t][l][u])
            H_lt[l] = np.sum(H_lt_u)
        H[t] = np.sum(H_lt) / n
    # H_R
    entropy_R = np.sum(H) / k
    return entropy_R

def Compactness(x, label):
    label = label.astype(np.int64)
    m = x.shape[0]
    n = x.shape[1]
    k = len(np.unique(label))
    value_label = np.unique(label)

    number_in_cluster = np.zeros(k, int)
    for i in range(m):
        for j in range(k):
            if label[i] == value_label[j]:
                number_in_cluster[j] += 1

    R = np.zeros(k)
    for i in range(k):
        R[i] = np.zeros((number_in_cluster[i], n))

    for i in range(k):
        count_x_in_rt = 0
        for j in range(m):
            if label[j] == value_label[i]:
                for v in range(n):
                    R[i][count_x_in_rt, v] = x[j, v]
                count_x_in_rt += 1

    dm = np.zeros(k)
    for i in range(k):
        for j in range(number_in_cluster[i]):
            for h in range(j + 1, number_in_cluster[i]):
                for l in range(n):
                    # Eq.38
                    dt = 0
                    if R[i][j, l] != R[i][h, l]:
                        dt = 1
                    dm[i] += dt ** 2

        dm[i] = dm[i] / (number_in_cluster[i] * (number_in_cluster[i] - 1))

    compactness = np.zeros(k)
    for i in range(k):
        compactness[i] = dm[i] * (number_in_cluster[i] / m)
    compactness = np.sum(compactness)

    return compactness

def Silhouette_score(x, label):
    label = label.astype(np.int64)
    silhouette_score_ = metrics.silhouette_score(x,label)
    return silhouette_score_

def Calinski_harabasz_score(x, label):
    label = label.astype(np.int64)
    calinski_harabasz_score = metrics.calinski_harabasz_score(x, label)
    return calinski_harabasz_score

def Davies_bouldin_score(x, label):
    label = label.astype(np.int64)
    davies_bouldin_score = metrics.davies_bouldin_score(x, label)
    return davies_bouldin_score