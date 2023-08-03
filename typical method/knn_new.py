import pdb

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import  KNeighborsClassifier
import matplotlib.pylab as plt
from loguru import logger
import os, sys
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
from cluster import split
def get_logger(output_file):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    while os.path.exists(output_file):
        output_file = output_file.replace('.log', '1.log')
    if output_file:
        logger.add(output_file, enqueue=True, format=log_format)
    return logger

output = f'log_knn/results_knn.log'
loger = get_logger(output)



def train(n_neigh, given_data, given_label):
    train_data, train_label, valid_data, valid_label, test_data, test_label = split(given_data, given_label, 0.6, 0.2, 0.2)
    knn = KNeighborsClassifier(n_neighbors=n_neigh, weights='distance', p=2)
    knn1 = KNeighborsClassifier(n_neighbors=n_neigh, weights='distance', p=2)
    knn.fit(train_data, train_label)
    kf = KFold(n_splits=5, shuffle=True, random_state=114514)
    val_scores = round(cross_val_score(knn1, valid_data, valid_label, cv=kf).mean(), 4)
    res = knn.predict(test_data)
    test_acc = round(accuracy_score(test_label, res), 4)
    loger.info(f'k={n_neigh}')
    loger.info(f'valid_score={val_scores*100}%.')
    loger.info(f'test_acc={test_acc*100}%.')
    confusion = confusion_matrix(test_label, res)
    for i in range(len(confusion)):
        if np.sum(confusion[i, :]) == 0:
            loger.info(f'There is no category{i} in test_set.')
        else:
            pre = round(confusion[i][i]/np.sum(confusion[i, :]), 4) * 100
            loger.info(f'precision_{i}={pre}%')
    return val_scores*100, test_acc*100

if __name__ == '__main__':
    data_3d = np.load('data_3d.npy')
    labels = np.load('labels.npy')
    num = []
    vals = []
    accs = []
    for k in range(1, 21):
        num.append(k)
        val, acc = train(k, data_3d, labels)
        vals.append(val)
        accs.append(acc)
    plt.figure()
    plt.xlabel('number of neighbors')  # x轴标题
    plt.ylabel('accuracy(%)')  # y轴标题
    # for a, b in zip(num, acc):
    #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    plt.plot(num, accs, marker='o', markersize=3)
    plt.plot(num, vals, marker='^', markersize=3)
    plt.legend(['accuracy', 'valid'])
    plt.savefig(f'img_knn/accuracy line graph for knn.jpg')
    plt.show()

    # plt.figure()
    # data_fin = []
    # label_fin = []
    # loger.info('this is test 2:')
    # for i in range(len(labels)):
    #     if labels[i] != 0:
    #         data_fin.append(data_3d[i])
    #         label_fin.append(labels[i] - 1)
    # accuracies = []
    # valids = []
    # data_fin = np.array(data_fin)
    # label_fin = np.array(label_fin)
    # for k in range(1, 21):
    #     val, acc = train(k, data_fin, label_fin)
    #     accuracies.append(acc)
    #     valids.append(val)
    # plt.xlabel('number of neighbors')  # x轴标题
    # plt.ylabel('accuracy(%)')  # y轴标题
    # # for a, b in zip(num, acc):
    # #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    # plt.plot(num, accuracies, marker='o', markersize=3)
    # plt.plot(num, valids, marker='^', markersize=3)
    # plt.legend(['accuracy', 'valid'])
    # plt.savefig(f'img_knn/accuracy line graph for knn 2.jpg')


