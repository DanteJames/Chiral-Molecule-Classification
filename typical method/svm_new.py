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
from sklearn.svm import SVC
def get_logger(output_file):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    while os.path.exists(output_file):
        output_file = output_file.replace('.log', '1.log')
    if output_file:
        logger.add(output_file, enqueue=True, format=log_format)
    return logger

output = f'log_svm/results_svm.log'
loger = get_logger(output)



def train(c, gamma, given_data, given_label):
    train_data, train_label, valid_data, valid_label, test_data, test_label = split(given_data, given_label, 0.6, 0.2, 0.2)
    svm = SVC(kernel='rbf', gamma=gamma, C=c)
    svm.fit(train_data, train_label)
    kf = KFold(n_splits=5, shuffle=True, random_state=114514)
    val_scores = round(cross_val_score(svm, valid_data, valid_label, cv=kf).mean(), 4)
    res = svm.predict(test_data)
    test_acc = round(accuracy_score(test_label, res), 4)
    loger.info(f'c = {c}, gamma = {gamma}')
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
    C = [0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    Gamma = [0.01, 0.05, 0.1, 0.5, 1.0]
    vals = []
    accs = []
    for c in C:
        for gamma in Gamma:
            val, acc = train(c, gamma, data_3d, labels)
            vals.append(val)
            accs.append(acc)
    plt.figure()
    plt.xlabel('c of svm')  # x轴标题
    plt.ylabel('accuracy(%)')  # y轴标题
    # for a, b in zip(num, acc):
    #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    plt.plot(C, accs[0:8], marker='o', markersize=3)
    plt.plot(C, accs[8:16], marker='^', markersize=3)
    plt.plot(C, accs[16:24], marker='o', markersize=3)
    plt.plot(C, accs[24:32], marker='^', markersize=3)
    plt.plot(C, accs[32:40], marker='o', markersize=3)
    plt.legend(['gamma=0.01', 'gamma=0.05', 'gamma=0.1', 'gamma=0.5', 'gamma=1.0'])
    plt.savefig(f'img_svm/accuracy line graph for svm.jpg')
    plt.figure()
    plt.xlabel('c')  # x轴标题
    plt.ylabel('valids(%)')  # y轴标题
    plt.plot(C, vals[0:8], marker='^', markersize=3)
    plt.plot(C, vals[8:16], marker='^', markersize=3)
    plt.plot(C, vals[16:24], marker='o', markersize=3)
    plt.plot(C, vals[24:32], marker='^', markersize=3)
    plt.plot(C, vals[32:40], marker='o', markersize=3)
    plt.legend(['gamma=0.01', 'gamma=0.05', 'gamma=0.1', 'gamma=0.5', 'gamma=1.0'])
    plt.savefig(f'img_svm/valid line graph for svm.jpg')
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
    # for c in C:
    #     for gamma in Gamma:
    #         val, acc = train(c, gamma, data_fin, label_fin)
    #         valids.append(val)
    #         accuracies.append(acc)
    # plt.figure()
    # plt.xlabel('c of svm')  # x轴标题
    # plt.ylabel('accuracy(%)')  # y轴标题
    # # for a, b in zip(num, acc):
    # #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    # plt.plot(C, accuracies[0:8], marker='o', markersize=3)
    # plt.plot(C, accuracies[8:16], marker='^', markersize=3)
    # plt.plot(C, accuracies[16:24], marker='o', markersize=3)
    # plt.plot(C, accuracies[24:32], marker='^', markersize=3)
    # plt.plot(C, accuracies[32:40], marker='o', markersize=3)
    # plt.legend(['gamma=0.01', 'gamma=0.05', 'gamma=0.1', 'gamma=0.5', 'gamma=1.0'])
    # plt.savefig(f'img_svm/accuracy line graph for svm_1.jpg')
    # plt.figure()
    # plt.xlabel('c')  # x轴标题
    # plt.ylabel('valids(%)')  # y轴标题
    # plt.plot(C, valids[0:8], marker='^', markersize=3)
    # plt.plot(C, valids[8:16], marker='^', markersize=3)
    # plt.plot(C, valids[16:24], marker='o', markersize=3)
    # plt.plot(C, valids[24:32], marker='^', markersize=3)
    # plt.plot(C, valids[32:40], marker='o', markersize=3)
    # plt.legend(['gamma=0.01', 'gamma=0.05', 'gamma=0.1', 'gamma=0.5', 'gamma=1.0'])
    # plt.savefig(f'img_svm/valid line graph for svm_1.jpg')
    # plt.show()


