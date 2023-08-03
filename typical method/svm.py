import pdb
from loguru import logger
import pickle
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pylab as plt
from sklearn.svm import SVC
import os
import sys
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
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

with open('chiral_data_3d.pkl', 'rb') as f:
    obj = pickle.load(f)
padding_value = 0.0
mask_value = 1.0

padded_data = []
mask = []

Smiles = obj['SMILES']
data = obj['pos']
label = obj['chiral type']
labels = []
dictionary = {}
max_len = 0
min_len = 100000
for d in data:
    max_len = max(len(d), max_len)
    min_len = min(len(d), min_len)
i = 0
for l in label:
    if l not in dictionary.keys():
        dictionary[l] = i
        i+=1

for l in label:
    labels.append(dictionary[l])
labels = np.array(labels)
# print(f'max_len = {max_len}')
# print(f'min_len = {min_len}')

for d in data:
    num_padding = max_len - len(d)
    # pdb.set_trace()
    padded_d = np.concatenate((d, np.full((num_padding, 3), padding_value)), axis=0)

    padded_d = padded_d.reshape(max_len*3)
    # pdb.set_trace()
    padded_data.append(padded_d)

    d_mask = [1.0] * len(d) + [mask_value] * num_padding
    mask.append(d_mask)

padded_data = np.array(padded_data)
mask = np.array(mask)
# center = []
# planar = []
# for i in range(len(labels)):


# pdb.set_trace()
def train(given_kernel, given_c, given_data, given_label):
    train_data, test_data, train_label, test_label = train_test_split(given_data, given_label, test_size=0.3)
    svm = SVC(kernel=given_kernel, C=given_c)
    svm.fit(train_data, train_label)
    test_acc = round(accuracy_score(test_label, svm.predict(test_data)), 4)
    loger.info(f'kernel = {given_kernel}')
    loger.info(f'C = {given_c}')
    loger.info(f'test_acc={test_acc*100}%.')
    confusion = confusion_matrix(test_label, svm.predict(test_data))
    # pdb.set_trace()
    for i in range(len(confusion)):
        pre = round(confusion[i][i] / np.sum(confusion[i, :]), 4) * 100
        loger.info(f'precision_{i}={pre}%')
    return test_acc*100

def test_1(sample_size = 300):
    loger.info('this is test 1:')
    cs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5,  1, 5,  10, 100]
    kernels = ['linear', 'rbf', 'sigmoid', 'poly']
    acc = []
    sample_indices = np.random.choice(len(padded_data), sample_size, replace=False)
    sample_data = padded_data[sample_indices]
    sample_label = labels[sample_indices]
    loger.info(f'sample size = {sample_size}')
    for k in kernels:
        for c in cs:
            acc.append(train(k, c, sample_data, sample_label))
    plt.figure()
    plt.xlabel('the c of svm')  # x轴标题
    plt.ylabel('accuracy(%)')  # y轴标题
    plt.plot(cs, acc[:10], marker='o', markersize=3)
    plt.plot(cs, acc[10:20], marker='D', markersize=3)
    plt.plot(cs, acc[20:30], marker='^', markersize=3)
    plt.plot(cs, acc[30:40], marker='^', markersize=3)
    plt.legend(['linear', 'rbf', 'sigmoid', 'poly'])
    plt.savefig(f'fig_svm/accuracy line graph for svm{sample_size}.jpg')
    plt.show()


def test_2():
    data_fin = []
    label_fin = []
    for i in range(len(labels)):
        if labels[i] != 0:
            data_fin.append(padded_data[i])
            label_fin.append(labels[i] - 1)
    loger.info('this is test 2:')
    cs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5,  1, 5,  10, 100]
    kernels = ['linear', 'rbf', 'sigmoid', 'poly']
    acc = []
    for k in kernels:
        for c in cs:
            acc.append(train(k, c, data_fin, label_fin))
    plt.figure()
    plt.xlabel('the c of svm')  # x轴标题
    plt.ylabel('accuracy(%)')  # y轴标题
    plt.plot(cs, acc[:10], marker='o', markersize=3)
    plt.plot(cs, acc[10:20], marker='D', markersize=3)
    plt.plot(cs, acc[20:30], marker='^', markersize=3)
    plt.plot(cs, acc[30:40], marker='^', markersize=3)
    plt.legend(['linear', 'rbf', 'sigmoid', 'poly'])
    plt.savefig(f'fig_svm/accuracy line graph for svm1.jpg')
    plt.show()

def train_prime(given_kernel, given_c, given_data, given_label):
    k = 5
    kf = KFold(n_splits=k, shuffle=True)

    scores = []
    for train_index, test_index in kf.split(given_data):
        train_data, test_data = given_data[train_index], given_data[test_index]
        train_label, test_label = given_label[train_index], given_label[test_index]

        svm = SVC(kernel=given_kernel, C=given_c)
        svm.fit(train_data, train_label)

        score = accuracy_score(test_label, svm.predict(test_data))
        scores.append(score)
    average_score = round(sum(scores) / len(scores), 4)
    loger.info(f'c={given_c},kernel={given_kernel},acc = {average_score*100}%.')
    return average_score*100

def test_3(sample_size):
    loger.info('this is test 3:')
    sample_indices = np.random.choice(len(padded_data), sample_size, replace=False)
    sample_data = padded_data[sample_indices]
    sample_label = labels[sample_indices]
    cs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100]
    kernels = ['linear', 'rbf', 'sigmoid', 'poly']
    acc = []
    for k in kernels:
        for c in cs:
            acc.append(train_prime(k, c, sample_data, sample_label))
    plt.figure()
    plt.xlabel('the c of svm')  # x轴标题
    plt.ylabel('accuracy(%)')  # y轴标题
    plt.plot(cs, acc[:10], marker='o', markersize=3)
    plt.plot(cs, acc[10:20], marker='D', markersize=3)
    plt.plot(cs, acc[20:30], marker='^', markersize=3)
    plt.plot(cs, acc[30:40], marker='^', markersize=3)
    plt.legend(['linear', 'rbf', 'sigmoid', 'poly'])
    plt.savefig(f'fig_svm/accuracy line graph for svm_val_{sample_size}.jpg')
    plt.show()

def test_4():
    data_fin = []
    label_fin = []
    loger.info('this is test 4:')
    for i in range(len(labels)):
        if labels[i] != 0:
            data_fin.append(padded_data[i])
            label_fin.append(labels[i] - 1)
    data_fin=np.array(data_fin)
    label_fin=np.array(label_fin)
    cs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100]
    kernels = ['linear', 'rbf', 'sigmoid', 'poly']
    acc = []
    for k in kernels:
        for c in cs:
            ac = train_prime(k, c, data_fin, label_fin)
        # pdb.set_trace()
            acc.append(ac)
    plt.figure()
    plt.xlabel('the c of svm')  # x轴标题
    plt.ylabel('accuracy(%)')  # y轴标题
    plt.plot(cs, acc[:10], marker='o', markersize=3)
    plt.plot(cs, acc[10:20], marker='D', markersize=3)
    plt.plot(cs, acc[20:30], marker='^', markersize=3)
    plt.plot(cs, acc[30:40], marker='^', markersize=3)
    plt.legend(['linear', 'rbf', 'sigmoid', 'poly'])
    plt.savefig(f'fig_svm/accuracy line graph for svm1_val.jpg')
    plt.show()

for num in [300, 500, 1000]:
    test_1(num)
test_2()
for s in [300, 500, 1000]:
    test_3(s)
test_4()