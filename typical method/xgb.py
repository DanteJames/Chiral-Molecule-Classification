import pdb
import pickle
import numpy as np
import xgboost as xgb
from loguru import logger
import os, sys
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from  sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
def get_logger(output_file):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    while os.path.exists(output_file):
        output_file = output_file.replace('.log', '1.log')
    if output_file:
        logger.add(output_file, enqueue=True, format=log_format)
    return logger

output = f'log_xgb/results_xgb.log'
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
print(len(dictionary))
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

def train(given_data, given_label, num_class, lr=0.001, m_depth = 6):
    train_data, test_data, train_label, test_label = train_test_split(given_data, given_label, test_size=0.4)
    train = xgb.DMatrix(train_data, label=train_label)
    test = xgb.DMatrix(test_data, label=test_label)
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': num_class,
        'max_depth': m_depth,
        'eta': lr,
        'gamma': 0,
        'min_child_weight': 1,
        'subsample': 1,
        'colsample_bytree': 1,
        'colsample_bylevel': 1,
        'lambda': 1,
        'alpha': 0,
        'nthread': -1,
        'eval_metric': 'merror',
        'seed': 0
    }
    model = xgb.train(list(params.items()), train, 10)

    res = model.predict(test)

    test_acc = round(accuracy_score(test_label, res), 4)
    loger.info(f'lr={lr}, depth = {m_depth}, acc = {test_acc*100}%')
    confusion = confusion_matrix(test_label, res)
    # pdb.set_trace()
    for i in range(len(confusion)):
        pre = round(confusion[i][i] / np.sum(confusion[i, :]), 4) * 100
        loger.info(f'precision_{i}={pre}%')
    return test_acc*100

def test_1(sample_size):
    loger.info('this is test_1:')
    sample_indices = np.random.choice(len(padded_data), sample_size, replace=False)
    sample_data = padded_data[sample_indices]
    sample_label = labels[sample_indices]
    loger.info(f'sample size = {sample_size}')
    accs = []
    lrs = [0.001,0.002, 0.004, 0.005, 0.01,0.02, 0.05, 0.1]
    for lr in lrs:
        acc = train(sample_data, sample_label, 4, lr)
        accs.append(acc)
    plt.figure()
    plt.xlabel('learning rate')  # x轴标题
    plt.ylabel('accuracy(%)')  # y轴标题
    plt.plot(lrs, accs, marker='o', markersize=3)
    plt.savefig(f'fig_xgb/new_accuracy line graph for xgb {sample_size}.jpg')
    plt.show()

def test_2():
    data_fin = []
    label_fin = []
    for i in range(len(labels)):
        if labels[i] != 0:
            data_fin.append(padded_data[i])
            label_fin.append(labels[i] - 1)
    loger.info('this is test_2:')
    lrs = [0.001,0.002, 0.004, 0.005, 0.01,0.02, 0.05, 0.1]
    accs = []
    for lr in lrs:
        acc = train(data_fin, label_fin, 3, lr)
        accs.append(acc)
    plt.figure()
    plt.xlabel('learning rate')  # x轴标题
    plt.ylabel('accuracy(%)')  # y轴标题
    plt.plot(lrs, accs, marker='o', markersize=3)
    plt.savefig(f'fig_xgb/new_accuracy line graph for xgb1.jpg')
    plt.show()

def test_3(sample_size):
    loger.info('this is test_3:')
    sample_indices = np.random.choice(len(padded_data), sample_size, replace=False)
    sample_data = padded_data[sample_indices]
    sample_label = labels[sample_indices]
    loger.info(f'sample size = {sample_size}')
    accs = []
    depth = [4, 5, 6, 7, 8]
    for d in depth:
        acc = train(sample_data, sample_label, 4, m_depth=d)
        accs.append(acc)
    plt.figure()
    plt.xlabel('depth')  # x轴标题
    plt.ylabel('accuracy(%)')  # y轴标题
    plt.plot(depth, accs, marker='o', markersize=3)
    plt.savefig(f'fig_xgb/accuracy line graph for xgb_depth {sample_size}.jpg')
    plt.show()

def test_4():
    data_fin = []
    label_fin = []
    for i in range(len(labels)):
        if labels[i] != 0:
            data_fin.append(padded_data[i])
            label_fin.append(labels[i] - 1)
    loger.info('this is test_4:')
    accs = []
    depth = [4, 5, 6, 7, 8]
    for d in depth:
        acc = train(data_fin, label_fin, num_class=3, m_depth=d)
        accs.append(acc)
    plt.figure()
    plt.xlabel('depth')  # x轴标题
    plt.ylabel('accuracy(%)')  # y轴标题
    plt.plot(depth, accs, marker='o', markersize=3)
    plt.savefig(f'fig_xgb/accuracy line graph for xgb_depth.jpg')
    plt.show()

def train_prime(given_data, given_label, num_class, lr=0.001, m_depth = 6):
    k = 5
    kf = KFold(n_splits=k, shuffle=True)

    scores = []
    for train_index, test_index in kf.split(given_data):
        train_data, test_data = given_data[train_index], given_data[test_index]
        train_label, test_label = given_label[train_index], given_label[test_index]

        train = xgb.DMatrix(train_data, label=train_label)
        test = xgb.DMatrix(test_data, label=test_label)
        params = {
            'booster': 'gbtree',
            'objective': 'multi:softmax',
            'num_class': num_class,
            'max_depth': m_depth,
            'eta': 0.1,
            'gamma': 0,
            'min_child_weight': 1,
            'subsample': 1,
            'colsample_bytree': 1,
            'colsample_bylevel': 1,
            'lambda': 1,
            'alpha': 0,
            'nthread': -1,
            'eval_metric': 'merror',
            'seed': 0
        }
        model = xgb.train(list(params.items()), train, 10)

        res = model.predict(test)


        score = accuracy_score(test_label, res)
        scores.append(score)
    average_score = round(sum(scores) / len(scores), 4)
    loger.info(f'lr={lr}, depth = {m_depth}, acc = {average_score * 100}%')
    return average_score*100

def test_5(sample_size):
    loger.info('this is test 5:')
    sample_indices = np.random.choice(len(padded_data), sample_size, replace=False)
    sample_data = padded_data[sample_indices]
    sample_label = labels[sample_indices]
    acc = []
    lrs = [0.001, 0.002, 0.004, 0.005, 0.01, 0.02, 0.05, 0.1]
    for lr in lrs:
        acc.append(train_prime(sample_data, sample_label,num_class=4, lr=lr))
    plt.figure()
    plt.xlabel('learning rate')  # x轴标题
    plt.ylabel('accuracy(%)')  # y轴标题
    plt.plot(lrs, acc, marker='o', markersize=3)
    plt.savefig(f'fig_xgb/new_accuracy line graph for xgb_lr_val{sample_size}.jpg')
    plt.show()

def test_6():
    loger.info('this is test 6:')
    data_fin = []
    label_fin = []
    for i in range(len(labels)):
        if labels[i] != 0:
            data_fin.append(padded_data[i])
            label_fin.append(labels[i] - 1)
    acc = []
    data_fin = np.array(data_fin)
    label_fin = np.array(label_fin)
    # pdb.set_trace()
    lrs = [0.001, 0.002, 0.004, 0.005, 0.01, 0.02, 0.05, 0.1]
    for lr in lrs:
        acc.append(train_prime(data_fin, label_fin,num_class=3, lr=lr, m_depth=6))
    plt.figure()
    plt.xlabel('learning rate')  # x轴标题
    plt.ylabel('accuracy(%)')  # y轴标题
    plt.plot(lrs, acc, marker='o', markersize=3)
    plt.savefig(f'fig_xgb/new_accuracy line graph for xgb1_lr_val.jpg')
    plt.show()

def test_7(sample_size):
    loger.info('this is test_7:')
    sample_indices = np.random.choice(len(padded_data), sample_size, replace=False)
    sample_data = padded_data[sample_indices]
    sample_label = labels[sample_indices]
    loger.info(f'sample size = {sample_size}')
    accs = []
    depth = [4, 5, 6, 7, 8]
    for d in depth:
        acc = train_prime(sample_data, sample_label, 4, m_depth=d)
        accs.append(acc)
    plt.figure()
    plt.xlabel('depth')  # x轴标题
    plt.ylabel('accuracy(%)')  # y轴标题
    plt.plot(depth, accs, marker='o', markersize=3)
    plt.savefig(f'fig_xgb/accuracy line graph for xgb_depth_val {sample_size}.jpg')
    plt.show()

def test_8():
    data_fin = []
    label_fin = []

    for i in range(len(labels)):
        if labels[i] != 0:
            data_fin.append(padded_data[i])
            label_fin.append(labels[i] - 1)
    data_fin = np.array(data_fin)
    label_fin = np.array(label_fin)
    loger.info('this is test_8:')
    accs = []
    depth = [4, 5, 6, 7, 8]
    for d in depth:
        acc = train_prime(data_fin, label_fin, num_class=3, m_depth=d)
        accs.append(acc)
    plt.figure()
    plt.xlabel('depth')  # x轴标题
    plt.ylabel('accuracy(%)')  # y轴标题
    plt.plot(depth, accs, marker='o', markersize=3)
    plt.savefig(f'fig_xgb/accuracy line graph for xgb1_depth_val.jpg')
    plt.show()

for s in [300, 500, 1000]:
    test_1(s)
test_2()
for s in [300, 500, 1000]:
    test_3(s)
test_4()
for s in [300, 500, 1000]:
    test_5(s)
test_6()
for s in [300, 500, 1000]:
    test_7(s)
test_8()
