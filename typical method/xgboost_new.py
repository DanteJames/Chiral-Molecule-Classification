import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pylab as plt
from loguru import logger
import os, sys
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
from cluster import split
from sklearn.svm import SVC
import xgboost as xgb
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

def valid_train(given_data, give_labels, lr):
    kf = KFold(n_splits=5, shuffle=True, random_state=114514)
    scores = []
    for train_index, test_index in kf.split(given_data):
        train_data, test_data = given_data[train_index], given_data[test_index]
        train_label, test_label = give_labels[train_index], give_labels[test_index]

        train = xgb.DMatrix(train_data, label=train_label)
        test = xgb.DMatrix(test_data, label=test_label)
        params = {
            'booster': 'gbtree',
            'objective': 'multi:softmax',
            'num_class': 4,
            'max_depth': 6,
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

        score = accuracy_score(test_label, res)
        scores.append(score)
    average_score = round(sum(scores) / len(scores), 4)
    return average_score
def train(given_data, given_label, num_class, lr=0.001, m_depth = 6):
    train_data, train_label, valid_data, valid_label, test_data, test_label = split(given_data, given_label, 0.6, 0.2,
                                                                                0.2)
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
    loger.info(f'lr={lr}, depth = {m_depth}, acc = {test_acc * 100}%')
    confusion = confusion_matrix(test_label, res)
    # pdb.set_trace()
    val_scores = valid_train(valid_data, valid_label, lr)
    for i in range(len(confusion)):
        if np.sum(confusion[i, :]) == 0:
            loger.info(f'There is no category{i} in test_set.')
        else:
            pre = round(confusion[i][i] / np.sum(confusion[i, :]), 4) * 100
            loger.info(f'precision_{i}={pre}%')
    return val_scores * 100, test_acc * 100

if __name__ == '__main__':
    data_3d = np.load('data_3d.npy')
    labels = np.load('labels.npy')
    accs = []
    vals = []
    etas = [0.00005, 0.00006, 0.00008, 0.0001, 0.00011]
    for eta in etas:
        val, acc = train(data_3d, labels, 4, eta)
        vals.append(val)
        accs.append(acc)
    plt.figure()
    plt.xlabel('number of neighbors')  # x轴标题
    plt.ylabel('accuracy(%)')  # y轴标题
    # for a, b in zip(num, acc):
    #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    plt.plot(etas, accs, marker='o', markersize=3)
    plt.plot(etas, vals, marker='^', markersize=3)
    plt.legend(['accuracy', 'valid'])
    plt.savefig(f'img_xgb/accuracy line graph for xgb1.jpg')
    plt.show()