import pdb
import random
import numpy as np
from itertools import compress
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
import pickle
from e3fp.pipeline import fprints_from_mol
from e3fp.fingerprint.metrics.array_metrics import *


def generate_scaffold(smiles, include_chirality=True):
    """
    Obtain Bemis-Murcko scaffold from smiles
    Args:
        smiles: smiles sequence
        include_chirality: Default=True

    Return:
        the scaffold of the given smiles.
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold

data_path = 'data_new.pkl'
with open(data_path, 'rb') as f:
    obj = pickle.load(f)
Smiles = obj['SMILES']

def split(data, labels, frac_train, frac_valid, frac_test):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    N = len(data)

    all_scaffolds = {}
    for i in range(N):
        scaffold = generate_scaffold(Smiles[i], include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    train_cutoff = frac_train * N
    valid_cutoff = (frac_train + frac_valid) * N
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)
    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    train_data = data[train_idx]
    valid_data = data[valid_idx]
    test_data = data[test_idx]
    train_labels = labels[train_idx]
    valid_labels = labels[valid_idx]
    test_labels = labels[test_idx]

    n_samples = len(labels)
    class_counts = np.bincount(labels)
    train_counts = np.round((class_counts / n_samples) * len(train_labels))
    test_counts = np.round((class_counts / n_samples) * len(test_labels))
    valid_counts = np.round((class_counts/ n_samples) * len(valid_labels))
    train_data_fin, train_labels_fin = [], []
    test_data_fin, test_labels_fin = [], []
    valid_data_fin, valid_labels_fin = [], []

    for label in range(len(class_counts)):
        indices = np.where(train_labels == label)[0]
        selected_indices = np.random.choice(indices, size=int(train_counts[label]), replace=True)
        train_data_fin.extend(train_data[selected_indices])
        train_labels_fin.extend(train_labels[selected_indices])

        indices = np.where(test_labels == label)[0]
        selected_indices = np.random.choice(indices, size=int(test_counts[label]), replace=True)
        test_data_fin.extend(test_data[selected_indices])
        test_labels_fin.extend(test_labels[selected_indices])

        indices = np.where(valid_labels == label)[0]
        selected_indices = np.random.choice(indices, size=int(valid_counts[label]), replace=True)
        valid_data_fin.extend(valid_data[selected_indices])
        valid_labels_fin.extend(valid_labels[selected_indices])

    train_data_fin = np.array(train_data_fin)
    train_labels_fin = np.array(train_labels_fin)
    test_data_fin = np.array(test_data_fin)
    test_labels_fin = np.array(test_labels_fin)
    valid_data_fin = np.array(valid_data_fin)
    valid_labels_fin = np.array(valid_labels_fin)

    # print('train', np.bincount(train_labels_fin))
    # print('test', np.bincount(test_labels_fin))
    # print('valid', np.bincount(valid_labels_fin))
    return train_data_fin, train_labels_fin, valid_data_fin, valid_labels_fin, test_data_fin, test_labels_fin

def build_split(given_label, frac_train=0.5, frac_test=0.5):
    np.testing.assert_almost_equal(frac_train + frac_test, 1.0)
    N = len(given_label)

    all_scaffolds = {}
    for i in range(N):
        scaffold = generate_scaffold(Smiles[i], include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=False)
    ]
    np.random.seed(123)
    np.random.shuffle(all_scaffold_sets)
    train_cutoff = frac_train * N
    train_idx, test_idx = [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)
    assert len(set(train_idx).intersection(set(test_idx))) == 0

    train_label, test_label = [], []
    for i in train_idx:
        train_label.append(given_label[i])

    for i in test_idx:
        test_label.append(given_label[i])

    train_label = np.array(train_label)
    test_label = np.array(test_label)
    n_samples = len(given_label)
    class_counts = np.bincount(given_label)
    train_counts = np.round((class_counts / n_samples) * len(train_label))
    test_counts = np.round((class_counts / n_samples) * len(test_label))

    train_fin, test_fin = [], []
    for label in range(len(class_counts)):
        indices = np.where(train_label == label)[0]
        selected_indices = np.random.choice(indices, size=int(train_counts[label]), replace=True)
        train_fin.extend(selected_indices)

        indices = np.where(test_label == label)[0]
        pdb.set_trace()
        selected_indices = np.random.choice(indices, size=int(test_counts[label]), replace=True)
        test_fin.extend(selected_indices)

    train_fin = np.array(train_idx)
    test_fin = np.array(test_idx)

    np.save('train_idx.npy', train_fin)
    np.save('test_idx.npy', test_fin)

labels = obj['TARGET']
build_split(labels)
