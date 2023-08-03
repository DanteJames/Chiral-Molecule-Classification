import pdb

from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
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

Smiles = []
with open('SMILES.txt', 'r') as f:
    for l in f.readlines():
        line = l.strip('\n')
        Smiles.append(line)

def split(given_data, given_label, frac_train=0.7, frac_test=0.3):
    np.testing.assert_almost_equal(frac_train + frac_test, 1.0)
    N = len(given_data)

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

    remaining_idx = []
    leng = 0
    for scaffold_set in all_scaffold_sets:
        remains = []
        zeroes = []
        for i in range(len(scaffold_set)):
            if given_label[scaffold_set[i]] != 0:
                remains.append(scaffold_set[i])
            elif given_label[scaffold_set[i]] == 0:
                zeroes.append(scaffold_set[i])
        num = len(remains)
        center_num = len(zeroes)
        # pdb.set_trace()
        if center_num >= num*2:
            random_indices = np.random.choice(zeroes, size=2*num, replace=False)
            remains.extend(random_indices)
        else:
            remains.extend(zeroes)
        if num == 0:
            random_indices = np.random.choice(zeroes, size=center_num//5, replace=False)
            remains.extend(random_indices)
        # print(len(remains))
        # pdb.set_trace()
        leng += len(remains)
        remaining_idx.append(remains)
        # if 0 in remains:
        #     print("yes")
        # else:
        #     print('No')
    np.random.seed(123)
    np.random.shuffle(remaining_idx)
    train_cutoff = frac_train * leng
    train_idx, test_idx = [], []
    for re in remaining_idx:
        if len(train_idx) + len(re) > train_cutoff:
            test_idx.extend(re)
        else:
            train_idx.extend(re)
    assert len(set(train_idx).intersection(set(test_idx))) == 0
    train_data = given_data[train_idx]
    train_label = given_label[train_idx]
    test_data = given_data[test_idx]
    test_label = given_label[test_idx]

    return train_data, train_label, test_data, test_label


data_3d = np.load('data_3d.npy')
labels = np.load('labels.npy')
train_data, train_label, test_data, test_label = split(data_3d, labels)
pdb.set_trace()
print(len(data_3d))
