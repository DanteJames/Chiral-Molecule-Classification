import pdb
import pickle
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import  KNeighborsClassifier
import matplotlib.pylab as plt
def save_list_to_file(lst, filename):
    with open(filename, 'w') as file:
        for item in lst:
            file.write(item + '\n')
Smiles = []
kick = []
i = 0
with open('SMILES.txt', 'r') as f:
    for i, l in enumerate(f.readlines()):
        line = l.strip('\n')
        if '*' in line:
            kick.append(i)
            continue
        Smiles.append(line)
data = np.load('data_3d.npy')
labels = np.load('labels.npy')
data = np.delete(data, kick, axis=0)
labels = np.delete(labels, kick, axis=0)
print(len(Smiles))
print(len(labels))
print(len(data))
print(kick)
# np.save('data_3d.npy', data)
# np.save('labels.npy', labels)
# save_list_to_file(Smiles, 'SMILES.txt')
