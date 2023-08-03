import pdb
import pickle
import os
from collections import defaultdict
import pandas as pd
import numpy as np
from e3fp.pipeline import fprints_from_mol
from e3fp.fingerprint.metrics.array_metrics import *
from rdkit import Chem
from rdkit.Chem import AllChem

def save_list_to_file(lst, filename):
    with open(filename, 'w') as file:
        for item in lst:
            file.write(item + '\n')


data_path = 'chiral_data_3d_aug.pkl'
with open(data_path, 'rb') as f:
    obj = pickle.load(f)
dictionary = {}
labels = []
# df = pd.read_csv('chiral_data.csv')
chiral_type = obj['chiral type']
i = 0
for l in chiral_type:
    if l not in dictionary.keys():
        dictionary[l] = i
        i+=1
fprint_params = {'bits': 4096, 'first': 5}
data_3d = []
smiles = obj['SMILES']
SMILES = []
data = Chem.JSONToMols(obj['mol'])
pdb.set_trace()
for i, l in enumerate(chiral_type):
    mol = data[i]
    mol.SetProp('_Name', obj["SMILES"][i])
    prints3 = fprints_from_mol(mol, fprint_params=fprint_params)
    for j in range(len(prints3)):
        data_3d.append(prints3[j].to_vector(sparse=False))
    smile = [smiles[i]] * len(prints3)
    SMILES+=smile
    label = [dictionary[l]] * len(prints3)
    labels += label

labels = np.array(labels)
np.save('labels.npy', labels)
print('labels:',len(labels))
# # data = np.load('morgan_fp.npy')
# # print(len(data))
# # pdb.set_trace()
data_3d = np.array(data_3d)
np.save('data_3d.npy', data_3d)
print('data_3d:',len(data_3d))
save_list_to_file(SMILES, 'SMILES.txt')
print('smiles:', len(SMILES))
#
# # data = []
# # for smiles in SMILES:
# #     mol = Chem.MolFromSmiles(smiles)
# #     fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
# #     data.append(fp)
# # data = np.array(data)
# # np.save('morgan_fp.npy', data)
# # print(len(labels))
# # print(len(data_3d))
# # print(len(data))