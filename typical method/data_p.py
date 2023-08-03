import pdb
import pickle
import os
from collections import defaultdict
import pandas as pd
import numpy as np
# from e3fp.pipeline import fprints_from_mol
# from e3fp.fingerprint.metrics.array_metrics import *
from rdkit import Chem
from rdkit.Chem import AllChem

data_path_1 = 'train_set_1.pkl'
with open(data_path_1, 'rb') as f:
    obj = pickle.load(f)
data_path = 'test_set_1.pkl'
with open(data_path, 'rb') as f:
    obj1 = pickle.load(f)
pdb.set_trace()
# pdb.set_trace()
label = obj['TARGET']
# mols = Chem.JSONToMols(obj['mol'])
# # pdb.set_trace()
# atm = []
# cor = []
# for mol in mols:
#     atoms = mol.GetAtoms()
#     symbols = [atom.GetSymbol() for atom in atoms]
#     # get the first conformer
#     conf_num = mol.GetNumConformers()
#     conformer = mol.GetConformer(0)
#     coordinates = conformer.GetPositions()
#     atm.append(symbols)
#     cor.append(coordinates)
# train_idx = np.load('train_idx.npy')
# test_idx = np.load('test_idx.npy')
# train_a = []
# train_c = []
# train_l = []
# test_a = []
# test_c = []
# test_l = []
# for i in range(len(train_idx)):
#     train_a.append(atm[train_idx[i]])
#     train_c.append(cor[train_idx[i]])
#     train_l.append(label[train_idx[i]])
# test_a = []
# test_c = []
# test_l = []
# for i in range(len(test_idx)):
#     test_a.append(atm[test_idx[i]])
#     test_c.append(cor[test_idx[i]])
#     test_l.append(label[test_idx[i]])
# train_set = {}
# test_set= {}
# train_set['atoms'] = train_a
# train_set['coordinates'] = train_c
# train_set['TARGET'] = train_l
# test_set['atoms'] = test_a
# test_set['coordinates'] = test_c
# test_set['TARGET'] = test_l
# # dictionary = {}
# # i = 0
# # for l in chiral_type:
# #     if l not in dictionary.keys():
# #         dictionary[l] = i
# #         i+=1
# # labels = []
# # for l in chiral_type:
# #     labels.append(dictionary[l])
# # obj['TARGET'] = labels
# # obj.pop('chiral type', None)

# train_path = 'train_set_1.pkl'
# with open(train_path, 'wb') as f:
#     pickle.dump(train_set, f)
# test_path = 'test_set_1.pkl'
# with open(test_path, 'wb') as f:
#     pickle.dump(test_set, f)
