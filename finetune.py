from unimol_tools import MolTrain, MolPredict
import pickle
data1 = 'train_set_1.pkl'
data2 = 'test_set_1.pkl'
with open(data1, 'rb') as f:
    train_set = pickle.load(f)
with open(data2, 'rb') as g:
    test_set = pickle.load(g)
clf = MolTrain(task='multiclass',
                data_type='molecule',
                epochs=20,
                learning_rate=0.0001,
                batch_size=16,
                early_stopping=5,
                metrics='acc',
                split='scaffold_random',
                save_path='./exp2',
                remove_hs=False,
              )
pred = clf.fit(data=train_set)
# currently support data with smiles based csv/txt file, and
# custom dict of {'atoms':[['C','C],['C','H','O']], 'coordinates':[coordinates_1,coordinates_2]}

clf = MolPredict(load_model='exp2/')
res = clf.predict(data=test_set)