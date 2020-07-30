#! /usr/bin/env python3

from time import time
import os
from copy import copy

import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF
import pickle

class Analyzer:
    def __init__(self, fname):
        self.data = np.load(fname, mmap_mode='r')
        self.rf = []
        self.distance = None
        self.pred = None
        self.rhos = None

    def train(self, size=10_000, n_estimators=10):
        clf = RF(n_estimators=n_estimators, n_jobs=-1)
        idx_spec = np.random.choice(self.data.shape[1], size, replace=False)
        rdm_spec = np.copy(self.data[:, idx_spec].T)
        synt_spec = np.copy(rdm_spec)
        for wl in synt_spec.T:
            np.random.shuffle(wl)

        labels = np.ones(2*size)
        labels[size:] = 0
        print('Training the model!', end=' ')
        clf.fit(np.vstack((rdm_spec, synt_spec)), labels)
        print('Training finished!', end=' ')
        self.rf.append(clf)

    def apply(self):
        self.pred = self.rf.apply(self.data.T)

    def analyze(self, n_spec):
        self.rhos = np.empty(self.pred[:n_spec, :].shape[0])
        for idx, leaves in enumerate(self.pred[:n_spec, :]):
            print(f'Computing the outlier score for the spectrum N° {idx}', end='\r')
            self.rhos[idx] = np.count_nonzero(self.pred[:n_spec, :] == leaves)

        self.rhos = 1 - self.rhos * (1 / self.pred[:n_spec, :].size)

    def clf_merging(self):
        dad_rf = copy(self.rf[0])
        for clf in self.rf[1:]:
            dad_rf.estimators_ += clf.estimators_
        dad_rf.n_estimators = len(dad_rf.estimators_)
        self.rf = dad_rf

ti = time()

## Random forest implementation
# Creating several clfs

rf = Analyzer('/home/edgar/zorro/MaNGAdata/spectra_bin_9.npy')

n_iles = 100

for step in range(n_iles):
    print(f'Training island forest {step:003}', end='\r')
    rf.train(n_estimators=200)

rf.clf_merging()

## Saving the model to disk
#if not os.path.exists(f'RF_{n_iles}.pkl'):
#    with open('RF_{n_iles}.pkl', 'wb') as file:
#        pickle.dump(rf, file)
#else:
## Loading the model
#    with open('RF_{n_iles}.pkl', 'rb') as file:
#        rf = pickle.load(file)

rf.apply()
rf.analyze(n_spec=216_871) #216_871) Change for taking everython autoatically

#np.save('rhos.npy', rf.rhos)
tf = time()
print(f'Runing time= {tf-ti:.2f} [s]')
