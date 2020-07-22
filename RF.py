#! /usr/bin/env python3

from time import time
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF
import pickle

class Analyzer:
    def __init__(self, fname):
        self.data = np.load(fname, mmap_mode='r')
        self.rf = None
        self.distance = None
        self.pred = None
        self.rhos = None

    def train(self, size=10_000):
        clf = RF(n_estimators=2_000, n_jobs=-1)
        idx_spec = np.random.choice(self.data.shape[1], size, replace=False)
        rdm_spec = np.copy(self.data[:, idx_spec].T)
        synt_spec = np.copy(rdm_spec)
        for wl in synt_spec:
            np.random.shuffle(wl)

        labels = np.ones(2*size)
        labels[size:] = 0
        print('Training the model!')
        clf.fit(np.vstack((rdm_spec, synt_spec)), labels)
        print('Training finished!')
        self.rf = clf

    def apply(self):
        self.pred = self.rf.apply(self.data.T)

    def analyze(self, n_spec):
        self.rhos = np.empty(self.pred[:n_spec, :].shape[0])
        for idx, leaves in enumerate(self.pred[:n_spec, :]):
            print(f'Computing the outlier score for the spectrum N° {idx}', end='\r')
            self.rhos[idx] = np.count_nonzero(self.pred[:n_spec, :] == leaves)

        self.rhos = 1 - self.rhos * (1 / self.pred[:n_spec, :].size)

ti = time()

## Random forest implementation

if not os.path.exists('RF.pkl'):
    rf = Analyzer('spectra_bin_9.npy')
    rf.train()
## Saving the model to disk
    with open('RF.pkl', 'wb') as file:
        pickle.dump(rf, file)
else:
## Loading the model
    with open('RF.pkl', 'rb') as file:
        rf = pickle.load(file)

rf.apply()
rf.analyze(n_spec=216_871)

np.save('rhos.npy', rf.rhos)
tf = time()
print(f'Runing time= {tf-ti:.2f} [s]')
