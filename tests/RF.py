#! /usr/bin/env python3
################################################################################
from copy import copy
import ctypes
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
import os
from time import time
################################################################################
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier as RF
################################################################################
from worker import initializer, worker
################################################################################

ti = time()

## Random forest implementation
# Creating several clfs

rf = Analyzer('/home/edgar/zorro/MaNGAdata/spectra_bin_9.npy')

n_iles = 209

for step in range(n_iles):
    print(f'Training island forest {step:003}', end='\r')
    rf.train(n_estimators=48)

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
rf.parallel_analyze() #216_871) Change for taking everython autoatically

np.save('rhos.npy', rf.rhos)
tf = time()
print(f'Runing time= {tf-ti:.2f} [s]')
