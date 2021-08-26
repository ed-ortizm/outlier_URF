#! /usr/bin/env python3
################################################################################
from configparser import ConfigParser, ExtendedInterpolation
from copy import copy
import os
from time import time
################################################################################
import numpy as np
import pickle
################################################################################
from src.unsupervisedRF.worker import initializer, worker
from src.unsupervisedRF import RF
################################################################################
ti = time()
################################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read('RF.ini')
############################################################################
# directories
model_directory = parser.get('directories', 'models')
############################################################################
# Parameters
number_estimators = parser.getint('parameters', 'estimators')
number_leafs = parser.getint('parameters', 'leafs')
number_islands = parser.getint('parameters', 'islands')
############################################################################
############################################################################
################################################################################
#Random forest implementation

# Creating several clfs
RF = RF.Analyzer('/home/edgar/zorro/MaNGAdata/spectra_bin_9.npy')
############################################################################
# Training a forests in different islands
for island in range(number_islands):
    print(f'Training island forest {island:003}', end='\r')
    rf.train(n_estimators=number_estimators)
# merging the islands
rf.clf_merging()
############################################################################
#Saving the model to disk
model_name = f'RF_{number_islands}'
if not os.path.exists(f'{model_directory}/{model_name}.pkl'):
   with open(f'{model_directory}/{model_name}.pkl', 'wb') as file:
       pickle.dump(rf, file)
#else:
## Loading the model
#    with open('RF_{n_iles}.pkl', 'rb') as file:
#        rf = pickle.load(file)

# rf.apply()
# rf.parallel_analyze() #216_871) Change for taking everython autoatically
#
# np.save('rhos.npy', rf.rhos)
tf = time()
print(f'Runing time= {tf-ti:.2f} [s]')
