import ctypes
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
################################################################################
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF
################################################################################

################################################################################
class Analyzer:
    def __init__(self, fname):
        """

        INPUTS

        OUTPUTS

        """
        # self.data = np.load(fname, mmap_mode='r')
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
        self.pred_shared = RawArray(ctypes.c_uint16, self.pred.ravel())

    def parallel_analyze(self):
        counter = mp.Value('i', 0)
        with mp.Pool(initializer=initializer, initargs=((self.pred_shared, self.pred.shape), counter),
                     processes=48) as pool:
            print(self.pred.shape)
            self.rhos = np.array(pool.map(worker, range(self.pred.shape[0])))

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
