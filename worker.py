import numpy as np

def initializer(in_app_mat, in_counter):
   global app_mat
   global counter
   global bools

   bools = np.empty(in_app_mat[1], dtype=np.bool)
   app_mat = np.ctypeslib.as_array(in_app_mat[0]).reshape(in_app_mat[1])
   counter = in_counter

def worker(idx):
    rhos = np.count_nonzero(np.equal(app_mat, app_mat[idx, :], out=bools))
    with counter.get_lock():
        counter.value += 1
        cnt_val = counter.value
    print(f'Outlier score for {cnt_val} spectra computed', end='\r')
    return 1 - rhos / app_mat.size
