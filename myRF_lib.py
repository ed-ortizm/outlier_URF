import numpy as np

def get_metadata(fnames):
    ids = []
    snrs = []
    plate_ifus = []
    for fname in fnames:
        bname = fname[:-4]
        ids.append(np.load(f'{bname}-id.npy'))
        snrs.append(np.load(f'{bname}-SNR.npy'))
        plate_ifus.append([bname.split('/')[-1]]*ids[-1].size)

    return np.hstack(ids), np.hstack(snrs), np.hstack(plate_ifus)
