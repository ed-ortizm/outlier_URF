import sys

import numpy as np
import astropy.io.fits as pyfits
import matplotlib
import matplotlib.pyplot as plt

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

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def spec_name(plate_ifu):
    cube = f'/home/edgar/zorro/MaNGAdata/all_data/manga-{plate_ifu}-LOGCUBE-HYB10-MILESHC-MASTARHC.fits'
    maps = f'/home/edgar/zorro/MaNGAdata/all_data/manga-{plate_ifu}-MAPS-HYB10-MILESHC-MASTARHC.fits'
    return cube, maps

def sp_plot(plate_ifu, ID, prospec):
    cube, maps = spec_name(plate_ifu)

    with pyfits.open(cube) as hdul:
        wl = hdul['WAVE'].data
        bin_id = hdul['BINID'].data
        ids = np.where(bin_id[0, :, :]==ID)
        flx = hdul['FLUX'].data[:, ids[0], ids[1]]

    ## Plotting spectra
    plt.figure(figsize=(24,18))
    plt.plot(wl, flx)
    plt.plot(prospec)
    plt.savefig(f'/home/edgar/zorro/MaNGAdata/outlier_URF/plateIFU_{plate_ifu}_ID_{ID}.png')
    plt.close()

    return ids
