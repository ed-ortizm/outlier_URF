#! /usr/bin/env python3

import sys
import os
import glob

import numpy as np
import astropy.io.fits as pyfits
import matplotlib
import matplotlib.pyplot as plt

from myRF_lib import get_metadata
from constants import data_proc, all_data_proc, n_wl


## Retrieve the name of the spectra

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
    plt.figure(figsize=(8,6), dpi=80)
    plt.plot(wl, flx)
    plt.plot(prospec)
    plt.savefig(f'spectra_{plate_ifu}.png')
    plt.close()


fnames = glob.glob(f'{all_data_proc}/*-*[0-9].npy')

print('Retrieving metadata')
ids, snrs, plate_ifus = get_metadata(fnames)

## Spaxels with high S/N (>10)

w10 = np.where(snrs > 10.)
ids = ids[w10]
snrs = snrs[w10]
plate_ifus = plate_ifus[w10]
srt_idx = np.argsort(snrs)
split = np.array_split(srt_idx, 10)
print(f'Size of split: {len(split)}. size of the bin: {split[9].size}')
idx = split[9][19985]



print('Loading outlier scores!')
rhos = np.load('rhos.npy', mmap_mode='r')
idx_prospec = np.argmax(rhos)

print('Extracting fluxes for the most outliying specterum')
prospec = np.load('spectra_bin_9.npy', mmap_mode = 'r')[:, idx_prospec]


print(f'the plate_ifu is {plate_ifus[idx]} & the id is {ids[idx]}')

plate_ifu = plate_ifus[idx]
ID = ids[idx]
## sp_plot(plate_ifu, ID, prospec)
