#!/usr/bin/env python3

import argparse
import os
import re
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import stats
from sklearn.decomposition import FastICA, PCA


parser = argparse.ArgumentParser(
    description=('Run temporal ICA on data.'),
    add_help=False,
)
arguments = parser.add_argument_group('Arguments')
arguments.add_argument(
    '-in',
    '--func',
    dest='fname',
    type=str,
    help=(
        'Complete path (absolute or relative) and name '
        'of the nifti file containing fMRI signal. Required.'
    ),
    required=True,
)
arguments.add_argument(
    '-m',
    '--mask',
    dest='mname',
    type=str,
    help=(
        'Complete path (absolute or relative) and name '
        'of the mask to limit ica to an area of the image. Optional.'
    ),
    default=None,
)
arguments.add_argument(
    '-merged',
    '--load_merged',
    dest='load_merged',
    type=bool,
    action='store_true',
    help=(
        'Check if there is a previous merged timeseries and if so load it.'
    ),
    default=False,
)
arguments.add_argument(
    '-iced',
    '--load_ica',
    dest='load_ica',
    type=bool,
    action='store_true',
    help=(
        'Given the number of components, check if there is a previous ICA run and if '
        'so load it.'
    ),
    default=False,
)
arguments.add_argument(
    '-n',
    '--ncomp',
    dest='ncomp',
    type=int,
    help=(
        'Number of components to extract. Optional. Default 15.'
    ),
    default=15,
)
arguments.add_argument(
    '-h', '--help', action='help', help='Show this help message and exit'
)

args = parser.parse_args()

fname = args.fname
mname = args.mname
ncomp = args.ncomp

indir = Path(os.path.dirname(fname))
filename = os.path.splitext(os.path.splitext(fname)[0])[0]

outdir = Path(os.path.join(indir, '..', '..', '..', 'ica'))
outdir.mkdir(parents=True, exist_ok=True)

fileprefix = fname.split('_run-')[0]
filesuffix = fname.split('_run-')[1][2:]

if args.load_merged and os.path.isfile(os.path.join(outdir, f'{fileprefix}_merged.nii.gz')):
    data = nib.load(fname).get_fdata()
    # Load mask if any or get non-zeroes from data and apply it
    mask = nib.load(mname).get_fdata() if mname is not None else np.squeeze(np.any(data, axis=-1))
else:
    data = {}
    run_regex = re.compile(r'_run-([^_]+)')
    runs_set = set()

    # Check folder
    for file_path in indir.iterdir():
        if file_path.is_file() and fileprefix in file_path.name:
            match = run_regex.search(file_path.name)
            if match:
                runs_set.add(match.group(1))

    runs = sorted(list(runs_set))

    # Load data
    if len(runs) == 0 or (len(runs) == 1 and not runs[0]):
        data[1] = nib.load(fname).get_fdata()
    else:
        for r in runs:
            runname = f'{fileprefix}_run-{r}_{filesuffix}'
            data[r] = nib.load(runname).get_fdata()

    # Load mask if any or get non-zeroes from data and apply it
    mask = nib.load(mname).get_fdata() if mname is not None else np.squeeze(np.any(data[1], axis=-1))
    for r in runs:
        data[r] = data[r][mask != 0]

    # Cat
    data = np.concatenate(list(data.values()), axis=-1)

# Squeeze & Transpose for tICA
tdata = np.squeeze(data).T

if args.load_ica and os.path.isfile(os.path.join(outdir, f'ICA_{ncomp}_voxs.nii.gz')):
    voxss = nib.load(os.path.join(outdir, f'ICA_{ncomp}_voxs.nii.gz')).get_fdata()[mask != 0]
    tss = np.genfromtxt(os.path.join(outdir, f'ICA_{ncomp}_tss'))
else:
    # PCA & ICA
    ppca = PCA(n_components=ncomp, svd_solver='full', copy=False)
    ppca.fit(tdata)

    voxs = ppca.components_.T
    s = ppca.explained_variance_
    ts = np.dot(np.dot(tdata, voxs), np.diag(1. / s))

    w_ts = (ts * s[None, :])
    pcadata = np.dot(w_ts, voxs.T)

    pcadata = stats.zscore(pcadata, axis=1)  # variance normalize voxels
    pcadata = stats.zscore(pcadata, axis=None)  # variance normalize everything

    ica = FastICA(n_components=ncomp, algorithm='parallel',
                  fun='logcosh', max_iter=500, random_state=42)

    tss = ica.fit_transform(pcadata)
    voxss = ica.mixing_

# Compute frequencies
f = nib.load(fname)
tr = f.header['pixdim'][4]
time = np.arange(tss.shape[0]) * tr

frequencies = np.fft.fftfreq(tss.shape[0], d=tr)[:tss.shape[0] // 2]

freqs = np.zeros_like(tss)

for i in range(15):
    fft_values = np.fft.fft(tss[:, i])
    freqs[:, i] = np.abs(fft_values[:tss.shape[0] // 2]) * 2 / tss.shape[0]

# Export
outdir = Path(os.path.join(indir, '..', '..', '..', 'ica'))
outdir.mkdir(parents=True, exist_ok=True)

np.savetxt(os.path.join(outdir, f'ICA_{ncomp}_tss'), tss, fmt="%.6f")
np.savetxt(os.path.join(outdir, f'ICA_{ncomp}_freqs'), freqs, fmt="%.6f")

tmp_header = deepcopy(f.header)
tmp_header['dim'][0] = 3
tmp_header['dim'][3] = ncomp
tmp_header['dim'][4] = 1

out = np.zeros(mask.shape + (ncomp,))
out[mask != 0] = voxss

out_img = nib.Nifti1Image(out, f.affine, tmp_header)
out_img.to_filename(os.path.join(outdir, f'ICA_{ncomp}_voxs.nii.gz'))

tmp_header = deepcopy(f.header)
tmp_header['dim'][4] = ncomp
out_img = nib.Nifti1Image(out[..., np.newaxis, :], f.affine, tmp_header)
out_img.to_filename(os.path.join(outdir, f'ICA_{ncomp}_voxs_4D.nii.gz'))

tmp_header['dim'][4] = data.shape[-1]
out_img = nib.Nifti1Image(out[..., np.newaxis, :], f.affine, tmp_header)
out_img.to_filename(os.path.join(outdir, f'{fileprefix}_merged.nii.gz'))


# Plot
width = 25
height = tss.shape[1] * (width / 16) * .9
fig, axes = plt.subplots(nrows=tss.shape[1], ncols=1, figsize=(width, height), sharex=True)

if ncomp == 1:
    axes = [axes]
# Plot each column in a separate row
for i in range(ncomp):
    axes[i].plot(time, tss[:, i])
    axes[i].set_title(f'Component {i+1}')

# Adjust layout and show the plot
plt.tight_layout()

plt.savefig(os.path.join(outdir, f'ICA_{ncomp}_tss.png'))
plt.close()

fig, axes = plt.subplots(nrows=tss.shape[1], ncols=1, figsize=(width, height), sharex=True)

if ncomp == 1:
    axes = [axes]
# Plot each column in a separate row
for i in range(ncomp):
    axes[i].plot(frequencies, freqs[:, i], label=f'TS {i+1}')
    axes[i].set_title(f'Component {i+1}')

# Adjust layout and show the plot
plt.tight_layout()

plt.savefig(os.path.join(outdir, f'ICA_{ncomp}_freqs.png'))
plt.close()
