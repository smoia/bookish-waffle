#!/usr/bin/env python3

import argparse
import os

import nibabel as nib
import numpy as np

parser = argparse.ArgumentParser(
    description=(
        'Compute DVARS of a nifti file.\n'
        'If motion parameters exist, also compute FD'
    ),
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
    dest='mask',
    type=str,
    help=(
        'Complete path (absolute or relative) and name '
        'of the mask to limit DVARS to an area of the image. Optional.'
    ),
    default=None,
)
arguments.add_argument(
    '-h', '--help', action='help', help='Show this help message and exit'
)

args = parser.parse_args()

fname = args.fname
mask = args.mask

indir = os.path.dirname(fname)
filename = os.path.splitext(os.path.splitext(fname)[0])[0]
origname = filename.replace('_mcf', '')

# Compute DVARS
movnii = nib.load(f'{filename}.nii.gz').get_fdata()

try:
    dvars = movnii[nib.load(mask).get_fdata() != 0] if mask is not None else movnii
except IndexError:
    raise IndexError(f'Cannot mask data with shape {movnii.shape} using mask '
                     f'with shape {mask.shape}')

dvars = np.sqrt(np.square(np.diff(dvars.mean(0))))
np.savetxt(f'{filename}_dvars.par', dvars, fmt="%.6f")

if os.path.exists(f'{filename}.par'):
    # Compute FD
    movpar = np.genfromtxt(f'{filename}.par')
    FD = np.abs(movpar[:-1] - movpar[1:]).sum(1)
    np.savetxt(f'{origname}_fd.par', FD, fmt="%.6f")
