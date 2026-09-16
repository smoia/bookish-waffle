#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from fastdtw import fastdtw
from joblib import Parallel, delayed
from scipy.spatial.distance import euclidean
from scipy.stats import zscore
from tslearn.barycenters import dtw_barycenter_averaging


def fit_linear_scale(source_signal, target_signal):
    """Fits linear amplitude scaling (gain & offset) from source to target via OLS."""
    A = np.vstack([source_signal, np.ones(len(source_signal))]).T
    gain, offset = np.linalg.lstsq(A, target_signal, rcond=None)[0]
    return gain, offset


def weighted_dba(X_neighbors, weights, init_center, max_iter=6):
    """
    Computes weighted DTW Barycenter Averaging for a set of time series.

    Parameters
    ----------
    X_neighbors : ndarray of shape (K, T)
        Normalized time series of the K spatial neighbors.
    weights : ndarray of shape (K,)
        Normalized Gaussian spatial weights for each neighbor (sum to 1.0).
    init_center : ndarray of shape (T,)
        Initial seed center (e.g., normalized probe time series).
    max_iter : int
        Number of DBA convergence iterations.
    """
    K, T = X_neighbors.shape
    center = init_center.copy()

    for _ in range(max_iter):
        # Accumulators for weighted barycenter update across time steps
        updated_center = np.zeros(T)
        weight_acc = np.zeros(T) + 1e-8  # Avoid division by zero

        for k in range(K):
            # Compute DTW warping path between current center and neighbor k
            _, path = fastdtw(center, X_neighbors[k], dist=euclidean)

            # Accumulate weighted values along the warping path
            for t_center, t_neighbor in path:
                updated_center[t_center] += weights[k] * X_neighbors[k, t_neighbor]
                weight_acc[t_center] += weights[k]

        # Update center estimate
        center = updated_center / weight_acc

    return center


def process_single_voxel(
    i, matrix_XT, coords_3d, probe_init, radius_mm, max_iter, gaussian, keep_residuals
):
    """Worker function to execute Spatial DBA for voxel index i."""
    # Spatial distance lookup in millimeter coordinates
    dist = np.linalg.norm(coords_3d - coords_3d[i], axis=1)
    mask_in_radius = dist <= radius_mm
    neighbor_indices = np.where(mask_in_radius)[0]
    neighbor_dists = dist[mask_in_radius]

    # Extract & normalize local neighborhood time series
    neighbor_series = matrix_XT[neighbor_indices]
    norm_neighbors = np.array([zscore(s) for s in neighbor_series])

    if gaussian:
        # Compute Gaussian spatial weights w_ij
        raw_weights = np.exp(-(neighbor_dists**2) / (2 * ((radius_mm / 3) ** 2)))
        spatial_weights = raw_weights / np.sum(raw_weights)
        # Perform Weighted DBA initialized with the probe
        dba_consensus = weighted_dba(
            norm_neighbors,
            weights=spatial_weights,
            init_center=probe_init,
            max_iter=max_iter,
        )
    else:
        # Compute DTW Barycenter Averaging initialized with the probe
        X_neighbors = norm_neighbors[:, :, np.newaxis]
        dba_consensus = dtw_barycenter_averaging(
            X_neighbors,
            init_barycenter=probe_init,
            max_iter=max_iter,
            metric='euclidean',
            verbose=False,
        ).flatten()

    # 3. Fit amplitude back to target voxel's dynamic range
    gain, offset = fit_linear_scale(dba_consensus, matrix_XT[i])

    dba_signal = dba_consensus * gain

    dba_signal = dba_signal + offset if keep_residuals else dba_signal
    return dba_signal


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
    '-ppg',
    '--ppg',
    dest='pname',
    type=str,
    help=(
        'Complete path (absolute or relative) and name '
        'of the file containing PPG signal. Required.'
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
    '-gaussian',
    '--gaussian',
    dest='gaussian',
    action='store_true',
    help=(
        'Apply a gaussian weight of sigma=radius/3 in the ROI for DBA filtering. '
        'Default not to.'
    ),
    default=False,
)
arguments.add_argument(
    '-residuals',
    '--keep_residuals',
    dest='keep_residuals',
    action='store_true',
    help=('Keep residuals in returning DBA. Default not to.'),
    default=False,
)
arguments.add_argument(
    '-r',
    '--radius_mm',
    dest='radius_mm',
    type=float,
    help=('Radius of ROI for DBA filtering. Optional. Default 8.'),
    default=8,
)
arguments.add_argument(
    '-maxiter',
    '--maxiter',
    dest='maxiter',
    type=int,
    help=('Maximum iterations for DBA convergence. Optional. Default 6.'),
    default=6,
)
arguments.add_argument(
    '-j',
    '--njobs',
    dest='njobs',
    type=int,
    help=('Number of jobs to use. Optional. Default uses 1/3 of available cores.'),
    default=None,
)
arguments.add_argument(
    '-h', '--help', action='help', help='Show this help message and exit'
)

args = parser.parse_args()

# Import
indir = Path(os.path.dirname(args.fname))

img = nib.load(args.fname)
tdata = img.get_fdata()
probe = np.loadtxt(args.pname).astype(np.float32)
probe_init = zscore(probe).reshape(-1, 1)

# Load mask if any or get non-zeroes from data and apply it
mask = (
    nib.load(args.mname).get_fdata()
    if args.mname is not None
    else np.squeeze(np.any(tdata, axis=-1))
)
grid_indices = np.argwhere(mask)
matrix_XT = tdata[mask != 0]
N, T = matrix_XT.shape

affine = img.affine

print(f'Extracted {N} active voxels across {T} temporal volumes.')

# Transform Voxel Grid Coordinates to Real-World mm (N x 3)
coords_homo = np.hstack([grid_indices, np.ones((N, 1))])
coords_3d = (affine @ coords_homo.T)[:3, :].T

total_cores = os.cpu_count() or 1
if args.n_jobs is None or args.n_jobs <= 0:
    n_workers = max(1, int(np.floor(0.30 * total_cores)))
else:
    n_workers = min(args.n_jobs, total_cores)


print(f'Executing Spatial DBA (Radius = {args.radius_mm}mm, n_jobs={n_workers})...')

results = Parallel(n_jobs=n_workers, batch_size=64)(
    delayed(process_single_voxel)(
        i,
        matrix_XT,
        coords_3d,
        probe_init,
        args.radius_mm,
        args.max_iter,
        args.gaussian,
        args.keep_residuals,
    )
    for i in range(N)
)

filtered_XT = np.array(results, dtype=np.float32)

# 4. Reconstruct and Save 4D NIfTI File
print('Reconstructing 4D volume...')
filtered_4d = np.zeros_like(tdata)
filtered_4d[mask] = filtered_XT

# Export
outdir = Path(os.path.join(indir, '..', '..', '..', 'dbafilter'))
outdir.mkdir(parents=True, exist_ok=True)

out_name = os.path.splitext(os.path.splitext(os.path.basename(args.fname))[0])[0]

out_img = nib.Nifti1Image(filtered_4d, img.affine, img.header)
out_img.to_filename(os.path.join(outdir, f'{out_name}_DBA.nii.gz'))
