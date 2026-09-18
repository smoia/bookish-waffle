#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# =====================================================================
# 1. Soft-DTW Loss Function (PyTorch differentiable DTW)
# =====================================================================
class SoftDTWLoss(nn.Module):
    """Soft-DTW loss function for smooth, differentiable dynamic time warping.
    Ref: Cuturi & Blondel (ICML 2017)
    """

    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma

    def forward(self, D):
        """D is the pairwise distance matrix between reconstructed series and Probe (B, T, M)"""
        B, N, M = D.shape
        gamma = self.gamma

        # Soft-min recurrence matrix
        R = torch.zeros((B, N + 1, M + 1), device=D.device) + 1e8
        R[:, 0, 0] = 0

        for i in range(1, N + 1):
            for j in range(1, M + 1):
                cost = D[:, i - 1, j - 1]
                # Soft-min operator: -gamma * logsumexp(-[r1, r2, r3] / gamma)
                r1 = R[:, i - 1, j]
                r2 = R[:, i, j - 1]
                r3 = R[:, i - 1, j - 1]
                stacked = torch.stack([r1, r2, r3], dim=-1)
                softmin = -gamma * torch.logsumexp(-stacked / gamma, dim=-1)
                R[:, i, j] = cost + softmin

        return R[:, N, M].mean()


# =====================================================================
# 2. 3D-Temporal Autoencoder Architecture
# =====================================================================
class ProbeVariationAutoencoder(nn.Module):
    """Network that filters 4D spacetime sub-blocks or 1D voxels into ppg variations."""

    def __init__(self, temporal_dim):
        super().__init__()
        # Encoder: compress temporal features
        self.encoder = nn.Sequential(
            nn.Linear(temporal_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        # Decoder: reconstruct time series constrained to ppg dynamics
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, temporal_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


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
    '-e',
    '--epochs',
    dest='epochs',
    type=int,
    help=('Number of training epochs for the autoencoder. Default 15.'),
    default=15,
)
arguments.add_argument(
    '-bs',
    '--batch_size',
    dest='batch_size',
    type=int,
    help=('Batch size for DataLoader training and evaluation. Default 512.'),
    default=512,
)
arguments.add_argument(
    '-lr',
    '--learning_rate',
    dest='lr',
    type=float,
    help=('Learning rate for the Adam optimizer. Default 1e-3.'),
    default=1e-3,
)
arguments.add_argument(
    '-g',
    '--gamma',
    dest='gamma',
    type=float,
    help=('Smoothing parameter for the Soft-DTW loss function. Default 0.1.'),
    default=0.1,
)
# arguments.add_argument(
#     '-j',
#     '--njobs',
#     dest='njobs',
#     type=int,
#     help=('Number of jobs to use. Optional. Default uses 1/3 of available cores.'),
#     default=None,
# )
arguments.add_argument(
    '-d',
    '--device',
    dest='device',
    choices=('cuda', 'cpu'),
    type=str,
    help=(
        "Computation device ('cuda' or 'cpu'), by default 'cuda' if GPU is available"
        "else 'cpu'"
    ),
    default=None,
)
arguments.add_argument(
    '-h', '--help', action='help', help='Show this help message and exit'
)

args = parser.parse_args()

if args.device is None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

indir = Path(os.path.dirname(args.fname))


print(f'Using compute device: {device}')

# Load 4D NIfTI & Probe
print('Loading NIfTI volume and ppg text file...')
img = nib.load(args.fname)
tdata = img.get_fdata().astype(np.float32)

ppg = np.loadtxt(args.pname).astype(np.float32)

# Mask background voxels
mask = (
    nib.load(args.mname).get_fdata()
    if args.mname is not None
    else np.squeeze(np.any(tdata, axis=-1))
)
matrix_XT = tdata[mask != 0]
N, T = matrix_XT.shape

print(f'Dataset extracted: {N} active voxels across {T} time points.')

# Normalize inputs to zero-mean, unit-variance for neural network training
means = np.mean(matrix_XT, axis=1, keepdims=True)
stds = np.std(matrix_XT, axis=1, keepdims=True) + 1e-8
norm_matrix_XT = (matrix_XT - means) / stds

norm_ppg = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)

# PyTorch Data Setup
tensor_data = torch.from_numpy(norm_matrix_XT).float()
tensor_ppg = torch.from_numpy(norm_ppg).float().to(device)

dataset = TensorDataset(tensor_data)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Model & Loss Initialization
model = ProbeVariationAutoencoder(temporal_dim=T).to(device)
soft_dtw_loss_fn = SoftDTWLoss(gamma=args.gamma)
mse_loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Prepare batch ppg tensor shape (1, M, 1)
ppg_expanded = tensor_ppg.unsqueeze(0).unsqueeze(-1)

print('Training Soft-DTW Autoencoder...')
model.train()
for epoch in range(args.epochs):
    total_loss = 0.0
    for (batch_x,) in loader:
        batch_x = batch_x.to(device)
        optimizer.zero_grad()

        # Forward pass: Reconstruct normalized time series
        reconstructed = model(batch_x)

        # Compute pairwise squared Euclidean distance matrix D between Reconstruct (B, T) and Probe (1, M)
        # Reshape for broadcasting: Reconstructed -> (B, T, 1), Probe -> (1, 1, M)
        rec_exp = reconstructed.unsqueeze(-1)  # (B, T, 1)
        prb_exp = tensor_ppg.unsqueeze(0).unsqueeze(0)  # (1, 1, M)

        # Distance matrix D: shape (B, T, M)
        D = (rec_exp - prb_exp) ** 2

        # Combined Loss: Soft-DTW aligns ppg shape; MSE keeps autoencoder grounded to input phase
        dtw_loss = soft_dtw_loss_fn(D)
        recon_loss = mse_loss_fn(reconstructed, batch_x)
        loss = dtw_loss + 0.5 * recon_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_x)

    avg_loss = total_loss / N
    print(f'Epoch [{epoch + 1}/{args.epochs}] - Loss: {avg_loss:.4f}')

# Inference & Reconstruction
print('Extracting filtered ppg variations across volume...')
model.eval()
filtered_norm_XT = np.zeros_like(matrix_XT)

eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
idx = 0
with torch.no_grad():
    for (batch_x,) in eval_loader:
        batch_x = batch_x.to(device)
        out = model(batch_x).cpu().numpy()
        filtered_norm_XT[idx : idx + len(out)] = out
        idx += len(out)

# Denormalize to restore original amplitude scale and temporal offset
filtered_XT = (filtered_norm_XT * stds) + means

# Reconstruct and Save 4D NIfTI File
print('Reconstructing 4D volume...')
filtered_4d = np.zeros_like(tdata)
filtered_4d[mask] = filtered_XT

# Export
outdir = Path(os.path.join(indir, '..', '..', '..', 'sdtwautoencoder'))
outdir.mkdir(parents=True, exist_ok=True)

out_name = os.path.splitext(os.path.splitext(os.path.basename(args.fname))[0])[0]

out_img = nib.Nifti1Image(filtered_4d, img.affine, img.header)
out_img.to_filename(os.path.join(outdir, f'{out_name}_DBA.nii.gz'))
