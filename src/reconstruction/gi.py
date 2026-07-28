"""Conventional ghost-imaging reconstruction"""

import numpy as np
import torch


def trad_gi_recon(patterns_full, buckets_raw, img_size, device=None):
    """
    Conventional ghost-imaging reconstruction

    Args:
        patterns_full: (N, N) numpy array, the full Hadamard matrix
        buckets_raw: (B, M) numpy array, the raw bucket signals (not normalized)
        img_size: int, the image side length
        device: torch device

    Returns:
        (B, 1, H, W) numpy array, the reconstructed images
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    N = patterns_full.shape[0]
    B, M = buckets_raw.shape

    # convert to torch tensors
    H_full = torch.from_numpy(patterns_full).float().to(device)  # (N, N)
    buckets = torch.from_numpy(buckets_raw).float().to(device)   # (B, M)

    # zero-pad up to N
    bucket_pad = torch.zeros(B, N, device=device, dtype=torch.float32)
    bucket_pad[:, :M] = buckets

    # inverse Hadamard transform
    img_vec = torch.matmul(bucket_pad, H_full)  # (B, N)

    # reshape into an image
    img = img_vec.view(B, img_size, img_size)  # (B, H, W)

    # remove the mean, then normalize
    img = img - img.mean(dim=[1,2], keepdim=True)
    img_min = img.amin(dim=[1,2], keepdim=True)
    img_max = img.amax(dim=[1,2], keepdim=True)
    img = (img - img_min) / (img_max - img_min + 1e-8)

    # add the channel dimension
    img = img.unsqueeze(1)  # (B, 1, H, W)

    return img.detach().cpu().numpy()