"""Conventional ghost-imaging reconstruction"""

import numpy as np
import torch


def trad_gi_recon(patterns, buckets_raw, img_size, device=None):
    """
    Conventional ghost-imaging reconstruction

    Args:
        patterns: (M, N) numpy array, the acquired Hadamard rows
        buckets_raw: (B, M) numpy array, the raw bucket signals (not normalized)
        img_size: int, the image side length
        device: torch device

    Returns:
        (B, 1, H, W) numpy array, the reconstructed images
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    B, M = buckets_raw.shape
    N = img_size * img_size
    if patterns.shape != (M, N):
        raise ValueError(
            f"patterns must match buckets as (M,N)=({M},{N}), got {patterns.shape}")

    # convert to torch tensors
    H_acquired = torch.from_numpy(patterns).float().to(device)   # (M, N)
    buckets = torch.from_numpy(buckets_raw).float().to(device)   # (B, M)

    # Adjoint of the acquired rows. This is exactly the old zero-pad @ H_full
    # computation without allocating the unused (N-M) rows.
    img_vec = torch.matmul(buckets, H_acquired)  # (B, N)

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
