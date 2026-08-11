# -*- coding: utf-8 -*-
"""GPU-side bucket computation.

Moves the per-sample 512×16384 matrix product (single-pixel acquisition) that
`base_dataset.__getitem__` evaluates on the CPU into a batched GPU call -- in training
the random augmentation applied every epoch forces the bucket to be recomputed every
epoch (caching it would freeze the augmentation), and doing that large matrix product
on the CPU is why carvana (4064 train images) trains slowly with the GPU left starving.

The noise-free path is verbatim identical to the bucket part of base_dataset:
  bucket_raw  = patterns @ vec(img)            # (M,)  per-sample
  bucket_norm = (raw - raw.min())/(raw.max()-raw.min()+1e-8)
The only differences: batched, on the GPU, and evaluated in fp32 outside autocast
(matches the CPU fp32 result, allclose<1e-4).
Note: the noise branch's sigma uses torch .std (unbiased, ddof=1) whereas base_dataset
   uses numpy .std (ddof=0); the two differ by sqrt(n/(n-1))≈1.00098 (+0.0085 dB), so
   "verbatim identical" holds for the noise-free path only.
"""
import torch

from src.utils.ghost_patterns import get_hadamard_matrix_cached


def build_phi(img_size, bucket_size, device, dtype=torch.float32, perm_seed=None):
    """Φ = the first M=bucket_size rows × N=img_size² columns of Hadamard, (M, N). Registered on device for reuse (buffer).
    A non-None perm_seed selects the fixed row permutation of the acquisition-order ablation (see ghost_patterns.get_hadamard_matrix)."""
    phi = get_hadamard_matrix_cached(int(img_size) * int(img_size), int(bucket_size), perm_seed)  # numpy (M, N)
    return torch.from_numpy(phi).to(device=device, dtype=dtype)


def compute_bucket_gpu(image, phi, eps=1e-8, noise_snr_db=None, noise_ref='full'):
    """image (B,1,H,W) in [0,1] on GPU; phi (M, H*W) on the same device -> bucket (B, M), per-sample min-max normalized.

    noise_snr_db: when not None, adds measurement-domain Gaussian noise to the raw bucket signal before normalization (the additive measurement-noise term),
    calibrated per sample from the signal standard deviation: sigma = std(ref) * 10^(-SNR/20).
    noise_ref: 'full' = std over all M coefficients (historical behaviour; on natural images it is dominated by row 0 = DC, so the SNR semantics drift across datasets);
    'ac' = std over rows 1..M-1 only (DC excluded), so the SNR is referenced directly to the information-carrying components and is comparable across datasets (physically, detector noise is unrelated to the signal std)."""
    B = image.size(0)
    flat = image.reshape(B, -1).to(phi.dtype)        # (B, H*W)
    raw = flat @ phi.t()                             # (B, M)
    if noise_snr_db is not None:
        ref = raw[:, 1:] if noise_ref == 'ac' else raw
        sigma = ref.std(dim=1, keepdim=True) * (10.0 ** (-float(noise_snr_db) / 20.0))
        raw = raw + sigma * torch.randn_like(raw)
    mn = raw.amin(dim=1, keepdim=True)
    mx = raw.amax(dim=1, keepdim=True)
    return (raw - mn) / (mx - mn + eps)
