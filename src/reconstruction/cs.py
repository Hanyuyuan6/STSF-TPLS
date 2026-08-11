"""Compressed-sensing reconstruction algorithms (CS ghost imaging, A is (M,N), with ADMM-L1 / FISTA-L1, DCT transform added, numerically safeguarded)"""

from __future__ import annotations
import numpy as np
import torch
from typing import Optional


def _soft_threshold_t(x: torch.Tensor, lam: float) -> torch.Tensor:
    """Soft-thresholding operator (torch)"""
    return torch.sign(x) * torch.clamp(torch.abs(x) - lam, min=0.0)


def _ensure_A_MN(patterns: np.ndarray, img_size: int) -> np.ndarray:
    """
    Make sure the measurement matrix has shape (M, N). One passed in as (N, M) is transposed automatically.
    N must equal img_size^2.
    """
    H2 = img_size * img_size
    if patterns.ndim != 2:
        raise ValueError(f"patterns should be 2-D, got {patterns.ndim} dimensions")
    r, c = patterns.shape
    # the target is (M, N)
    if c == H2:      # (M, N) already correct
        return patterns
    if r == H2:      # (N, M) -> transpose
        return patterns.T
    raise AssertionError(f"patterns has shape {patterns.shape}, which cannot match N=img_size^2={H2} as its second dimension")


def dct_2d_matrix(n: int) -> np.ndarray:
    """
    Build the 2-D DCT transform matrix Psi, of size (n*n, n*n).
    It takes an image vector into the DCT domain; Psi is orthogonal.
    """
    import scipy.fftpack

    # build the 1-D DCT matrix
    def dct_matrix_1d(size):
        return scipy.fftpack.dct(np.eye(size), norm='ortho')

    D = dct_matrix_1d(n)  # (n, n)
    # the 2-D DCT matrix is the Kronecker product
    Psi = np.kron(D, D)   # (n*n, n*n)
    return Psi


@torch.no_grad()
def fista_l1_recon(
    patterns: np.ndarray,           # the raw measurement matrix (M, N)
    buckets_raw: np.ndarray,        # (B, M)
    img_size: int,
    l1_weight: float = 0.01,
    steps: int = 100,
    step_scale: float = 1.0,     # tstep=step_scale/L;1.0=the standard 1/L step (L=N exactly). Once wrongly set to 0.01 -> ~100x understepping, returning an approximate back-projection
    device: Optional[torch.device | str] = None,
) -> np.ndarray:
    """
    FISTA-L1 sparse reconstruction, on top of the DCT:
        min_alpha 0.5||A Psi alpha - y||_2^2 + λ||alpha||_1
    where
        alpha: DCT coefficients (B, N)
        reconstructed image x = Psi alpha
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    patterns = _ensure_A_MN(patterns, img_size)   # (M, N)
    M, N = patterns.shape
    B = buckets_raw.shape[0]
    H = W = img_size

    # build the DCT transform matrix Psi and its transpose
    Psi = dct_2d_matrix(img_size)                  # (N, N)
    Psi_torch = torch.from_numpy(Psi).float().to(device)
    Psi_T = Psi_torch.t()                           # (N, N)

    # recast the measurement matrix as A Psi
    A_Psi = torch.from_numpy(patterns).float().to(device) @ Psi_torch  # (M, N)
    AT_Psi = A_Psi.t()                                                    # (N, M)

    y = torch.from_numpy(buckets_raw).float().to(device)                # (B, M)

    def A_Psi_x(alpha: torch.Tensor) -> torch.Tensor:
        # alpha: (B, N) -> (B, M)
        return alpha @ AT_Psi

    def AT_Psi_z(z: torch.Tensor) -> torch.Tensor:
        # z: (B, M) -> (B, N)
        return z @ A_Psi

    # Lipschitz constant L = ||A_Psi||_2^2. Psi (DCT) is orthogonal and Phi Phi^T = N*I (the first M rows of the natural-order Hadamard matrix),
    # so L = N holds exactly -- no estimate and no power iteration needed.
    # L was once hardcoded to 1.0 (a factor N=16384 too small), which put tstep at 82x the 2/L divergence bound: the iteration overflowed to nan,
    # and the nan_to_num below then silently squashed it to 0 -> at img_size=128 it returned a result bit-for-bit identical to an all-black image, and raised nothing at all.
    L = float(N)
    tstep = float(step_scale) / L
    lam = float(l1_weight)

    alpha = torch.zeros(B, N, device=device)  # initialize the sparse coefficients alpha
    z = alpha.clone()
    t = 1.0

    for _ in range(steps):
        Az = A_Psi_x(z)           # (B, M)
        grad = AT_Psi_z(Az - y)   # (B, N)
        alpha_next = _soft_threshold_t(z - tstep * grad, lam * tstep)
        t_next = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        z = alpha_next + ((t - 1.0) / t_next) * (alpha_next - alpha)
        alpha, t = alpha_next, t_next

    # transform back to the pixel domain, x = Psi alpha
    x = alpha @ Psi_T        # (B, N)
    img = x.view(B, H, W)
    img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img_min = img.amin(dim=[1, 2], keepdim=True)
    img_max = img.amax(dim=[1, 2], keepdim=True)
    denom = torch.clamp(img_max - img_min, min=1e-8)
    img = torch.clamp((img - img_min) / denom, 0.0, 1.0)

    return img.unsqueeze(1).detach().cpu().numpy()


@torch.no_grad()
def admm_l1_recon(
    patterns: np.ndarray,
    buckets_raw: np.ndarray,
    img_size: int,
    l1_weight: float = 0.01,
    rho: float = 1.0,
    steps: int = 100,
    device: Optional[torch.device | str] = None,
) -> np.ndarray:
    """
    ADMM-L1 sparse reconstruction, on top of the DCT:
        min_alpha 0.5||A Psi alpha - y||_2^2 + λ||alpha||_1
    where
        alpha: DCT coefficients (B, N)
        reconstructed image x = Psi alpha
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    patterns = _ensure_A_MN(patterns, img_size)
    M, N = patterns.shape
    B = buckets_raw.shape[0]
    H = W = img_size

    Psi = dct_2d_matrix(img_size)
    Psi_torch = torch.from_numpy(Psi).float().to(device)
    Psi_T = Psi_torch.t()

    A_Psi = torch.from_numpy(patterns).float().to(device) @ Psi_torch
    AT_Psi = A_Psi.t()
    y = torch.from_numpy(buckets_raw).float().to(device)

    I_M = torch.eye(M, device=device)
    inv_term = torch.linalg.inv(I_M + (A_Psi @ AT_Psi) / rho)

    lam = float(l1_weight)

    alpha = torch.zeros(B, N, device=device)
    z = torch.zeros(B, N, device=device)
    u = torch.zeros(B, N, device=device)

    for _ in range(steps):
        v = z - u
        vAT = v @ AT_Psi
        right = y - vAT
        mid = right @ inv_term
        alpha = v + (mid @ A_Psi) / rho

        z = _soft_threshold_t(alpha + u, lam / rho)
        u = u + alpha - z

    x = alpha @ Psi_T
    img = x.view(B, H, W)
    img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img_min = img.amin(dim=[1, 2], keepdim=True)
    img_max = img.amax(dim=[1, 2], keepdim=True)
    denom = torch.clamp(img_max - img_min, min=1e-8)
    img = torch.clamp((img - img_min) / denom, 0.0, 1.0)

    return img.unsqueeze(1).detach().cpu().numpy()