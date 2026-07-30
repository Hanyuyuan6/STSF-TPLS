"""Hadamard pattern generation utilities"""

import numpy as np
import functools
import logging


@functools.lru_cache(maxsize=32)
def get_hadamard_matrix_cached(N, M, perm_seed=None):
    """
    Generate the Hadamard measurement matrix (cached; perm_seed is part of the cache key)

    Args:
        N: total number of image pixels
        M: number of measurements (bucket_size)
        perm_seed: for the acquisition-order ablation -- when not None, applies a fixed seeded permutation to the M rows

    Returns:
        (M, N) numpy array
    """
    return get_hadamard_matrix(N, M, perm_seed)


def get_hadamard_matrix(N, M, perm_seed=None):
    """
    Generate the Hadamard measurement matrix

    Args:
        N: total number of image pixels
        M: number of measurements (bucket_size)
        perm_seed: when not None, uses an independent RandomState(perm_seed) to apply one
            fixed row permutation to the first M rows (acquisition-order ablation). This function is
            the single source of truth for Φ -- CPU/GPU, training/evaluation all go through here,
            so the permutation stays consistent along the whole chain automatically.

    Returns:
        (M, N) numpy array, the first M Hadamard bases (optionally reordered by perm_seed)
    """
    if not isinstance(N, (int, np.integer)) or N <= 0 or N & (N - 1):
        raise ValueError(
            f"N must be a positive power of two so the returned rows remain orthogonal; got {N!r}"
        )
    if not isinstance(M, (int, np.integer)) or not 1 <= M <= N:
        raise ValueError(f"M must be an integer satisfying 1 <= M <= N; got M={M!r}, N={N!r}")
    n = int(N)

    # Sylvester natural-order entry H[r,c] = (-1)^popcount(r & c). Computing
    # only the requested rectangle avoids materialising the full n×n matrix
    # (n=16384 would otherwise require about 1 GiB in float32).
    if n <= 2 ** 8:
        uint, shifts = np.uint8, (4, 2, 1)
    elif n <= 2 ** 16:
        uint, shifts = np.uint16, (8, 4, 2, 1)
    elif n <= 2 ** 32:
        uint, shifts = np.uint32, (16, 8, 4, 2, 1)
    else:
        uint, shifts = np.uint64, (32, 16, 8, 4, 2, 1)
    parity = np.bitwise_and(
        np.arange(M, dtype=uint)[:, None], np.arange(N, dtype=uint)[None, :])
    for shift in shifts:
        parity ^= parity >> shift
    H_sub = 1.0 - 2.0 * (parity & 1).astype(np.float32)

    # acquisition-order ablation: fixed seeded row permutation (independent RNG, leaves the global random state untouched)
    if perm_seed is not None:
        perm = np.random.RandomState(int(perm_seed)).permutation(M)
        H_sub = H_sub[perm]
        logging.info(f"Hadamard row permutation enabled: perm_seed={perm_seed} (M={M})")

    return H_sub.astype(np.float32)
