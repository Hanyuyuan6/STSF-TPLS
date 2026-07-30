# -*- coding: utf-8 -*-
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# In-project helpers/functions (make sure your project paths are set up correctly)
from src.utils.config_parser import load_config          # parse the YAML/JSON config
from src.utils.ghost_patterns import get_hadamard_matrix # build/fetch the Hadamard pattern matrix
from src.reconstruction import trad_gi_recon, admm_l1_recon  # implementations of the two reconstruction methods

# SciPy is only needed when a .mat file has to be read
try:
    import scipy.io as sio
except Exception:
    # scipy is missing from the environment, so sio stays None; a clear error is raised if the user reads a .mat
    sio = None


def to_uint8_gray(array_01: np.ndarray) -> np.ndarray:
    """
    Convert a float grayscale image in the [0,1] range to an 8-bit image (uint8).
    - first clip to [0,1]
    - then map linearly to [0,255] and round
    """
    arr = np.clip(array_01, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _usable_bucket_shape(value):
    """Whether the array can be normalized to [B, M] by ``load_buckets``."""
    shape = np.asarray(value).shape
    return len(shape) in (1, 2) or (len(shape) == 3 and 1 in shape)


def _select_keyed_array(data, requested_key, format_name):
    """Select one usable named array, failing closed on ambiguous archives."""
    visible = {key: value for key, value in data.items() if not key.startswith('__')}
    shapes = ', '.join(f"{key}:{np.asarray(value).shape}" for key, value in visible.items())
    if requested_key is not None:
        if requested_key not in visible:
            raise KeyError(
                f"Requested key {requested_key!r} not found in {format_name}; "
                f"available key/shape entries: {shapes or 'none'}")
        return visible[requested_key]

    usable = {key: value for key, value in visible.items() if _usable_bucket_shape(value)}
    if len(usable) == 1:
        return next(iter(usable.values()))
    usable_shapes = ', '.join(
        f"{key}:{np.asarray(value).shape}" for key, value in usable.items())
    if not usable:
        raise ValueError(
            f"No usable bucket array in {format_name}; key/shape entries: {shapes or 'none'}")
    raise ValueError(
        f"Ambiguous {format_name}: multiple usable bucket arrays "
        f"({usable_shapes}). Pass --mat_key explicitly.")


def load_buckets(path, mat_key=None) -> np.ndarray:
    """
    Load an experimental bucket-signal file. Several formats are accepted, and all of them end up as a 2-D [B, M] array:
    - B: number of samples (batch size), which is B=1 in most cases
    - M: number of measurements (measurement entries), e.g. M=N=H*W at full sampling

    Supported formats:
    - .npy: an array written by np.save
    - .npz: an archive written by np.savez; one usable array is auto-selected, otherwise pass mat_key
    - .txt/.csv: text/comma-separated, read with numpy.loadtxt
    - .mat: a MATLAB file; one usable variable is auto-selected, otherwise pass mat_key

    Returns:
    - arr: np.ndarray of shape [B, M], dtype float32
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Bucket file not found: {path}")

    suf = p.suffix.lower()
    if suf == ".npy":
        # read the numpy binary directly
        arr = np.load(p)
    elif suf == ".npz":
        with np.load(p) as data:
            arr = np.array(_select_keyed_array(data, mat_key, 'NPZ file'))
    elif suf in [".txt", ".csv"]:
        # text read; CSV is comma-separated, TXT lets numpy infer the delimiter
        arr = np.loadtxt(p, dtype=float, delimiter=',' if suf == '.csv' else None)
    elif suf == ".mat":
        # read the MATLAB file
        if sio is None:
            # scipy is not installed, so .mat files cannot be read
            raise ImportError("scipy is required to read .mat files: run pip install scipy first")
        data = sio.loadmat(p)
        arr = _select_keyed_array(data, mat_key, 'MAT file')
        # fold MATLAB's possible column/row-vector shapes into a numpy array and drop the length-1 dimensions
        arr = np.array(arr)
        if arr.ndim > 2:
            # for cases such as [M,1,1], squeeze first
            arr = np.squeeze(arr)
    else:
        raise ValueError(f"Unsupported bucket file extension: {suf}")

    # unify the dtype
    arr = np.array(arr, dtype=np.float32)

    # unify the shape to [B, M]
    if arr.ndim == 1:
        # a 1-D vector for a single sample -> add an axis to act as the batch
        arr = arr[None, :]  # [1, M]
    elif arr.ndim == 2:
        # two possibilities:
        # 1) already [B, M], nothing to do
        # 2) [M, 1] or [1, M], which can be squeezed and then re-checked
        if 1 in arr.shape:
            arr = np.squeeze(arr)
            if arr.ndim == 1:
                arr = arr[None, :]
            elif arr.ndim != 2:
                # the shape is malformed after squeezing
                raise ValueError(f"Cannot make sense of the bucket array shape: {arr.shape}")
        # otherwise it is taken to be [B, M] already
    elif arr.ndim == 3:
        # for cases such as [B, 1, M] or [B, M, 1], squeeze down to [B, M]
        if 1 in arr.shape:
            arr = np.squeeze(arr)
            if arr.ndim == 1:
                arr = arr[None, :]
            elif arr.ndim != 2:
                raise ValueError(f"Cannot make sense of the bucket array shape: {arr.shape}")
        else:
            # other 3-D shapes are not supported for now
            raise ValueError(f"Unsupported bucket array shape: {arr.shape}")
    else:
        # more than 3 or fewer than 1 dimensions is outside the supported range
        raise ValueError(f"Unsupported number of bucket array dimensions: {arr.ndim}")

    # no transpose is applied here: if your data is [M, B], transpose it yourself or save it as [B, M] in the first place

    return arr  # [B, M]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Prediction (experimental bucket signal -> reconstructed image), accepts .mat/.npy/.npz/.txt/.csv")
    parser.add_argument('--config', type=str, required=True, help='path to the config file')
    parser.add_argument('--method', type=str, required=True, choices=['tradgi', 'admm-l1'], help='reconstruction method')
    parser.add_argument('--bucket_file', type=str, required=True, help='path to the bucket-signal file')
    parser.add_argument('--mat_key', type=str, default=None,
                        help='bucket-array key for .mat/.npz; required when multiple usable arrays exist')
    parser.add_argument('--img_size', type=int, default=None, help='override the image size from the config (square)')
    parser.add_argument('--reg_weight', type=float, default=0.01, help='ADMM L1 regularization weight (admm-l1 only)')
    parser.add_argument('--steps', type=int, default=100, help='number of ADMM iterations (admm-l1 only)')
    parser.add_argument('--rho', type=float, default=1.0, help='ADMM penalty parameter (admm-l1 only)')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--save_dir', type=str, required=True, help='directory the reconstructions are written to')
    parser.add_argument('--save_npy', action='store_true', help='also save the raw float reconstruction as .npy')
    parser.add_argument('--clip_percentile', type=float, default=None, help='percentile clip before normalization, e.g. 1.0 clips to the [1%, 99%] range')
    parser.add_argument('--transpose', action='store_true', help='transpose before saving (swap width and height)')
    parser.add_argument('--rot90', type=int, default=0, help='rotate counter-clockwise by 90*k degrees before saving, k=0/1/2/3')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    cfg = load_config(args.config)

    img_size = args.img_size if args.img_size is not None else cfg['data']['img_size']
    N = img_size * img_size

    # read the bucket signal
    buckets_raw = load_buckets(args.bucket_file, mat_key=args.mat_key)  # [B, M]
    B, M = buckets_raw.shape
    logging.info(f"Bucket signal read: B={B}, M={M}, target pixels N={N}, sampling rate SR={M/N:.2%}")

    # method and patterns
    if args.method == 'tradgi':
        if M > N:
            # oversampled: cannot be used by tradgi as is (a Hadamard basis holds at most N orthogonal patterns)
            raise ValueError(f"tradgi does not support oversampling: M(={M}) > N(={N}). Use admm-l1, or select the first N measurements beforehand.")
        patterns = get_hadamard_matrix(N, M)
        logging.info(f"tradgi: using the acquired Hadamard rectangle ({M}, {N}) directly.")

    else:
        # admm-l1 handles any M <= N (M>N runs too but is usually pointless); build the (M, N) pattern matrix as before
        patterns = get_hadamard_matrix(N, M)

    # reconstruction
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.method == 'tradgi':
        recon = trad_gi_recon(patterns, buckets_raw, img_size, device)
    else:
        recon = admm_l1_recon(
            patterns, buckets_raw, img_size,
            l1_weight=args.reg_weight, rho=args.rho, steps=args.steps, device=device
        )

    # unify to [B, H, W]
    if recon.ndim == 4:
        recon = recon[:, 0, :, :]
    elif recon.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected number of dimensions in the reconstruction: {recon.shape}")

    # post-processing and saving
    for i in range(B):
        img = recon[i].astype(np.float32)  # [H, W]

        # optional rotation/transpose
        if args.rot90 % 4 != 0:
            img = np.rot90(img, k=args.rot90 % 4)
        if args.transpose:
            img = img.T

        img_to_save = img.copy()

        # optional percentile clipping
        if args.clip_percentile is not None and 0 < args.clip_percentile < 50:
            low = np.percentile(img_to_save, args.clip_percentile)
            high = np.percentile(img_to_save, 100 - args.clip_percentile)
            if high > low:
                img_to_save = np.clip(img_to_save, low, high)

        # linear normalization to [0,1]
        mn, mx = float(img_to_save.min()), float(img_to_save.max())
        if mx > mn:
            img_01 = (img_to_save - mn) / (mx - mn)
        else:
            img_01 = np.zeros_like(img_to_save, dtype=np.float32)

        u8 = to_uint8_gray(img_01)
        out_png = save_dir / f"{args.method}_recon_{i:06d}.png"
        Image.fromarray(u8).save(out_png)

        if args.save_npy:
            out_npy = save_dir / f"{args.method}_recon_{i:06d}.npy"
            np.save(out_npy, img.astype(np.float32))

    logging.info(f"Done, written to: {save_dir}")


if __name__ == '__main__':
    main()
