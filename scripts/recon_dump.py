"""recon_dump.py — dump classical reconstructions as a standard segmentation dataset (for task-adapted recon-then-seg)

Independent of any segmentation checkpoint: for every image of the given dataset/split
    1) compute bucket_raw with the same Φ as in training (Sylvester natural-order Hadamard, get_hadamard_matrix(N, M));
    2) reconstruct according to --method:
         hsi -> src.reconstruction.trad_gi_recon   (Hadamard inverse transform; HSI=tradgi in run_all.sh)
         cs  -> src.reconstruction.admm_l1_recon   (ADMM-L1+DCT; paper parameters l1=0.01, rho=1.0, steps=100)
    3) reconstructions go to  <out_root>/<method>_<dataset>/<split>/images/<stem>.png (grayscale 128x128, 0-255)
       GT masks go to <out_root>/<method>_<dataset>/<split>/masks/<stem>.png  (same stem, same size;
       binary=0/255, multi-class=evenly spaced gray levels 0/128/255, matching the data_rev mask encoding, so
       FolderSegDataset('custom') / WBCDataset('wbc') can read them straight back).

Idempotent: skipped when images+masks both exist already; PNGs are written atomically (tmp+replace), an interrupt leaves no half-written file.

Usage (from the repository root, PYTHONPATH=repository root):
    python scripts/recon_dump.py --dataset carvana --split test --method hsi --limit 8
    python scripts/recon_dump.py --dataset carvana --split train --method cs
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.datasets.dataset_factory import get_dataset
from src.reconstruction import trad_gi_recon, admm_l1_recon
from src.utils.ghost_patterns import get_hadamard_matrix

# dataset metadata, matching configs/experiments/rev_*_traditional.yaml
DATASET_META = {
    'carvana': {'root': 'data_rev/carvana', 'classes': 1},
    'wbc':     {'root': 'data_rev/wbc',     'classes': 3},
    'mnist':   {'root': 'data_rev/mnist',   'classes': 1},
}


def to_uint8_gray(arr01):
    """float [0,1] -> uint8 grayscale (same logic as to_uint8_gray in scripts/reconstruct_eval.py)"""
    arr = np.clip(arr01, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def save_png_atomic(u8_hw, path: Path):
    """Write a tmp file first, then os.replace, so an interrupt cannot leave a truncated PNG"""
    tmp = path.with_suffix('.png.tmp')
    Image.fromarray(u8_hw).save(tmp, format='PNG')  # 2D uint8 -> automatically 'L' grayscale
    os.replace(tmp, path)


def sample_stem(dataset, idx):
    """folder-style datasets -> the image file stem; MNIST (samples are torchvision indices) -> mnist_<idx:05d>"""
    entry = dataset.samples[idx]
    if isinstance(entry, (tuple, list)) and hasattr(entry[0], 'stem'):
        return entry[0].stem
    return f"mnist_{int(entry):05d}"


def encode_mask_png(mask_idx_hw, num_classes):
    """class index -> gray encoding: binary 0/255; multi-class evenly spaced gray levels (wbc: 0/128/255)"""
    if num_classes <= 1:
        return (mask_idx_hw.astype(np.uint8) * 255)
    levels = np.rint(np.linspace(0, 255, num_classes)).astype(np.uint8)
    return levels[mask_idx_hw.astype(np.int64)]


def reconstruct_batch(method, buckets_bm, img_size, patterns, device, cs_args):
    """Returns (B,1,H,W) float in [0,1]; the algorithms are reused straight from src.reconstruction (as in the paper)"""
    if method == 'hsi':
        return trad_gi_recon(patterns, buckets_bm, img_size, device)
    if method == 'cs':
        return admm_l1_recon(
            patterns, buckets_bm, img_size,
            l1_weight=cs_args['l1_weight'], rho=cs_args['rho'],
            steps=cs_args['steps'], device=device,
        )
    raise ValueError(f"unknown method: {method}")


def main():
    parser = argparse.ArgumentParser(description="dump classical reconstructions as a segmentation dataset (independent of any ckpt)")
    parser.add_argument('--dataset', required=True, choices=list(DATASET_META.keys()))
    parser.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    parser.add_argument('--method', required=True, choices=['hsi', 'cs'])
    parser.add_argument('--out_root', default='data_recon', help='output root directory (default data_recon)')
    parser.add_argument('--data_root', default=None,
                        help='source data root (default data_rev/<dataset>, matching rev_*_traditional.yaml)')
    parser.add_argument('--bucket_size', type=int, default=512, help='M (default 512 = 3.13%% @128x128)')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--limit', type=int, default=0, help='process only the first N images of the split (0=all; for a dry run)')
    parser.add_argument('--batch_size', type=int, default=64, help='reconstruction batch size')
    parser.add_argument('--device', default=None, help='cuda/cpu (auto by default)')
    # CS(ADMM-L1) parameters, defaults matching the reconstruction baseline of run_all.sh: --reg_weight 0.01 --rho 1.0 --steps 100
    parser.add_argument('--cs_l1_weight', type=float, default=0.01)
    parser.add_argument('--cs_rho', type=float, default=1.0)
    parser.add_argument('--cs_steps', type=int, default=100)
    # noise-augmented TA dump: inject AC-ref Gaussian into the bucket BEFORE reconstruction
    # (sigma = std(raw[:,1:]) * 10^(-SNR/20)), matching the AC reference of evaluate.py /
    # ta_noise_eval. numpy std here is ddof=0 vs torch ddof=1 there: sigma differs by
    # sqrt(M/(M-1)) ~ 1.001, i.e. +0.0085 dB (see src/utils/bucket.py).
    parser.add_argument('--noise_snr_db', type=float, default=None,
                        help='fixed measurement-domain SNR (dB) injected before recon; None=clean (original behavior)')
    parser.add_argument('--noise_mixed', action='store_true',
                        help='per-image SNR sampled uniformly from [--noise_mixed_lo, --noise_mixed_hi] dB')
    parser.add_argument('--noise_mixed_lo', type=float, default=10.0)
    parser.add_argument('--noise_mixed_hi', type=float, default=40.0)
    parser.add_argument('--noise_seed', type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    _noise_rng = np.random.default_rng(args.noise_seed)

    meta = DATASET_META[args.dataset]
    data_root = args.data_root or meta['root']
    num_classes = meta['classes']
    N = args.img_size * args.img_size
    M = args.bucket_size
    logging.info(f"dataset={args.dataset} split={args.split} method={args.method} "
                 f"M={M}/{N} ({M / N:.2%}) device={device}")

    # same Φ as in training: BaseSegmentationDataset internally uses get_hadamard_matrix_cached(N, M, perm_seed=None)
    dataset = get_dataset(
        args.dataset,
        root_dir=data_root,
        bucket_size=M,
        img_size=args.img_size,
        num_classes=num_classes,
        mode=args.split,
        preload=False,
        augmentation=None,
        transform=None,
        compute_bucket=True,
    )

    out_split = Path(args.out_root) / f"{args.method}_{args.dataset}" / args.split
    img_dir = out_split / 'images'
    msk_dir = out_split / 'masks'
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    # measurement matrix: hsi takes the full (N,N) (inverse transform at the reconstruction end), cs takes (M,N) (the same first M rows as the bucket)
    if args.method == 'hsi':
        patterns = get_hadamard_matrix(N, N)
    else:
        patterns = get_hadamard_matrix(N, M)

    cs_args = {'l1_weight': args.cs_l1_weight, 'rho': args.cs_rho, 'steps': args.cs_steps}

    total = len(dataset)
    n_take = min(total, args.limit) if args.limit > 0 else total
    logging.info(f"split holds {total} images, processing the first {n_take} this run -> {out_split}")

    dumped, skipped = 0, 0
    pending = []  # (stem, bucket_raw(M,), mask_idx(H,W))

    def flush():
        nonlocal dumped
        if not pending:
            return
        buckets = np.stack([b for _, b, _ in pending]).astype(np.float32)  # (B, M)
        if args.noise_mixed or args.noise_snr_db is not None:
            ref = buckets[:, 1:]  # AC-ref (exclude DC), matches evaluate.py / ta_noise_eval convention
            if args.noise_mixed:
                snr = _noise_rng.uniform(args.noise_mixed_lo, args.noise_mixed_hi, size=(buckets.shape[0], 1))
            else:
                snr = args.noise_snr_db
            sigma = ref.std(axis=1, keepdims=True) * (10.0 ** (-snr / 20.0))
            buckets = (buckets + sigma * _noise_rng.standard_normal(buckets.shape)).astype(np.float32)
        recon = reconstruct_batch(args.method, buckets, args.img_size, patterns, device, cs_args)
        for (stem, _, mask_idx), rec in zip(pending, recon):
            save_png_atomic(to_uint8_gray(rec[0]), img_dir / f"{stem}.png")
            save_png_atomic(encode_mask_png(mask_idx, num_classes), msk_dir / f"{stem}.png")
            dumped += 1
        logging.info(f"  dumped {dumped}/{n_take - skipped} (skipped {skipped})")
        pending.clear()

    for idx in range(n_take):
        stem = sample_stem(dataset, idx)
        if (img_dir / f"{stem}.png").exists() and (msk_dir / f"{stem}.png").exists():
            skipped += 1
            continue
        sample = dataset[idx]  # bucket_raw = Φ @ image (CPU path, noise-free, no augmentation, deterministic)
        pending.append((stem, sample['bucket_raw'], sample['mask']))
        if len(pending) >= args.batch_size:
            flush()
    flush()

    meta_out = {
        'dataset': args.dataset, 'split': args.split, 'method': args.method,
        'bucket_size': M, 'img_size': args.img_size, 'sampling_rate': M / N,
        'phi': 'sylvester_natural_first_M_rows (get_hadamard_matrix)',
        'cs_params': cs_args if args.method == 'cs' else None,
        'num_classes': num_classes,
        'mask_encoding': '0/255' if num_classes <= 1 else
                         f"levels={np.rint(np.linspace(0, 255, num_classes)).astype(int).tolist()}",
        'total_in_split': total, 'processed': n_take,
        'dumped_this_run': dumped, 'skipped_existing': skipped,
    }
    with open(out_split / '_dump_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)
    logging.info(f"done: {dumped} new, {skipped} skipped. meta -> {out_split / '_dump_meta.json'}")


if __name__ == '__main__':
    main()
