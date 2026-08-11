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

Idempotent only after a complete hash inventory validates. Partial runs are rewritten;
PNGs are written atomically (tmp+replace), so an interrupt leaves no half-written file.

Usage (from the repository root, PYTHONPATH=repository root):
    python scripts/recon_dump.py --dataset carvana --split test --method hsi --limit 8
    python scripts/recon_dump.py --dataset carvana --split train --method cs
"""

import argparse
import hashlib
import inspect
import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.datasets.dataset_factory import get_dataset
from src.reconstruction import trad_gi_recon, admm_l1_recon
from src.utils.ghost_patterns import get_hadamard_matrix
from src.utils.reconstruction_manifest import (
    build_reconstruction_inventory,
    inventory_sha256,
    validate_reconstruction_manifest,
)

# dataset metadata, matching configs/experiments/rev_*_traditional.yaml
DATASET_META = {
    'carvana': {'root': 'data_rev/carvana', 'classes': 1},
    'wbc':     {'root': 'data_rev/wbc',     'classes': 3},
    'mnist':   {'root': 'data_rev/mnist',   'classes': 1},
}


def to_uint8_gray(arr01):
    """float [0,1] -> uint8 grayscale (same logic as to_uint8_gray in scripts/reconstruct_eval.py)"""
    arr = np.asarray(arr01)
    if not np.isfinite(arr).all():
        raise FloatingPointError("reconstruction contains NaN or Inf; refusing to publish a PNG")
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def save_png_atomic(u8_hw, path: Path):
    """Write a tmp file first, then os.replace, so an interrupt cannot leave a truncated PNG"""
    tmp = path.with_suffix('.png.tmp')
    Image.fromarray(u8_hw).save(tmp, format='PNG')  # 2D uint8 -> automatically 'L' grayscale
    os.replace(tmp, path)


def write_json_atomic(payload, path: Path):
    """Atomically publish a JSON manifest."""
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write('\n')
    os.replace(tmp, path)


def generation_signature(generation):
    """Stable digest covering every setting that can change dumped pixels."""
    encoded = json.dumps(
        generation, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def validate_existing_generation(meta_path: Path, has_outputs, expected_signature):
    """Reject legacy or differently generated outputs instead of mixing them."""
    if not meta_path.exists():
        if has_outputs:
            raise RuntimeError(
                f"Found reconstruction PNGs under {meta_path.parent} but no generation "
                "manifest. Refusing to mix legacy/unverified outputs; use a new --out_root "
                "or remove the old output directory after reviewing it."
            )
        return None

    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            previous = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot validate existing generation manifest {meta_path}: {exc}") from exc

    actual = previous.get('generation_signature')
    if actual != expected_signature:
        raise RuntimeError(
            f"Existing output generation signature does not match this run "
            f"(existing={actual or 'missing'}, requested={expected_signature}). "
            "Refusing to create a stale/mixed dataset; use a new --out_root or remove "
            "the reviewed old output directory."
        )
    return previous


def sample_stem(dataset, idx):
    """folder-style datasets -> the image file stem; MNIST (samples are torchvision indices) -> mnist_<idx:05d>"""
    entry = dataset.samples[idx]
    if isinstance(entry, (tuple, list)) and hasattr(entry[0], 'stem'):
        return entry[0].stem
    return f"mnist_{int(entry):05d}"


def update_source_content_hash(hasher, dataset, idx):
    """Hash the source bytes (or decoded sample) so same-name edits invalidate a dump."""
    entry = dataset.samples[idx]
    paths = []
    if isinstance(entry, (tuple, list)):
        paths = [Path(value) for value in entry if isinstance(value, (str, os.PathLike))]
    if paths and all(path.is_file() for path in paths):
        for path in paths:
            hasher.update(path.name.encode('utf-8'))
            with open(path, 'rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    hasher.update(chunk)
        return

    # Index-backed datasets such as MNIST have no per-sample source path.
    image, mask = dataset._load_pair(idx)
    for value in (image, mask):
        array = np.asarray(value)
        hasher.update(str(array.shape).encode('ascii'))
        hasher.update(str(array.dtype).encode('ascii'))
        hasher.update(array.tobytes(order='C'))


def implementation_fingerprint(dataset):
    """Hash the local implementation files that participate in pixel generation."""
    files = {
        Path(__file__).resolve(),
        Path(inspect.getfile(get_hadamard_matrix)).resolve(),
        Path(inspect.getfile(trad_gi_recon)).resolve(),
        Path(inspect.getfile(admm_l1_recon)).resolve(),
    }
    for cls in dataset.__class__.__mro__:
        if cls is not object:
            files.add(Path(inspect.getfile(cls)).resolve())
    digest = hashlib.sha256()
    for path in sorted(files, key=str):
        digest.update(path.name.encode('utf-8'))
        digest.update(path.read_bytes())
    return digest.hexdigest()


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

    if args.noise_mixed and args.noise_snr_db is not None:
        parser.error('--noise_mixed and --noise_snr_db are mutually exclusive')
    finite_values = {
        '--cs_l1_weight': args.cs_l1_weight,
        '--cs_rho': args.cs_rho,
        '--noise_mixed_lo': args.noise_mixed_lo,
        '--noise_mixed_hi': args.noise_mixed_hi,
    }
    if args.noise_snr_db is not None:
        finite_values['--noise_snr_db'] = args.noise_snr_db
    for label, value in finite_values.items():
        if not math.isfinite(value):
            parser.error(f'{label} must be finite')
    if args.batch_size <= 0 or args.cs_steps <= 0:
        parser.error('--batch_size and --cs_steps must be positive')
    if args.cs_l1_weight < 0 or args.cs_rho <= 0:
        parser.error('--cs_l1_weight must be non-negative and --cs_rho must be positive')
    if args.noise_mixed_lo > args.noise_mixed_hi:
        parser.error('--noise_mixed_lo must be <= --noise_mixed_hi')

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    meta = DATASET_META[args.dataset]
    data_root = args.data_root or meta['root']
    num_classes = meta['classes']
    N = args.img_size * args.img_size
    M = args.bucket_size
    if not 0 < M <= N:
        parser.error(f'--bucket_size must be in [1, {N}], got {M}')
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

    total = len(dataset)
    sample_hasher = hashlib.sha256()
    source_content_hasher = hashlib.sha256()
    for idx in range(total):
        sample_hasher.update(f"{idx}\0{sample_stem(dataset, idx)}\n".encode('utf-8'))
        update_source_content_hash(source_content_hasher, dataset, idx)

    if args.noise_mixed:
        noise_config = {
            'mode': 'mixed_uniform_db', 'lo_db': args.noise_mixed_lo,
            'hi_db': args.noise_mixed_hi, 'seed': args.noise_seed,
        }
    elif args.noise_snr_db is not None:
        noise_config = {
            'mode': 'fixed_db', 'snr_db': args.noise_snr_db, 'seed': args.noise_seed,
        }
    else:
        noise_config = {'mode': 'clean'}

    cs_args = {'l1_weight': args.cs_l1_weight, 'rho': args.cs_rho, 'steps': args.cs_steps}
    generation = {
        'schema_version': 3,
        'dataset': args.dataset,
        'split': args.split,
        'method': args.method,
        'source_root': str(Path(data_root).resolve()),
        'source_sample_count': total,
        'source_sample_ids_sha256': sample_hasher.hexdigest(),
        'source_content_sha256': source_content_hasher.hexdigest(),
        'implementation_sha256': implementation_fingerprint(dataset),
        'bucket_size': M,
        'img_size': args.img_size,
        'phi': 'sylvester_natural_first_M_rows',
        'cs_params': cs_args if args.method == 'cs' else None,
        'num_classes': num_classes,
        'mask_encoding': '0/255' if num_classes <= 1 else
                         f"levels={np.rint(np.linspace(0, 255, num_classes)).astype(int).tolist()}",
        'noise': noise_config,
        'reconstruction_batch_size': args.batch_size,
        'device': str(device),
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
    }
    signature = generation_signature(generation)
    meta_path = out_split / '_dump_meta.json'
    has_outputs = any(img_dir.glob('*.png')) or any(msk_dir.glob('*.png'))
    previous = validate_existing_generation(meta_path, has_outputs, signature)
    if previous is not None and previous.get('complete') is True:
        validate_reconstruction_manifest(out_split, expected_signature=signature)
        logging.info(f"verified complete reconstruction dump -> {out_split}")
        return
    if previous is not None and has_outputs:
        logging.warning(
            "A matching but incomplete dump exists; every selected pair will be rewritten."
        )

    # Publish the signature before the first PNG. A killed run can only be resumed
    # with an identical generation contract.
    write_json_atomic({
        'generation_signature': signature,
        'generation': generation,
        'complete': False,
        'status': 'in_progress',
    }, meta_path)

    # Both HSI (adjoint) and CS use exactly the acquired first M Hadamard rows.
    patterns = get_hadamard_matrix(N, M)

    n_take = min(total, args.limit) if args.limit > 0 else total
    logging.info(f"split holds {total} images, processing the first {n_take} this run -> {out_split}")

    dumped, skipped = 0, 0
    pending = []  # (dataset index, stem, bucket_raw(M,), mask_idx(H,W))

    def flush():
        nonlocal dumped
        if not pending:
            return
        buckets = np.stack([b for _, _, b, _ in pending]).astype(np.float32)  # (B, M)
        if args.noise_mixed or args.noise_snr_db is not None:
            # Derive an independent RNG stream from the dataset index. This makes
            # noisy output invariant to batch size, skips, and interrupted resumes.
            for row, (sample_idx, _, _, _) in enumerate(pending):
                sample_rng = np.random.default_rng(
                    np.random.SeedSequence([args.noise_seed, sample_idx])
                )
                snr = (sample_rng.uniform(args.noise_mixed_lo, args.noise_mixed_hi)
                       if args.noise_mixed else args.noise_snr_db)
                ref_std = buckets[row, 1:].std()
                sigma = ref_std * (10.0 ** (-snr / 20.0))
                buckets[row] += sigma * sample_rng.standard_normal(buckets.shape[1])
        recon = reconstruct_batch(args.method, buckets, args.img_size, patterns, device, cs_args)
        for (_, stem, _, mask_idx), rec in zip(pending, recon):
            save_png_atomic(to_uint8_gray(rec[0]), img_dir / f"{stem}.png")
            save_png_atomic(encode_mask_png(mask_idx, num_classes), msk_dir / f"{stem}.png")
            dumped += 1
        logging.info(f"  dumped {dumped}/{n_take}")
        pending.clear()

    for idx in range(n_take):
        stem = sample_stem(dataset, idx)
        sample = dataset[idx]  # bucket_raw = Φ @ image (CPU path, noise-free, no augmentation, deterministic)
        pending.append((idx, stem, sample['bucket_raw'], sample['mask']))
        if len(pending) >= args.batch_size:
            flush()
    flush()

    inventory = build_reconstruction_inventory(out_split)
    if n_take == total:
        expected_stems = [sample_stem(dataset, idx) for idx in range(total)]
        actual_stems = [entry['stem'] for entry in inventory]
        if len(expected_stems) != len(set(expected_stems)) or actual_stems != sorted(expected_stems):
            raise RuntimeError(
                "complete reconstruction inventory does not exactly match source sample stems"
            )
    meta_out = {
        'generation_signature': signature,
        'generation': generation,
        'complete': n_take == total,
        'status': 'complete' if n_take == total else 'partial_limit',
        'dataset': args.dataset, 'split': args.split, 'method': args.method,
        'bucket_size': M, 'img_size': args.img_size, 'sampling_rate': M / N,
        'phi': 'sylvester_natural_first_M_rows (get_hadamard_matrix)',
        'cs_params': cs_args if args.method == 'cs' else None,
        'num_classes': num_classes,
        'mask_encoding': '0/255' if num_classes <= 1 else
                         f"levels={np.rint(np.linspace(0, 255, num_classes)).astype(int).tolist()}",
        'total_in_split': total, 'processed': n_take,
        'dumped_this_run': dumped, 'skipped_existing': skipped,
        'pair_count': len(inventory), 'file_count': 2 * len(inventory),
        'inventory_sha256': inventory_sha256(inventory),
        'inventory': inventory,
    }
    write_json_atomic(meta_out, meta_path)
    if n_take == total:
        validate_reconstruction_manifest(out_split, expected_signature=signature)
    logging.info(f"done: {dumped} new, {skipped} skipped. meta -> {meta_path}")


if __name__ == '__main__':
    main()
