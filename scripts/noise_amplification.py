"""Recompute the released measurement-to-reconstruction noise-amplification diagnostic."""

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.dataset_factory import get_dataset
from src.utils.ghost_patterns import get_hadamard_matrix
from src.utils.gradient_diagnostics import atomic_write_json
from src.utils.noise_diagnostics import noise_amplification_metrics


def _summary(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("carvana", "mnist", "wbc"), required=True)
    parser.add_argument("--root", default=None, help="dataset root (default: data_rev/<dataset>)")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--bucket_size", type=int, default=512)
    parser.add_argument("--snr_db", type=float, default=20.0)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--noise_seeds", type=int, nargs="+", default=None,
        help="independent noise seeds (default: 1 2 3, matching the released protocol)",
    )
    seed_group.add_argument(
        "--noise_seed", type=int, default=None,
        help="single-seed diagnostic compatibility mode",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.bucket_size <= 1 or args.img_size <= 0 or args.batch_size <= 0:
        raise SystemExit("bucket_size, img_size, and batch_size must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("max_samples must be positive when supplied")
    noise_seeds = (
        [args.noise_seed]
        if args.noise_seed is not None
        else (args.noise_seeds if args.noise_seeds is not None else [1, 2, 3])
    )
    if len(noise_seeds) != len(set(noise_seeds)):
        raise SystemExit("noise seeds must be unique")

    n_pixels = args.img_size * args.img_size
    if args.bucket_size > n_pixels:
        raise SystemExit("bucket_size cannot exceed img_size squared")
    classes = 3 if args.dataset == "wbc" else 1
    root = args.root or f"data_rev/{args.dataset}"
    dataset = get_dataset(
        args.dataset,
        root_dir=root,
        bucket_size=args.bucket_size,
        img_size=args.img_size,
        num_classes=classes,
        mode=args.split,
        preload=False,
        augmentation=None,
        transform=None,
        compute_bucket=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    phi = get_hadamard_matrix(n_pixels, args.bucket_size).astype(np.float32, copy=False)
    clean_batches = []
    collected = 0
    for batch in loader:
        clean = batch["bucket_raw"].detach().cpu().numpy().astype(np.float32, copy=False)
        remaining = None if args.max_samples is None else args.max_samples - collected
        if remaining is not None and remaining <= 0:
            break
        if remaining is not None:
            clean = clean[:remaining]
        clean_batches.append(clean.copy())
        collected += len(clean)

    if not clean_batches:
        raise RuntimeError("dataset produced no samples")
    records_by_seed = {}
    sample_records = []
    for noise_seed in noise_seeds:
        rng = np.random.default_rng(noise_seed)
        seed_records = []
        sample_index = 0
        for clean in clean_batches:
            # Match torch.std(...), used by evaluate.py/ta_noise_eval.py (Bessel correction).
            ac_reference = np.std(clean[:, 1:], axis=1, ddof=1, keepdims=True)
            sigma = ac_reference * (10.0 ** (-args.snr_db / 20.0))
            noise = rng.standard_normal(clean.shape, dtype=np.float32) * sigma
            noisy = clean + noise
            batch_records = noise_amplification_metrics(
                clean, noisy, phi, ta_uint8_roundtrip=True
            )
            for record in batch_records:
                record["sample_index"] = sample_index
                record["noise_seed"] = noise_seed
                sample_index += 1
                seed_records.append(record)
                sample_records.append(record)
        records_by_seed[str(noise_seed)] = seed_records
    metric_names = [
        "relative_if",
        "relative_ta",
        "linear_ratio",
        "first_order_ratio",
        "normalized_if_shift",
        "normalized_ta_shift",
        "normalized_shift_ratio",
        "measurement_range",
        "reconstruction_range",
    ]
    payload = {
        "protocol": "identical AC-referenced measurement noise applied before both lifts",
        "dataset": args.dataset,
        "root": str(root),
        "split": args.split,
        "img_size": args.img_size,
        "bucket_size": args.bucket_size,
        "snr_db": args.snr_db,
        "noise_seeds": noise_seeds,
        "sample_count_per_seed": collected,
        "record_count": len(sample_records),
        "diagnostic_subset": args.max_samples is not None,
        "metric_definition": {
            "relative_if": "RMS(noisy_measurement-clean_measurement) / clean_measurement_range",
            "relative_ta": "RMS(noisy_adjoint-clean_adjoint) / clean_adjoint_range",
            "linear_ratio": "relative_ta / relative_if",
            "first_order_ratio": "sqrt(M) / N * R_IF / R_TA",
            "normalized_shift": "RMS difference after independent min-max normalization; the TA lift then uses the deployed 8-bit round-trip",
            "first_order_scope": "first-order orthogonal-row approximation; realized linear and post-normalization shifts are also reported",
        },
        "aggregation": "full selected split for each seed; overall summary pools equal-size seed blocks",
        "summary": {
            name: _summary([record[name] for record in sample_records])
            for name in metric_names
        },
        "per_seed_summary": {
            seed: {
                name: _summary([record[name] for record in records])
                for name in metric_names
            }
            for seed, records in records_by_seed.items()
        },
        "samples": sample_records,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    atomic_write_json(args.output, payload)
    print(
        f"wrote {args.output} ({collected} samples x {len(noise_seeds)} noise seeds)"
    )


if __name__ == "__main__":
    main()
