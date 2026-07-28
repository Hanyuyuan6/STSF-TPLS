"""Measure single-sequence inference latency of a trained STSF checkpoint.

Reports batch-1 latency (warmup + CUDA-synchronized timing) so the figure quoted
in the paper is reproducible.

Run:
  PYTHONPATH=. python scripts/measure_latency.py \
      --ckpt checkpoints/rev_carvana_tpls_s42/best.pth --device cuda
"""
import argparse
import time

import numpy as np
import torch

import src.models as models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to best.pth")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--runs", type=int, default=500)
    args = ap.parse_args()

    try:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    except Exception:
        import warnings
        warnings.warn("weights_only=True failed; falling back to a full (unsafe pickle) load. Only load checkpoints from a source you trust.")
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = getattr(models, cfg["model"]["name"])(**cfg["model"]["params"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval().to(args.device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    x = torch.randn(1, cfg["data"]["bucket_size"], device=args.device)  # one measurement sequence
    cuda = args.device == "cuda"

    with torch.no_grad():
        for _ in range(args.warmup):
            model(x)
        if cuda:
            torch.cuda.synchronize()
        ts = []
        for _ in range(args.runs):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if cuda:
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)

    ts = np.asarray(ts)
    dev = torch.cuda.get_device_name(0) if cuda else "CPU"
    print(f"{cfg['model']['name']}  {n_params:.2f}M params  device={dev}  batch=1")
    print(f"latency (ms): mean {ts.mean():.2f}  median {np.median(ts):.2f}  "
          f"p10 {np.percentile(ts, 10):.2f}  p90 {np.percentile(ts, 90):.2f}  "
          f"min {ts.min():.2f}  (n={args.runs})")


if __name__ == "__main__":
    main()
