"""End-to-end per-sequence inference latency: image-free (STSF, SPIFS) vs
reconstruct-then-segment (HSI/GI, ADMM-L1). Batch=1, RTX 4090, CUDA-synchronized.

Operators (Hadamard sensing matrix, DCT basis, ADMM inverse term) are PREloaded onto
the GPU once and excluded from timing -- they are fixed and precomputed in any deployed
system; what we time is the marginal per-measurement-sequence cost. Models are randomly
initialized (forward latency is weight-independent), so no checkpoint is needed.

Run:  PYTHONPATH=. python scripts/bench_latency.py   (from the repo root)
"""
import time
import numpy as np
import torch
import yaml

import src.models as models
from src.utils.ghost_patterns import get_hadamard_matrix
from src.reconstruction.cs import dct_2d_matrix, _soft_threshold_t, _ensure_A_MN

DEVICE = "cuda"
IMG = 128
M = 512
N = IMG * IMG          # 16384
B = 1
WARMUP, RUNS = 50, 200
torch.backends.cudnn.benchmark = True


def make(cfg_path):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    name = cfg["model"]["name"]
    return name, getattr(models, name)(**cfg["model"]["params"]).eval().to(DEVICE)


def call(model, x):
    out = model(x)
    return out["logits"] if isinstance(out, dict) else out


# ---- models (random init; latency is weight-independent) ----
n_stsf, stsf = make("configs/experiments/rev_carvana_tpls.yaml")       # GRUUNetPP
n_spifs, spifs = make("configs/experiments/rev_carvana_fcn.yaml")      # FCNUNetPP
n_seg, seg = make("configs/experiments/rev_carvana_traditional.yaml")  # BaselineUNetPP

bucket = torch.randn(B, M, device=DEVICE)

# ---- preloaded operators (setup, NOT timed) ----
H_full = torch.from_numpy(get_hadamard_matrix(N, N)).float().to(DEVICE)        # GI: (N,N)
patterns = _ensure_A_MN(get_hadamard_matrix(N, M), IMG)                        # ADMM: (M,N)
Psi = torch.from_numpy(dct_2d_matrix(IMG)).float().to(DEVICE)
A_Psi = torch.from_numpy(patterns).float().to(DEVICE) @ Psi
AT_Psi, Psi_T = A_Psi.t().contiguous(), Psi.t().contiguous()
inv_term = torch.linalg.inv(torch.eye(M, device=DEVICE) + (A_Psi @ AT_Psi) / 1.0)


def gi_core(bk):
    pad = torch.zeros(B, N, device=DEVICE); pad[:, :M] = bk
    img = (pad @ H_full).view(B, IMG, IMG)
    img = img - img.mean(dim=[1, 2], keepdim=True)
    mn = img.amin(dim=[1, 2], keepdim=True); mx = img.amax(dim=[1, 2], keepdim=True)
    return ((img - mn) / (mx - mn + 1e-8)).unsqueeze(1)


def admm_core(bk, steps=100, rho=1.0, lam=0.01):
    y = bk
    alpha = torch.zeros(B, N, device=DEVICE); z = torch.zeros_like(alpha); u = torch.zeros_like(alpha)
    for _ in range(steps):
        v = z - u
        mid = (y - v @ AT_Psi) @ inv_term
        alpha = v + (mid @ A_Psi) / rho
        z = _soft_threshold_t(alpha + u, lam / rho); u = u + alpha - z
    img = torch.nan_to_num((alpha @ Psi_T).view(B, IMG, IMG))
    mn = img.amin(dim=[1, 2], keepdim=True); mx = img.amax(dim=[1, 2], keepdim=True)
    return torch.clamp((img - mn) / torch.clamp(mx - mn, min=1e-8), 0, 1).unsqueeze(1)


def timeit(fn):
    with torch.no_grad():
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(RUNS):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    return np.asarray(ts)


cases = [
    ("STSF (ours, image-free)", lambda: call(stsf, bucket)),
    ("SPIFS (image-free)",       lambda: call(spifs, bucket)),
    ("HSI/GI -> segment",        lambda: call(seg, gi_core(bucket))),
    ("ADMM-L1 (100 it) -> segment", lambda: call(seg, admm_core(bucket))),
]

print(f"device={torch.cuda.get_device_name(0)}  batch={B}  warmup={WARMUP} runs={RUNS}")
print(f"models: STSF={n_stsf} SPIFS={n_spifs} segmenter={n_seg}\n")
res = {}
for name, fn in cases:
    t = timeit(fn); res[name] = t
    print(f"{name:32s}  mean {t.mean():7.2f} ms   median {np.median(t):7.2f}   "
          f"std {t.std():5.2f}   p90 {np.percentile(t,90):7.2f}")

base = res["STSF (ours, image-free)"].mean()
print("\nrelative to STSF (lower latency is better):")
for name, t in res.items():
    print(f"  {name:32s}  {t.mean()/base:5.2f}x")
