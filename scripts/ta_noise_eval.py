# -*- coding: utf-8 -*-
"""TA (fixed-physics lift) measurement-domain noise robustness — the reconstruct-then-segment arm
of the regime-reversal analysis.

Noise model and calibration identical to `evaluate.py --noise_ref ac` (the image-free arm), so both
families read measurements corrupted at the same SNR. The realizations are NOT shared: this arm uses
batch_size=64 while the image-free arm uses the per-dataset cfg batch size, and on CUDA the Philox
offset advances per kernel launch, so an equal --noise_seed still yields a different draw after the
first batch. The comparison is matched-SNR, averaged over 3 noise seeds x 3 training seeds.
  clean-trained ckpt -> add ac-referenced SNR noise to bucket_raw -> reconstruct (tradgi / admm-l1)
  -> 8-bit roundtrip (mirrors the PNG the segmenter was trained on) -> BaselineUNetPP -> metrics json.
  sigma = std(raw[:, 1:]) * 10^(-SNR/20)     # row 0 = DC is excluded: that is the "ac" reference

Config via env vars (defaults reproduce the released grid):
  SEEDS="42 43 44"  DATASETS="carvana mnist wbc"  METHODS="hsi cs"  SNRS="40 30 20 10"  NSEEDS="1 2 3"
Output: results/ta_noise/ta_{ds}_{method}_s{seed}_snr{snr}_ns{ns}_test.json  (idempotent; skips done)
Gracefully skips (logs MISS) when a ckpt is absent.

Example (one cell):  SEEDS=42 DATASETS=mnist METHODS=hsi SNRS=20 NSEEDS=1 python -m scripts.ta_noise_eval"""
import json, os, sys, torch
sys.path.insert(0, os.getcwd())
from torch.utils.data import DataLoader
from src.datasets.dataset_factory import get_dataset
from src.utils.ghost_patterns import get_hadamard_matrix
from src.reconstruction import admm_l1_recon
from src.metrics.segmentation_metrics import batch_segmentation_metrics
import src.models as models
import numpy as np

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG = 128; N = IMG * IMG; M = 512
SNRS = [int(s) for s in os.environ.get('SNRS', '40 30 20 10').split()]
NSEEDS = [int(s) for s in os.environ.get('NSEEDS', '1 2 3').split()]
SEEDS = [int(s) for s in os.environ.get('SEEDS', '42 43 44').split()]
DATASETS = os.environ.get('DATASETS', 'carvana mnist wbc').split()
METHODS = os.environ.get('METHODS', 'hsi cs').split()
OUTDIR = os.environ.get('OUTDIR', 'results/ta_noise')
os.makedirs(OUTDIR, exist_ok=True)
H_gpu = None

def adjoint_recon(noisy_raw):  # gi.py verbatim semantics
    global H_gpu
    if H_gpu is None:
        H_gpu = torch.from_numpy(get_hadamard_matrix(N, M).astype(np.float32)).to(DEV)
    B = noisy_raw.size(0)
    img = (noisy_raw @ H_gpu).view(B, IMG, IMG)
    img = img - img.mean(dim=[1, 2], keepdim=True)
    mn = img.amin(dim=[1, 2], keepdim=True); mx = img.amax(dim=[1, 2], keepdim=True)
    return ((img - mn) / (mx - mn + 1e-8)).unsqueeze(1)

@torch.no_grad()
def run(ds, method, seed):
    ckpath = f'checkpoints/ta_{ds}_{method}_s{seed}/best.pth'
    if not os.path.exists(ckpath):
        print('MISS', ckpath, flush=True); return
    ck = torch.load(ckpath, map_location=DEV, weights_only=True)
    cfg = ck['config']
    Model = getattr(models, cfg['model']['name'])
    seg = Model(**cfg['model']['params']).to(DEV); seg.load_state_dict(ck['model_state_dict']); seg.eval()
    ncls = int(cfg['data']['classes']); thr = float(cfg.get('inference', {}).get('threshold', 0.5))
    dset = get_dataset(ds, root_dir=f'data_rev/{ds}', bucket_size=M, img_size=IMG, num_classes=ncls,
                       mode='test', preload=False, augmentation=None, transform=None, compute_bucket=True)
    dl = DataLoader(dset, batch_size=64, shuffle=False, num_workers=4)
    A_np = get_hadamard_matrix(N, M).astype(np.float32) if method == 'cs' else None
    for snr in SNRS:
        for ns in NSEEDS:
            out = f'{OUTDIR}/ta_{ds}_{method}_s{seed}_snr{snr}_ns{ns}_test.json'
            if os.path.exists(out): print('skip', out, flush=True); continue
            torch.manual_seed(ns)
            preds, gts = [], []
            for batch in dl:
                raw = batch['bucket_raw'].float().to(DEV)
                ref = raw[:, 1:]
                sigma = ref.std(dim=1, keepdim=True) * (10.0 ** (-snr / 20.0))
                noisy = raw + sigma * torch.randn_like(raw)
                if method == 'hsi':
                    rec = adjoint_recon(noisy)
                else:
                    rec_np = admm_l1_recon(A_np, noisy.cpu().numpy(), IMG,
                                           l1_weight=0.01, rho=1.0, steps=100, device=DEV)
                    rec = torch.from_numpy(np.asarray(rec_np)).float().to(DEV)
                rec = torch.floor(torch.clamp(rec, 0, 1) * 255.0 + 0.5) / 255.0
                logits = seg(rec)['logits']
                p = (torch.sigmoid(logits) > thr).long().squeeze(1) if ncls == 1 else torch.argmax(logits, 1)
                preds.append(p.cpu()); gts.append(batch['mask'].long())
            P = torch.cat(preds).numpy(); G = torch.cat(gts).numpy()
            met = batch_segmentation_metrics(P, G, num_classes=max(2, ncls))
            json.dump({'experiment': f'ta_{ds}_{method}_s{seed}_snr{snr}_ns{ns}', 'noise_ref': 'ac',
                       'noise_snr_db': snr, 'noise_seed': ns, 'train_seed': seed, 'split': 'test',
                       'metrics': {k: float(v) for k, v in met.items()}}, open(out, 'w'), indent=2)
            print('->', out, round(met['miou_fg'] * 100, 2), flush=True)

if __name__ == '__main__':
    # Guard required: the DataLoader below uses num_workers>0, and on a spawn platform (Windows/macOS)
    # an unguarded module-level driver re-imports this file in every worker and recurses.
    for seed in SEEDS:
        for ds in DATASETS:
            for method in METHODS:
                run(ds, method, seed)
    print('TA_NOISE_DONE', flush=True)
