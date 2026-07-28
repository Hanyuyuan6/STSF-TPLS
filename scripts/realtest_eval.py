# -*- coding: utf-8 -*-
"""Rigorous, reproducible real-hardware MNIST digit-1 eval (rebuilds realtest_metrics on verifiable ground).

Target: MNISTSegDataset test index 8 = digit "1". Physical DMD acquisition read from --real_dir:
  B_STSF-TPLS_MNIST_image_000008_run00{1,2}.mat  (keys: B raw bipolar, d_B differential).
  The .mat captures are gitignored and not shipped — supply your own (see README).

Three lifting strategies, ALL preprocessing bit-aligned to the repo (verified against
base_dataset.py / bucket.py / predict_from_bucket.py):
  STSF   = GRUUNetPP  (checkpoints/rev_mnist_tpls_m512_s42)  <- bucket = minmax(d_B[:512])
  SPIFS  = FCNUNetPP  (checkpoints/rev_mnist_fcn_m512_s42)   <- bucket = minmax(d_B[:512])
  TA-HSI = tradgi recon(d_B[:512] raw) -> ta_mnist_hsi segmenter (recon-image input)

CRUX (why not just reuse predict_from_bucket): DL bucket models train on `patterns @ image` with
patterns in {+1,-1} (Sylvester Hadamard) -> physical analog is the DIFFERENTIAL bucket d_B, NOT the
all-positive B. load_bucket() picks the first >=M 1-D vector by mat-key order and could grab B by
mistake. Here we FORCE d_B explicitly.

GT mask: MNISTSegDataset test index 8 -> (digit>0) -> resize128 NEAREST -> (>127). NOT literal
torchvision test[8] — the test pool is subsampled, so dataset idx 8 maps to torchvision idx 40;
literal test[8] is a "5". See build_gt() for the derivation.
fg IoU: positive-class IoU = |pred & gt| / |pred | gt|  (== metrics iou_per[1]).

sim_upper = clean simulated measurement of the SAME digit-1 image (phi @ img128, no hw noise) through
the same three methods -> isolates the pure sim->real hardware gap on one identical target.

Run on the host that holds the checkpoints:
  python -m scripts.realtest_eval --real_dir path/to/captures --out_dir realtest_out
Outputs: <out_dir>/{metrics.json, <run>_<method>_mask.png, <run>_<method>_recon.png, montage.png}
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.predict_from_bucket import _safe_minmax_norm          # noqa: E402  DL normalization, bit-aligned
from src.utils.ghost_patterns import get_hadamard_matrix           # noqa: E402  natural-order Sylvester (M,N)
from src.reconstruction import trad_gi_recon                       # noqa: E402
import src.models as models                                        # noqa: E402
import torchvision                                                 # noqa: E402

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG = 128
N = IMG * IMG
M = 512
# Defaults only — both are overridable on the CLI. The real-hardware .mat captures are gitignored
# (see .gitignore `*.mat`) and never shipped; a public clone has none, so the hardware arms below
# fall through the graceful skip path and only the simulated upper bound is reported unless you
# pass --real_dir at your own captures.
OUT = ROOT / 'realtest_out'
REAL = ROOT / 'test'   # local default: the two MNIST captures live here (gitignored), alongside the
                       # pytest suite, so local reproduction works with no --real_dir.

CKPT = {  # method -> checkpoint directory name (takes the *best*.pth inside it)
    'STSF':   'rev_mnist_tpls_m512_s42',
    'SPIFS':  'rev_mnist_fcn_m512_s42',
    'TA_HSI': 'ta_mnist_hsi_s42',
}


def find_pth(dirname):
    d = ROOT / 'checkpoints' / dirname
    cands = sorted(glob.glob(str(d / '*best*.pth'))) or sorted(glob.glob(str(d / '*.pth')))
    if not cands:
        raise FileNotFoundError(f'no .pth under {d}')
    return cands[0]


def load_model(dirname):
    pth = find_pth(dirname)
    ck = torch.load(pth, map_location=DEV, weights_only=True)
    cfg = ck['config']
    Model = getattr(models, cfg['model']['name'])
    m = Model(**cfg['model']['params']).to(DEV).eval()
    sd = ck['model_state_dict'] if 'model_state_dict' in ck else ck
    m.load_state_dict(sd)
    thr = float(cfg.get('inference', {}).get('threshold', 0.5))
    itype = getattr(m, 'input_type', 'bucket')
    return m, thr, itype, cfg['model']['name'], pth


def seg_logits(model, x):
    out = model(x)
    return out['logits'] if isinstance(out, dict) else out


def build_gt():
    """CORRECT target = MNISTSegDataset test-index 8 (subsampled), NOT literal torchvision test[8].
    specific_datasets.py:191-196 -> test pool=range(10000), cap=2000, sel=linspace(0,9999,2000).round();
    dataset idx 8 -> torchvision idx sel[8]=40 = digit-1 (fg~6.1%). Literal test[8] is a '5' (the trap)."""
    ds = torchvision.datasets.MNIST(str(ROOT / 'data_rev' / '_mnist_raw'), train=False, download=True)
    sel = np.linspace(0, len(ds) - 1, 2000).round().astype(int)   # == specific_datasets.py:195
    orig_idx = int(sel[8])                                        # dataset test idx 8 -> torchvision idx
    pil, label = ds[orig_idx]                            # PIL 'L' 28x28
    print(f'[GT] dataset test idx 8 -> torchvision idx {orig_idx}')
    arr28 = np.array(pil)
    img128 = np.array(pil.resize((IMG, IMG), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    mpil = Image.fromarray(((arr28 > 0).astype(np.uint8) * 255), mode='L')
    mpil = mpil.resize((IMG, IMG), Image.Resampling.NEAREST)
    gt = (np.array(mpil) > 127).astype(np.uint8)
    return img128, gt, int(label)


def fg_iou(pred, gt):
    p = pred.astype(bool); g = gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter) / float(union + 1e-8)


def read_dB(matpath):
    md = loadmat(str(matpath))
    if 'd_B' not in md:
        raise KeyError(f'{matpath} has no d_B (keys={[k for k in md if not k.startswith("__")]})')
    v = np.asarray(md['d_B']).squeeze().astype(np.float32)
    if v.ndim != 1 or v.shape[0] < M:
        raise ValueError(f'd_B bad shape {v.shape}')
    return v[:M]                                         # raw differential bucket, first M (natural order)


def predict_bucket(model, thr, bucket_raw_M):
    b = _safe_minmax_norm(bucket_raw_M)                  # (M,) minmax -> matches the training bucket_norm
    x = torch.from_numpy(b).float().unsqueeze(0).to(DEV)  # (1, M)
    with torch.no_grad():
        lg = seg_logits(model, x)
        mask = (torch.sigmoid(lg) > thr).long().squeeze(1)[0].cpu().numpy().astype(np.uint8)
    return mask


def predict_ta(model, thr, bucket_raw_M, H_full):
    pad = np.zeros((1, N), dtype=np.float32); pad[:, :M] = bucket_raw_M[None, :]
    rec = trad_gi_recon(H_full, pad, IMG, DEV)           # (1,1,H,W) in [0,1]
    recimg = rec[0, 0]
    rt = torch.from_numpy(recimg[None, None]).float().to(DEV)
    with torch.no_grad():
        lg = seg_logits(model, rt)
        mask = (torch.sigmoid(lg) > thr).long().squeeze(1)[0].cpu().numpy().astype(np.uint8)
    return mask, recimg


def main():
    global OUT, REAL
    ap = argparse.ArgumentParser(description='Real-hardware MNIST digit-1 eval (STSF / SPIFS / TA-HSI).')
    ap.add_argument('--real_dir', default=str(REAL),
                    help='directory holding the real-hardware .mat captures (gitignored; supply your own)')
    ap.add_argument('--out_dir', default=str(OUT), help='where to write masks / recons / metrics.json')
    args = ap.parse_args()
    REAL, OUT = Path(args.real_dir), Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    img128, gt, label = build_gt()
    Image.fromarray((gt * 255).astype(np.uint8)).save(OUT / 'gt_mask.png')
    Image.fromarray((img128 * 255).astype(np.uint8)).save(OUT / 'target_img.png')
    # NB: the torchvision index is printed by build_gt() itself (it is local to that function);
    # referencing it here raised NameError and aborted the run right after the first line of
    # output -- which made the script look like it had started successfully. Fixed 2026-07-23.
    print(f'[GT] MNISTSegDataset test idx 8, label={label} '
          f'gt_fg_pct={100*gt.mean():.2f}%')

    phi = get_hadamard_matrix(N, M).astype(np.float32)   # (M, N) natural-order Sylvester
    H_full = get_hadamard_matrix(N, N).astype(np.float32)
    sim_bucket = (phi @ img128.reshape(-1)).astype(np.float32)  # (M,) raw clean measurement

    models_loaded = {}
    for meth, dirn in CKPT.items():
        mdl, thr, itype, name, pth = load_model(dirn)
        models_loaded[meth] = (mdl, thr, itype)
        print(f'[ckpt] {meth}: {name} input_type={itype} thr={thr}  <- {Path(pth).name}')

    runs = {'sim': sim_bucket}
    for rk in ('run001', 'run002'):
        p = REAL / f'B_STSF-TPLS_MNIST_image_000008_{rk}.mat'
        if p.exists():
            runs[rk] = read_dB(p)
        else:
            print(f'[skip] {rk}: {p} not found — pass --real_dir to point at your captures')
    if len(runs) < 2:
        print('[warn] no real capture found; only the simulated upper bound will be reported')

    results = {}
    masks = {}
    for rk, bucket in runs.items():
        results[rk] = {}
        masks[rk] = {}
        for meth, (mdl, thr, itype) in models_loaded.items():
            if meth == 'TA_HSI':
                mask, rec = predict_ta(mdl, thr, bucket, H_full)
                Image.fromarray((rec * 255).clip(0, 255).astype(np.uint8)).save(OUT / f'{rk}_{meth}_recon.png')
            else:
                mask = predict_bucket(mdl, thr, bucket)
            iou = fg_iou(mask, gt)
            results[rk][meth] = round(iou, 4)
            masks[rk][meth] = mask
            Image.fromarray((mask * 255).astype(np.uint8)).save(OUT / f'{rk}_{meth}_mask.png')
            print(f'  {rk:7s} {meth:7s} fg_IoU={iou:.4f}  pred_fg%={100*mask.mean():.2f}')

    # run-to-run pairIoU (run-to-run stability): IoU(run001_mask, run002_mask) per method.
    # Needs BOTH captures; with only one in --real_dir it is undefined, not zero.
    pair = {}
    if 'run001' in masks and 'run002' in masks:
        for meth in CKPT:
            m1, m2 = masks['run001'][meth], masks['run002'][meth]
            inter = np.logical_and(m1.astype(bool), m2.astype(bool)).sum()
            union = np.logical_or(m1.astype(bool), m2.astype(bool)).sum()
            pair[meth] = round(float(inter) / float(union + 1e-8), 4)
        print(f'[pairIoU run001-vs-run002] {pair}')
    else:
        pair = None
        print('[pairIoU] skipped — needs both run001 and run002 in --real_dir')

    out = {
        'exp': 'real_hardware_mnist_image_000008_REBUILT',
        'gt': 'MNISTSegDataset test index 8 -> torchvision index 40 (digit-1); NOT literal torchvision test[8]',
        'gt_fg_pct': round(100 * float(gt.mean()), 2),
        'M': M, 'sr': f'{M/N:.3%}', 'img_size': IMG,
        'protocol': 'STSF/SPIFS=minmax(d_B[:512]); TA_HSI=tradgi(d_B[:512])->ta_mnist_hsi; natural-order Hadamard; clean training',
        'real': {k: v for k, v in results.items() if k != 'sim'},
        'sim_upper': results['sim'],
        'pairIoU_run001_run002': pair,
    }
    (OUT / 'metrics.json').write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print('\n[SAVED]', OUT / 'metrics.json')
    print(json.dumps(out, indent=2, ensure_ascii=False))

    # montage: rows = whichever runs were actually found, cols = GT | STSF | SPIFS | TA_HSI.
    # `runs` is populated only from captures present on disk (see :173-178), so the row list must
    # follow it — hardcoding run001/run002 raised KeyError right after metrics.json was written,
    # i.e. exactly on the ":180 continue without captures" path the README makes the default.
    order = ['STSF', 'SPIFS', 'TA_HSI']
    rows = [rk for rk in ('run001', 'run002', 'sim') if rk in masks]
    tiles = []
    for rk in rows:
        rowtiles = [(gt * 255).astype(np.uint8)]
        for meth in order:
            rowtiles.append((masks[rk][meth] * 255).astype(np.uint8))
        tiles.append(np.hstack(rowtiles))
    Image.fromarray(np.vstack(tiles)).save(OUT / 'montage.png')
    print('[SAVED]', OUT / 'montage.png', f'(rows {"/".join(rows)}; cols GT|{"|".join(order)})')


if __name__ == '__main__':
    main()
