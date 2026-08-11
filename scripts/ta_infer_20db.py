#!/usr/bin/env python
"""Standalone TA (recon-then-segment) inference on pre-dumped noisy reconstructions.
Bypasses the config-baked data paths: loads model from ckpt['config'], reads PNGs
from --img_dir (e.g. data_recon_20db/hsi_<ds>/test/images), saves predicted masks.
Preprocessing parity with FolderSegDataset: Image.convert('L') -> float/255, no aug.
Usage: python -m scripts.ta_infer_20db --ckpt checkpoints/ta_carvana_hsi_s42/best.pth \
         --img_dir data_recon_20db/hsi_carvana/test/images --out_dir ta20/carvana
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import src.models as models

p = argparse.ArgumentParser()
p.add_argument('--ckpt', required=True)
p.add_argument('--img_dir', required=True)
p.add_argument('--out_dir', required=True)
p.add_argument('--thr', type=float, default=0.5)
a = p.parse_args()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
ck = torch.load(a.ckpt, map_location=dev, weights_only=True)
cfg = ck['config']
Model = getattr(models, cfg['model']['name'])
model = Model(**cfg['model']['params']).to(dev)
model.load_state_dict(ck['model_state_dict'])
model.eval()
ncls = int(cfg['data'].get('classes', 1))

out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
files = sorted(Path(a.img_dir).glob('*.png'))
assert files, f'no PNGs in {a.img_dir}'
with torch.no_grad():
    for f in files:
        img = np.asarray(Image.open(f).convert('L'), dtype=np.float32) / 255.0
        x = torch.from_numpy(img)[None, None].to(dev)
        outp = model(x)
        if isinstance(outp, dict):
            logits = outp['logits']
        elif isinstance(outp, (list, tuple)):
            logits = outp[0]
        else:
            logits = outp
        if ncls <= 1 or logits.shape[1] == 1:
            pred = (torch.sigmoid(logits) > a.thr).long().squeeze().cpu().numpy().astype(np.uint8)
            png = (pred * 255).astype(np.uint8)
        else:
            pred = torch.argmax(logits, 1).squeeze().cpu().numpy().astype(np.uint8)
            png = (pred * (255 // max(ncls - 1, 1))).astype(np.uint8)
        Image.fromarray(png, mode='L').save(out / f.name)
print('OK', len(files), '->', a.out_dir, '| model', cfg['model']['name'], 'ncls', ncls)
