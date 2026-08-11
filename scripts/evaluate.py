import argparse
import logging
import re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from src.utils.config_parser import load_config
from src.datasets.dataset_factory import get_dataset
import src.models as models
from src.metrics.segmentation_metrics import batch_segmentation_metrics
from src.utils.checkpoint import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Validation-set evaluation (PA, Dice, IoU, MPA)")
    parser.add_argument('--ckpt_path', type=str, required=True, help='path to the model weights file')
    parser.add_argument('--save_vis_dir', type=str, default=None, help='visualization output directory (saves image/pred/mask)')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='which split to evaluate; released numbers are computed on the test split')
    parser.add_argument('--out_json', type=str, default=None, help='write the metrics to this JSON (merge_results.py ingests it only when --noise_snr_db was set)')
    parser.add_argument('--noise_snr_db', type=float, default=None,
                        help='measurement-domain Gaussian noise SNR (dB) at eval time; for the train-clean/eval-noisy robustness sweep, off by default')
    parser.add_argument('--noise_ref', type=str, default='full', choices=['full', 'ac'],
                        help="reference for the SNR sigma: full=std over the whole sequence (historical behaviour), ac=AC std after dropping the DC row (comparable across datasets)")
    parser.add_argument('--noise_seed', type=int, default=None,
                        help='random seed for the evaluation noise (seeds the draw when not None, making the noise reproducible; several seeds give the error bars)')
    parser.add_argument('--allow_unsafe_pickle', action='store_true',
                        help='allow weights_only=False only for a checkpoint you independently trust')
    args = parser.parse_args()

    if args.noise_seed is not None:
        torch.manual_seed(args.noise_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = load_checkpoint(
        args.ckpt_path, map_location=device, allow_unsafe_pickle=args.allow_unsafe_pickle)
    cfg = ckpt['config']
    bucket_on_gpu = bool(cfg['data'].get('bucket_on_gpu', False))

    # perm_seed reaches Phi only on the GPU bucket path (see the build_phi call below); the CPU
    # dataset would be constructed without it and hand back UNPERMUTED buckets while the checkpoint
    # was trained on permuted ones. Same guard as train.py -- refuse rather than mis-evaluate.
    if not bucket_on_gpu and cfg['data'].get('perm_seed') is not None:
        raise SystemExit(
            f"checkpoint config sets perm_seed={cfg['data'].get('perm_seed')} but bucket_on_gpu=false. "
            f"The permutation is only applied on the GPU bucket path, so this evaluation would use a "
            f"different acquisition order than training. Set data.bucket_on_gpu: true.")

    base_root = str(Path(cfg['data']['train_dir']).parent)
    require_reconstruction_manifest = bool(
        cfg['data'].get('require_reconstruction_manifest', False)
        or str(cfg['training'].get('experiment_name', '')).startswith('ta_')
    )
    val_set = get_dataset(
        cfg['data']['dataset'],
        root_dir=base_root,
        bucket_size=cfg['data']['bucket_size'],
        img_size=cfg['data']['img_size'],
        num_classes=cfg['data']['classes'],
        mode=args.split,
        preload=False,
        augmentation=None,
        transform=None,
        compute_bucket=not bucket_on_gpu,
        require_reconstruction_manifest=require_reconstruction_manifest,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True
    )

    Model = getattr(models, cfg['model']['name'])
    model = Model(**cfg['model']['params']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    input_type = getattr(model, 'input_type', 'bucket')
    phi = None
    if bucket_on_gpu and input_type == 'bucket':
        from src.utils.bucket import build_phi
        phi = build_phi(cfg['data']['img_size'], cfg['data']['bucket_size'], device,
                        perm_seed=cfg['data'].get('perm_seed'))

    preds, masks = [], []

    save_dir = Path(args.save_vis_dir) if args.save_vis_dir else None
    if save_dir:
        (save_dir / 'image').mkdir(parents=True, exist_ok=True)
        (save_dir / 'pred').mkdir(parents=True, exist_ok=True)
        (save_dir / 'mask').mkdir(parents=True, exist_ok=True)

    idx = 0
    thr = cfg.get('inference', {}).get('threshold', 0.5)
    num_classes = int(cfg['data']['classes'])

    # scale label indices onto 0-255 so the saved mask PNGs are viewable
    if num_classes > 1:
        vis_scale = 255.0 / (num_classes - 1)
    else:
        vis_scale = 255.0

    # Fail loudly instead of silently dropping the noise. Injection happens ONLY on the
    # bucket_on_gpu branch below; on the other two branches --noise_snr_db would be ignored
    # while the output JSON still stamps the REQUESTED snr/ref/seed (see the metrics dump),
    # i.e. a clean number wearing a noisy label. The CPU-bucket branch is unreachable (every
    # shipped config inherits bucket_on_gpu=true); the image-input branch IS reachable, and this
    # guard deliberately turns that combination into a hard error instead of a mislabelled number.
    # For image-domain arms (BaselineUNetPP: ta_* / rev_*_traditional) the noise must be
    # injected upstream at the measurement — use scripts/ta_noise_eval.py.
    if args.noise_snr_db is not None and not (input_type == 'bucket' and bucket_on_gpu):
        raise SystemExit(
            f"--noise_snr_db={args.noise_snr_db} cannot be honoured here: this checkpoint has "
            f"input_type={input_type!r}, bucket_on_gpu={bucket_on_gpu}. Measurement-domain noise "
            f"is only injected on the image-free GPU-bucket path. Writing the requested SNR into "
            f"the results JSON while evaluating clean data would mislabel the number. "
            f"For reconstruct-then-segment arms use: python -m scripts.ta_noise_eval"
        )

    # Companion tripwire: 'ac' noise excludes DC via raw[:,1:], which assumes DC sits at row 0.
    # A seeded row permutation (perm_seed) moves DC off row 0, so 'ac' would then exclude a random
    # AC coefficient and KEEP DC, mis-scaling sigma. No shipped config combines the two (perm123
    # sets no eval noise), so this is a tripwire, not a behaviour change.
    if (args.noise_snr_db is not None and args.noise_ref == 'ac'
            and cfg['data'].get('perm_seed') is not None):
        raise SystemExit(
            f"--noise_ref=ac with perm_seed={cfg['data'].get('perm_seed')} is unsafe: the AC "
            f"reference excludes DC assuming DC is row 0, but the row permutation moves DC off row 0, "
            f"so raw[:,1:] would drop an AC coefficient and keep DC. Use --noise_ref full instead."
        )

    for batch in val_loader:
        image = batch['image'].to(device).float()
        mask = batch['mask'].to(device).long()

        if input_type != 'bucket':
            x = image
        elif bucket_on_gpu:
            from src.utils.bucket import compute_bucket_gpu
            x = compute_bucket_gpu(image, phi, noise_snr_db=args.noise_snr_db, noise_ref=args.noise_ref)
        else:
            x = batch['bucket'].to(device).float()
        out = model(x)
        logits = out['logits']

        if num_classes == 1:
            pred = (torch.sigmoid(logits) > thr).long().squeeze(1)
        else:
            pred = torch.argmax(logits, dim=1)

        preds.append(pred.cpu())
        masks.append(mask.cpu())

        if save_dir is not None:
            img_bchw = image.detach().cpu().numpy()
            p = pred.detach().cpu().numpy()
            m = mask.detach().cpu().numpy()

            B = p.shape[0]
            for i in range(B):
                img_chw = img_bchw[i]
                if img_chw.ndim == 3 and img_chw.shape[0] == 1:
                    img_hw = img_chw[0]
                elif img_chw.ndim == 2:
                    img_hw = img_chw
                else:
                    img_hw = np.squeeze(img_chw)

                img_u8 = np.clip(img_hw * 255.0 + 0.5, 0, 255).astype(np.uint8)
                Image.fromarray(img_u8, mode='L').save(save_dir / 'image' / f'image_{idx + i:06d}.png')

                # ========== modified: apply the visualization scaling ==========
                if num_classes == 1:
                    pred_u8 = (p[i] * 255).astype(np.uint8)
                else:
                    pred_u8 = (p[i] * vis_scale).astype(np.uint8)  # scale into the visualization range

                Image.fromarray(pred_u8, mode='L').save(save_dir / 'pred' / f'pred_{idx + i:06d}.png')

                if num_classes == 1:
                    mask_u8 = (m[i] * 255).astype(np.uint8)
                else:
                    mask_u8 = (m[i] * vis_scale).astype(np.uint8)  # scale into the visualization range

                Image.fromarray(mask_u8, mode='L').save(save_dir / 'mask' / f'mask_{idx + i:06d}.png')
                # ========== end of modification ==========

            idx += B

    P = torch.cat(preds, 0).numpy()
    G = torch.cat(masks, 0).numpy()
    metrics = batch_segmentation_metrics(P, G, num_classes=max(2, num_classes))

    print(f"==== Eval [{args.split}] ====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    if args.out_json:
        import json
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        experiment_name = cfg['training'].get('experiment_name', '')
        seed_match = re.search(r'_s(\d+)$', experiment_name)
        if seed_match is None:
            seed_match = re.search(r'_s(\d+)', str(args.ckpt_path))
        train_seed = ckpt.get('seed')
        if train_seed is None and seed_match is not None:
            train_seed = int(seed_match.group(1))
        with open(outp, 'w') as f:
            json.dump({
                'ckpt': args.ckpt_path,
                'split': args.split,
                'experiment_name': experiment_name,
                'model': cfg['model']['name'],
                'dataset': cfg['data']['dataset'],
                'bucket_size': cfg['data']['bucket_size'],
                'sampling_rate': cfg['data']['bucket_size'] / (cfg['data']['img_size'] ** 2),
                'train_seed': train_seed,
                'samples': int(P.shape[0]),
                'noise_snr_db': args.noise_snr_db,
                'noise_ref': args.noise_ref,
                'noise_seed': args.noise_seed,
                'perm_seed': cfg['data'].get('perm_seed'),
                'metrics': {k: float(v) for k, v in metrics.items()},
            }, f, indent=2)
        print(f"saved {outp}")


if __name__ == '__main__':
    main()
