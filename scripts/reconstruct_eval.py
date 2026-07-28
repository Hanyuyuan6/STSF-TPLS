import argparse
import logging
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils.config_parser import load_config
from src.datasets.dataset_factory import get_dataset
from src.utils.ghost_patterns import get_hadamard_matrix
from src.reconstruction import trad_gi_recon, admm_l1_recon, fista_l1_recon
import src.models as models
from src.metrics.segmentation_metrics import batch_segmentation_metrics


def to_uint8_gray(array_01):
    """Input a float array (0~1), output a uint8 grayscale image"""
    arr = np.clip(array_01, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


# ========== modified: added the vis_scale argument ==========
def save_single_images(gt_imgs, rec_imgs, seg_gts, seg_preds, save_dir, prefix="", vis_scale=255.0):
    """Save a handful of images and masks; kept compatible with the old interface"""
    os.makedirs(save_dir, exist_ok=True)
    num_samples = len(gt_imgs)
    for i in range(num_samples):
        Image.fromarray(to_uint8_gray(gt_imgs[i])).save(os.path.join(save_dir, f'GT_{i + 1}.png'))
        Image.fromarray(to_uint8_gray(rec_imgs[i])).save(os.path.join(save_dir, f'{prefix}_{i + 1}.png'))

        # apply the visualization scaling
        gt_mask_vis = (seg_gts[i] * vis_scale).astype(np.uint8)
        pred_mask_vis = (seg_preds[i] * vis_scale).astype(np.uint8)

        Image.fromarray(gt_mask_vis).save(os.path.join(save_dir, f'GT_mask_{i + 1}.png'))
        Image.fromarray(pred_mask_vis).save(os.path.join(save_dir, f'{prefix}_mask_{i + 1}.png'))

    logging.info(f"Saved {num_samples} images to {save_dir}")


# ========== end of modification ==========


def ensure_dirs(root):
    """Make sure the directory structure for the results exists"""
    (root / 'image').mkdir(parents=True, exist_ok=True)
    (root / 'recon').mkdir(parents=True, exist_ok=True)
    (root / 'mask').mkdir(parents=True, exist_ok=True)
    (root / 'pred').mkdir(parents=True, exist_ok=True)


# ========== modified: added the visualization scaling ==========
def save_batch_all(image_bchw, recon_bchw, mask_bhw, pred_bhw, root_dir, start_idx, num_classes):
    """Save every source image, reconstruction, mask and prediction of one batch; handles binary and multi-class"""
    B = image_bchw.shape[0]

    # compute the visualization scaling factor
    if num_classes > 1:
        vis_scale = 255.0 / (num_classes - 1)
    else:
        vis_scale = 255.0

    for i in range(B):
        img_hw = image_bchw[i, 0] if image_bchw.ndim == 4 and image_bchw.shape[1] == 1 else np.squeeze(image_bchw[i])
        Image.fromarray(to_uint8_gray(img_hw)).save(root_dir / 'image' / f'image_{start_idx + i:06d}.png')

        rec_hw = recon_bchw[i, 0] if recon_bchw.ndim == 4 and recon_bchw.shape[1] == 1 else np.squeeze(recon_bchw[i])
        Image.fromarray(to_uint8_gray(rec_hw)).save(root_dir / 'recon' / f'recon_{start_idx + i:06d}.png')

        # apply the visualization scaling
        m_u8 = (mask_bhw[i] * vis_scale).astype(np.uint8)
        Image.fromarray(m_u8, mode='L').save(root_dir / 'mask' / f'mask_{start_idx + i:06d}.png')

        p_u8 = (pred_bhw[i] * vis_scale).astype(np.uint8)
        Image.fromarray(p_u8, mode='L').save(root_dir / 'pred' / f'pred_{start_idx + i:06d}.png')


# ========== end of modification ==========


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Classical reconstruction -> segmentation evaluation (no TV)")
    parser.add_argument('--ckpt_path', type=str, required=True, help='path to the model weights')
    parser.add_argument('--config', type=str, required=True, help='path to the config file')
    parser.add_argument('--method', type=str, required=True,
                        choices=['tradgi', 'admm-l1', 'fista-l1'], help='reconstruction method')
    parser.add_argument('--sampling_rate', type=float, default=None, help='sampling rate (overrides the config)')
    parser.add_argument('--save_vis_dir', type=str, default=None, help='directory the visualization results are saved to')
    parser.add_argument('--reg_weight', type=float, default=0.01, help='L1 regularization weight (fista/admm)')
    parser.add_argument('--steps', type=int, default=100, help='number of iterations (fista/admm)')
    parser.add_argument('--rho', type=float, default=1.0, help='ADMM penalty parameter (admm)')
    parser.add_argument('--fista_step_scale', type=float, default=1.0, help='FISTA step-size scaling (tstep=step_scale/L; 1.0=the standard 1/L step)')
    parser.add_argument('--num_samples', type=int, default=None, help='cap the number of evaluated samples')
    parser.add_argument('--num_vis', type=int, default=6, help='maximum number of comparison panels')
    parser.add_argument('--save_all', action='store_true', help='save the visualization of every sample (image/recon/mask/pred)')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='which split to evaluate; the final numbers of the paper are reported on test')
    parser.add_argument('--out_json', type=str, default=None, help='write the metrics to this JSON (standalone: merge_results.py only ingests noise evals)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = load_config(args.config)
    assert cfg['model']['name'] == 'BaselineUNetPP', "this script only supports BaselineUNetPP"

    img_size = cfg['data']['img_size']
    N = img_size * img_size
    if args.sampling_rate is not None:
        M = int(N * args.sampling_rate)
        M = max(1, min(M, N))
    else:
        M = cfg['data']['bucket_size']
    actual_sr = M / N
    logging.info(f"Sampling rate: {actual_sr:.2%} ({M}/{N})")

    if args.save_vis_dir:
        vis_root = Path(args.save_vis_dir)
        vis_root.mkdir(parents=True, exist_ok=True)
        method_dir = vis_root / f"{args.method}_sr{actual_sr:.3f}"
        method_dir.mkdir(exist_ok=True)
        ensure_dirs(method_dir)
    else:
        method_dir = None

    base_root = str(Path(cfg['data']['train_dir']).parent)
    val_set = get_dataset(
        cfg['data']['dataset'],
        root_dir=base_root,
        bucket_size=M,
        img_size=img_size,
        num_classes=cfg['data']['classes'],
        mode=args.split,
        preload=False
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True
    )

    # safe loader first; full training checkpoints embed a config dict, so fall back only for files you trust.
    try:
        ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=True)
    except Exception:
        import warnings
        warnings.warn("weights_only=True failed; falling back to a full (unsafe pickle) load. Only load checkpoints from a source you trust.")
        ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    Model = getattr(models, cfg['model']['name'])
    seg_model = Model(**cfg['model']['params']).to(device)
    seg_model.load_state_dict(ckpt['model_state_dict'])
    seg_model.eval()

    if args.method == 'tradgi':
        patterns_full = get_hadamard_matrix(N, N)
        patterns = None
    else:
        patterns_full = None
        patterns = get_hadamard_matrix(N, M)

    all_preds, all_masks = [], []
    vis_gt_imgs, vis_rec_imgs, vis_gt_masks, vis_pred_masks = [], [], [], []
    vis_count = 0
    sample_count = 0

    # ========== added: compute the visualization scaling factor ==========
    num_classes = cfg['data']['classes']
    if num_classes > 1:
        vis_scale = 255.0 / (num_classes - 1)
    else:
        vis_scale = 255.0
    # ========== end of modification ==========

    pbar = tqdm(val_loader, desc=f'[{args.method.upper()}] reconstruction eval')

    for batch_idx, batch in enumerate(pbar):
        if args.num_samples and sample_count >= args.num_samples:
            break

        if 'bucket_raw' in batch:
            buckets_raw = batch['bucket_raw'].numpy()
        else:
            logging.warning("bucket_raw not found, falling back to the normalized bucket")
            buckets_raw = batch['bucket'].numpy()

        gt_images = batch['image'].numpy()
        gt_masks = batch['mask'].numpy()
        B = buckets_raw.shape[0]

        if args.method == 'tradgi':
            recon_images = trad_gi_recon(patterns_full, buckets_raw, img_size, device)
        elif args.method == 'admm-l1':
            recon_images = admm_l1_recon(
                patterns, buckets_raw, img_size,
                l1_weight=args.reg_weight, rho=args.rho, steps=args.steps, device=device
            )
        elif args.method == 'fista-l1':
            recon_images = fista_l1_recon(
                patterns, buckets_raw, img_size,
                l1_weight=args.reg_weight, steps=args.steps, step_scale=args.fista_step_scale, device=device
            )
        else:
            raise ValueError(f"unknown method: {args.method}")

        if recon_images.ndim == 3:
            recon_images = recon_images[:, np.newaxis, :, :]
        elif recon_images.ndim == 4 and recon_images.shape[1] != 1:
            recon_images = recon_images[:, :1, :, :]

        recon_tensor = torch.from_numpy(recon_images).float().to(device)
        seg_output = seg_model(recon_tensor)
        logits = seg_output['logits']

        if cfg['data']['classes'] == 1:
            thr = cfg.get('inference', {}).get('threshold', 0.5)
            seg_preds = (torch.sigmoid(logits) > thr).long().squeeze(1)
        else:
            seg_preds = torch.argmax(logits, dim=1)

        seg_preds_np = seg_preds.cpu().numpy()

        all_preds.append(seg_preds_np)
        all_masks.append(gt_masks)

        if method_dir is not None:
            if args.save_all:
                save_batch_all(
                    image_bchw=gt_images,
                    recon_bchw=recon_images,
                    mask_bhw=gt_masks,
                    pred_bhw=seg_preds_np,
                    root_dir=method_dir,
                    start_idx=sample_count,
                    num_classes=cfg['data']['classes']
                )
            else:
                for i in range(min(B, max(0, args.num_vis - vis_count))):
                    vis_gt_imgs.append(gt_images[i, 0])
                    vis_rec_imgs.append(recon_images[i, 0])
                    vis_gt_masks.append(gt_masks[i])
                    vis_pred_masks.append(seg_preds_np[i])
                    vis_count += 1

        sample_count += B
        pbar.set_postfix({'samples': sample_count})

    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    metrics = batch_segmentation_metrics(
        all_preds, all_masks, num_classes=max(2, cfg['data']['classes'])
    )

    print(f"\n{'=' * 60}")
    print(f"Reconstruction method: {args.method.upper()}")
    print(f"Sampling rate: {actual_sr:.2%} ({M}/{N})")
    print(f"Evaluated samples: {sample_count}")
    print(f"{'=' * 60}")
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")
    print(f"{'=' * 60}\n")

    if args.out_json:
        import json
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, 'w') as f:
            json.dump({
                'method': args.method, 'split': args.split,
                'sampling_rate': actual_sr, 'samples': int(sample_count),
                'dataset': cfg['data']['dataset'],
                'metrics': {k: float(v) for k, v in metrics.items()},
            }, f, indent=2)
        logging.info(f"Metrics JSON: {outp}")

    if method_dir is not None:
        if not args.save_all and vis_count > 0:
            # ========== modified: pass the vis_scale argument ==========
            save_single_images(
                vis_gt_imgs, vis_rec_imgs, vis_gt_masks, vis_pred_masks,
                method_dir, prefix=args.method.upper(), vis_scale=vis_scale
            )
            # ========== end of modification ==========

            for i in range(min(vis_count, args.num_vis)):
                fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                axes[0, 0].imshow(vis_gt_imgs[i], cmap='gray')
                axes[0, 0].set_title('Ground Truth')
                axes[0, 0].axis('off')

                axes[0, 1].imshow(vis_rec_imgs[i], cmap='gray')
                axes[0, 1].set_title(f'{args.method.upper()} (SR={actual_sr:.1%})')
                axes[0, 1].axis('off')

                error = np.abs(vis_gt_imgs[i] - vis_rec_imgs[i])
                axes[0, 2].imshow(error, cmap='hot')
                axes[0, 2].set_title(f'Error (MAE={error.mean():.3f})')
                axes[0, 2].axis('off')

                # ========== modified: matplotlib visualization scaling ==========
                axes[1, 0].imshow(vis_gt_masks[i] * vis_scale / 255.0, cmap='gray', vmin=0, vmax=1)
                axes[1, 0].set_title('GT Mask')
                axes[1, 0].axis('off')

                axes[1, 1].imshow(vis_pred_masks[i] * vis_scale / 255.0, cmap='gray', vmin=0, vmax=1)
                axes[1, 1].set_title('Predicted Mask')
                axes[1, 1].axis('off')

                overlay = np.zeros((img_size, img_size, 3), dtype=np.float32)
                if vis_gt_masks[i].max() > 0:
                    overlay[..., 0] = (vis_gt_masks[i] * vis_scale) / 255.0
                if vis_pred_masks[i].max() > 0:
                    overlay[..., 1] = (vis_pred_masks[i] * vis_scale) / 255.0
                # ========== end of modification ==========

                axes[1, 2].imshow(overlay)
                axes[1, 2].set_title('Overlay (R=GT, G=Pred)')
                axes[1, 2].axis('off')

                plt.suptitle(f'Sample {i + 1}', fontsize=14)
                plt.tight_layout()
                plt.savefig(method_dir / f'comparison_{i + 1}.png', dpi=150, bbox_inches='tight')
                plt.close()

        result_file = method_dir / f"results_{args.method}_sr{actual_sr:.3f}.txt"
        with open(result_file, 'w') as f:
            f.write(f"Method: {args.method.upper()}\n")
            f.write(f"Sampling Rate: {actual_sr:.2%} ({M}/{N})\n")
            f.write(f"Samples: {sample_count}\n")
            f.write("-" * 40 + "\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")
        logging.info(f"Results saved to: {result_file}")


if __name__ == '__main__':
    main()