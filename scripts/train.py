import os
import argparse
import logging
import torch
from torch.utils.data import DataLoader

from src.utils.config_parser import load_config
from src.utils.seed import seed_everything, seed_worker
from src.datasets.dataset_factory import get_dataset
from src.utils.transforms import build_transforms
import src.models as models
import src.losses as losses
from src.trainer import SegmentationTrainer
from src.utils.model_validator import validate_model_architecture

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")  # configure the log format and set the level to INFO

def _str_to_number(v):
    """Try to convert a string to int or float; return the original value on failure"""
    if isinstance(v, str):
        s = v.strip()   # strip the leading/trailing whitespace
        # check whether it is an integer (negatives included)
        if s.lstrip('-').isdigit():
            try:
                return int(s)   # convert to int
            except Exception:
                pass
        try:
            return float(s)     # convert to float
        except Exception:
            return v
    return v

def _map_numbers(obj):
    """Recursively convert numeric strings inside a dict or list into numbers"""
    if isinstance(obj, dict):
        return {k: _map_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_map_numbers(x) for x in obj]
    return _str_to_number(obj)

def main(args):
    seed_everything(args.seed)  # fix the random seed so the experiment is reproducible

    cfg = load_config(args.config)  # load the config file

    if args.dataset:
        cfg['data']['dataset'] = args.dataset  # command line overrides the dataset name
    if args.epochs:
        cfg['training']['epochs'] = args.epochs  # command line overrides the number of epochs
    if args.batch_size:
        cfg['training']['batch_size'] = args.batch_size  # command line overrides the batch size

    # one checkpoint directory per seed, so several seeds do not overwrite each other
    cfg['training']['experiment_name'] = f"{cfg['training']['experiment_name']}_s{args.seed}"

    # bucket_on_gpu: move the bucket matmul to the GPU (trainer computes it on the fly), dataset skips the CPU-side computation (carvana speedup)
    bucket_on_gpu = bool(cfg['data'].get('bucket_on_gpu', False))

    # perm_seed (acquisition-order ablation) and bucket_noise_snr_db (training noise) are read from
    # the config by the trainer, i.e. on the GPU bucket path only -- the CPU dataset accepts both as
    # constructor arguments that nothing here passes. Running such a config with bucket_on_gpu=false
    # would therefore train WITHOUT the permutation or the noise and report it as if it had them.
    # Fail loudly instead of producing a silently mislabelled run.
    if not bucket_on_gpu:
        ignored = [k for k in ('perm_seed', 'bucket_noise_snr_db') if cfg['data'].get(k) is not None]
        if ignored:
            raise SystemExit(
                f"config sets {', '.join(ignored)} but bucket_on_gpu=false. Those settings are only "
                f"honoured on the GPU bucket path, so this run would silently ignore them. Set "
                f"data.bucket_on_gpu: true, or remove {', '.join(ignored)}.")

    logging.info("Config loaded successfully")

    aug_train = build_transforms(
        cfg.get('augmentation', {}).get('train', {}),
        cfg['data']['img_size']
    ) if cfg.get('augmentation', {}).get('train', {}).get('enable', False) else None  # training-set augmentation pipeline, or None

    aug_val = build_transforms(
        cfg.get('augmentation', {}).get('val', {}),
        cfg['data']['img_size']
    ) if cfg.get('augmentation', {}).get('val', {}).get('enable', False) else None  # val-set augmentation pipeline, or None

    from pathlib import Path
    base_root = str(Path(cfg['data']['train_dir']).parent)  # take the parent of the train directory as the data root

    logging.info(f"Creating dataset: {cfg['data']['dataset']}")

    train_set = get_dataset(
        cfg['data']['dataset'],
        root_dir=base_root,
        bucket_size=cfg['data']['bucket_size'],
        img_size=cfg['data']['img_size'],
        num_classes=cfg['data']['classes'],
        mode='train',
        preload=cfg['data'].get('preload', False),
        augmentation=aug_train,
        transform=None,
        compute_bucket=not bucket_on_gpu
    )  # build the training dataset object

    val_set = get_dataset(
        cfg['data']['dataset'],
        root_dir=base_root,
        bucket_size=cfg['data']['bucket_size'],
        img_size=cfg['data']['img_size'],
        num_classes=cfg['data']['classes'],
        mode='val',
        preload=False,
        augmentation=aug_val,
        transform=None,
        compute_bucket=not bucket_on_gpu
    )  # build the validation dataset object

    g = torch.Generator()
    g.manual_seed(args.seed)  # fix the DataLoader shuffle order; together with seed_worker this makes it reproducible

    train_loader = DataLoader(
        train_set,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg['data']['num_workers'] > 0,
        worker_init_fn=seed_worker,
        generator=g
    )  # training data loader: multi-worker, shuffled, drops the trailing samples that do not fill a batch

    val_loader = DataLoader(
        val_set,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
        persistent_workers=cfg['data']['num_workers'] > 0
    )  # validation data loader: no shuffling, multi-worker with pinned memory

    logging.info(f"Train samples: {len(train_set)}")
    logging.info(f"Val samples: {len(val_set)}")

    device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else 'cpu')  # select the training device
    logging.info(f"Using device: {device}")

    Model = getattr(models, cfg['model']['name'])
    model = Model(**cfg['model']['params']).to(device)  # instantiate the model and move it onto the device

    logging.info("Validating model architecture...")
    is_valid, msg = validate_model_architecture(
        model,
        cfg['data']['bucket_size'],
        cfg['data']['img_size'],
        cfg['data']['classes'],
        device
    )  # validate the model input/output shapes

    if not is_valid:
        logging.error(f"Model validation failed: {msg}")
        return
    else:
        logging.info(f"Model validation passed: {msg}")

    seg_cfg = cfg['training']['loss']['seg']
    SegLoss = getattr(losses, seg_cfg['name'])
    seg_criterion = SegLoss(**seg_cfg['params'])  # load the segmentation loss

    aux_cfg = cfg['training']['loss']['aux_recon']
    AuxLoss = getattr(losses, aux_cfg['name'])
    aux_criterion = AuxLoss(**aux_cfg['params'])  # load the auxiliary reconstruction loss

    lr = cfg['training'].get('learning_rate', 5e-4)
    wd = cfg['training'].get('weight_decay', 0.0)
    try:
        lr = float(lr)
    except Exception:
        raise ValueError(f"learning_rate must be numeric, got: {lr} (type: {type(lr)})")
    try:
        wd = float(wd)
    except Exception:
        raise ValueError(f"weight_decay must be numeric, got: {wd} (type: {type(wd)})")  # validate the learning rate and weight decay

    opt_params = _map_numbers(cfg['training'].get('optimizer_params', {}) or {})  # parse the extra optimizer parameters

    Optim = getattr(torch.optim, cfg['training']['optimizer'])
    optimizer = Optim(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
        **opt_params
    )  # instantiate the optimizer

    Schd = getattr(torch.optim.lr_scheduler, cfg['training']['scheduler'])
    schd_params = _map_numbers(cfg['training'].get('scheduler_params', {}) or {})
    scheduler = Schd(optimizer, **schd_params)  # instantiate the learning-rate scheduler

    trainer = SegmentationTrainer(
        model, optimizer, seg_criterion, aux_criterion,
        train_loader, val_loader, scheduler, device, cfg
    )  # build the trainer object

    trainer.train()  # start training

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single-pixel (Hadamard) semantic segmentation training")
    parser.add_argument('--config', type=str, required=True, help="path to the config file")
    parser.add_argument('--dataset', type=str, default=None, help="override the dataset name")
    parser.add_argument('--epochs', type=int, default=None, help="override the number of epochs")
    parser.add_argument('--batch_size', type=int, default=None, help="override the batch size")
    parser.add_argument('--seed', type=int, default=42, help="random seed (keeps the checkpoint directories apart when running several seeds)")

    args = parser.parse_args()
    main(args)