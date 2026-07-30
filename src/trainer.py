import torch
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from src.metrics.segmentation_metrics import batch_segmentation_metrics
from src.utils.tb_logger import TBLogger
from src.utils.model_utils import count_parameters
from src.utils.gradient_diagnostics import append_jsonl, measure_loss_gradients


class SegmentationTrainer:
    """Unified trainer, supporting configurable global weights and adaptive weight scheduling"""

    def __init__(self, model, optimizer, criterion_seg, criterion_aux,
                 train_loader, val_loader, scheduler, device, config):
        self.model = model
        self.optimizer = optimizer
        self.criterion_seg = criterion_seg
        self.criterion_aux = criterion_aux
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.device = device
        self.cfg = config

        self.epochs = int(config['training']['epochs'])
        self.amp = bool(config['training']['amp'] and self.device.type == 'cuda')
        self.grad_clip = float(config['training'].get('gradient_clip', 1.0))
        max_steps = config['training'].get('max_steps_per_epoch')
        self.max_steps_per_epoch = int(max_steps) if max_steps is not None else None
        if self.max_steps_per_epoch is not None and self.max_steps_per_epoch <= 0:
            raise ValueError("training.max_steps_per_epoch must be a positive integer")
        self.num_classes = int(config['data']['classes'])
        self.input_type = getattr(model, 'input_type', 'bucket')
        # bucket_on_gpu: compute the bucket in batch on the GPU (speeds up carvana; augmentation -> it must be recomputed every epoch -> put it on the GPU)
        self.bucket_on_gpu = bool(config['data'].get('bucket_on_gpu', False)) and self.input_type == 'bucket'
        self.phi = None
        if self.bucket_on_gpu:
            from src.utils.bucket import build_phi
            self.phi = build_phi(config['data']['img_size'], config['data']['bucket_size'], self.device,
                                 perm_seed=config['data'].get('perm_seed'))
        # measurement-domain training noise (optional train-noisy variant; default None = off). Injected in the training step only, validation stays clean.
        self.train_bucket_snr_db = config['data'].get('bucket_noise_snr_db')

        loss_cfg = config['training']['loss']
        self.base_seg_weight = float(loss_cfg.get('seg_weight', 1.0))
        self.base_aux_weight = float(loss_cfg.get('aux_recon_weight', 0.0))

        self.adaptive_loss = loss_cfg.get('adaptive_loss', {}) or {}
        self.use_adaptive = bool(self.adaptive_loss.get('enable', False)) and (self.base_aux_weight > 0.0)

        if self.use_adaptive:
            s1r = float(self.adaptive_loss.get('stage1_ratio', 0.3))
            s2r = float(self.adaptive_loss.get('stage2_ratio', 0.3))
            s1e = max(0, int(self.epochs * max(0.0, min(1.0, s1r))))
            s2e = max(0, int(self.epochs * max(0.0, min(1.0, s2r))))
            s1e = min(s1e, self.epochs)
            s2e = min(s2e, self.epochs - s1e)
            self.stage1_epochs = s1e
            self.stage2_epochs = s2e
            self.stage1_weights = self._parse_pair(self.adaptive_loss.get('stage1_weights', [0.1, 0.9]))
            self.stage2_weights = self._parse_pair(self.adaptive_loss.get('stage2_weights', [0.5, 0.5]))
            self.stage3_weights = self._parse_pair(self.adaptive_loss.get('stage3_weights', [0.9, 0.1]))
        else:
            if bool(self.adaptive_loss.get('enable', False)) and self.base_aux_weight <= 0.0:
                logging.warning("Adaptive weighting was requested, but aux_recon_weight=0; adaptive weighting will be disabled.")
            self.use_adaptive = False

        diagnostic_cfg = config['training'].get('gradient_diagnostics', {}) or {}
        diagnostic_path = diagnostic_cfg.get('jsonl')
        self.gradient_diagnostics_path = Path(diagnostic_path) if diagnostic_path else None
        self.gradient_run_label = str(diagnostic_cfg.get('run_label', 'run'))
        self.gradient_seed = int(diagnostic_cfg.get('seed', config['training'].get('seed', 42)))
        self.gradient_global_step = 0
        self.gradient_stepwise_schedule = bool(diagnostic_cfg.get('stepwise_schedule', False))
        self.gradient_total_steps = None
        if self.gradient_diagnostics_path is not None:
            if self.base_aux_weight <= 0.0:
                raise ValueError("gradient diagnostics require aux_recon_weight > 0")
            if self.gradient_diagnostics_path.exists():
                raise FileExistsError(
                    f"refusing to append to existing gradient diagnostics: "
                    f"{self.gradient_diagnostics_path}"
                )
        if self.gradient_stepwise_schedule:
            if self.gradient_diagnostics_path is None:
                raise ValueError("stepwise adaptive scheduling is restricted to an explicit diagnostic run")
            if not self.use_adaptive:
                raise ValueError("stepwise adaptive scheduling requires adaptive_loss.enable=true")
            if self.max_steps_per_epoch is None:
                raise ValueError("stepwise adaptive scheduling requires max_steps_per_epoch")
            self.gradient_total_steps = self.epochs * self.max_steps_per_epoch
            self.gradient_stage1_end, self.gradient_stage2_end = self._stepwise_boundaries(
                self.gradient_total_steps,
                self.adaptive_loss.get('stage1_ratio', 0.3),
                self.adaptive_loss.get('stage2_ratio', 0.3),
            )
            if not (0 < self.gradient_stage1_end < self.gradient_stage2_end < self.gradient_total_steps):
                raise ValueError(
                    "stepwise phase ratios must produce three non-empty phases over the configured steps"
                )

        self.ckpt_dir = Path(config['training']['checkpoint_dir']) / config['training']['experiment_name']
        requested_tb = bool(config['logging'].get('use_tensorboard', True))
        tb_dir = Path(config['logging'].get('tb_logdir', 'runs')) / config['training']['experiment_name']
        if config['training'].get('refuse_existing_output', False):
            self.ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                self.ckpt_dir.mkdir()
            except FileExistsError as exc:
                raise FileExistsError(
                    f"refusing to reuse existing checkpoint directory: {self.ckpt_dir}"
                ) from exc
            if requested_tb:
                tb_dir.parent.mkdir(parents=True, exist_ok=True)
                try:
                    tb_dir.mkdir()
                except FileExistsError as exc:
                    raise FileExistsError(
                        f"refusing to reuse existing TensorBoard directory: {tb_dir}"
                    ) from exc
        else:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.use_tb = requested_tb
        if self.use_tb:
            try:
                self.tb = TBLogger(
                    config['logging'].get('tb_logdir', 'runs'),
                    config['training']['experiment_name'],
                    config
                )
                self.tb.define_metric("train/*", "epoch")
                self.tb.define_metric("val/*", "epoch")
            except Exception as e:
                logging.warning(f"TensorBoard initialization failed: {e}")
                self.tb = None
                self.use_tb = False
        else:
            self.tb = None

        self.scaler = torch.amp.GradScaler(device='cuda', enabled=self.amp)

        logging.info(f"Model: {config['model']['name']}")
        logging.info(f"Parameters: {count_parameters(self.model):,}")
        logging.info(f"Input type: {self.input_type}")
        logging.info(f"Global segmentation weight seg_weight: {self.base_seg_weight}")
        logging.info(f"Global auxiliary weight aux_recon_weight: {self.base_aux_weight}")
        logging.info(f"Adaptive weighting: {'enabled' if self.use_adaptive else 'disabled'}")

        self.best_metric = -1.0
        self.patience = 0
        self.early_stop = int(config['training'].get('early_stopping_patience', 20))

    @staticmethod
    def _parse_pair(x):
        try:
            a, b = float(x[0]), float(x[1])
        except Exception:
            a, b = 0.5, 0.5
        return a, b

    @staticmethod
    def _stepwise_boundaries(total_steps, stage1_ratio, stage2_ratio):
        stage1 = max(0.0, min(1.0, float(stage1_ratio)))
        stage2 = max(0.0, min(1.0, float(stage2_ratio)))
        return (
            round(total_steps * stage1),
            round(total_steps * min(1.0, stage1 + stage2)),
        )

    def _get_adaptive_weights(self, epoch):
        if not self.use_adaptive:
            return self.base_seg_weight, self.base_aux_weight

        if epoch <= self.stage1_epochs:
            seg_w, aux_w = self.stage1_weights
        elif epoch <= self.stage1_epochs + self.stage2_epochs:
            seg_w, aux_w = self.stage2_weights
        else:
            seg_w, aux_w = self.stage3_weights

        return self.base_seg_weight * seg_w, self.base_aux_weight * aux_w

    def _phase_name(self, epoch):
        if not self.use_adaptive:
            return "fixed"
        if epoch <= self.stage1_epochs:
            return "early"
        if epoch <= self.stage1_epochs + self.stage2_epochs:
            return "middle"
        return "late"

    def _get_stepwise_weights(self, global_step):
        """Return the phase and weights for a one-based diagnostic optimizer step."""
        if not self.gradient_stepwise_schedule or self.gradient_total_steps is None:
            raise RuntimeError("stepwise gradient scheduling is not enabled")
        if not 1 <= global_step <= self.gradient_total_steps:
            raise ValueError(
                f"global_step must be in [1, {self.gradient_total_steps}], got {global_step}"
            )
        if global_step <= self.gradient_stage1_end:
            phase, (seg_w, aux_w) = "early", self.stage1_weights
        elif global_step <= self.gradient_stage2_end:
            phase, (seg_w, aux_w) = "middle", self.stage2_weights
        else:
            phase, (seg_w, aux_w) = "late", self.stage3_weights
        return phase, (self.base_seg_weight * seg_w, self.base_aux_weight * aux_w)

    def train(self):
        logging.info(f"Starting training: {self.epochs} epochs")

        for epoch in range(1, self.epochs + 1):
            seg_w, aux_w = self._get_adaptive_weights(epoch)
            if self.use_adaptive:
                logging.info(f"Epoch {epoch} top-level weights: seg={seg_w:.3f}, aux={aux_w:.3f}")

            train_loss, train_metrics = self._train_epoch(epoch, seg_w, aux_w)
            if self.gradient_stepwise_schedule:
                _, (val_seg_w, val_aux_w) = self._get_stepwise_weights(
                    min(self.gradient_global_step, self.gradient_total_steps)
                )
            else:
                val_seg_w, val_aux_w = seg_w, aux_w
            val_loss, val_metrics = self._validate_epoch(epoch, val_seg_w, val_aux_w)

            if self.scheduler:
                if 'ReduceLROnPlateau' in self.scheduler.__class__.__name__:
                    self.scheduler.step(val_metrics['mdice'])
                else:
                    self.scheduler.step()

            current_metric = val_metrics['mdice']
            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric
                self.patience = 0
                logging.info(f"New best model! Dice: {self.best_metric:.4f}")
            else:
                self.patience += 1

            self._save_checkpoint(epoch, is_best, val_metrics)
            if self.patience >= self.early_stop:
                logging.info(f"Early stopping triggered (patience={self.early_stop})")
                break

        if self.tb:
            self.tb.finish()

        logging.info(f"Training finished. Best Dice: {self.best_metric:.4f}")

    def _get_input(self, batch, noisy=False):
        image = batch['image'].to(self.device).float()
        mask = batch['mask'].to(self.device).long()
        if self.input_type != 'bucket':
            x = image
        elif self.bucket_on_gpu:
            from src.utils.bucket import compute_bucket_gpu
            snr = self.train_bucket_snr_db if noisy else None
            x = compute_bucket_gpu(image, self.phi, noise_snr_db=snr)  # computed on the GPU (outside autocast, fp32, matching the CPU path)
        else:
            x = batch['bucket'].to(self.device).float()
        return x, image, mask

    def _train_epoch(self, epoch, seg_weight, aux_weight):
        self.model.train()
        total_loss = 0.0
        total_seg_loss = 0.0
        total_aux_loss = 0.0
        num_aux_samples = 0
        _train_noisy = self.train_bucket_snr_db is not None
        preds, gts = [], []
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}/{self.epochs}")

        for batch_idx, batch in enumerate(pbar):
            if self.max_steps_per_epoch is not None and batch_idx >= self.max_steps_per_epoch:
                break
            runtime_seg_weight = float(seg_weight)
            runtime_aux_weight = float(aux_weight)
            runtime_phase = self._phase_name(epoch)
            if self.gradient_stepwise_schedule:
                runtime_phase, (runtime_seg_weight, runtime_aux_weight) = (
                    self._get_stepwise_weights(self.gradient_global_step + 1)
                )
            x, image, mask = self._get_input(batch, noisy=_train_noisy)

            with torch.amp.autocast(device_type='cuda', enabled=self.amp):
                output = self.model(x)
                logits = output['logits']

                seg_loss = self.criterion_seg(logits, mask)
                loss = runtime_seg_weight * seg_loss
                total_seg_loss += float(seg_loss.item())

                aux_loss = None
                if runtime_aux_weight > 0 and output.get('aux_recon') is not None:
                    aux_loss = self.criterion_aux(output['aux_recon'], image)
                    loss = loss + runtime_aux_weight * aux_loss
                    total_aux_loss += float(aux_loss.item())
                    num_aux_samples += 1

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite training loss at epoch={epoch}, batch={batch_idx + 1}: "
                    f"{float(loss.detach().cpu())}"
                )

            if self.gradient_diagnostics_path is not None:
                if aux_loss is None:
                    raise RuntimeError(
                        "gradient diagnostics require an auxiliary reconstruction output and loss"
                    )
                geometry = measure_loss_gradients(
                    seg_loss,
                    aux_loss,
                    self.model.parameters(),
                    seg_weight=runtime_seg_weight,
                    aux_weight=runtime_aux_weight,
                )
                append_jsonl(
                    self.gradient_diagnostics_path,
                    {
                        "run_label": self.gradient_run_label,
                        "seed": self.gradient_seed,
                        "epoch": epoch,
                        "step_in_epoch": batch_idx + 1,
                        "global_step": self.gradient_global_step + 1,
                        "phase": runtime_phase,
                        "seg_weight": runtime_seg_weight,
                        "aux_weight": runtime_aux_weight,
                        "seg_loss": float(seg_loss.detach().item()),
                        "aux_loss": float(aux_loss.detach().item()),
                        **geometry,
                    },
                )
                self.gradient_global_step += 1

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            if self.grad_clip and self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.item())
            num_batches += 1

            with torch.no_grad():
                if self.num_classes == 1:
                    pred = (torch.sigmoid(logits) > self.cfg['inference']['threshold']).long().squeeze(1)
                else:
                    pred = torch.argmax(logits, dim=1)
                preds.append(pred.cpu())
                gts.append(mask.cpu())

            pbar.set_postfix({'loss': f"{float(loss.item()):.4f}",
                              'seg_w': f"{runtime_seg_weight:.2f}",
                              'aux_w': f"{runtime_aux_weight:.2f}"})

        P = torch.cat(preds, 0).numpy()
        G = torch.cat(gts, 0).numpy()
        metrics = batch_segmentation_metrics(P, G, num_classes=max(2, self.num_classes))

        if num_batches == 0:
            raise RuntimeError("training loader produced no complete batches")
        avg_loss = total_loss / num_batches
        avg_seg_loss = total_seg_loss / num_batches
        avg_aux_loss = total_aux_loss / max(1, num_aux_samples) if num_aux_samples > 0 else 0

        finite_values = [avg_loss, avg_seg_loss, avg_aux_loss, *metrics.values()]
        if not all(np.isfinite(float(value)) for value in finite_values):
            raise FloatingPointError(f"non-finite training summary at epoch={epoch}")

        if self.tb:
            log_dict = {
                'train/loss': avg_loss,
                'train/seg_loss': avg_seg_loss,
                'train/pa': metrics['pa'],
                'train/dice': metrics['mdice'],
                'train/iou': metrics['miou'],
                'epoch': epoch,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'seg_weight_runtime': float(seg_weight),
                'aux_weight_runtime': float(aux_weight),
            }
            if num_aux_samples > 0:
                log_dict['train/aux_loss'] = avg_aux_loss
            self.tb.log_metrics(log_dict)

        logging.info(
            f"[Train] Epoch {epoch}: loss={avg_loss:.4f}, seg={avg_seg_loss:.4f}, "
            f"aux={avg_aux_loss:.4f}, PA={metrics['pa']:.4f}, Dice={metrics['mdice']:.4f}"
        )

        return avg_loss, metrics

    @torch.no_grad()
    def _validate_epoch(self, epoch, seg_weight, aux_weight):
        self.model.eval()
        total_loss = 0.0
        total_seg_loss = 0.0
        total_aux_loss = 0.0
        num_aux_samples = 0
        preds, gts = [], []

        pbar = tqdm(self.val_loader, desc=f"Val Epoch {epoch}")

        for i, batch in enumerate(pbar):
            x, image, mask = self._get_input(batch)
            output = self.model(x)
            logits = output['logits']

            seg_loss = self.criterion_seg(logits, mask)
            loss = float(seg_weight) * seg_loss
            total_seg_loss += float(seg_loss.item())

            if aux_weight > 0 and output.get('aux_recon') is not None:
                aux_loss = self.criterion_aux(output['aux_recon'], image)
                loss = loss + float(aux_weight) * aux_loss
                total_aux_loss += float(aux_loss.item())
                num_aux_samples += 1

            total_loss += float(loss.item())

            if self.num_classes == 1:
                pred = (torch.sigmoid(logits) > self.cfg['inference']['threshold']).long().squeeze(1)
            else:
                pred = torch.argmax(logits, dim=1)

            preds.append(pred.cpu())
            gts.append(mask.cpu())

            if i == 0 and self.tb and epoch % self.cfg['logging'].get('log_images_every_n_epochs', 5) == 0:
                num_show = min(self.cfg['logging'].get('max_log_images', 6), image.size(0))
                imgs = {
                    'val/image': image[:num_show],
                    'val/gt': mask[:num_show].unsqueeze(1).float(),
                    'val/pred': pred[:num_show].unsqueeze(1).float()
                }
                if output.get('aux_recon') is not None:
                    imgs['val/aux_recon'] = output['aux_recon'][:num_show]

                # pass the num_classes argument, used for the label scaling
                self.tb.log_images(imgs, step=epoch, num_classes=self.num_classes)

        P = torch.cat(preds, 0).numpy()
        G = torch.cat(gts, 0).numpy()
        metrics = batch_segmentation_metrics(P, G, num_classes=max(2, self.num_classes))

        avg_loss = total_loss / max(1, len(self.val_loader))
        avg_seg_loss = total_seg_loss / max(1, len(self.val_loader))
        avg_aux_loss = total_aux_loss / max(1, num_aux_samples) if num_aux_samples > 0 else 0

        finite_values = [avg_loss, avg_seg_loss, avg_aux_loss, *metrics.values()]
        if not all(np.isfinite(float(value)) for value in finite_values):
            raise FloatingPointError(f"non-finite validation summary at epoch={epoch}")

        if self.tb:
            log_dict = {
                'val/loss': avg_loss,
                'val/seg_loss': avg_seg_loss,
                'val/pa': metrics['pa'],
                'val/dice': metrics['mdice'],
                'val/iou': metrics['miou'],
                'val/dice_fg': metrics['mdice_fg'],
                'val/iou_fg': metrics['miou_fg'],
                'val/mpa': metrics['mpa'],
                'epoch': epoch,
                'seg_weight_runtime': float(seg_weight),
                'aux_weight_runtime': float(aux_weight),
            }
            if num_aux_samples > 0:
                log_dict['val/aux_loss'] = avg_aux_loss
            self.tb.log_metrics(log_dict)

        logging.info(
            f"[Val]   Epoch {epoch}: loss={avg_loss:.4f}, seg={avg_seg_loss:.4f}, "
            f"aux={avg_aux_loss:.4f}, PA={metrics['pa']:.4f}, Dice={metrics['mdice']:.4f}"
        )

        return avg_loss, metrics

    def _save_checkpoint(self, epoch, is_best, metrics):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.cfg,
            'metrics': metrics
        }

        latest_path = self.ckpt_dir / 'latest.pth'
        torch.save(state, latest_path)

        if is_best:
            best_path = self.ckpt_dir / 'best.pth'
            torch.save(state, best_path)

            import json
            metrics_path = self.ckpt_dir / 'metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump({
                    'dice': float(metrics['mdice']),
                    'iou': float(metrics['miou']),
                    'dice_fg': float(metrics['mdice_fg']),
                    'iou_fg': float(metrics['miou_fg']),
                    'pa': float(metrics['pa']),
                    'mpa': float(metrics['mpa']),
                    'epoch': epoch
                }, f, indent=2)

        save_interval = int(self.cfg['training'].get('save_interval', 10))
        if epoch % save_interval == 0:
            epoch_path = self.ckpt_dir / f'epoch_{epoch}.pth'
            torch.save(state, epoch_path)
