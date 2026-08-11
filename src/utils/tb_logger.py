"""TensorBoard logger. Exposes four methods -- define_metric / log_metrics / log_images / finish --
for the training loop to record scalars and images. Logs are written to log_dir/experiment_name/."""
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    def __init__(self, log_dir, experiment_name, config):
        self.config = config
        self.writer = SummaryWriter(log_dir=str(Path(log_dir) / experiment_name))
        self._step = 0

    def define_metric(self, *args, **kwargs):
        # TensorBoard needs no metric declared up front; the empty implementation is kept for compatibility with older call sites
        pass

    def log_metrics(self, metrics_dict):
        # use the 'epoch' entry of the dict as the global step (both the training and validation log_dict carry epoch)
        step = int(metrics_dict.get('epoch', self._step))
        self._step = step
        for k, v in metrics_dict.items():
            if k == 'epoch':
                continue
            try:
                self.writer.add_scalar(k, float(v), step)
            except (TypeError, ValueError):
                pass
        self.writer.flush()

    def log_images(self, images_dict, step, num_classes=None):
        # normalize the GT/predicted masks to [0,1] for TB display (multi-class divides by num_classes-1, binary stays {0,1})
        mask_norm = (1.0 / (num_classes - 1)) if (num_classes is not None and num_classes > 1) else 1.0
        for k, v in images_dict.items():
            t = v.detach().cpu().float()
            if 'gt' in k or 'pred' in k:
                t = t * mask_norm
            if t.dim() == 4 and t.size(1) == 1:
                t = t.repeat(1, 3, 1, 1)  # single channel to three channels
            n = min(t.size(0), self.config['logging'].get('max_log_images', 6))
            t = t[:n].clamp(0, 1)
            self.writer.add_images(k, t, step)
        self.writer.flush()

    def finish(self):
        self.writer.close()
