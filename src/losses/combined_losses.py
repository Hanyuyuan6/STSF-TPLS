import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.losses import ssim_loss


class CombinedReconLoss(nn.Module):
    """
    Auxiliary 2-D reconstruction loss on the intermediate layer, made of an L1, an L2 and an SSIM term.
    pred and target are expected to have shape (B,1,H,W), with values in [0,1].
    """

    def __init__(self, l1_weight=0.3, l2_weight=0.3, ssim_weight=0.4):
        super().__init__()  # initialize through the parent constructor
        self.l1_w = l1_weight  # L1 loss weight
        self.l2_w = l2_weight  # L2 loss weight
        self.ssim_w = ssim_weight  # SSIM loss weight

    def forward(self, pred, target):
        # if pred or target is None, return a zero loss, placed on target's device or on the CPU
        if pred is None or target is None:
            device = target.device if isinstance(target, torch.Tensor) else 'cpu'
            return torch.tensor(0.0, device=device)

        loss = 0.0  # start the total loss at 0
        if self.l1_w > 0:
            loss += self.l1_w * F.l1_loss(pred, target)  # compute the L1 loss and accumulate it with its weight
        if self.l2_w > 0:
            loss += self.l2_w * F.mse_loss(pred, target)  # compute the L2 loss and accumulate it with its weight
        if self.ssim_w > 0:
            # structural similarity loss (SSIM), window size 5, averaged then accumulated with its weight
            loss += self.ssim_w * ssim_loss(pred, target, window_size=5).mean()
        return loss  # return the weighted combined loss