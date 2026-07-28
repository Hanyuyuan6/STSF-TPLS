import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()  # initialize the parent class
        self.smooth = smooth  # smoothing term, guards against division by zero

    def forward(self, logits, target):
        if logits.size(1) == 1:  # binary case, a single channel
            probs = torch.sigmoid(logits)  # a sigmoid over the logits gives the probabilities
            target = target.float().unsqueeze(1)  # cast target to float and add the channel dimension
        else:  # multi-class case
            probs = torch.softmax(logits, dim=1)  # a softmax over the logits gives the class probabilities
            # one-hot encode the class labels, permute the dimensions into (B, C, H, W) and cast to float
            target = F.one_hot(target, num_classes=logits.size(1)).permute(0, 3, 1, 2).float()

        inter = (probs * target).sum(dim=(0, 2, 3))  # the overlap between prediction and ground truth (the intersection)
        denom = probs.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))  # |A|+|B|, the Dice denominator -- NOT the union (|A|+|B|-|A∩B|)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)  # the Dice coefficient
        return 1 - dice.mean()  # the Dice loss (1 - the mean Dice coefficient)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()  # initialize the parent class
        self.a = alpha  # the α parameter, balances the positive/negative sample weights
        self.g = gamma  # the γ parameter, tunes how much easy vs. hard samples count

    def forward(self, logits, target):
        if logits.size(1) == 1:  # binary case
            bce = F.binary_cross_entropy_with_logits(logits.squeeze(1), target.float(), reduction='none')  # the per-sample BCE loss
            p = torch.sigmoid(logits.squeeze(1))  # predicted probability
            pt = target * p + (1 - target) * (1 - p)  # the predicted probability of the ground-truth label
            at = target * self.a + (1 - target) * (1 - self.a)  # the weight α of the ground-truth label
            # Focal Loss: the hard samples get more weight
            return (at * (1 - pt) ** self.g * bce).mean()
        else:  # multi-class case
            ce = F.cross_entropy(logits, target, reduction='none')  # the per-sample cross-entropy loss
            pt = torch.exp(-ce)  # the predicted probability of the correct class
            # the multi-class Focal Loss
            return (self.a * (1 - pt) ** self.g * ce).mean()

class CombinedSegmentationLoss(nn.Module):
    """
    Segmentation loss combining Dice and cross-entropy, for the binary case (Sigmoid+BCE+Dice) and the multi-class case (Softmax+CE+Dice).
    Focal Loss and class weights are optional.
    """
    def __init__(self, dice_weight=0.5, ce_weight=0.5, focal_weight=0.0, focal_gamma=2.0, class_weights=None):
        super().__init__()  # initialize the parent class
        self.dw, self.cw, self.fw = dice_weight, ce_weight, focal_weight  # the individual loss weights
        self.dice = DiceLoss()  # the Dice loss instance
        self.focal = FocalLoss(gamma=focal_gamma)  # the Focal loss instance, with gamma set
        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)  # the class weights as a tensor
        else:
            w = None
        # register the cross-entropy weights as a buffer, so training never updates them
        self.register_buffer('ce_weights', w, persistent=False)

    def forward(self, logits, target):
        loss = 0.0  # start the total loss at 0
        if self.cw > 0:
            if logits.size(1) == 1:  # binary, use BCE
                loss += self.cw * F.binary_cross_entropy_with_logits(logits.squeeze(1), target.float())
            else:  # multi-class, use the weighted cross-entropy
                loss += self.cw * F.cross_entropy(logits, target, weight=self.ce_weights)
        if self.dw > 0:
            loss += self.dw * self.dice(logits, target)  # add the Dice loss
        if self.fw > 0:
            loss += self.fw * self.focal(logits, target)  # add the Focal loss
        return loss  # return the weighted combined loss