import numpy as np  # numpy, for the numerical work
from sklearn.metrics import confusion_matrix  # the confusion-matrix routine

def _metrics_from_cm(cm):
    pa = np.diag(cm).sum() / (cm.sum() + 1e-8)  # pixel accuracy PA: the diagonal sum over the total pixel count, plus a small constant against division by zero
    mpa = np.nanmean(np.diag(cm) / (cm.sum(axis=1) + 1e-8))  # mean pixel accuracy MPA: the per-class accuracies averaged, with a small constant against division by zero and NaNs ignored
    iou_per = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm) + 1e-8)  # per-class IoU, the denominator being the size of the prediction-label union
    miou = np.nanmean(iou_per)  # mean IoU, NaNs ignored
    dice_per = 2*np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) + 1e-8)  # per-class Dice, the denominator being the prediction and label pixel counts
    mdice = np.nanmean(dice_per)  # mean Dice, NaNs ignored
    # foreground-only (protocol B, background class 0 dropped): with a sparse foreground the background IoU of ~0.99 inflates the 2-class mean, so the paper reports the foreground value
    miou_fg = float(np.nanmean(iou_per[1:])) if len(iou_per) > 1 else float(miou)
    mdice_fg = float(np.nanmean(dice_per[1:])) if len(dice_per) > 1 else float(mdice)
    return dict(pa=float(pa), mpa=float(mpa), miou=float(miou), mdice=float(mdice),
                miou_fg=miou_fg, mdice_fg=mdice_fg)  # the 2-class means plus the foreground-only ones

def batch_segmentation_metrics(predictions, targets, num_classes=2):
    flat_p = predictions.reshape(-1)  # flatten the predictions into a 1-D array
    flat_t = targets.reshape(-1)  # flatten the ground-truth labels into a 1-D array
    cm = confusion_matrix(flat_t, flat_p, labels=list(range(num_classes)))  # confusion matrix, over the explicitly given class labels
    return _metrics_from_cm(cm)  # derive every metric from the confusion matrix and return them