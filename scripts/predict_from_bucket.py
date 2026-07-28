import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat

import src.models as models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")  # set the logging format and level


def _safe_minmax_norm(x: np.ndarray) -> np.ndarray:
    """Safely normalize a 1-D array to [0,1], taking care of NaNs and infinities"""
    x = np.asarray(x, dtype=np.float32).squeeze()  # cast to float32 and drop the redundant dimensions
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)  # replace NaNs and infinities with 0
    if x.ndim != 1:
        raise ValueError(f"Expected a 1-D bucket vector, got shape {x.shape}")  # make sure it is a 1-D array
    xmin, xmax = float(x.min()), float(x.max())  # compute the minimum and the maximum
    denom = max(xmax - xmin, 1e-8)  # guard against division by zero, the smallest denominator being 1e-8
    return (x - xmin) / denom  # normalize into the [0,1] range


def load_bucket(path: str, M: int) -> np.ndarray:
    """
    Read a bucket signal from a .npy/.txt/.mat file and return a normalized 1-D vector of length M.
    For .mat files every variable is scanned: a 1-D vector of length exactly M is preferred, otherwise a longer one is truncated.
    """
    ext = os.path.splitext(path)[-1].lower()  # get the file extension

    if ext == '.npy':
        b = np.load(path)  # read the npy file
    elif ext == '.txt':
        b = np.loadtxt(path)  # read the txt file
    elif ext == '.mat':
        mat = loadmat(path)  # read the mat file
        candidates_eq = []  # candidate arrays of length exactly M
        candidates_ge = []  # candidate arrays longer than M
        for k, v in mat.items():
            if k.startswith('__'):
                continue  # skip the metadata entries of the mat file
            a = np.array(v).squeeze()  # convert to an array and drop the redundant dimensions
            if a.ndim == 1:
                if a.size == M:
                    candidates_eq.append(a)  # the length is exactly M
                elif a.size > M:
                    candidates_ge.append(a)  # longer than M
        if candidates_eq:
            b = candidates_eq[0]  # prefer the first one whose length is exactly M
        elif candidates_ge:
            b = candidates_ge[0][:M]  # otherwise truncate the first array longer than M
        else:
            raise ValueError(f"No suitable 1-D bucket signal found in the MAT file (length =={M} or >={M})")
    else:
        raise ValueError("Only .npy/.txt/.mat files are supported")

    b = np.asarray(b).squeeze()  # convert to an array and drop the redundant dimensions
    if b.ndim != 1:
        raise ValueError(f"The bucket signal that was read is not 1-D, shape={b.shape}")
    if b.shape[0] < M:
        raise ValueError(f"Bucket signal too short: {b.shape[0]} < expected {M}")
    if b.shape[0] > M:
        b = b[:M]  # cut off the surplus

    b = _safe_minmax_norm(b)  # normalize
    return b


def main():
    parser = argparse.ArgumentParser(description="Inference straight from a bucket signal (bucket-type models only)")
    parser.add_argument('--bucket_path', type=str, required=True, help="path to the bucket-signal file (.npy/.txt/.mat)")
    parser.add_argument('--ckpt_path', type=str, required=True, help="path to the model weights .pth")
    parser.add_argument('--save_path', type=str, default='./seg_pred.png', help="path the prediction is saved to")
    parser.add_argument('--threshold', type=float, default=None, help="binary threshold, None falls back to the config default")
    args = parser.parse_args()

    save_path = Path(args.save_path)  # resolve the save path
    save_path.parent.mkdir(parents=True, exist_ok=True)  # make sure the save directory exists

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # pick the device

    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=True)  # weights_only=True: only tensors/basic types are deserialized, blocking code execution by a malicious .pth (trusted weights only)
    cfg = ckpt['config']  # read the config
    Model = getattr(models, cfg['model']['name'])  # look the model class up dynamically
    model = Model(**cfg['model']['params']).to(device)  # instantiate the model and move it to the device
    model.load_state_dict(ckpt['model_state_dict'])  # load the model parameters
    model.eval()  # switch to evaluation mode

    if getattr(model, 'input_type', 'bucket') != 'bucket':
        raise RuntimeError("This script only supports bucket-type models; use the matching script for non-bucket models.")

    M = int(cfg['data']['bucket_size'])  # length of the bucket signal
    num_classes = int(cfg['data']['classes'])  # number of classes

    b = load_bucket(args.bucket_path, M)  # read and normalize the bucket signal, shape (M,)
    x = torch.from_numpy(b).float().unsqueeze(0).to(device)  # to a tensor, shape (1, M)

    with torch.no_grad():  # turn off gradient computation to save memory
        out = model(x)  # forward pass
        if isinstance(out, dict) and 'logits' in out:
            logits = out['logits']  # take the logits
        else:
            logits = out  # fall back for models that return the logits directly

        if num_classes == 1:
            thr = args.threshold if args.threshold is not None else float(cfg['inference']['threshold'])  # get the threshold
            pred = (torch.sigmoid(logits) > thr).long().squeeze(1).cpu().numpy()  # binary thresholding, shape (B,H,W)
            pred_img = (pred[0] * 255).astype(np.uint8)  # to a uint8 grayscale image
        else:
            pred = torch.argmax(logits, dim=1).cpu().numpy()  # multi-class, take the most probable class, shape (B,H,W)
            scale = 255 // max(1, num_classes - 1)  # map the class indices onto 0-255 gray levels
            pred_img = (pred[0].astype(np.uint8) * scale)  # to a uint8 grayscale image

    Image.fromarray(pred_img).save(save_path)  # save the predicted image
    logging.info(f"Result saved to: {save_path}")  # log the save path


if __name__ == '__main__':
    main()  # run the main function