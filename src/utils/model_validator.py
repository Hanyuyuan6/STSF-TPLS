"""
Model validation utilities
Used to check that the model architecture is correct and that the dataset format is valid
"""

import torch
import logging
import numpy as np
from typing import Tuple


def validate_model_architecture(model, bucket_size: int, img_size: int,
                                num_classes: int, device: str = 'cpu') -> Tuple[bool, str]:
    """
    Validate that the model's forward structure and output format are correct
    Args:
      model: model to validate
      bucket_size: dimension of the bucket input vector
      img_size: image height and width (assumed square)
      num_classes: number of classes
      device: compute device, default 'cpu'
    Returns:
      Tuple[bool, str], whether validation passed plus a message
    """
    model = model.to(device)
    model.eval()
    batch_size = 2  # batch size used for the test
    input_type = getattr(model, 'input_type', 'bucket')  # read the model's input type, default 'bucket'

    # build the input tensor according to the input type
    if input_type == 'bucket':
        x = torch.randn(batch_size, bucket_size).to(device)
    elif input_type == 'image':
        x = torch.randn(batch_size, 1, img_size, img_size).to(device)
    else:
        return False, f"Unknown input type: {input_type}"

    try:
        with torch.no_grad():
            output = model(x)  # forward pass through the model

        # the output must be a dict
        if not isinstance(output, dict):
            return False, "Model output must be a dict"

        # the output dict must contain the 'logits' key
        if 'logits' not in output:
            return False, "Output dict must contain the 'logits' key"

        logits = output['logits']
        # logits are expected to have shape (batch_size, num_classes, H, W); a single class means 1 channel
        expected_channels = num_classes if num_classes > 1 else 1
        expected_shape = (batch_size, expected_channels, img_size, img_size)
        if logits.shape != expected_shape:
            return False, f"Wrong logits shape: expected {expected_shape}, got {tuple(logits.shape)}"

        # check the output for NaN or Inf
        if torch.isnan(logits).any():
            return False, "Output contains NaN"
        if torch.isinf(logits).any():
            return False, "Output contains Inf"

        # if the auxiliary output aux_recon is present, check its shape and range
        if 'aux_recon' in output and output['aux_recon'] is not None:
            aux = output['aux_recon']
            expected_aux_shape = (batch_size, 1, img_size, img_size)
            if aux.shape != expected_aux_shape:
                return False, f"Wrong auxiliary output shape: expected {expected_aux_shape}, got {tuple(aux.shape)}"
            # the auxiliary output should normally sit around [0,1]; warn when it strays far
            if aux.min() < -0.1 or aux.max() > 1.1:
                logging.warning(f"Auxiliary output range is off: [{aux.min():.2f}, {aux.max():.2f}]")

        return True, "Model validation passed"

    except Exception as e:
        return False, f"Model forward pass failed: {str(e)}"


def validate_dataset(dataset, num_samples: int = 5) -> Tuple[bool, str]:
    """
    Validate the dataset format and that the key fields are correct
    Args:
      dataset: dataset object to validate, must support indexing
      num_samples: number of samples to spot-check, default 5
    Returns:
      Tuple[bool, str], whether validation passed plus a message
    """
    try:
        num_samples = min(num_samples, len(dataset))  # keep the sample count from exceeding the dataset size
        for i in range(num_samples):
            sample = dataset[i]
            # 'image' and 'mask' are always present; 'bucket' only when the dataset computes it on the
            # CPU (compute_bucket=True). Every shipped config sets bucket_on_gpu=true, so the datasets
            # are built with compute_bucket=False and carry no 'bucket' key -- requiring it there would
            # fail every valid dataset.
            required = ['image', 'mask'] + (['bucket'] if getattr(dataset, 'compute_bucket', False) else [])
            for key in required:
                if key not in sample:
                    return False, f"Sample is missing key: {key}"

            image = sample['image']
            mask = sample['mask']

            # each present field must be an np.ndarray or a torch.Tensor
            if not isinstance(image, (np.ndarray, torch.Tensor)):
                return False, f"Wrong image type: {type(image)}"
            if not isinstance(mask, (np.ndarray, torch.Tensor)):
                return False, f"Wrong mask type: {type(mask)}"

            if 'bucket' in sample:
                bucket = sample['bucket']
                if not isinstance(bucket, (np.ndarray, torch.Tensor)):
                    return False, f"Wrong bucket type: {type(bucket)}"
                # bucket should be a 1-D vector
                if len(bucket.shape) != 1:
                    return False, f"Wrong bucket dimensions: {bucket.shape}"
            # image should be 3-D (channel, height, width)
            if len(image.shape) != 3:
                return False, f"Wrong image dimensions: {image.shape}"
            # mask should be 2-D (height, width)
            if len(mask.shape) != 2:
                return False, f"Wrong mask dimensions: {mask.shape}"

        return True, f"Dataset validation passed ({num_samples} samples checked)"
    except Exception as e:
        return False, f"Dataset validation failed: {str(e)}"