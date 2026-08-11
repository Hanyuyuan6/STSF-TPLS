import torch.nn as nn
from abc import ABC, abstractmethod

class BaseSegmentationModel(nn.Module, ABC):
    """
    Abstract base class defining the common interface and attributes of a segmentation model.
    - input_type: the model's input type, either 'bucket' or 'image'.
    - forward() should return a dict containing:
      'logits': the segmentation prediction produced by the network, of shape (B, C, H, W);
      'aux_recon': the auxiliary reconstruction, of shape (B, 1, H, W), or None.
    """
    input_type: str = 'bucket'  # default input type is 'bucket'

    def __init__(self, img_size, num_classes):
        super().__init__()  # call the initializer of the parent nn.Module
        self.img_size = img_size  # store the input image size (width or height, assumed square)
        self.num_classes = num_classes  # segmentation classes, background included

    @abstractmethod
    def forward(self, x): ...
    # abstract method that every subclass must implement: the forward pass, taking x and returning a dict