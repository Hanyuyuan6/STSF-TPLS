"""BaselineUNetPP - conventional-pipeline baseline model"""

import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseSegmentationModel
from .unetpp_blocks import UNetPP


class BaselineUNetPP(BaseSegmentationModel):
    """
    Conventional-pipeline baseline: feed the reconstructed image straight into segmentation
    Used to assess how well the conventional reconstruction + segmentation pipeline performs
    """

    input_type = 'image'

    def __init__(self,
                 img_size=128,
                 classes=1,
                 in_ch=1,
                 bilinear=True,
                 # NOTE: use_se / use_cbam / use_aspp / droppath_prob are declared by every ta_* and
                 # rev_*_traditional config but land in **kwargs here -- this model hardcodes them to
                 # off below. They ARE live in GRUUNetPP / LiftUNetPP, which is what makes the name
                 # collision a trap. Every shipped config sets them to the same values hardcoded here,
                 # so no released run differs; changing them in a config would do nothing.
                 **kwargs):  # ignore the remaining arguments
        super().__init__(img_size, classes)

        # a simple input adaptation layer
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # the UNet++ body (without any of the advanced modules)
        self.head = UNetPP(
            in_ch=64,
            base=64,
            num_classes=classes,
            bilinear=bilinear,
            use_se=False,  # the baseline model uses no SE
            use_cbam=False,  # the baseline model uses no CBAM
            use_aspp=False,  # the baseline model uses no ASPP
            droppath_prob=0  # the baseline model uses no DropPath
        )

    def forward(self, image):
        """
        Forward pass
        Args:
            image: (B, 1, H, W) the reconstructed image
        Returns:
            dict: contains 'logits' and 'aux_recon' (None)
        """
        x = self.stem(image)
        logits = self.head(x)

        if logits.size(-1) != self.img_size:
            logits = F.interpolate(
                logits,
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False
            )

        return {'logits': logits, 'aux_recon': None}