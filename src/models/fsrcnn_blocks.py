"""
FSRCNN (Fast Super-Resolution CNN) feature extraction module
Feature extraction block designed after the paper "Image-free single-pixel segmentation"
"""

import torch
import torch.nn as nn


class FSRCNNBlock(nn.Module):
    """
    FSRCNN feature extraction block, made of five stages:
    1. feature extraction (5×5 convolution)
    2. shrinking (1×1 convolution reducing the number of channels)
    3. non-linear mapping (several 3×3 convolution layers, with optional attention and dropout)
    4. expanding (1×1 convolution restoring the number of channels)
    5. deconvolution (upsampling, or a plain convolution that keeps the size)
    Reference: Dong et al. "Accelerating the Super-Resolution Convolutional Neural Network"
    """

    def __init__(self,
                 in_channels=1,
                 out_channels=64,
                 d=56,
                 s=12,
                 m=4,
                 upscale_factor=1,
                 use_se=False,
                 use_cbam=False,
                 dropout_rate=0.0):
        super(FSRCNNBlock, self).__init__()

        # check that the arguments are valid
        assert in_channels >= 1, f"the number of input channels must be >=1, got {in_channels}"
        assert out_channels >= 1, f"the number of output channels must be >=1, got {out_channels}"
        assert d >= 1, f"the feature dimension d must be >=1, got {d}"
        assert s >= 1, f"the shrinking dimension s must be >=1, got {s}"
        assert m >= 1, f"the number of mapping layers m must be >=1, got {m}"

        self.in_channels = in_channels
        self.out_channels = out_channels

        # 1. feature extraction layer: 5x5 convolution to d channels, keeping the spatial size, followed by BatchNorm and PReLU
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(d),
            nn.PReLU()
        )

        # 2. shrinking layer: 1x1 convolution down to s channels to cut the compute, followed by BatchNorm and PReLU
        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1, bias=False),
            nn.BatchNorm2d(s),
            nn.PReLU()
        )

        # 3. non-linear mapping layers: m 3x3 convolution blocks, each a convolution, BatchNorm and PReLU
        mapping_layers = []
        for i in range(m):
            mapping_layers.extend([
                nn.Conv2d(s, s, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(s),
                nn.PReLU()
            ])

            # insert an attention module (SE or CBAM) at the middle layer
            if i == m // 2:
                if use_se:
                    from .unetpp_blocks import SEBlock
                    mapping_layers.append(SEBlock(s))
                elif use_cbam:
                    from .unetpp_blocks import CBAMBlock
                    mapping_layers.append(CBAMBlock(s))

            # for every layer but the last, add Dropout2d according to dropout_rate to prevent overfitting
            if dropout_rate > 0 and i < m - 1:
                mapping_layers.append(nn.Dropout2d(dropout_rate))

        self.mapping = nn.Sequential(*mapping_layers)

        # 4. expanding layer: 1x1 convolution restoring the channels back to d, followed by BatchNorm and PReLU
        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1, bias=False),
            nn.BatchNorm2d(d),
            nn.PReLU()
        )

        # 5. deconvolution layer: if upscale_factor>1 a transposed convolution does the upsampling; otherwise a plain convolution keeps the size
        if upscale_factor > 1:
            self.deconvolution = nn.ConvTranspose2d(
                d, out_channels,
                kernel_size=9,
                stride=upscale_factor,
                padding=4,
                output_padding=upscale_factor - 1,
                bias=False
            )
        else:
            self.deconvolution = nn.Conv2d(
                d, out_channels,
                kernel_size=9,
                padding=4,
                bias=False
            )

        # the final batch normalization and ReLU activation
        self.final_bn = nn.BatchNorm2d(out_channels)
        self.final_act = nn.ReLU(inplace=True)

        # weight initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize the weights of the convolution and normalization layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.numel() > 0:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None and m.bias.numel() > 0:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                if m.weight.numel() > 0:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None and m.bias.numel() > 0:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None and m.weight.numel() > 0:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None and m.bias.numel() > 0:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        The forward pass

        Input:
            x: the input feature map, of shape (B, C, H, W)

        Returns:
            the output feature map, of shape (B, out_channels, H', W')
        """
        # make sure the input is a 4-D tensor and that the channels match
        assert x.dim() == 4, f"the input must be a 4-D tensor, current number of dimensions: {x.dim()}"
        assert x.size(1) == self.in_channels, \
            f"input channel mismatch: expected {self.in_channels}, got {x.size(1)}"

        # run the five FSRCNN stages in order
        x = self.feature_extraction(x)  # feature extraction
        x = self.shrinking(x)           # channel shrinking
        x = self.mapping(x)             # non-linear mapping
        x = self.expanding(x)           # channel expanding
        x = self.deconvolution(x)       # deconvolution / upsampling
        x = self.final_bn(x)            # batch normalization
        x = self.final_act(x)           # ReLU activation

        return x