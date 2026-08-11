"""
FCNUNetPP - the SPIFS baseline network (Liu et al., "Image-free single-pixel
segmentation", Opt. Laser Technol. 157:108600, 2023), re-implemented here so the
comparison runs under one shared acquisition and optimization protocol.
It follows a three-stage structure: Encoder + FSRCNN + UNet++
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseSegmentationModel
from .unetpp_blocks import UNetPP
from .fsrcnn_blocks import FSRCNNBlock


class FCNUNetPP(BaseSegmentationModel):
    """
    Single-pixel imaging semantic segmentation network (paper-architecture version)

    Network structure:
    1. Encoder: encodes the bucket signal into 2D features
    2. FSRCNN: feature extraction and enhancement
    3. Auxiliary reconstruction head: intermediate supervision signal
    4. UNet++: the segmentation network
    """

    input_type = 'bucket'

    def __init__(self,
                 bucket_size,
                 img_size=128,
                 classes=1,
                 init_feat_hw=None,
                 init_feat_ch=None,
                 fsrcnn_d=56,
                 fsrcnn_s=12,
                 fsrcnn_m=4,
                 unetpp_in_ch=64,
                 unetpp_base=64,
                 bilinear=True,
                 droppath_prob=0.0,
                 use_aux_recon_head=False,
                 # NOTE: use_se / use_cbam / use_aspp are declared by the rev_*_fcn configs but land
                 # in **kwargs here -- this model hardcodes them to off below. They ARE live in
                 # GRUUNetPP / LiftUNetPP. Every shipped config sets them to the hardcoded values, so
                 # no released run differs; changing them in a config would do nothing.
                 **kwargs):
        super().__init__(img_size, classes)

        if init_feat_hw is None:
            init_feat_hw = 16 if img_size <= 128 else 32
        if init_feat_ch is None:
            init_feat_ch = 1

        self.bucket_size = bucket_size
        self.init_feat_hw = init_feat_hw
        self.init_feat_ch = init_feat_ch
        self.use_aux = use_aux_recon_head

        self.init_feat_dim = init_feat_ch * init_feat_hw * init_feat_hw

        # 1. Encoder
        self.encoder = nn.Sequential(
            nn.Linear(bucket_size, self.init_feat_dim),
            nn.LayerNorm(self.init_feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # 2. Spatial Upsample
        target_hw = min(img_size // 2, 128)
        if init_feat_hw < target_hw:
            scale = target_hw // init_feat_hw
            self.spatial_upsample = nn.Sequential(
                nn.ConvTranspose2d(init_feat_ch, 16, kernel_size=4, stride=scale, padding=scale // 2),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            )
            fsrcnn_in_ch = 16
            fsrcnn_in_hw = target_hw
        else:
            self.spatial_upsample = nn.Identity()
            fsrcnn_in_ch = init_feat_ch
            fsrcnn_in_hw = init_feat_hw

        # 3. FSRCNN feature extraction
        final_upscale = img_size // fsrcnn_in_hw
        self.feature_extraction = FSRCNNBlock(
            in_channels=fsrcnn_in_ch,
            out_channels=unetpp_in_ch,
            d=fsrcnn_d,
            s=fsrcnn_s,
            m=fsrcnn_m,
            upscale_factor=max(1, final_upscale),
            dropout_rate=droppath_prob
        )

        # 4. auxiliary reconstruction head (placed after FSRCNN, before UNet++)
        if self.use_aux:
            self.aux_head = nn.Sequential(
                nn.Conv2d(unetpp_in_ch, unetpp_in_ch // 2, 3, 1, 1),
                nn.BatchNorm2d(unetpp_in_ch // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(unetpp_in_ch // 2, 1, 1),
                nn.Sigmoid()
            )
        else:
            self.aux_head = None

        # 5. UNet++ segmentation network (simplified, without the advanced modules)
        self.segmentation = UNetPP(
            in_ch=unetpp_in_ch,
            base=unetpp_base,
            num_classes=classes,
            bilinear=bilinear,
            use_se=False,      # no SE inside FCN
            use_cbam=False,    # no CBAM inside FCN
            use_aspp=False,    # no ASPP inside FCN
            droppath_prob=0
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, bucket):
        B = bucket.size(0)

        # 1. encode
        feat = self.encoder(bucket)
        feat = feat.view(B, self.init_feat_ch, self.init_feat_hw, self.init_feat_hw)

        # 2. spatial upsampling
        feat = self.spatial_upsample(feat)

        # 3. FSRCNN feature extraction
        feat = self.feature_extraction(feat)

        # 4. auxiliary reconstruction output (intermediate supervision)
        aux = None
        if self.aux_head is not None:
            aux = self.aux_head(feat)
            if aux.size(-1) != self.img_size:
                aux = F.interpolate(aux, size=(self.img_size, self.img_size),
                                  mode='bilinear', align_corners=False)

        # 5. UNet++ segmentation
        logits = self.segmentation(feat)
        if logits.size(-1) != self.img_size:
            logits = F.interpolate(logits, size=(self.img_size, self.img_size),
                                  mode='bilinear', align_corners=False)

        return {'logits': logits, 'aux_recon': aux}