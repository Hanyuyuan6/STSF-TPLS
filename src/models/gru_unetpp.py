import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseSegmentationModel
from .unetpp_blocks import (
    UNetPP,
    SEBlock, CBAMBlock,
    ConvNeXtLiteBlock,
    ResidualECA, ResidualSCSE,
)
from .attn_lift import AttnLift


class GRUUNetPP(BaseSegmentationModel):
    input_type = 'bucket'

    def __init__(self,
                 bucket_size,
                 img_size=128,
                 classes=1,
                 hidden_size=256,
                 num_layers=3,
                 init_feat_hw=16,
                 init_feat_ch=32,
                 bilinear=True,
                 use_se=False,
                 use_cbam=False,
                 use_eca=True,
                 use_scse=True,
                 use_multi_scale=True,
                 use_residual=True,
                 use_aux_recon_head=False,
                 dropout=0.1,
                 ms_blocks=2,
                 dp_prob=0.0,
                 upsample_target_ratio=0.25,
                 upsample_target_size=None,
                 seq_pool='meanlast',
                 seed_ch=16,
                 attn_dim=288,
                 attn_depth=4,
                 attn_heads=8,
                 **kwargs):
        super().__init__(img_size, classes)

        self.bucket_size = int(bucket_size)                        # bucket size (total sequence length)
        self.img_size = int(img_size)                              # target image size (height equals width)
        self.classes = int(classes)                                # number of classes
        self.init_feat_hw = int(init_feat_hw)                      # height/width of the initial feature map
        self.init_feat_ch = int(init_feat_ch)                      # number of initial feature channels
        self.use_multi_scale = bool(use_multi_scale)               # whether to use the multi-scale enhancement module
        # ⚠️ INERT / legacy key, not a knob: no post-hoc residual gate is implemented and no code in this class reads the attribute;
        # flipping it builds a bit-identical network (measured: use_residual=False vs True → 36,843,495 params, identical state_dict key
        # hashes; as a control, flipping use_scse does change the param count → the probe is valid). 29 configs declare
        # `use_residual: false`, the headline rev_*_tpls among them — they match training exactly, so **no published number is affected**;
        # the key is still accepted because the config embedded in the released checkpoints carries it, so dropping it on the yaml side
        # would make the two disagree. Accepting it explicitly (rather than letting **kwargs swallow it silently) is deliberate: a config
        # key quietly absorbed by **kwargs is how an architecture mismatch goes unnoticed until the numbers are already published.
        self.use_residual = bool(use_residual)                     # INERT: see above; no gate exists
        self.use_aux = bool(use_aux_recon_head)                    # whether to use the auxiliary reconstruction head
        self.ms_blocks = int(ms_blocks)                            # number of multi-scale blocks
        self.use_se = bool(use_se)                                 # whether to use the SE attention module
        self.use_cbam = bool(use_cbam)                             # whether to use the CBAM attention module
        self.use_eca = bool(use_eca)                               # whether to use the ECA attention module
        self.use_scse = bool(use_scse)                             # whether to use the SCSE attention module
        self.dp_prob = float(dp_prob)                              # DropPath probability
        self.upsample_target_ratio = float(upsample_target_ratio) # upsampling ratio
        self.upsample_target_size = int(upsample_target_size) if upsample_target_size is not None else None  # upsampling target size
        # ⚠️ INERT in this class (same status as use_residual above): both belonged to the original
        # meanlast + Linear + FSRCNN lift. AttnLift consumes the full token sequence and builds its own
        # query grid, so neither attribute is read after this line — setting them in a config changes
        # nothing here. They ARE live in LiftUNetPP, which still uses the seed-then-conv lifts.
        self.seq_pool = str(seq_pool)                              # INERT here; live in LiftUNetPP
        self.seed_ch = int(seed_ch)                                # INERT here; live in LiftUNetPP

        self.seq_len = self._compute_seq_len(self.bucket_size)    # compute the sequence length (number of GRU time steps)
        self.input_size = self.bucket_size // self.seq_len        # input feature length per time step
        feat_dim = self.init_feat_ch * self.init_feat_hw * self.init_feat_hw  # feature dimension after flattening

        # input projection layer, mapping the input of each time step to the hidden dimension
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),    # linear transform
            nn.LayerNorm(hidden_size),                   # normalization
            nn.GELU(),                                  # activation
            nn.Dropout(dropout)                         # dropout to prevent overfitting
        )

        # bidirectional GRU encoder
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True  # bidirectional GRU
        )
        gru_out_dim = hidden_size * 2  # output dimension of the bidirectional GRU

        # 1D->2D lift: cross-attention (16x16 query grid ← the GRU's T temporal tokens), replacing the
        # original meanlast + Linear + FSRCNN. It was chosen on the strength of the paper's lift ablation;
        # the original FSRCNN lift is kept in LiftUNetPP(lift='gru') for ablation / baseline reproduction.
        self.lift = AttnLift(
            gru_out_dim, self.init_feat_hw, self.init_feat_ch,
            dim=attn_dim, depth=attn_depth, heads=attn_heads, dropout=dropout,
        )

        # multi-scale enhancement module, a stack of ConvNeXtLiteBlocks
        ms_modules = []
        if self.use_multi_scale:
            for _ in range(self.ms_blocks):
                ms_modules.append(ConvNeXtLiteBlock(self.init_feat_ch, drop_path=self.dp_prob))
        self.ms_enhance = nn.Sequential(*ms_modules) if self.use_multi_scale else nn.Identity()

        # attention stack, supporting any combination of SE, CBAM, ECA and SCSE
        attn_modules = []
        if self.use_se:
            attn_modules.append(SEBlock(self.init_feat_ch))
        if self.use_cbam:
            attn_modules.append(CBAMBlock(self.init_feat_ch))
        if self.use_eca:
            attn_modules.append(ResidualECA(self.init_feat_ch, drop_prob=self.dp_prob))
        if self.use_scse:
            attn_modules.append(ResidualSCSE(self.init_feat_ch, drop_prob=self.dp_prob))
        self.attn_stack = nn.Sequential(*attn_modules) if len(attn_modules) > 0 else nn.Identity()

        # build the adaptive upsampling module, doubling the feature-map size in a loop until the target size is reached
        self.adaptive_upsample = self._build_adaptive_upsample()

        # optional auxiliary reconstruction head, used to assist training
        if self.use_aux:
            self.aux_head = nn.Sequential(
                nn.Conv2d(self.init_feat_ch, self.init_feat_ch // 2, 3, 1, 1),  # convolution reducing the channels
                nn.BatchNorm2d(self.init_feat_ch // 2),                         # normalization
                nn.ReLU(inplace=True),                                          # activation
                nn.Conv2d(self.init_feat_ch // 2, 1, 1),                        # output 1 channel
                nn.Sigmoid()                                                   # Sigmoid normalizing to [0,1]
            )
        else:
            self.aux_head = None

        # U-Net++ backbone segmentation head
        self.head = UNetPP(
            in_ch=self.init_feat_ch,
            base=64,
            num_classes=classes,
            bilinear=bilinear,
            use_se=False,
            use_cbam=False,
            use_aspp=False,
            droppath_prob=0.0
        )

        self._init_weights()  # weight initialization

    # ⚠️ SHARED TAIL — duplicated verbatim in src/models/lift_unetpp.py
    #    (_compute_seq_len · _build_adaptive_upsample · _init_weights · the ms_enhance/attn/aux/UNet++ tail).
    #    LiftUNetPP(lift='attn') must stay numerically identical to this model so the lift ablation
    #    (paper Table II) is a fair, capacity-matched comparison. EDIT BOTH FILES TOGETHER — a silent
    #    divergence here biases the ablation. The invariant to preserve: built from the same config and seed,
    #    LiftUNetPP(lift='attn') and this model must produce identical parameter counts and outputs.
    # compute the GRU sequence length, preferring the fixed common values, so bucket_size divides evenly and the sequence length stays sensible
    def _compute_seq_len(self, bucket_size: int) -> int:
        preferred = [64, 32, 16, 8]
        for s in preferred:
            if bucket_size % s == 0 and (bucket_size // s) >= 8:
                return s
        for s in range(128, 1, -1):
            if bucket_size % s == 0:
                return s
        return 1

    # build the adaptive upsampling module, enlarging the feature map to the target size step by step with transposed convolutions
    def _build_adaptive_upsample(self):
        modules = []
        current = self.init_feat_hw
        if self.upsample_target_size is not None:
            target = max(self.init_feat_hw, int(self.upsample_target_size))
        else:
            target = max(4, int(round(self.img_size * self.upsample_target_ratio)))
            target = max(target, self.init_feat_hw)
        while current < target:
            modules.extend([
                nn.ConvTranspose2d(self.init_feat_ch, self.init_feat_ch, 4, 2, 1),  # 2x upsampling convolution kernel
                nn.BatchNorm2d(self.init_feat_ch),                                  # normalization
                nn.ReLU(inplace=True)                                               # activation
            ])
            current *= 2
        return nn.Sequential(*modules) if modules else nn.Identity()

    # weight initialization, using a different scheme per layer type
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)           # Xavier uniform initialization of the linear weights
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)            # biases initialized to 0
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Kaiming normal initialization of the convolution layers
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for p in m.parameters():
                    if p.ndim >= 2:
                        nn.init.orthogonal_(p)               # orthogonal initialization of the GRU weights
                    else:
                        nn.init.normal_(p, std=0.01)        # normal initialization of the biases

    # forward pass: takes the bucket sequence, returns the segmentation logits and the auxiliary reconstruction (if enabled)
    def forward(self, bucket):
        B = bucket.size(0)                              # batch size

        x = bucket.view(B, self.seq_len, self.input_size)  # reshape to (B, sequence length, input length per step)
        x = self.input_proj(x)                            # input projection to the hidden dimension
        x, _ = self.gru(x)                                # GRU encoding, output (B, sequence length, 2*hidden_size)

        # 1D->2D lift: cross-attention (a 16x16 query grid ← the GRU's T temporal tokens) -> (B, init_feat_ch, hw, hw)
        feat = self.lift(x)

        if self.use_multi_scale:
            feat = self.ms_enhance(feat)                     # processed by the multi-scale enhancement module
        feat = self.attn_stack(feat)                         # processed by the attention stack

        feat = self.adaptive_upsample(feat)                  # adaptive upsampling to the target size

        aux = None
        if self.aux_head is not None:
            aux_feat = F.interpolate(feat, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)  # bilinear interpolation back to the original image size
            aux = self.aux_head(aux_feat)                     # output of the auxiliary head

        logits = self.head(feat)                              # output of the U-Net++ backbone segmentation head

        if logits.size(-1) != self.img_size:
            logits = F.interpolate(logits, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)  # make sure the output size matches

        return {'logits': logits, 'aux_recon': aux}           # return the segmentation result and the auxiliary reconstruction (may be None)