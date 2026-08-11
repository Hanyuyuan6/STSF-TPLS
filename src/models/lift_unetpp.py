"""LiftUNetPP — skeleton for comparing 1D→2D lift modules (shared-GRU-front-end version).

Where it sits in the architecture (our choice): the interchangeable module goes **after the GRU and before ConvNeXt**,
i.e. the `meanlast pooling + seq_to_seed(Linear) + reshape + conv_lift(FSRCNN)` stretch of the original `gru_unetpp.py`.

- **shared front end** (identical for every variant): `bucket → input_proj → biGRU → seq (B, T, gru_dim=2*hidden)`.
- **swappable lift module**: `lift(seq) -> feat (B, init_feat_ch, init_feat_hw, init_feat_hw)`, (32,16,16) by default.
- **shared tail** (reused verbatim from GRUUNetPP): `ms_enhance(ConvNeXt) → attn_stack → adaptive_upsample → [aux] → UNet++`.

Since the GRU is shared, the swappable modules are matched to a parameter budget of ≈ 4.4M (= the original seq_to_seed 4.19M + conv_lift 0.17M).
Six variants: gru (reference) / srconv / attn / inr / mamba (SSM, 2024) / kan (Kolmogorov-Arnold, 2024).
"""

import math

import numpy as np
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
from .fsrcnn_blocks import FSRCNNBlock
from .attn_lift import AttnLift


# ============================================================================
# Utilities  — NOTE: SHARED TAIL, duplicated verbatim from src/models/gru_unetpp.py
#    (_compute_seq_len · _build_adaptive_upsample · _init_weights · the ms_enhance/attn/aux/UNet++ tail).
#    This model's lift='attn' path must stay numerically identical to GRUUNetPP so the released
#    lift ablation is fair. EDIT BOTH FILES TOGETHER — a silent divergence biases the ablation.
#    The invariant to preserve: built from the same config and seed, this model with lift='attn' and
#    GRUUNetPP must produce identical parameter counts and outputs. (The complementary invariant — that
#    the six lifts differ from each other — is what the lift ablation itself depends on.)
# ============================================================================
def _compute_seq_len(bucket_size: int) -> int:
    preferred = [64, 32, 16, 8]
    for s in preferred:
        if bucket_size % s == 0 and (bucket_size // s) >= 8:
            return s
    for s in range(128, 1, -1):
        if bucket_size % s == 0:
            return s
    return 1


def _pool(seq, mode='meanlast'):
    """(B, T, C) -> (B, C) or (B, 2C). meanlast = concatenate [mean, last state]."""
    if mode == 'meanlast':
        return torch.cat([seq.mean(dim=1), seq[:, -1]], dim=1)
    if mode == 'last':
        return seq[:, -1]
    return seq.mean(dim=1)


# ============================================================================
# lift #0: gru (reference) — pool + Linear seed + FSRCNN (= the original STSF lift, with the GRU factored out)
# ============================================================================
class GRULift(nn.Module):
    def __init__(self, gru_dim, init_feat_hw, init_feat_ch,
                 seq_pool='meanlast', seed_ch=16, dropout=0.1, dp_prob=0.0, **kwargs):
        super().__init__()
        self.hw = int(init_feat_hw)
        self.init_feat_ch = int(init_feat_ch)
        self.seq_pool = str(seq_pool)
        self.seed_ch = int(seed_ch)
        agg_dim = gru_dim * (2 if seq_pool == 'meanlast' else 1)
        seed_dim = self.seed_ch * self.hw * self.hw
        self.seq_to_seed = nn.Sequential(
            nn.Linear(agg_dim, seed_dim), nn.LayerNorm(seed_dim), nn.GELU(), nn.Dropout(dropout))
        self.conv_lift = FSRCNNBlock(in_channels=self.seed_ch, out_channels=self.init_feat_ch,
                                     d=56, s=12, m=4, upscale_factor=1, dropout_rate=dp_prob)

    def forward(self, seq):
        B = seq.size(0)
        seed = self.seq_to_seed(_pool(seq, self.seq_pool))
        feat = seed.view(B, self.seed_ch, self.hw, self.hw)
        return self.conv_lift(feat)


# ============================================================================
# lift #5: srconv — pool + Linear seed + RDB + PixelShuffle (a strong generic super-resolution control)
# ============================================================================
class _RDB(nn.Module):
    def __init__(self, ch, growth=32, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(ch + i * growth, growth, 3, 1, 1), nn.ReLU(inplace=True)))
        self.lff = nn.Conv2d(ch + n_layers * growth, ch, 1)

    def forward(self, x):
        feats = [x]
        for layer in self.layers:
            feats.append(layer(torch.cat(feats, dim=1)))
        return x + self.lff(torch.cat(feats, dim=1))


class SRConvLift(nn.Module):
    def __init__(self, gru_dim, init_feat_hw, init_feat_ch, seq_pool='meanlast',
                 base_hw=8, base_ch=56, n_rdb=2, growth=32, dropout=0.1, dp_prob=0.0, **kwargs):
        super().__init__()
        self.hw = int(init_feat_hw)
        self.init_feat_ch = int(init_feat_ch)
        self.seq_pool = str(seq_pool)
        self.base_hw = int(base_hw)
        self.base_ch = int(base_ch)
        assert self.hw % self.base_hw == 0
        agg_dim = gru_dim * (2 if seq_pool == 'meanlast' else 1)
        seed_dim = self.base_ch * self.base_hw * self.base_hw
        self.to_seed = nn.Sequential(
            nn.Linear(agg_dim, seed_dim), nn.LayerNorm(seed_dim), nn.GELU(), nn.Dropout(dropout))
        self.rdbs = nn.Sequential(*[_RDB(self.base_ch, growth) for _ in range(n_rdb)])
        ups, cur = [], self.base_hw
        while cur < self.hw:
            ups += [nn.Conv2d(self.base_ch, self.base_ch * 4, 3, 1, 1), nn.PixelShuffle(2), nn.ReLU(inplace=True)]
            cur *= 2
        self.upsample = nn.Sequential(*ups) if ups else nn.Identity()
        self.out_conv = nn.Conv2d(self.base_ch, self.init_feat_ch, 3, 1, 1)

    def forward(self, seq):
        B = seq.size(0)
        s = self.to_seed(_pool(seq, self.seq_pool)).view(B, self.base_ch, self.base_hw, self.base_hw)
        return self.out_conv(self.upsample(self.rdbs(s)))


# ============================================================================
# lift #3: attn — best in the pilot; moved out into attn_lift.py as the main model's lift and imported back here
# ============================================================================
# AttnLift / _CrossBlock live in attn_lift.py (the GRUUNetPP main line and this ablation framework share one implementation).


# ============================================================================
# lift #2: inr — pool→latent z; a coordinate MLP (Fourier features) renders point by point
# ============================================================================
class INRLift(nn.Module):
    def __init__(self, gru_dim, init_feat_hw, init_feat_ch, seq_pool='meanlast',
                 latent_dim=256, n_freq=16, hidden=960, depth=5, dropout=0.1, **kwargs):
        super().__init__()
        self.hw = int(init_feat_hw)
        self.init_feat_ch = int(init_feat_ch)
        self.latent_dim = int(latent_dim)
        self.n_freq = int(n_freq)
        self.seq_pool = str(seq_pool)
        agg_dim = gru_dim * (2 if seq_pool == 'meanlast' else 1)
        self.to_latent = nn.Sequential(
            nn.Linear(agg_dim, self.latent_dim), nn.LayerNorm(self.latent_dim), nn.GELU())
        lin = torch.linspace(-1, 1, self.hw)
        yy, xx = torch.meshgrid(lin, lin, indexing='ij')
        self.register_buffer('coords', torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1))
        self.register_buffer('freq_bands', (2.0 ** torch.arange(self.n_freq).float()) * float(np.pi))
        in_dim = self.latent_dim + 2 + 4 * self.n_freq
        layers = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, self.init_feat_ch)]
        self.mlp = nn.Sequential(*layers)

    def _posenc(self):
        c = self.coords
        xb = c[..., None] * self.freq_bands
        return torch.cat([c, torch.sin(xb).flatten(1), torch.cos(xb).flatten(1)], dim=-1)

    def forward(self, seq):
        B = seq.size(0)
        z = self.to_latent(_pool(seq, self.seq_pool))
        pe = self._posenc()
        P = pe.size(0)
        inp = torch.cat([z[:, None, :].expand(B, P, self.latent_dim), pe[None].expand(B, P, -1)], dim=-1)
        out = self.mlp(inp)
        return out.transpose(1, 2).reshape(B, self.init_feat_ch, self.hw, self.hw)


# ============================================================================
# lift #A: mamba — a selective state space (Mamba/Vision-Mamba, 2024) processes the GRU sequence, then lifts
# ============================================================================
class _MambaBlock(nn.Module):
    """Simplified Mamba (S6): in_proj → causal conv1d → SiLU → selective SSM (input-dependent Δ,B,C) → gating → out_proj.
    A pure-torch sequential scan (T≤64, so the cost is negligible). See Gu & Dao 2023 / Vision Mamba 2401.09417."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = int(d_state)
        self.dt_rank = max(1, d_model // 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        A = torch.arange(1, self.d_state + 1).float().repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _ssm(self, x):                                   # x: (B, L, d_inner)
        B, L, DI = x.shape
        A = -torch.exp(self.A_log)                       # (DI, d_state)
        dbl = self.x_proj(x)                             # (B, L, dt_rank+2*ds)
        dt, Bm, Cm = torch.split(dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                # (B, L, DI)
        h = x.new_zeros(B, DI, self.d_state)
        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)                          # (B,DI,ds)
            dBx = (dt[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1)) * x[:, t].unsqueeze(-1)  # (B,DI,ds)
            h = dA * h + dBx
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))                      # (B,DI)
        y = torch.stack(ys, dim=1)                        # (B, L, DI)
        return y + x * self.D

    def forward(self, x):                                 # (B, L, d_model)
        L = x.size(1)
        xz = self.in_proj(x)
        xx, z = xz.chunk(2, dim=-1)
        xx = self.conv1d(xx.transpose(1, 2))[..., :L].transpose(1, 2)
        xx = F.silu(xx)
        y = self._ssm(xx) * F.silu(z)
        return self.out_proj(y)


class MambaLift(nn.Module):
    def __init__(self, gru_dim, init_feat_hw, init_feat_ch, seq_pool='meanlast',
                 d_model=320, n_blocks=2, seed_ch=16, dropout=0.1, dp_prob=0.0, **kwargs):
        super().__init__()
        self.hw = int(init_feat_hw)
        self.init_feat_ch = int(init_feat_ch)
        self.seq_pool = str(seq_pool)
        self.seed_ch = int(seed_ch)
        self.in_proj = nn.Linear(gru_dim, d_model)
        self.blocks = nn.ModuleList([_MambaBlock(d_model) for _ in range(n_blocks)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_blocks)])
        agg_dim = d_model * (2 if seq_pool == 'meanlast' else 1)
        seed_dim = self.seed_ch * self.hw * self.hw
        self.to_seed = nn.Sequential(
            nn.Linear(agg_dim, seed_dim), nn.LayerNorm(seed_dim), nn.GELU(), nn.Dropout(dropout))
        self.conv_lift = FSRCNNBlock(in_channels=self.seed_ch, out_channels=self.init_feat_ch,
                                     d=56, s=12, m=4, upscale_factor=1, dropout_rate=dp_prob)

    def forward(self, seq):
        B = seq.size(0)
        x = self.in_proj(seq)
        for blk, nrm in zip(self.blocks, self.norms):
            x = x + blk(nrm(x))
        seed = self.to_seed(_pool(x, self.seq_pool)).view(B, self.seed_ch, self.hw, self.hw)
        return self.conv_lift(seed)


# ============================================================================
# lift #B: kan — a Kolmogorov-Arnold (learnable B-spline edges, 2024) lift bottleneck
# ============================================================================
class _KANLinear(nn.Module):
    """An efficient-kan style KAN linear layer: a base (SiLU) path + a B-spline path (fixed grid). See Liu et al. 2404.19756."""

    def __init__(self, in_f, out_f, grid_size=5, spline_order=3, grid_range=(-1.0, 1.0)):
        super().__init__()
        self.in_f, self.out_f = int(in_f), int(out_f)
        self.grid_size, self.spline_order = int(grid_size), int(spline_order)
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1).float() * h + grid_range[0])
        self.register_buffer('grid', grid.expand(self.in_f, -1).contiguous())  # (in_f, G+2o+1)
        self.base_weight = nn.Parameter(torch.empty(self.out_f, self.in_f))
        self.spline_weight = nn.Parameter(torch.empty(self.out_f, self.in_f, grid_size + spline_order))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, 0.0, 0.1)
        self.base_act = nn.SiLU()

    def _b_splines(self, x):                              # x: (B, in_f) -> (B, in_f, G+order)
        g = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= g[:, :-1]) & (x < g[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - g[:, :-(k + 1)]) / (g[:, k:-1] - g[:, :-(k + 1)]) * bases[:, :, :-1]) + \
                    ((g[:, k + 1:] - x) / (g[:, k + 1:] - g[:, 1:-k]) * bases[:, :, 1:])
        return bases

    def forward(self, x):                                 # (B, in_f) -> (B, out_f)
        base = F.linear(self.base_act(x), self.base_weight)
        spline = torch.einsum('bik,oik->bo', self._b_splines(x), self.spline_weight)
        return base + spline


class KANLift(nn.Module):
    def __init__(self, gru_dim, init_feat_hw, init_feat_ch, seq_pool='meanlast',
                 kan_dim=448, seed_ch=16, dropout=0.1, dp_prob=0.0, **kwargs):
        super().__init__()
        self.hw = int(init_feat_hw)
        self.init_feat_ch = int(init_feat_ch)
        self.seq_pool = str(seq_pool)
        self.seed_ch = int(seed_ch)
        agg_dim = gru_dim * (2 if seq_pool == 'meanlast' else 1)
        self.down = nn.Linear(agg_dim, kan_dim)
        self.kan = _KANLinear(kan_dim, kan_dim)
        seed_dim = self.seed_ch * self.hw * self.hw
        self.up = nn.Linear(kan_dim, seed_dim)
        self.conv_lift = FSRCNNBlock(in_channels=self.seed_ch, out_channels=self.init_feat_ch,
                                     d=56, s=12, m=4, upscale_factor=1, dropout_rate=dp_prob)

    def forward(self, seq):
        B = seq.size(0)
        h = torch.tanh(self.down(_pool(seq, self.seq_pool)))   # squash into [-1,1] to line up with the KAN grid
        h = self.kan(h)
        seed = self.up(h).view(B, self.seed_ch, self.hw, self.hw)
        return self.conv_lift(seed)


# ============================================================================
# registry + factory
# ============================================================================
LIFT_REGISTRY = {
    'gru': GRULift,
    'srconv': SRConvLift,
    'attn': AttnLift,
    'inr': INRLift,
    'mamba': MambaLift,
    'kan': KANLift,
}


def build_lift(lift_type, gru_dim, init_feat_hw, init_feat_ch, **lift_kwargs):
    if lift_type not in LIFT_REGISTRY:
        raise ValueError(f"unknown lift_type: {lift_type}; available: {list(LIFT_REGISTRY)}")
    return LIFT_REGISTRY[lift_type](gru_dim, init_feat_hw, init_feat_ch, **lift_kwargs)


# ============================================================================
# Skeleton: shared GRU front end + swappable lift + shared tail
# ============================================================================
class LiftUNetPP(BaseSegmentationModel):
    input_type = 'bucket'

    def __init__(self,
                 bucket_size,
                 img_size=128,
                 classes=1,
                 init_feat_hw=16,
                 init_feat_ch=32,
                 bilinear=True,
                 hidden_size=256,
                 num_layers=3,
                 seq_pool='meanlast',
                 dropout=0.1,
                 lift_type='gru',
                 lift_kwargs=None,
                 use_se=False,
                 use_cbam=False,
                 use_eca=True,
                 use_scse=True,
                 use_multi_scale=True,
                 use_aux_recon_head=False,
                 ms_blocks=2,
                 dp_prob=0.0,
                 upsample_target_ratio=0.25,
                 upsample_target_size=None,
                 **kwargs):
        super().__init__(img_size, classes)
        self.bucket_size = int(bucket_size)
        self.img_size = int(img_size)
        self.classes = int(classes)
        self.init_feat_hw = int(init_feat_hw)
        self.init_feat_ch = int(init_feat_ch)
        self.lift_type = str(lift_type)
        self.seq_pool = str(seq_pool)
        self.use_multi_scale = bool(use_multi_scale)
        self.use_aux = bool(use_aux_recon_head)
        self.ms_blocks = int(ms_blocks)
        self.use_se = bool(use_se)
        self.use_cbam = bool(use_cbam)
        self.use_eca = bool(use_eca)
        self.use_scse = bool(use_scse)
        self.dp_prob = float(dp_prob)
        self.upsample_target_ratio = float(upsample_target_ratio)
        self.upsample_target_size = int(upsample_target_size) if upsample_target_size is not None else None

        # ---- shared front end: input_proj + biGRU ----
        self.seq_len = _compute_seq_len(self.bucket_size)
        self.input_size = self.bucket_size // self.seq_len
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_size, hidden_size), nn.LayerNorm(hidden_size),
            nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0, bidirectional=True)
        gru_dim = hidden_size * 2

        # ---- swappable lift module ----
        lk = dict(lift_kwargs) if lift_kwargs else {}
        lk.setdefault('seq_pool', self.seq_pool)
        lk.setdefault('dropout', dropout)
        lk.setdefault('dp_prob', self.dp_prob)
        self.lift = build_lift(self.lift_type, gru_dim, self.init_feat_hw, self.init_feat_ch, **lk)

        # ---- shared tail (reused verbatim from GRUUNetPP) ----
        ms_modules = [ConvNeXtLiteBlock(self.init_feat_ch, drop_path=self.dp_prob)
                      for _ in range(self.ms_blocks)] if self.use_multi_scale else []
        self.ms_enhance = nn.Sequential(*ms_modules) if ms_modules else nn.Identity()

        attn_modules = []
        if self.use_se:
            attn_modules.append(SEBlock(self.init_feat_ch))
        if self.use_cbam:
            attn_modules.append(CBAMBlock(self.init_feat_ch))
        if self.use_eca:
            attn_modules.append(ResidualECA(self.init_feat_ch, drop_prob=self.dp_prob))
        if self.use_scse:
            attn_modules.append(ResidualSCSE(self.init_feat_ch, drop_prob=self.dp_prob))
        self.attn_stack = nn.Sequential(*attn_modules) if attn_modules else nn.Identity()

        self.adaptive_upsample = self._build_adaptive_upsample()

        if self.use_aux:
            self.aux_head = nn.Sequential(
                nn.Conv2d(self.init_feat_ch, self.init_feat_ch // 2, 3, 1, 1),
                nn.BatchNorm2d(self.init_feat_ch // 2), nn.ReLU(inplace=True),
                nn.Conv2d(self.init_feat_ch // 2, 1, 1), nn.Sigmoid())
        else:
            self.aux_head = None

        self.head = UNetPP(in_ch=self.init_feat_ch, base=64, num_classes=classes, bilinear=bilinear,
                           use_se=False, use_cbam=False, use_aspp=False, droppath_prob=0.0)

        self._init_weights()

    def _build_adaptive_upsample(self):
        modules = []
        current = self.init_feat_hw
        if self.upsample_target_size is not None:
            target = max(self.init_feat_hw, int(self.upsample_target_size))
        else:
            target = max(4, int(round(self.img_size * self.upsample_target_ratio)))
            target = max(target, self.init_feat_hw)
        while current < target:
            modules += [nn.ConvTranspose2d(self.init_feat_ch, self.init_feat_ch, 4, 2, 1),
                        nn.BatchNorm2d(self.init_feat_ch), nn.ReLU(inplace=True)]
            current *= 2
        return nn.Sequential(*modules) if modules else nn.Identity()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for p in m.parameters():
                    if p.ndim >= 2:
                        nn.init.orthogonal_(p)
                    else:
                        nn.init.normal_(p, std=0.01)

    def forward(self, bucket):
        B = bucket.size(0)
        x = bucket.view(B, self.seq_len, self.input_size)
        x = self.input_proj(x)
        seq, _ = self.gru(x)                              # (B, T, gru_dim)

        feat = self.lift(seq)                             # (B, init_feat_ch, hw, hw)
        if self.use_multi_scale:
            feat = self.ms_enhance(feat)
        feat = self.attn_stack(feat)
        feat = self.adaptive_upsample(feat)

        aux = None
        if self.aux_head is not None:
            aux_feat = F.interpolate(feat, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
            aux = self.aux_head(aux_feat)

        logits = self.head(feat)
        if logits.size(-1) != self.img_size:
            logits = F.interpolate(logits, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        return {'logits': logits, 'aux_recon': aux}
