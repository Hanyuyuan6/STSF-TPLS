"""AttnLift — cross-attention 1D→2D lift, used as the lift of the STSF main model.

Takes the temporal token sequence seq (B, T, gru_dim) coming out of the GRU; a 16×16 grid of learnable queries
cross-attends over those T tokens, so each spatial position aggregates the temporal information it needs, giving (B, init_feat_ch, hw, hw).

Chosen over the other lifts on the strength of the paper's lift ablation; the alternatives are all
implemented in LiftUNetPP, so the comparison can be re-run from this repo.
Shared by GRUUNetPP (the main line) and LiftUNetPP (the ablation framework).
Lineage: Perceiver / DETR style query-set cross-attention, the counterpart of the attention-based SPI reconstruction line
(Dual-Scale Transformer SPI, Hybrid-Attention CS).
"""

import torch
import torch.nn as nn


class _CrossBlock(nn.Module):
    """A single cross-attention block: query q attends to key/value kv (= GRU tokens) via MHA + residual + MLP (pre-norm)."""

    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.nq = nn.LayerNorm(dim)
        self.nk = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, dim))

    def forward(self, q, kv):
        kn = self.nk(kv)
        a, _ = self.attn(self.nq(q), kn, kn, need_weights=False)
        q = q + a
        return q + self.mlp(self.n2(q))


class AttnLift(nn.Module):
    """16×16 query grid ← cross-attention over the GRU's T temporal tokens. seq (B,T,gru_dim) -> (B,init_feat_ch,hw,hw)."""

    # `seq_pool` is INERT here and is named explicitly rather than left to **kwargs so that a config
    # setting it is visible in the signature: the other lifts pool the sequence down to a seed vector,
    # while this one cross-attends over the whole token sequence and has nothing to pool.
    def __init__(self, gru_dim, init_feat_hw, init_feat_ch, seq_pool='meanlast',
                 dim=288, depth=4, heads=8, mlp_ratio=4.0, dropout=0.1, **kwargs):
        super().__init__()
        self.hw = int(init_feat_hw)
        self.init_feat_ch = int(init_feat_ch)
        self.dim = int(dim)
        self.kv_proj = nn.Linear(gru_dim, dim)
        self.query = nn.Parameter(torch.randn(1, self.hw * self.hw, dim) * 0.02)
        self.tok_pos = nn.Parameter(torch.randn(1, 256, dim) * 0.02)   # temporal positional embedding (up to 256 steps)
        self.blocks = nn.ModuleList([_CrossBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Conv2d(dim, init_feat_ch, 1)

    def forward(self, seq):
        B, T, _ = seq.shape
        kv = self.kv_proj(seq) + self.tok_pos[:, :T]              # (B, T, dim)
        q = self.query.expand(B, -1, -1).contiguous()            # (B, hw*hw, dim)
        for blk in self.blocks:
            q = blk(q, kv)
        q = self.norm(q).transpose(1, 2).reshape(B, self.dim, self.hw, self.hw)
        return self.proj(q)
