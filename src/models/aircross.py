"""
AirCross (Phase A): meteorology-conditioned attention model for hourly stack NOx.

Input : (B, T, F) window of T consecutive overpass slots with F features
        (ERA5 weather, calendar encodings, and optionally plant gross load).
Output: (B,) predicted log1p(NOx kg/h) at the last slot.

Architecture
1. Linear token projection + learned position embedding.
2. Transformer encoder (self-attention over the T slots).
3. Meteorology-conditioned cross-attention pooling: the query is a projection of the
   target-slot meteorology only (wind u, v, speed, pressure, temperature, boundary-layer
   height), the keys/values are the encoded tokens. With `use_cross_attention=False`
   (ablation) the encoded last token is used directly.
4. MLP regression head.

`forward(..., return_attention=True)` also returns the cross-attention weights (B, T).
"""

import torch
import torch.nn as nn


class AirCross(nn.Module):
    def __init__(self, n_features: int, met_idx: list[int], seq_len: int, embed_dim: int = 64,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1,
                 use_cross_attention: bool = True):
        super().__init__()
        self.met_idx = list(met_idx)
        self.use_cross_attention = use_cross_attention
        self.proj = nn.Linear(n_features, embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        layer = nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim * 4, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.met_query = nn.Linear(len(self.met_idx), embed_dim)
        self.xattn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(embed_dim, 1))

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        tokens = self.encoder(self.proj(x) + self.pos)
        attn_w = None
        if self.use_cross_attention:
            q = self.met_query(x[:, -1:, self.met_idx])            # (B, 1, D)
            pooled, attn_w = self.xattn(q, tokens, tokens, need_weights=True)
            h = self.norm(pooled.squeeze(1) + tokens[:, -1])
            attn_w = attn_w.squeeze(1)
        else:
            h = self.norm(tokens[:, -1])
        out = self.head(h).squeeze(-1)
        return (out, attn_w) if return_attention else out
