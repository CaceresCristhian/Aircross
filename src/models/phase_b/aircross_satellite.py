"""
PHASE B (NOT USED IN ANY REPORTED RESULT).
Satellite version of AirCross. It will be trained only after real Sentinel-5P TROPOMI
patches are ingested (see README, Roadmap). Phase A experiments use src/models/aircross.py.
"""
"""
AirCross Model Architecture:
- SpatialViTEncoder (Self-Attention over satellite column density patches)
- WindDirectedCrossAttention (WDCA: Wind conditioning for plume advection & upwind localization)
- StoichiometricCrossAttention (Multi-pollutant cross-species alignment)
- TemporalSelfAttentionAggregator (Temporal overpass sequence weighting)
- Multi-Task Prediction Heads (Stack Emission kg/h & Coordinate Localization Error)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialViTEncoder(nn.Module):
    """
    Self-Attention Spatial Vision Transformer encoder over local satellite column density patches.
    Input: (B, C, H, W) where C = channels (NO2, CO, SO2), H = W = grid size (e.g. 15x15)
    Output: (B, N, D) spatial token sequence.
    """
    def __init__(self, in_channels=3, grid_size=15, patch_size=3, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.num_patches = (grid_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.patch_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        tokens = self.patch_proj(x) # (B, D, H', W')
        tokens = tokens.flatten(2).transpose(1, 2) # (B, num_patches, D)
        tokens = tokens + self.pos_embed
        out = self.transformer(tokens)
        return self.norm(out) # (B, N, D)

class WindDirectedCrossAttention(nn.Module):
    """
    Cross-Attention module: Meteorological state (u, v, PBLH) queries spatial plume tokens.
    Includes forward advection head (plume mass flow) and inverse traceback head (localization).
    """
    def __init__(self, embed_dim=128, num_heads=4, meteo_dim=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Meteorological vector projection to query space
        self.meteo_proj = nn.Sequential(
            nn.Linear(meteo_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Cross-Attention layers
        self.cross_attn_forward = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_inverse = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Coordinate localization regression head from inverse attention
        self.loc_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2) # delta_x, delta_y offset in km from center
        )

    def forward(self, spatial_tokens, meteo_vec):
        # spatial_tokens: (B, N, D)
        # meteo_vec: (B, meteo_dim) -> [u_wind, v_wind, pblh_norm, surface_pressure_norm]
        q_forward = self.meteo_proj(meteo_vec).unsqueeze(1) # (B, 1, D)
        
        # Inverse wind vector for upwind traceback
        meteo_inv = meteo_vec.clone()
        meteo_inv[:, 0] = -meteo_inv[:, 0] # invert u
        meteo_inv[:, 1] = -meteo_inv[:, 1] # invert v
        q_inverse = self.meteo_proj(meteo_inv).unsqueeze(1) # (B, 1, D)

        # 1. Forward Cross-Attention (Disentangles downwind plume advection)
        attn_out_fwd, weights_fwd = self.cross_attn_forward(
            query=q_forward,
            key=spatial_tokens,
            value=spatial_tokens
        )
        fwd_repr = self.norm1(attn_out_fwd.squeeze(1)) # (B, D)

        # 2. Inverse Cross-Attention (Traces back to upwind source apex for localization)
        attn_out_inv, weights_inv = self.cross_attn_inverse(
            query=q_inverse,
            key=spatial_tokens,
            value=spatial_tokens
        )
        inv_repr = self.norm2(attn_out_inv.squeeze(1)) # (B, D)
        pred_coords_offset = self.loc_head(inv_repr) # (B, 2) -> (delta_km_x, delta_km_y)

        return fwd_repr, pred_coords_offset, weights_fwd, weights_inv

class TemporalSelfAttentionAggregator(nn.Module):
    """
    Self-Attention over temporal observation dates to aggregate multiple satellite overpasses.
    """
    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

    def forward(self, seq_tokens, mask=None):
        # seq_tokens: (B, T, D)
        B = seq_tokens.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, seq_tokens], dim=1) # (B, T+1, D)
        
        attn_out, _ = self.self_attn(tokens, tokens, tokens)
        out = self.norm(attn_out[:, 0]) # CLS pooled token
        return out

class AirCrossModel(nn.Module):
    """
    Full AirCross Architecture:
    SpatialViT -> WDCA (Wind Cross-Attn) -> Multi-Pollutant Fusion -> Temporal Self-Attn -> Multi-Task Heads.
    """
    def __init__(self, in_channels=3, grid_size=15, patch_size=3, embed_dim=128, num_heads=4):
        super().__init__()
        self.spatial_encoder = SpatialViTEncoder(
            in_channels=in_channels,
            grid_size=grid_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        self.wdca = WindDirectedCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            meteo_dim=4
        )
        self.temporal_aggregator = TemporalSelfAttentionAggregator(
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        
        # Emission prediction head (Predicts actual stack kg/h NOx)
        self.emission_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, satellite_patches, meteo_vectors):
        """
        satellite_patches: (B, T, C, H, W)
        meteo_vectors: (B, T, 4) -> [u, v, pblh, pressure]
        """
        B, T, C, H, W = satellite_patches.shape
        
        # Process each temporal overpass
        temp_repr_list = []
        coord_offsets_list = []

        for t in range(T):
            patches_t = satellite_patches[:, t] # (B, C, H, W)
            meteo_t = meteo_vectors[:, t]       # (B, 4)

            spatial_tokens = self.spatial_encoder(patches_t) # (B, N, D)
            fwd_repr, pred_coords, _, _ = self.wdca(spatial_tokens, meteo_t)
            temp_repr_list.append(fwd_repr)
            coord_offsets_list.append(pred_coords)

        # Stack over time: (B, T, D)
        seq_tokens = torch.stack(temp_repr_list, dim=1)
        # Average predicted coordinate offsets across valid overpasses
        avg_coord_offset = torch.stack(coord_offsets_list, dim=1).mean(dim=1) # (B, 2)

        # Temporal self-attention aggregation
        agg_repr = self.temporal_aggregator(seq_tokens) # (B, D)
        pred_emissions = self.emission_head(agg_repr).squeeze(-1) # (B,)

        return {
            "pred_emissions_kgh": pred_emissions,
            "pred_coords_offset_km": avg_coord_offset
        }

# ----------------- Benchmark Baselines for IEEE Paper -----------------

class MLPBaseline(nn.Module):
    """Standard Multi-Layer Perceptron (Flattened Satellite + Meteo)."""
    def __init__(self, in_dim=15*15*3 + 4, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.loc_net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x_flat):
        return {
            "pred_emissions_kgh": self.net(x_flat).squeeze(-1),
            "pred_coords_offset_km": self.loc_net(x_flat)
        }

class CNNBaseline(nn.Module):
    """Standard 2D Convolutional Network (ResNet-like blocks)."""
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc_emiss = nn.Linear(64, 1)
        self.fc_loc = nn.Linear(64, 2)

    def forward(self, x):
        feat = self.conv(x).flatten(1)
        return {
            "pred_emissions_kgh": self.fc_emiss(feat).squeeze(-1),
            "pred_coords_offset_km": self.fc_loc(feat)
        }

