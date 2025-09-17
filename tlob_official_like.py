
"""
TLOB (official-like) PyTorch module for LOB classification.
This file mirrors the key architectural choices from the official TLOB (tlob.py):
- BiN input normalization
- Dual self-attention (temporal first, then spatial) with PreNorm residuals
- MLPLOB (feature-MLP + temporal-MLP) as FFN
- Sinusoidal positional encoding (temporal)

Author: ChatGPT
"""
from typing import Optional, Tuple
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BilinearNorm(nn.Module):
    def __init__(self, d_features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.w_c = nn.Parameter(torch.tensor(0.5))
        self.w_r = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.ones(1, 1, d_features))
        self.beta  = nn.Parameter(torch.zeros(1, 1, d_features))

    def forward(self, x: Tensor) -> Tensor:
        mean_t = x.mean(dim=1, keepdim=True)
        std_t  = x.var(dim=1, keepdim=True, unbiased=False).sqrt().clamp_min(self.eps)
        Xc = (x - mean_t) / std_t
        mean_f = x.mean(dim=2, keepdim=True)
        std_f  = x.var(dim=2, keepdim=True, unbiased=False).sqrt().clamp_min(self.eps)
        Xr = (x - mean_f) / std_f
        y = self.w_c * Xc + self.w_r * Xr
        return y * self.gamma + self.beta

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

class PreNorm(nn.Module):
    def __init__(self, dim: int, module: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.module = module
    def forward(self, x: Tensor) -> Tensor:
        return x + self.module(self.norm(x))

class TemporalSelfAttention(nn.Module):
    def __init__(self, d_features: int, d_model: int, n_heads: int, dropout: float, max_len: int):
        super().__init__()
        self.proj_in = nn.Linear(d_features, d_model)
        self.posenc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.proj_out = nn.Linear(d_model, d_features)
        self.drop = nn.Dropout(dropout)
    def forward(self, x: Tensor) -> Tensor:
        h = self.proj_in(x)
        h = self.posenc(h)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = self.drop(self.proj_out(h))
        return h

class SpatialSelfAttention(nn.Module):
    def __init__(self, seq_len: int, d_features: int, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.proj_in = nn.Linear(seq_len, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.proj_out = nn.Linear(d_model, seq_len)
        self.drop = nn.Dropout(dropout)
    def forward(self, x: Tensor) -> Tensor:
        h = x.transpose(1, 2)           # [B, D, T]
        h, _ = self.attn(self.proj_in(h), self.proj_in(h), self.proj_in(h), need_weights=False)
        h = self.drop(self.proj_out(h)) # [B, D, T]
        return h.transpose(1, 2)        # [B, T, D]

class FeatureMLP(nn.Module):
    def __init__(self, d_features: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_features),
            nn.Dropout(dropout),
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class TemporalMLP(nn.Module):
    def __init__(self, seq_len: int, hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(seq_len, hidden)
        self.fc2 = nn.Linear(hidden, seq_len)
        self.drop = nn.Dropout(dropout)
    def forward(self, x: Tensor) -> Tensor:
        y = x.transpose(1, 2)            # [B, D, T]
        y = self.fc2(self.drop(F.gelu(self.fc1(y))))
        y = self.drop(y).transpose(1, 2)  # [B, T, D]
        return y

class MLPLOB(nn.Module):
    def __init__(self, d_features: int, seq_len: int, hidden: int, dropout: float):
        super().__init__()
        self.fmlp = FeatureMLP(d_features, hidden, dropout)
        self.tmlp = TemporalMLP(seq_len, hidden, dropout)
    def forward(self, x: Tensor) -> Tensor:
        return self.fmlp(x) + self.tmlp(x)

class TLOBBlock(nn.Module):
    def __init__(self, d_features: int, seq_len: int, d_model: int, n_heads: int, mlp_hidden: int, dropout: float):
        super().__init__()
        self.bin = BilinearNorm(d_features)
        self.temporal = PreNorm(d_features, TemporalSelfAttention(d_features, d_model, n_heads, dropout, max_len=seq_len))
        self.spatial  = PreNorm(d_features, SpatialSelfAttention(seq_len, d_features, d_model, n_heads, dropout))
        self.mlp = PreNorm(d_features, MLPLOB(d_features, seq_len, mlp_hidden, dropout))
    def forward(self, x: Tensor) -> Tensor:
        x = self.bin(x)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.mlp(x)
        return x

class TLOB(nn.Module):
    def __init__(self, d_features: int, seq_len: int, d_model: int = 128, n_heads: int = 4,
                 mlp_hidden: int = 256, n_layers: int = 2, dropout: float = 0.1, n_classes: int = 3,
                 head_hidden: int = 256):
        super().__init__()
        self.blocks = nn.ModuleList([
            TLOBBlock(d_features, seq_len, d_model, n_heads, mlp_hidden, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_features)
        self.head = nn.Sequential(
            nn.Linear(d_features, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_classes),
        )
    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = blk(x)                  # [B, T, D]
        x = self.final_norm(x)
        x = x.mean(dim=1)              # [B, D]
        return self.head(x)            # [B, 3]
