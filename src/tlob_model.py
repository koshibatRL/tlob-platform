"""
Refactored TLOB model (bugfix: avoid shadowing torch.nn.functional as F).

File name suggestion: tlob_model.py
Paired preprocessing utilities: lob_dataset.py

Key fix in this version
-----------------------
- In BiN.forward, local variable name `F` (feature dim) shadowed the imported
  module alias `F = torch.nn.functional`, causing `AttributeError: 'int' object
  has no attribute 'softplus'`. We renamed the local variable to `C` (channels)
  and keep `F` for the functional module import.

Other highlights
----------------
- Device/dtype safe; no global DEVICE.
- Alternating Feature/Temporal Transformer blocks; last pair reduces dims by
  `reduce_factor` (default 4).
- Optional sinusoidal or learned positional embeddings.
- Attention maps can be stored via `store_att=True`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import math
import torch
from torch import nn
import torch.nn.functional as F


# ----------------------------- Utilities ------------------------------------

def sinusoidal_positional_embedding(seq_len: int, dim: int, *, device=None, dtype=None) -> torch.Tensor:
    """Standard sinusoidal positional embeddings: [1, S, dim]."""
    if dim % 2 != 0:
        raise ValueError(f"Positional embedding requires even dim, got {dim}.")
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    pe = torch.zeros(seq_len, dim, device=device, dtype=dtype)
    position = torch.arange(0, seq_len, device=device, dtype=dtype).unsqueeze(1)  # [S,1]
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # [1,S,dim]


class Residual(nn.Module):
    """Residual wrapper with optional projection if dims differ."""
    def __init__(self, in_dim: int, out_dim: int, p_drop: float = 0.0) -> None:
        super().__init__()
        self.proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, fx: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(x)) + fx


class FeedForward(nn.Module):
    """Simple MLP block with GELU and dropout; supports out_dim != in_dim."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: Optional[int] = None, p_drop: float = 0.0) -> None:
        super().__init__()
        out_dim = out_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """PreNorm Transformer block (MHA + FFN) with residuals.

    Expects input shape [B, tokens, dim] where *dim* == embed_dim for nn.MultiheadAttention.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        out_dim: Optional[int] = None,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed dim ({dim}) must be divisible by num_heads ({num_heads}).")

        out_dim = out_dim or dim
        self.dim = dim
        self.out_dim = out_dim

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(proj_dropout)

        hidden = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, hidden, out_dim=out_dim, p_drop=ffn_dropout)

        # Residuals (support out_dim != dim via projection on the *second* residual)
        self.res1 = Residual(in_dim=dim, out_dim=dim, p_drop=proj_dropout)
        self.res2 = Residual(in_dim=dim, out_dim=out_dim, p_drop=proj_dropout)

    def forward(self, x: torch.Tensor, need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention
        y, att = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=need_weights, average_attn_weights=False)
        x = self.res1(x, self.drop1(y))  # shape: [B, tokens, dim]
        # FFN + residual (may change embed dim)
        y2 = self.ffn(self.norm2(x))  # [B, tokens, out_dim]
        x = self.res2(x, y2)          # project skip if needed, result [B, tokens, out_dim]
        return x, att if need_weights else None


class BiN(nn.Module):
    """Bi-directional Normalization for inputs [B, F, S].

    - Time-wise normalization (per feature channel along S) -> affine (gamma_f, beta_f)
    - Feature-wise normalization (per time step along F)     -> affine (gamma_t, beta_t)
    - Positive-weighted mix (y1, y2) via softplus
    """
    def __init__(self, num_features: int, seq_len: int, eps: float = 1e-4, y_init: float = 0.5) -> None:
        super().__init__()
        self.F = num_features
        self.S = seq_len
        self.eps = eps

        # Affine params
        self.gamma_f = nn.Parameter(torch.ones(1, num_features, 1))
        self.beta_f  = nn.Parameter(torch.zeros(1, num_features, 1))
        self.gamma_t = nn.Parameter(torch.ones(1, 1, seq_len))
        self.beta_t  = nn.Parameter(torch.zeros(1, 1, seq_len))

        # Softplus-parameterized positive mix weights
        inv_sp = torch.log(torch.expm1(torch.tensor(float(y_init))))
        self.raw_y1 = nn.Parameter(inv_sp.clone())
        self.raw_y2 = nn.Parameter(inv_sp.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"BiN expects [B, F, S], got {tuple(x.shape)}")
        B, C, S = x.shape  # C = channels (features), avoid shadowing F module
        if C != self.F or S != self.S:
            raise ValueError(f"BiN was initialized for [*, {self.F}, {self.S}], got [*, {C}, {S}].")

        # Time-direction (per feature)
        mean_t = x.mean(dim=2, keepdim=True)
        var_t  = x.var(dim=2, keepdim=True, unbiased=False)
        z2     = (x - mean_t) / torch.sqrt(var_t + self.eps)
        x2     = self.gamma_f * z2 + self.beta_f

        # Feature-direction (per time)
        mean_f = x.mean(dim=1, keepdim=True)
        var_f  = x.var(dim=1, keepdim=True, unbiased=False)
        z1     = (x - mean_f) / torch.sqrt(var_f + self.eps)
        x1     = self.gamma_t * z1 + self.beta_t

        y1 = F.softplus(self.raw_y1) + self.eps
        y2 = F.softplus(self.raw_y2) + self.eps
        return y1 * x1 + y2 * x2


@dataclass
class TLOBConfig:
    hidden_dim: int
    num_layers: int
    seq_len: int
    num_features: int
    num_heads: int = 4
    use_sinusoidal_pos_emb: bool = True
    reduce_factor: int = 4
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    ffn_dropout: float = 0.0
    mlp_ratio: float = 4.0


class TLOB(nn.Module):
    """Refactored TLOB classifier.

    Pipeline:
      1) BiN over [B, F, S]
      2) Linear embed F -> H => [B, S, H]
      3) Add positional encodings (learned or sinusoidal)
      4) For each layer i in [0..L-1]:
           a) Feature-side TransformerBlock (embed_dim = H, maybe shrink H on last)
           b) Swap -> [B, H', S]
           c) Temporal-side TransformerBlock (embed_dim = S, maybe shrink S on last)
           d) Swap -> [B, S', H']
      5) Flatten [B, S', H'] and MLP head -> 3 logits
    """
    def __init__(self, cfg: TLOBConfig) -> None:
        super().__init__()
        H_dim, S_len, F_dim = cfg.hidden_dim, cfg.seq_len, cfg.num_features

        if H_dim % cfg.num_heads != 0:
            raise ValueError(f"hidden_dim ({H_dim}) must be divisible by num_heads ({cfg.num_heads}).")
        if S_len % cfg.num_heads != 0:
            raise ValueError(f"seq_len ({S_len}) must be divisible by num_heads ({cfg.num_heads}).")
        if cfg.reduce_factor <= 1:
            raise ValueError("reduce_factor must be >= 2 to be meaningful.")
        if (H_dim % cfg.reduce_factor) != 0 or (S_len % cfg.reduce_factor) != 0:
            raise ValueError(
                f"hidden_dim and seq_len must be divisible by reduce_factor ({cfg.reduce_factor})."
            )

        self.cfg = cfg

        # 1) BiN over [B, F, S]
        self.bin = BiN(num_features=F_dim, seq_len=S_len)

        # 2) Feature embedding F -> H
        self.embed = nn.Linear(F_dim, H_dim)

        # 3) Positional encodings [1,S,H]
        if cfg.use_sinusoidal_pos_emb:
            pe = sinusoidal_positional_embedding(S_len, H_dim)
            self.register_buffer("pos_emb", pe, persistent=False)
            self.learned_pos = None
        else:
            self.pos_emb = None
            self.learned_pos = nn.Parameter(torch.randn(1, S_len, H_dim) * 0.02)

        # 4) Alternating axial stack
        self.feature_blocks = nn.ModuleList()
        self.temporal_blocks = nn.ModuleList()
        H_cur, S_cur = H_dim, S_len
        for i in range(cfg.num_layers):
            last = (i == cfg.num_layers - 1)
            out_H = (H_cur // cfg.reduce_factor) if last else H_cur
            out_S = (S_cur // cfg.reduce_factor) if last else S_cur

            self.feature_blocks.append(
                TransformerBlock(
                    dim=H_cur,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    out_dim=out_H,
                    attn_dropout=cfg.attn_dropout,
                    proj_dropout=cfg.proj_dropout,
                    ffn_dropout=cfg.ffn_dropout,
                )
            )
            self.temporal_blocks.append(
                TransformerBlock(
                    dim=S_cur,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    out_dim=out_S,
                    attn_dropout=cfg.attn_dropout,
                    proj_dropout=cfg.proj_dropout,
                    ffn_dropout=cfg.ffn_dropout,
                )
            )
            # Update for next iteration's expectations (post-swap dims)
            H_cur, S_cur = out_H, out_S

        # 5) Classification head over flattened [B, S_final * H_final]
        final_dim = (cfg.hidden_dim // cfg.reduce_factor) * (cfg.seq_len // cfg.reduce_factor)
        layers: List[nn.Module] = []
        d = final_dim
        while d > 128:
            layers += [nn.Linear(d, d // 4), nn.GELU(), nn.Dropout(cfg.proj_dropout)]
            d //= 4
        layers += [nn.Linear(d, 3)]
        self.head = nn.Sequential(*layers)

        # For external inspection (optional)
        self.last_attentions: List[torch.Tensor] = []

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Xavier init for Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, *, store_att: bool = False) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, S, F]
            store_att: if True, stores attention maps from each block in self.last_attentions
        Returns:
            [B, 3] logits
        """
        if x.dim() != 3:
            raise ValueError(f"Expected [B, S, F], got {tuple(x.shape)}")
        B, S_in, F_in = x.shape
        if S_in != self.cfg.seq_len or F_in != self.cfg.num_features:
            raise ValueError(
                f"Input must be [*, {self.cfg.seq_len}, {self.cfg.num_features}], got [*, {S_in}, {F_in}]."
            )

        # 1) BiN expects [B, F, S]
        x = x.permute(0, 2, 1)  # [B, F, S]
        x = self.bin(x)
        x = x.permute(0, 2, 1)  # [B, S, F]

        # 2) Embed F->H
        x = self.embed(x)  # [B, S, H0]

        # 3) Positional encodings
        if getattr(self, "pos_emb", None) is not None:
            pe = self.pos_emb.to(device=x.device, dtype=x.dtype)
            x = x + pe
        else:
            x = x + self.learned_pos

        # 4) Alternating axial blocks
        if store_att:
            self.last_attentions = []
        for f_blk, t_blk in zip(self.feature_blocks, self.temporal_blocks):
            # Feature-side on [B, S, H]
            x, att_f = f_blk(x, need_weights=store_att)
            if store_att and att_f is not None:
                self.last_attentions.append(att_f.detach())
            # swap -> [B, H, S]
            x = x.transpose(1, 2)
            # Temporal-side on [B, H, S] treating embed_dim=S
            x, att_t = t_blk(x, need_weights=store_att)
            if store_att and att_t is not None:
                self.last_attentions.append(att_t.detach())
            # swap back -> [B, S, H]
            x = x.transpose(1, 2)

        # 5) Flatten + head
        B, S_fin, H_fin = x.shape
        x = x.reshape(B, S_fin * H_fin)
        logits = self.head(x)
        return logits


if __name__ == "__main__":
    # Minimal smoke test
    cfg = TLOBConfig(
        hidden_dim=64,
        num_layers=2,
        seq_len=32,
        num_features=16,
        num_heads=4,
        use_sinusoidal_pos_emb=True,
        reduce_factor=4,
        attn_dropout=0.0,
        proj_dropout=0.0,
        ffn_dropout=0.0,
    )
    model = TLOB(cfg)
    x = torch.randn(8, cfg.seq_len, cfg.num_features)
    out = model(x, store_att=True)
    print(out.shape)  # [8,3]
    print(len(model.last_attentions))  # should be 2 * num_layers
