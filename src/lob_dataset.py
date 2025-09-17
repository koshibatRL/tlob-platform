"""
Refactored LOB windowing + labeling utilities (device/dtype safe, vectorized, no globals).

Functions
---------
- make_lob_window_dataset: Build [N, lookback, F] windows and 3-class labels from L1 quotes.
- train_val_split_by_time: Time-ordered split (no shuffle) returning tuples.
- macro_f1: Macro-averaged F1 without external deps.

Key changes
-----------
- Removed global DEVICE usage. Return CPU tensors; caller moves to device.
- Vectorized rolling means via convolution; no Python loops over samples.
- Sliding windows via numpy.sliding_window_view for speed & clarity.
- Shape and boundary checks with informative errors.
- Robust epsilon computation and NaN handling for spread.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple 1D rolling mean using convolution (valid mode).

    Returns array of length len(arr) - window + 1 where out[j] = mean(arr[j:j+window]).
    """
    if window <= 0:
        raise ValueError("window must be >= 1")
    if arr.ndim != 1:
        raise ValueError("_rolling_mean expects 1D array")
    if len(arr) < window:
        return np.empty(0, dtype=arr.dtype)
    kernel = np.ones(window, dtype=arr.dtype) / float(window)
    # 'valid' -> no padding; precision remains float32 if input is float32
    return np.convolve(arr, kernel, mode="valid")


def _cumulative_nanmean(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative mean along 1D array ignoring NaNs (past-only, no lookahead).

    Returns
    -------
    mean : np.ndarray
        mean[i] = nanmean(arr[: i+1]) if any finite values exist up to i, else NaN.
    count : np.ndarray
        count[i] = number of finite values in arr[: i+1].
    """
    if arr.ndim != 1:
        raise ValueError("_cumulative_nanmean expects 1D array")
    mask = np.isfinite(arr)
    csum = np.cumsum(np.where(mask, arr, 0.0), dtype=np.float64)
    ccnt = np.cumsum(mask.astype(np.int64))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = csum / np.maximum(ccnt, 1)
    out[ccnt == 0] = np.nan
    return out, ccnt


@dataclass
class LabelingConfig:
    horizon_steps: int             # H: forecast horizon (steps)
    smooth_len: int = 5            # W: smoothing window for m_past/m_future
    epsilon_mode: str = "avg_spread"  # one of {"avg_spread", "zero", "nonzero"}
    epsilon_scale: float = 0.5        # scale for avg_spread mode

    def validate(self) -> None:
        if self.horizon_steps < 0:
            raise ValueError("horizon_steps must be >= 0")
        if self.smooth_len < 1:
            raise ValueError("smooth_len must be >= 1")
        if self.epsilon_mode not in {"avg_spread", "zero", "nonzero"}:
            raise ValueError(f"unknown epsilon_mode: {self.epsilon_mode}")
        if self.epsilon_scale < 0:
            raise ValueError("epsilon_scale must be >= 0")


def make_lob_window_dataset(
    df: pd.DataFrame,
    feature_count: int,
    lookback: int,
    b1_index: int,
    a1_index: int,
    labeling: LabelingConfig,
) -> Tuple[Tensor, Tensor, np.ndarray]:
    """Construct dataset windows and labels for TLOB-like training.

    Target (per TLOB Sec.4):
      m_past(t)   = mean of last W mids up to t
      m_future(t) = mean of W mids starting right after (t+H)
      delta       = (m_future - m_past) / m_past
      label       = 2 if delta >= eps, 0 if delta < -eps, else 1

    Args
    ----
    df : DataFrame whose first `feature_count` columns are LOB features (float-like).
    feature_count : int, number of features to take from df's leftmost columns.
    lookback : int, window length (timesteps) for X.
    b1_index, a1_index : column indices (within the selected feature slice) for best bid/ask.
    labeling : LabelingConfig, controls horizon H, smoothing W, and epsilon.

    Returns
    -------
    X : torch.FloatTensor of shape [N, lookback, feature_count]
    y : torch.LongTensor of shape [N]
    t : np.ndarray of dtype datetime64[ns] with the end timestamp for each sample
    """
    labeling.validate()

    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if feature_count < 1:
        raise ValueError("feature_count must be >= 1")
    if df.shape[1] < feature_count:
        raise ValueError("df must have at least feature_count columns")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index must be a DatetimeIndex")

    # Extract numeric values as float32 matrix [T, F]
    vals = df.iloc[:, :feature_count].to_numpy(dtype=np.float32, copy=True)
    T, F = vals.shape

    if not (0 <= b1_index < F and 0 <= a1_index < F):
        raise IndexError("b1_index/a1_index out of range for selected features")

    b1 = vals[:, b1_index]
    a1 = vals[:, a1_index]
    mid = (b1 + a1) * 0.5  # [T]
    spread = (a1 - b1)     # [T]

    # Past-only cumulative average spread (no lookahead leakage)
    cum_avg_spread, cum_count = _cumulative_nanmean(spread.astype(np.float64))

    H = int(labeling.horizon_steps)
    W = int(max(1, labeling.smooth_len))

    # Determine valid i range (end index of input window)
    #  - need past W for m_past => i >= W-1
    #  - need future [i+H+1, i+H+W] => i <= T - (H+W) - 1
    #  - need input window [i-lookback+1, i] => i >= lookback-1
    start_i = max(lookback - 1, W - 1)
    end_i_excl = T - (H + W)
    if end_i_excl - start_i <= 0:
        raise ValueError(
            "Insufficient rows for given lookback/H/W. "
            f"Need T > max(lookback-1, W-1) + H + W; got T={T}, lookback={lookback}, H={H}, W={W}."
        )

    # Rolling means for mid
    past_ma = _rolling_mean(mid, W)   # length T-W+1, corresponds to end index i
    fut_ma  = _rolling_mean(mid, W)   # reuse; we will index by start position

    # Build vector of target indices
    i_range = np.arange(start_i, end_i_excl, dtype=np.int64)           # [N]
    past_idx = i_range - (W - 1)                                       # indices into past_ma
    fut_idx  = i_range + H + 1                                         # start indices into fut_ma

    m_past = past_ma[past_idx].astype(np.float32)
    m_future = fut_ma[fut_idx].astype(np.float32)

    # Ratio change
    denom = np.where(m_past == 0.0, 1e-12, m_past)
    delta = (m_future - m_past) / denom

    # Epsilon per-sample (ratio)
    if labeling.epsilon_mode == "avg_spread":
        eps_spread = cum_avg_spread[i_range].astype(np.float32)
        if not np.all(np.isfinite(eps_spread)):
            raise ValueError("Cumulative average spread is undefined (no finite past values) for some samples.")
        eps = (eps_spread / np.maximum(1e-12, m_past)) * float(labeling.epsilon_scale)
    elif labeling.epsilon_mode == "zero":
        eps = np.zeros_like(delta, dtype=np.float32)
    elif labeling.epsilon_mode == "nonzero":
        eps = np.full_like(delta, 0.002, dtype=np.float32)
    else:
        # Should not happen due to validate(), but keep for safety
        raise ValueError(f"unknown epsilon_mode: {labeling.epsilon_mode}")

    # 3-class labels (0=Down, 1=Stay, 2=Up)
    y_np = np.where(delta >= eps, 2, np.where(delta < -eps, 0, 1)).astype(np.int64)

    # Build input windows via sliding view along time axis
    try:
        from numpy.lib.stride_tricks import sliding_window_view
    except Exception as e:  # pragma: no cover
        raise RuntimeError("NumPy >= 1.20 is required for sliding_window_view") from e

    all_win = sliding_window_view(vals, window_shape=lookback, axis=0)  # [T-lookback+1, F, lookback]
    s_idx = i_range - (lookback - 1)                                    # window start indices
    X_np = all_win[s_idx]                                               # [N, F, lookback]

    # Transpose to [N, lookback, F] expected by the model
    X_np = np.transpose(X_np, (0, 2, 1))                                # [N, lookback, F]

    # Ensure contiguous (torch.from_numpy may dislike exotic strides)
    X_np = np.ascontiguousarray(X_np, dtype=np.float32)

    # Convert to tensors on CPU; caller can .to(device)
    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np).to(torch.long)
    t = df.index[i_range].to_numpy(dtype="datetime64[ns]")

    return X, y, t


def train_val_split_by_time(
    X: Tensor, y: Tensor, t: np.ndarray, val_ratio: float = 0.2
) -> Tuple[Tuple[Tensor, Tensor, np.ndarray], Tuple[Tensor, Tensor, np.ndarray]]:
    """Time-ordered split (no shuffle). Returns (train, val) tuples.

    Assumes X, y, t are aligned and t is non-decreasing.
    """
    if not (len(X) == len(y) == len(t)):
        raise ValueError("X, y, t must have the same length")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0,1)")

    n = len(X)
    k = max(1, int(round(n * (1.0 - val_ratio))))
    k = min(k, n - 1)  # ensure both splits non-empty

    return (X[:k], y[:k], t[:k]), (X[k:], y[k:], t[k:])


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    """Compute macro-averaged F1 for integer class labels.

    Handles empty-class edge cases by contributing 0.0 for that class.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    f1s = []
    for c in range(int(n_classes)):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        if tp + fp + fn == 0:
            f1s.append(0.0)
            continue
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall    = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = 0.0 if (precision + recall) == 0 else (2.0 * precision * recall) / (precision + recall)
        f1s.append(float(f1))
    return float(np.mean(f1s) if len(f1s) > 0 else 0.0)


if __name__ == "__main__":  # basic smoke test
    # Build a tiny fake LOB df
    rng = pd.date_range("2025-01-01", periods=200, freq="min")
    bid = np.linspace(100, 101, num=200, dtype=np.float32)
    ask = bid + 0.02
    extra = np.random.randn(200, 3).astype(np.float32)
    feats = np.c_[bid, ask, extra]
    df = pd.DataFrame(feats, index=rng, columns=["b1","a1","f2","f3","f4"])

    cfg = LabelingConfig(horizon_steps=5, smooth_len=5, epsilon_mode="avg_spread", epsilon_scale=0.5)
    X, y, t = make_lob_window_dataset(df, feature_count=5, lookback=16, b1_index=0, a1_index=1, labeling=cfg)
    (Xtr, ytr, ttr), (Xva, yva, tva) = train_val_split_by_time(X, y, t, val_ratio=0.2)
    print(X.shape, y.shape, t.shape)
    print(Xtr.shape, Xva.shape)
