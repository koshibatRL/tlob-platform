"""
End-to-end script for TLOB: smoke test + training loop + logging & model export
+ confusion matrix export for validation and an optional second CSV (e.g., 2502).

Simplified per user request:
- Remove --lob-depth (assume 10 for all provided data)
- Remove --features (infer from CSV; synth data uses 40 cols for depth=10)
- Remove --reduce option (fixed to 4)

File name: run_tlob_smoke.py

Usage examples:
  # Smoke (forward + one opt step) on synth data
  python run_tlob_smoke.py --mode smoke --seq-len 32 --hidden 64 --layers 2 --heads 4

  # Train on real CSV (2501) and evaluate on another CSV (2502)
  python run_tlob_smoke.py --mode train \
      --train-csv ../data/monthly/BTCUSDT_book_2501_1min.csv.gz \
      --eval-csv  ../data/monthly/BTCUSDT_book_2502_1min.csv.gz \
      --seq-len 32 --hidden 64 --layers 2 --heads 4 \
      --epochs 5 --batch 256 --lr 2e-4 --val 0.2 --device auto

Outputs are saved under logs/<JST timestamp>/:
  - run.log, config.json, summary.json
  - model_best.pt, model_config.json
  - confusion_matrix_valid.png
  - confusion_matrix_<eval-name>.png (if --eval-csv provided)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import random
from copy import deepcopy
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
import torch.nn.functional as F
import matplotlib.pyplot as plt

from lob_dataset import (
    LabelingConfig,
    make_lob_window_dataset,
    train_val_split_by_time,
    macro_f1,
)
from tlob_model import TLOB, TLOBConfig


# ------------------------- Utilities -------------------------

def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(arg: str | None = None) -> str:
    if arg and arg.lower() not in {"auto", ""}:
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mk_logdir(base: str = "logs") -> Path:
    ts = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d_%H%M%S")
    path = Path(base) / ts
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_lob_df(csv_path: str) -> pd.DataFrame:
    """Load LOB csv.gz (depth=10) with expected column ordering.

    Columns: b1..b10, bq1..b10, a1..a10, aq1..aq10
    """
    LOB_DEPTH = 10
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True).ffill()
    cols = (
        [f"b{i+1}" for i in range(LOB_DEPTH)] +
        [f"bq{i+1}" for i in range(LOB_DEPTH)] +
        [f"a{i+1}" for i in range(LOB_DEPTH)] +
        [f"aq{i+1}" for i in range(LOB_DEPTH)]
    )
    df = df[cols]
    return df


def configure_logger(logdir: Path) -> logging.Logger:
    logger = logging.getLogger("run")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # Clear previous handlers if re-running in same process
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(logdir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, batch_size: int = 256, return_preds: bool = False):
    model.eval()
    dev = model_device(model)
    preds = []
    for i in range(0, len(X), batch_size):
        xb = X[i : i + batch_size].to(dev)
        logits = model(xb)  # [B, 3]
        preds.append(logits.argmax(-1).cpu().numpy())
    y_pred = np.concatenate(preds) if preds else np.empty((0,), dtype=int)
    y_true = y.cpu().numpy()
    acc = float((y_pred == y_true).mean()) if len(y_true) else float("nan")
    f1 = macro_f1(y_true, y_pred, n_classes=3) if len(y_true) else float("nan")
    if return_preds:
        return {"acc": acc, "macro_f1": f1, "y_true": y_true, "y_pred": y_pred}
    return {"acc": acc, "macro_f1": f1}


def plot_confusion_matrix(y_true, y_pred, class_names=("Down", "Stay", "Up"), normalize="row", save_path: Path | None = None):
    """
    normalize: "row"（行方向％）, "all"（全体％）, None（生のカウント）
    Saves figure to save_path if provided.
    """
    C = len(class_names)
    cm = np.zeros((C, C), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(C)); ax.set_yticks(range(C))
    ax.set_xticklabels(class_names, rotation=0)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (3×3)")

    if normalize == "row":
        denom = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_pct = cm / denom
    elif normalize == "all":
        total = max(1, cm.sum())
        cm_pct = cm / total
    else:
        cm_pct = None

    vmax = cm.max() if cm.size else 1
    for i in range(C):
        for j in range(C):
            text = f"{cm[i, j]}"
            if cm_pct is not None:
                text += f"\n({cm_pct[i, j]*100:.1f}%)"
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if cm[i, j] > vmax/2 else "black",
                    fontsize=10)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    plt.close(fig)
    return cm


# ----------------------- Data synth (depth=10) ---------------

def make_synth_df_lob10(T: int) -> pd.DataFrame:
    LOB_DEPTH = 10
    rng = pd.date_range("2025-01-01", periods=T, freq="min")
    # b1..b10 & a1..a10 with mild trend/noise; volumes bq/aq as noise
    bid1 = np.linspace(100, 101, num=T, dtype=np.float32) + 0.05 * np.sin(np.linspace(0, 10, T)).astype(np.float32)
    ask1 = bid1 + 0.02
    # Build ladders by adding small offsets
    bids = [bid1 + 0.001 * i for i in range(LOB_DEPTH)]
    asks = [ask1 + 0.001 * i for i in range(LOB_DEPTH)]
    bq = [np.abs(np.random.randn(T)).astype(np.float32) for _ in range(LOB_DEPTH)]
    aq = [np.abs(np.random.randn(T)).astype(np.float32) for _ in range(LOB_DEPTH)]

    data = np.c_[*bids, *bq, *asks, *aq].astype(np.float32)
    cols = (
        [f"b{i+1}" for i in range(LOB_DEPTH)] +
        [f"bq{i+1}" for i in range(LOB_DEPTH)] +
        [f"a{i+1}" for i in range(LOB_DEPTH)] +
        [f"aq{i+1}" for i in range(LOB_DEPTH)]
    )
    return pd.DataFrame(data, index=rng, columns=cols)


# ----------------------- Trainer -----------------------------

def train_tlob(
    df: pd.DataFrame,
    lookback: int,
    horizon_steps: int,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 2e-4,
    val_ratio: float = 0.2,
    seed: int = 42,
    device: str = "cpu",
    logdir: Path | None = None,
    logger: logging.Logger | None = None,
    eval_df: pd.DataFrame | None = None,
    eval_name: str | None = None,
):
    set_seed(seed)

    # Determine indices and feature_count from df
    feature_count = df.shape[1]
    b1_index = df.columns.get_loc("b1")
    a1_index = df.columns.get_loc("a1")

    labeling = LabelingConfig(
        horizon_steps=horizon_steps,
        smooth_len=5,
        epsilon_mode="avg_spread",
        epsilon_scale=0.5,
    )

    X, y, t = make_lob_window_dataset(
        df, feature_count, lookback, b1_index, a1_index, labeling
    )
    (Xtr, ytr, ttr), (Xva, yva, tva) = train_val_split_by_time(
        X, y, t, val_ratio=val_ratio
    )

    cfg = TLOBConfig(
        hidden_dim=d_model,
        num_layers=n_layers,
        seq_len=lookback,
        num_features=feature_count,
        num_heads=n_heads,
        reduce_factor=4,                 # fixed reduce=4
        use_sinusoidal_pos_emb=True,
    )
    model = TLOB(cfg).to(device)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    best = {"epoch": -1, "score": -1.0, "state": None}

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        total = 0.0
        seen = 0
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            logits = model(xb)  # [B, 3]
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = len(idx)
            total += float(loss.detach().cpu()) * bs
            seen += bs

        sched.step()

        val = evaluate(model, Xva, yva, batch_size=batch_size)
        score = val["macro_f1"]
        msg = f"[{ep:02d}] train_loss={total/max(1,seen):.4f}  val_acc={val['acc']:.4f}  val_f1={val['macro_f1']:.4f}"
        (logger.info if logger else print)(msg)
        if np.isfinite(score) and score > best["score"]:
            best = {
                "epoch": ep,
                "score": float(score),
                "state": deepcopy(model.state_dict()),
            }

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    # ---- Save trained model ----
    if logdir is not None:
        # Save state_dict on CPU + model config
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(state, logdir / "model_best.pt")
        cfg_json = {
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            "seq_len": cfg.seq_len,
            "num_features": cfg.num_features,
            "num_heads": cfg.num_heads,
            "reduce_factor": cfg.reduce_factor,
            "use_sinusoidal_pos_emb": cfg.use_sinusoidal_pos_emb,
        }
        (logdir / "model_config.json").write_text(json.dumps(cfg_json, ensure_ascii=False, indent=2))
        (logger.info if logger else print)(f"Saved model to {logdir / 'model_best.pt'}")

    # ---- Final validation + confusion matrix ----
    final = evaluate(model, Xva, yva, batch_size=batch_size, return_preds=True)
    cm_path = None
    if logdir is not None:
        cm_path = logdir / "confusion_matrix_valid.png"
        plot_confusion_matrix(final["y_true"], final["y_pred"], class_names=("Down","Stay","Up"), normalize="row", save_path=cm_path)
        (logger.info if logger else print)(f"Saved validation confusion matrix to {cm_path}")

    # ---- Optional evaluation on another CSV/df ----
    extra_cm_path = None
    extra_metrics = None
    if eval_df is not None:
        feature_count_ev = eval_df.shape[1]
        b1_index_ev = eval_df.columns.get_loc("b1")
        a1_index_ev = eval_df.columns.get_loc("a1")
        Xev, yev, tev = make_lob_window_dataset(eval_df, feature_count_ev, lookback, b1_index_ev, a1_index_ev, labeling)
        extra = evaluate(model, Xev, yev, batch_size=batch_size, return_preds=True)
        extra_name = (eval_name or "eval").replace("/", "_")
        if logdir is not None:
            extra_cm_path = logdir / f"confusion_matrix_{extra_name}.png"
            plot_confusion_matrix(extra["y_true"], extra["y_pred"], class_names=("Down","Stay","Up"), normalize="row", save_path=extra_cm_path)
            (logger.info if logger else print)(f"Saved extra confusion matrix to {extra_cm_path}")
        extra_metrics = {k: extra[k] for k in ("acc", "macro_f1")}

    # Store summary JSON
    if logdir is not None:
        (logdir / "summary.json").write_text(json.dumps({
            "best_epoch": best["epoch"],
            "best_macro_f1": best["score"],
            "final_val": {k: (float(final[k]) if isinstance(final[k], (int,float,np.floating)) else None) for k in ("acc","macro_f1")},
            "valid_confusion_matrix": str(cm_path.name) if cm_path else None,
            "extra_metrics": {k: (float(v) if isinstance(v, (int,float,np.floating)) else None) for k,v in (extra_metrics or {}).items()},
            "extra_confusion_matrix": str(extra_cm_path.name) if extra_cm_path else None,
        }, ensure_ascii=False, indent=2))

    summary = {
        "val_best_epoch": best["epoch"],
        "val_best_macro_f1": best["score"],
        "val_metrics_final": {k: final[k] for k in ("acc","macro_f1")},
        "confusion_matrix_path": str(cm_path) if cm_path else None,
        "extra_metrics": extra_metrics,
        "extra_confusion_matrix_path": str(extra_cm_path) if extra_cm_path else None,
    }
    return model, summary


# ----------------------- CLI entry ---------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "train"], default="smoke")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--val", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logs", type=str, default="../logs")
    p.add_argument("--train-csv", type=str, default="", help="Path to training csv.gz (e.g., ...2501_1min.csv.gz)")
    p.add_argument("--eval-csv", type=str, default="", help="Optional extra eval csv.gz (e.g., ...2502_1min.csv.gz)")
    args = p.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)

    # Create per-run log dir in JST and logger
    logdir = mk_logdir(args.logs)
    logger = configure_logger(logdir)
    logger.info(f"Device selected: {device}")

    # Save run config
    cfg_dump = vars(args).copy()
    cfg_dump["device_resolved"] = device
    (logdir / "config.json").write_text(json.dumps(cfg_dump, ensure_ascii=False, indent=2))

    L = args.seq_len

    # Model constraints (reduce fixed to 4)
    assert L % args.heads == 0, "seq-len must be divisible by heads"
    assert args.hidden % args.heads == 0, "hidden must be divisible by heads"
    assert L % 4 == 0 and args.hidden % 4 == 0, "seq-len and hidden must be divisible by 4 (reduce=4)"

    # Data
    if args.train_csv:
        df = load_lob_df(args.train_csv)
        feature_count = df.shape[1]
        b1_index = df.columns.get_loc("b1")
        a1_index = df.columns.get_loc("a1")
    else:
        df = make_synth_df_lob10(T=max(500, L * 10))
        feature_count = df.shape[1]
        b1_index = 0
        a1_index = 20  # depth=10 -> a1 starts at index 20

    if args.mode == "smoke":
        lab = LabelingConfig(
            horizon_steps=5, smooth_len=5, epsilon_mode="avg_spread", epsilon_scale=0.5
        )
        X, y, t = make_lob_window_dataset(
            df, feature_count=feature_count, lookback=L, b1_index=b1_index, a1_index=a1_index, labeling=lab
        )
        logger.info(f"Dataset: {tuple(X.shape)} {tuple(y.shape)} ({len(t)})")

        cfg = TLOBConfig(
            hidden_dim=args.hidden,
            num_layers=args.layers,
            seq_len=L,
            num_features=feature_count,
            num_heads=args.heads,
            reduce_factor=4,
            use_sinusoidal_pos_emb=True,
        )
        model = TLOB(cfg).to(device)

        # Forward + one step
        batch = min(args.batch, len(X))
        x = X[:batch].to(dtype=torch.float32, device=device)
        yb = y[:batch].to(dtype=torch.long, device=device)

        logits = model(x)
        logger.info(f"Logits: {tuple(logits.shape)}")

        loss = nn.CrossEntropyLoss()(logits, yb)
        logger.info(f"Loss: {float(loss.item())}")

        opt = AdamW(model.parameters(), lr=args.lr)
        for s in range(args.steps):
            opt.zero_grad()
            logits = model(x)
            loss = nn.CrossEntropyLoss()(logits, yb)
            loss.backward()
            opt.step()
            logger.info(f"Step {s+1}/{args.steps}: loss={float(loss.item()):.4f}")

    else:  # train
        # Prepare optional eval CSV name for CM file naming
        eval_df = load_lob_df(args.eval_csv) if args.eval_csv else None
        if args.eval_csv:
            name = Path(args.eval_csv).name
            eval_name = name[:-7] if name.endswith(".csv.gz") else Path(args.eval_csv).stem
        else:
            eval_name = None

        model, summary = train_tlob(
            df=df,
            lookback=L,
            horizon_steps=5,
            d_model=args.hidden,
            n_heads=args.heads,
            n_layers=args.layers,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            val_ratio=args.val,
            seed=args.seed,
            device=device,
            logdir=logdir,
            logger=logger,
            eval_df=eval_df,
            eval_name=eval_name,
        )
        logger.info(f"Best epoch: {summary['val_best_epoch']}")
        logger.info(f"Best macro_f1: {summary['val_best_macro_f1']}")
        logger.info(f"Final val: {summary['val_metrics_final']}")
        if summary.get("confusion_matrix_path"):
            logger.info(f"Validation CM: {summary['confusion_matrix_path']}")
        if summary.get("extra_confusion_matrix_path"):
            logger.info(f"Extra CM: {summary['extra_confusion_matrix_path']}")


if __name__ == "__main__":
    main()
