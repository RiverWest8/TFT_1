

"""
Temporal Fusion Transformer (TFT) pipeline
==========================================
python3 TFT.py   --max_epochs 10   --batch_size 256   --max_encoder_length 64   --check_val_every_n_epoch 1   --log_every_n_steps 50   --num_workers 12   --prefetch_factor 2   --enable_perm_importance false   --perm_len 288   --fi_max_batches 15   --resume false   --gcs_data_prefix gs://river-ml-bucket/Data/CleanedData   --gcs_output_prefix gs://river-ml-bucket/Dissertation/Feature_Ablation/f_$(date -u +%s)


This script trains a single TFT model that jointly predicts:
  • realised volatility (quantile regression on an asinh‑transformed target)
  • the direction of the next period’s price move (binary classification)

It expects three parquet files:
    ▸ /Users/riverwest-gomila/Desktop/Data/CleanedData/universal_train.parquet
    ▸ /Users/riverwest-gomila/Desktop/Data/CleanedData/universal_val.parquet
    ▸ /Users/riverwest-gomila/Desktop/Data/CleanedData/universal_test.parquet

Required columns (exact names or common aliases will be auto‑detected):
    asset         : categorical asset identifier (aliases: symbol, ticker, instrument)
    Time          : timestamp (parsed to pandas datetime)
    realised_vol  : one‑period realised volatility target
    direction     : 0/1 label for next‑period price direction
plus any engineered numeric features

What the script does:
  1) Loads the parquet splits and standardises column names.
  2) Adds a per‑asset integer `time_idx` required by PyTorch‑Forecasting.
  3) Builds a `TimeSeriesDataSet` with **two targets**: ["realised_vol", "direction"].
     • Target normalisation:
         – realised_vol: `GroupNormalizer(..., transformation="asinh", scale_by_group=True)`
           (applies asinh in the normaliser, per asset)
         – direction: identity (no transform)
     • A per‑asset median `rv_scale` is also attached for **fallback decoding only**
       if a normaliser decode is unavailable.
  4) Fits a TemporalFusionTransformer with a dual head: 7 quantiles for vol and one
     logit for direction, using `LearnableMultiTaskLoss(AsymmetricQuantileLoss, LabelSmoothedBCE)`.
  5) Saves checkpoints, metrics, predictions, and quick feature‑importance summaries.

Adjust hyper‑parameters in the CONFIG section below.
"""

import warnings
warnings.filterwarnings("ignore")
from lightning.pytorch.callbacks import TQDMProgressBar
import os
from pathlib import Path
from typing import List
import json
import numpy as np
import pandas as pd
import math
import pandas as _pd
pd = _pd  # Ensure pd always refers to pandas module
import lightning.pytorch as pl





BATCH_SIZE   = 128
MAX_EPOCHS   = 35
EARLY_STOP_PATIENCE = 10
PERM_BLOCK_SIZE = 288

# --- Allowlist PF objects for torch.load under PyTorch>=2.6 (weights_only=True default) ---
try:
    from torch.serialization import add_safe_globals, safe_globals as _safe_globals_ctx
except Exception:
    _safe_globals_ctx = None
    def add_safe_globals(_): pass

try:
    from pytorch_forecasting.data.encoders import MultiNormalizer as _PFMultiNormalizer
    add_safe_globals([_PFMultiNormalizer])
except Exception:
    _PFMultiNormalizer = None

# Extra belt-and-braces: swallow BrokenPipe errors on stdout.flush() if any other lib calls it.
try:
    import sys
    _orig_flush = sys.stdout.flush
    def _safe_flush(*a, **k):
        try:
            return _orig_flush(*a, **k)
        except BrokenPipeError:
            return None
    sys.stdout.flush = _safe_flush
except Exception:
    pass

from torch.utils.data._utils.collate import default_collate

def _clean_none(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _clean_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        cleaned = [_clean_none(v) for v in obj if v is not None]
        return type(obj)(v for v in cleaned if v is not None)
    return obj

def make_safe_collate(base=None):
    base = base or default_collate
    def _safe(batch):
        batch = [b for b in batch if b is not None]
        batch = [_clean_none(b) for b in batch]
        batch = [b for b in batch if b not in (None, (), [], {})]
        return base(batch)
    return _safe

# Route all prints to stderr and swallow BrokenPipe to avoid crashes during teardown
import builtins as _builtins, sys as _sys
__orig_print = _builtins.print
def __safe_print(*args, **kwargs):
    try:
        if "file" not in kwargs or kwargs["file"] is None:
            kwargs["file"] = _sys.stderr   # default to stderr
        return __orig_print(*args, **kwargs)
    except BrokenPipeError:
        return None
_builtins = _builtins  # keep a name bound
_builtins.print = __safe_print

from lightning.pytorch import Trainer, seed_everything
from torchmetrics.classification import BinaryAUROC
import torch
import torch.nn as nn
import torch.nn.functional as F
def _permute_series_inplace(df: pd.DataFrame, col: str, block: int, group_col: str = "asset") -> None:
    if col not in df.columns:
        return
    if group_col not in df.columns:
        vals = df[col].values.copy()
        np.random.shuffle(vals)
        df[col] = vals
        return
    for _, idx in df.groupby(group_col, observed=True).groups.items():
        idx = np.asarray(list(idx))
        if block and block > 1:
            shift = np.random.randint(1, max(2, len(idx)))
            df.loc[idx, col] = df.loc[idx, col].values.take(np.arange(len(idx)) - shift, mode='wrap')
        else:
            vals = df.loc[idx, col].values.copy()
            np.random.shuffle(vals)
            df.loc[idx, col] = vals


# helper (put once near the top of the file, or inline if you prefer)
def _first_not_none(d, keys):
    for k in keys:
        v = d.get(k, None)
        if v is not None:
            return v
    return None

# ------------------ Imports ------------------

from pytorch_forecasting import (
    BaseModel,
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)
from pytorch_forecasting.data import MultiNormalizer, TorchNormalizer

from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger
# ---- Compatibility stub: safely neutralize TQDMProgressBar ----
try:
    from lightning.pytorch.callbacks import TQDMProgressBar as _RealTQDM
    # If the real class exists but progress bars are globally disabled, alias a no-op wrapper
    class TQDMProgressBar(pl.Callback):
        def __init__(self, *args, **kwargs):
            super().__init__()
        # No methods overridden → progress bar will not be used/initialized
except Exception:
    class TQDMProgressBar(pl.Callback):
        def __init__(self, *args, **kwargs):
            super().__init__()

# --- Cloud / performance helpers ---
import fsspec
try:
    import gcsfs  # ensure GCS protocol is registered with fsspec
except Exception:
    gcsfs = None

# CUDA perf knobs / utils
import shutil
import argparse

import pytorch_forecasting as pf
import inspect

VOL_QUANTILES = [0.05, 0.165, 0.25, 0.50, 0.75, 0.835, 0.95] #The quantiles which yielded our best so far
#Q50_IDX = VOL_QUANTILES.index(0.50)  
#VOL_QUANTILES = [0.05, 0.15, 0.35, 0.50, 0.65, 0.85, 0.95]
Q50_IDX = VOL_QUANTILES.index(0.50)
# Floor used when computing decoded QLIKE to avoid blow-ups when y or ŷ ~ 0
EVAL_VOL_FLOOR = 1e-6

# Composite metric weights (override via --metric_weights "w_mae,w_rmse,w_qlike")
COMP_WEIGHTS = (1.0, 1.0, 0.004)  # default: slightly emphasise QLIKE for RV focus

def composite_score(mae, rmse, qlike,
                    b_mae=None, b_rmse=None, b_qlike=None,
                    weights=None, eps=1e-12):
    """If baselines are provided → normalised composite (for FI).
       Else → absolute composite (for validation)."""
    w_mae, w_rmse, w_qlike = (weights or COMP_WEIGHTS)
    if b_mae is None or b_rmse is None or b_qlike is None:
        return w_mae*float(mae) + w_rmse*float(rmse) + w_qlike*float(qlike)
    mae_r   = float(mae)   / max(eps, float(b_mae))
    rmse_r  = float(rmse)  / max(eps, float(b_rmse))
    qlike_r = float(qlike) / max(eps, float(b_qlike))
    return w_mae*mae_r + w_rmse*rmse_r + w_qlike*qlike_r
# -----------------------------------------------------------------------
# Ensure a robust "identity" transformation for GroupNormalizer
# -----------------------------------------------------------------------
#
# Different versions of PyTorch‑Forecasting store transformations as a
# dictionary mapping *name* ➜ {"forward": fn, "inverse": fn}.  Some older
# releases omit "identity" entirely, which triggers a KeyError.  Other
# versions include it but as a bare function instead of a dict, which then
# breaks later when `.setdefault()` is called on it.  The logic below
# handles both cases safely.
#

from pytorch_forecasting.data.encoders import GroupNormalizer

# Add a robust log1p mapping as well (for PF versions that lack it or store a bare function)
if ("log1p" not in GroupNormalizer.TRANSFORMATIONS
    or not isinstance(GroupNormalizer.TRANSFORMATIONS["log1p"], dict)):
    GroupNormalizer.TRANSFORMATIONS["log1p"] = {
        "forward": lambda x: torch.log1p(x) if torch.is_tensor(x) else np.log1p(x),
        "inverse": lambda x: torch.expm1(x) if torch.is_tensor(x) else np.expm1(x),
    }
if hasattr(GroupNormalizer, "INVERSE_TRANSFORMATIONS"):
    GroupNormalizer.INVERSE_TRANSFORMATIONS.setdefault(
        "log1p",
        lambda x: torch.expm1(x) if torch.is_tensor(x) else np.expm1(x),
    )

def _extract_norm_from_dataset(ds):
    """
    Return the *volatility* GroupNormalizer used for the first target (realised_vol).
    Works across PF versions where target_normalizer may be:
      - a GroupNormalizer,
      - a MultiNormalizer holding a list in .normalizers / .normalization,
      - a dict mapping target name -> normalizer.
    Preference order: GroupNormalizer for realised_vol, else first normalizer.
    """
    tn = getattr(ds, "target_normalizer", None)
    if tn is None:
        raise ValueError("TimeSeriesDataSet has no target_normalizer")

    # dict mapping target name -> normalizer
    if isinstance(tn, dict):
        for v in tn.values():
            if isinstance(v, GroupNormalizer):
                return v
        return next(iter(tn.values()))

    # MultiNormalizer (PF)
    try:
        from pytorch_forecasting.data import MultiNormalizer as _PFMulti
    except Exception:
        _PFMulti = MultiNormalizer

    if isinstance(tn, _PFMulti):
        norms = getattr(tn, "normalizers", None) or getattr(tn, "normalization", None) or getattr(tn, "_normalizers", None)
        if isinstance(norms, (list, tuple)) and norms:
            for n in norms:
                if isinstance(n, GroupNormalizer):
                    return n
            return norms[0]
        return tn  # unknown container

    # already a single normalizer
    return tn


def _point_from_quantiles(vol_q: torch.Tensor) -> torch.Tensor:
    """
    Enforce non-decreasing quantiles along last dim and return the median (q=0.5).
    Assumes VOL_QUANTILES has q=0.50 at index 3.
    """
    vol_q = torch.cummax(vol_q, dim=-1).values
    return vol_q[..., 3]  # Q50_IDX

#EXTRACT HEADSSSSS
def _extract_heads(pred):
    """
    Return (p_vol, p_dir) as 1D tensors [B] on DEVICE.

    Handles outputs as:
      • list/tuple: [vol(…,K=7), dir(…,K or 1)]
      • tensor [B, 2, K]        (vol quantiles + dir replicated or single)
      • tensor [B, 1, K]        (vol only)
      • tensor [B, K]           (vol only)
      • tensor [B, K+1]         (concatenated: first 7 vol, last 1 dir)
      • tensor [B, 1, K+1]      (concatenated with a singleton middle dim)
    """
    import torch

    K = 7         # len(VOL_QUANTILES)
    MID = 3       # Q50 index when K=7

    def _pick_median_q(vol_q: torch.Tensor) -> torch.Tensor:
        # make sure last dim has the K quantiles
        if vol_q.ndim == 3 and vol_q.size(1) == 1:
            vol_q = vol_q.squeeze(1)            # [B, K]
        if vol_q.ndim == 3 and vol_q.size(-1) == 1 and vol_q.size(1) == K:
            vol_q = vol_q.squeeze(-1)           # [B, K]
        if vol_q.ndim == 2 and vol_q.size(-1) >= K:
            v = vol_q[:, :K]                    # keep exactly 7
        elif vol_q.ndim == 2 and vol_q.size(-1) == K:
            v = vol_q
        else:
            v = vol_q.reshape(vol_q.size(0), -1)[:, :K]
        # enforce monotone and take median
        v = torch.cummax(v, dim=-1).values
        return v[:, MID]                        # [B]

    def _pick_dir_logit(dir_t):
        if dir_t is None:
            return None
        if torch.is_tensor(dir_t) and dir_t.ndim == 3 and dir_t.size(1) == 1:
            dir_t = dir_t.squeeze(1)            # [B, K] or [B, 1]
        if torch.is_tensor(dir_t) and dir_t.ndim == 3 and dir_t.size(-1) == 1:
            dir_t = dir_t.squeeze(-1)           # [B, L]
        if torch.is_tensor(dir_t) and dir_t.ndim == 2:
            # if replicated across K, take middle; if 1, take that; else take last
            L = dir_t.size(-1)
            if L >= K:
                return dir_t[:, MID]
            return dir_t[:, -1]
        if torch.is_tensor(dir_t) and dir_t.ndim == 1:
            return dir_t
        if torch.is_tensor(dir_t):
            return dir_t.reshape(dir_t.size(0), -1)[:, -1]
        return None

    # ---------------- cases ----------------

    # Case A: list/tuple from PF: [vol, dir]
    if isinstance(pred, (list, tuple)):
        vol_q = pred[0]
        dir_t = pred[1] if len(pred) > 1 else None
        if torch.is_tensor(vol_q) and vol_q.ndim == 3 and vol_q.size(1) == 1:
            vol_q = vol_q.squeeze(1)            # [B, K]
        p_vol = _pick_median_q(vol_q) if torch.is_tensor(vol_q) else None
        p_dir = _pick_dir_logit(dir_t)
        return p_vol, p_dir

    if not torch.is_tensor(pred):
        return None, None

    t = pred

    # Squeeze a singleton prediction-length dim if present
    if t.ndim == 4 and t.size(1) == 1:
        t = t.squeeze(1)                         # [B, C, D]
    if t.ndim == 3 and t.size(1) == 1 and t.size(-1) >= 1:
        t = t[:, 0, :]                           # [B, D]

    # Case B: [B, 2, K]  → first = vol quantiles, second = dir (K or 1 replicated)
    if t.ndim == 3 and t.size(1) == 2:
        vol_q = t[:, 0, :]                       # [B, K] (or [B, >=K], we’ll trim)
        dir_t = t[:, 1, :]
        # sometimes dir_t is shape [B, 1] replicated to [B, K]; handle both
        if dir_t.ndim == 2 and dir_t.size(-1) == 1:
            dir_t = dir_t[:, 0]
        p_vol = _pick_median_q(vol_q)
        p_dir = _pick_dir_logit(dir_t)
        return p_vol, p_dir

    # Case C: [B, K, 2]  → last dim=2 (vol, dir)
    if t.ndim == 3 and t.size(-1) == 2 and t.size(1) >= K:
        vol_q = t[:, :K, 0]                      # [B, K]
        dir_t = t[:, :K, 1]                      # [B, K] (or fewer)
        p_vol = _pick_median_q(vol_q)
        p_dir = _pick_dir_logit(dir_t)
        return p_vol, p_dir

    # Case D: [B, K+1] (concat: first K vol, last 1 dir) or [B, K] (vol-only)
    if t.ndim == 2:
        D = t.size(-1)
        if D >= K + 1:
            vol_q = t[:, :K]
            d_log = t[:, K]
            return _pick_median_q(vol_q), d_log
        if D == K:
            return _pick_median_q(t), None
        if D == 1:
            return t.squeeze(-1), None
        # Unknown 2D layout → try to peel first K as vol and last as dir
        vol_q = t[:, :K] if D > K else t
        d_log = t[:, -1] if D > K else None
        return _pick_median_q(vol_q), d_log

    # Fallback: flatten and try our best
    if t.ndim >= 1:
        flat = t.reshape(t.size(0), -1)
        vol_q = flat[:, :K]
        d_log = flat[:, K] if flat.size(-1) > K else None
        return _pick_median_q(vol_q), d_log

    return None, None


    


# -----------------------------------------------------------------------
# Robust manual inverse for GroupNormalizer (fallback when decode fails)
# -----------------------------------------------------------------------
@torch.no_grad()
def manual_inverse_transform_groupnorm(normalizer, y: torch.Tensor, group_ids: torch.Tensor | None):
    """
    Reverse GroupNormalizer:
      1) destandardize: y -> y*scale + center (center may be None when center=False)
      2) inverse transform: asinh -> sinh, identity -> no-op, else via registry
    Works with scale_by_group=True (index by group_ids) or global scale.
    """
    y_ = y.squeeze(-1) if y.ndim > 1 else y

    # right after: y_ = y.squeeze(-1) if y.ndim > 1 else y
    dev = None
    if torch.is_tensor(y_):
        dev = y_.device
    if group_ids is not None and torch.is_tensor(group_ids):
        dev = group_ids.device

    center = getattr(normalizer, "center", None)
    scale  = getattr(normalizer, "scale",  None)

    if dev is not None:
        if isinstance(center, torch.Tensor) and center.device != dev:
            center = center.to(dev)
        if isinstance(scale, torch.Tensor) and scale.device != dev:
            scale = scale.to(dev)

    if scale is None:
        x = y_
    else:
        if group_ids is not None and torch.is_tensor(group_ids):
            g = group_ids
            if g.ndim > 1 and g.size(-1) == 1:
                g = g.squeeze(-1)
            g = g.long()
            # Ensure index tensor `g` is on the same device as scale/center
            if isinstance(scale, torch.Tensor) and g.device != scale.device:
                g = g.to(scale.device)
            if isinstance(center, torch.Tensor) and g.device != center.device:
                g = g.to(center.device)
            s = scale[g] if isinstance(scale, torch.Tensor) else scale
            c = center[g] if isinstance(center, torch.Tensor) else 0.0
        else:
            s = scale
            c = center if isinstance(center, torch.Tensor) else 0.0
        x = y_ * s + c

    tfm = getattr(normalizer, "transformation", None)
    if tfm == "log1p":
        x = torch.expm1(x)
    elif tfm in (None, "identity"):
        pass
    else:
        try:
            inv = type(normalizer).TRANSFORMATIONS[tfm]["inverse"]
            x = inv(x)
        except Exception:
            pass

    return x.view_as(y)

@torch.no_grad()
def _get_best_ckpt_path(trainer) -> str | None:
    """Return path to the best checkpoint chosen by `val_qlike_overall` if available.
    Fallback to any `best_model_path`, else newest local .ckpt in LOCAL_CKPT_DIR.
    """
    best_path = None
    try:
        for cb in getattr(trainer, "callbacks", []):
            if isinstance(cb, ModelCheckpoint):
                mon = getattr(cb, "monitor", None)
                bmp = getattr(cb, "best_model_path", "")
                if mon == "val_qlike_overall" and bmp:
                    return bmp
        if not best_path:
            for cb in getattr(trainer, "callbacks", []):
                if isinstance(cb, ModelCheckpoint):
                    bmp = getattr(cb, "best_model_path", "")
                    if bmp:
                        best_path = bmp
                        break
    except Exception:
        pass
    if best_path:
        return best_path
    try:
        ckpts = sorted(LOCAL_CKPT_DIR.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if ckpts:
            return str(ckpts[0])
    except Exception:
        pass
    return None


@torch.no_grad()
def _collect_predictions(
    model,
    dataloader,
    vol_normalizer=None,
    vol_norm=None,
    id_to_name=None,
    out_path=None,
    cal_scale=None,
    per_asset_scales=None,
    **kwargs
):
    """
    Robust exporter that tolerates PF's different batch shapes:
      (x, y, *rest), ((x, y), *rest), dict-only, or nested lists/tuples.
    """
    import pandas as pd
    model.eval()

    # accept either alias
    if vol_normalizer is None:
        vol_normalizer = vol_norm or kwargs.get("vol_norm", None)
    if vol_normalizer is None:
        raise RuntimeError("vol_normalizer/vol_norm must be provided to _collect_predictions")

    # --- helpers to peel nested structures safely ---
    def _find_first_dict(obj):
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, (list, tuple)):
            for it in obj:
                got = _find_first_dict(it)
                if got is not None:
                    return got
        return None

    def _find_first_tensor(obj):
        import torch
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, (list, tuple)):
            for it in obj:
                got = _find_first_tensor(it)
                if got is not None:
                    return got
        return None

    all_g, all_yv, all_pv, all_yd, all_pd, all_t = [], [], [], [], [], []

    for batch in dataloader:
        if batch is None:
            continue

        # 1) locate x dict and (optional) y tensor anywhere inside the batch
        x = None
        yb = None
        if isinstance(batch, dict):
            x = batch
        else:
            x = _find_first_dict(batch)
            yb = _find_first_tensor(batch)

        if not isinstance(x, dict):
            # can’t use this batch
            continue

        # 2) group ids
        groups = None
        for k in ("groups", "group_ids", "group_id"):
            try:
                if k in x and x[k] is not None:
                    groups = x[k]
                    break
            except Exception:
                pass
        if groups is None:
            continue
        g = groups[0] if isinstance(groups, (list, tuple)) and len(groups) > 0 else groups
        import torch
        while torch.is_tensor(g) and g.ndim > 1 and g.size(-1) == 1:
            g = g.squeeze(-1)
        if not torch.is_tensor(g):
            continue

        # 3) targets from decoder, else fallback to yb
        y_vol_t, y_dir_t = None, None
        try:
            dec_t = x.get("decoder_target", None)
        except Exception:
            dec_t = None

        if torch.is_tensor(dec_t):
            y = dec_t
            if y.ndim == 3 and y.size(-1) == 1:
                y = y[..., 0]
            if y.ndim == 2 and y.size(1) >= 1:
                y_vol_t = y[:, 0]
                if y.size(1) > 1:
                    y_dir_t = y[:, 1]
        elif isinstance(dec_t, (list, tuple)) and len(dec_t) >= 1:
            y_vol_t = dec_t[0]
            if torch.is_tensor(y_vol_t):
                if y_vol_t.ndim == 3 and y_vol_t.size(-1) == 1:
                    y_vol_t = y_vol_t[..., 0]
                if y_vol_t.ndim == 2 and y_vol_t.size(-1) == 1:
                    y_vol_t = y_vol_t[:, 0]
            if len(dec_t) > 1 and torch.is_tensor(dec_t[1]):
                y_dir_t = dec_t[1]
                if y_dir_t.ndim == 3 and y_dir_t.size(-1) == 1:
                    y_dir_t = y_dir_t[..., 0]
                if y_dir_t.ndim == 2 and y_dir_t.size(-1) == 1:
                    y_dir_t = y_dir_t[:, 0]

        if y_vol_t is None and torch.is_tensor(yb):
            yy = yb
            if yy.ndim == 3 and yy.size(1) == 1:
                yy = yy[:, 0, :]
            if yy.ndim == 2 and yy.size(1) >= 1:
                y_vol_t = yy[:, 0]
                if y_dir_t is None and yy.size(1) > 1:
                    y_dir_t = yy[:, 1]

        if y_vol_t is None:
            # no realised_vol target — skip
            continue

        # 4) forward + head extraction
        try:
            y_hat = model(x)
        except Exception:
            continue
        pred = getattr(y_hat, "prediction", y_hat)
        if isinstance(pred, dict) and "prediction" in pred:
            pred = pred["prediction"]

        p_vol, p_dir = _extract_heads(pred)
        if p_vol is None:
            continue

        # 5) accumulate on CPU
        L = g.shape[0]
        all_g.append(g.reshape(L).detach().cpu())
        all_yv.append(y_vol_t.reshape(L).detach().cpu())
        all_pv.append(p_vol.reshape(L).detach().cpu())

        if y_dir_t is not None and p_dir is not None:
            y_flat = y_dir_t.reshape(-1)
            p_flat = p_dir.reshape(-1)
            L2 = min(L, y_flat.numel(), p_flat.numel())
            if L2 > 0:
                all_yd.append(y_flat[:L2].detach().cpu())
                all_pd.append(p_flat[:L2].detach().cpu())

        # Optional time index
        try:
            dec_time = x.get("decoder_time_idx", None)
            if dec_time is None:
                dec_time = x.get("decoder_relative_idx", None)
            if torch.is_tensor(dec_time):
                tvec = dec_time
                while tvec.ndim > 1 and tvec.size(-1) == 1:
                    tvec = tvec.squeeze(-1)
                all_t.append(tvec.reshape(-1)[:L].detach().cpu())
        except Exception:
            pass

    if not all_g:
        raise RuntimeError("No predictions collected — dataloader yielded no usable batches.")

    # assemble
    g_cpu  = torch.cat(all_g)
    yv_cpu = torch.cat(all_yv)
    pv_cpu = torch.cat(all_pv)

    # decode (robust)
    yv_dec = safe_decode_vol(yv_cpu.unsqueeze(-1), vol_normalizer, g_cpu.unsqueeze(-1)).squeeze(-1)
    pv_dec = safe_decode_vol(pv_cpu.unsqueeze(-1), vol_normalizer, g_cpu.unsqueeze(-1)).squeeze(-1)

    floor_val = float(globals().get("EVAL_VOL_FLOOR", 1e-6))
    yv_dec = torch.clamp(yv_dec, min=floor_val)
    pv_dec = torch.clamp(pv_dec, min=floor_val)

    # global/per-asset calib (optional)
    if cal_scale is None:
        cal_scale = kwargs.get("calibration_scale", None)
    if cal_scale is not None:
        try:
            pv_dec = pv_dec * float(cal_scale)
        except Exception:
            pass

    if per_asset_scales is None:
        per_asset_scales = kwargs.get("per_asset_calibration", None)

    id_to_name = id_to_name or {}
    assets = [id_to_name.get(int(i), str(int(i))) for i in g_cpu.numpy().tolist()]

    if isinstance(per_asset_scales, dict) and len(per_asset_scales) > 0:
        try:
            _map = {str(k): float(v) for k, v in per_asset_scales.items()}
            scales_vec = torch.tensor([_map.get(str(a), 1.0) for a in assets], dtype=pv_dec.dtype)
            pv_dec = pv_dec * scales_vec
        except Exception as _e:
            print(f"[WARN] per-asset calibration failed, skipping: {_e}")

    t_cpu = torch.cat(all_t) if all_t else None

    df = pd.DataFrame({
        "asset": assets,
        "time_idx": t_cpu.numpy().tolist() if t_cpu is not None else [None] * len(assets),
        "y_vol": yv_dec.numpy().tolist(),
        "y_vol_pred": pv_dec.numpy().tolist(),
    })

    if all_yd and all_pd:
        yd_all = torch.cat(all_yd)
        pd_all = torch.cat(all_pd)
        try:
            if torch.isfinite(pd_all).any() and (pd_all.min() < 0 or pd_all.max() > 1):
                pd_all = torch.sigmoid(pd_all)
        except Exception:
            pd_all = torch.sigmoid(pd_all)
        pd_all = torch.clamp(pd_all, 0.0, 1.0)
        Lm = min(len(df), yd_all.numel(), pd_all.numel())
        df = df.iloc[:Lm].copy()
        df["y_dir"] = yd_all[:Lm].numpy().tolist()
        df["y_dir_prob"] = pd_all[:Lm].numpy().tolist()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"✓ Saved {len(df)} predictions → {out_path}")
        return out_path
    return df

# ---- single-worker export loader (stable & safe) ----
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate


def make_export_loader(dl):
    """
    Rebuild `dl` as a single-worker DataLoader and *normalize* sample layout so
    PyTorch-Forecasting's _collate_fn never hits `batches[0][1][1]` when weight is missing.

    We coerce each sample to the canonical shape:
        (x, (y, weight))    # weight may be None

    Additionally, we filter out None samples inside a batch so `default_collate`
    never sees `NoneType`.
    """
    from torch.utils.data import DataLoader
    from torch.utils.data._utils.collate import default_collate

    ds = dl.dataset
    bs = getattr(dl, "batch_size", 128) or 128
    pf_collate = getattr(ds, "_collate_fn", None)  # PF's own collate, if present

    def _massage_samples(samples):
        fixed = []
        for s in samples:
            if s is None:
                continue
            # target formats we see: (x, y), (x, (y,)), (x, (y, w)), (x, y, w)
            if isinstance(s, tuple):
                if len(s) == 3:
                    x, y, w = s
                    # if y already a (y, w2) tuple, keep its y; otherwise wrap
                    if isinstance(y, tuple):
                        y = (y[0], w if len(y) < 2 else y[1])
                    else:
                        y = (y, w)
                    fixed.append((x, y))
                    continue
                if len(s) == 2:
                    x, y = s
                    if isinstance(y, tuple):
                        # (y,) -> (y, None); (y,w,...) -> (y,w)
                        if len(y) == 1:
                            y = (y[0], None)
                        else:
                            y = (y[0], y[1])
                    else:
                        y = (y, None)
                    fixed.append((x, y))
                    continue
            # Drop truly empty containers
            if isinstance(s, (list, tuple, dict)) and len(s) == 0:
                continue
            fixed.append(s)
        return fixed

    def fixed_collate(samples):
        # 1) filter Nones up-front
        samples = [s for s in samples if s is not None]
        # 2) normalise structure
        samples = _massage_samples(samples)
        if not samples:
            # return a dummy minimal batch the downstream loop will skip
            import torch
            return {"groups": torch.empty(0, dtype=torch.long)}

        # 3) Prefer PF's collate if available; else fall back to a "safe" default collate
        if pf_collate is not None:
            try:
                return pf_collate(samples)
            except Exception as e:
                print(f"[EXPORT] PF collate failed ({e}); falling back to default_collate.")
        # Default collate after cleaning
        return default_collate(samples)

    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=0,              # single worker to avoid hangs
        pin_memory=False,
        drop_last=False,
        collate_fn=fixed_collate,   # our normalized + safe collate
    )

@torch.no_grad()
def _save_predictions_from_best(trainer, dataloader, split_name: str, out_path: Path | str,
                                id_to_name: dict | None = None):
    """
    GOOD1-STYLE EXPORT:
      • load best ckpt on CPU
      • iterate dataloader with num_workers=0
      • forward pass
      • take median RV quantile (+ optional direction)
      • decode with dataset.target_normalizer
      • write ONE parquet at out_path
    """
    ckpt = _get_best_ckpt_path(trainer)
    if ckpt is None:
        raise RuntimeError("Could not resolve best checkpoint path.")
    print(f"Loading best checkpoint for {split_name}: {ckpt}")

    # Safe load for PyTorch 2.6 (weights_only=True default)
    try:
        if _safe_globals_ctx is not None and _PFMultiNormalizer is not None:
            with _safe_globals_ctx([_PFMultiNormalizer]):
                model = TemporalFusionTransformer.load_from_checkpoint(ckpt, map_location="cpu")
        else:
            model = TemporalFusionTransformer.load_from_checkpoint(ckpt, map_location="cpu")
    except TypeError:
        # Older Lightning/PyTorch
        model = TemporalFusionTransformer.load_from_checkpoint(ckpt, map_location="cpu", weights_only=False)

    model.eval()

    # Resolve normalizer from THIS dataset (good1 style: use dataset’s own)
    vol_norm = _extract_norm_from_dataset(dataloader.dataset)

    export_loader = make_export_loader(dataloader)

    # Collect predictions (classic/simple)
    df = _classic_collect(model, export_loader, vol_norm=vol_norm, id_to_name=id_to_name or {})

    # Save
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"✓ Saved {len(df)} predictions → {out_path}")

    return out_path

@torch.no_grad()
def _classic_collect(model, dataloader, vol_norm, id_to_name: dict):
    """
    Minimal, robust collector (good1.py style):
      • supports PF batch layouts (x, y, …) or dict-only (x)
      • pulls groups & decoder_target from x
      • forward once per batch; extract heads with _extract_heads(...)
      • decode realised_vol via vol_norm
    """
    import pandas as pd
    all_assets, all_tidx, all_y_dec, all_p_dec, all_yd, all_pd = [], [], [], [], [], []

    def _get_x_y(batch):
        # Accept (x, y, …) or x
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            x = batch[0]
            y = batch[1] if len(batch) > 1 else None
            return x, y
        if isinstance(batch, dict):
            return batch, None
        return None, None

    batch_i = 0
    for batch in dataloader:
        batch_i += 1
        if batch_i % 50 == 0:
            print(f"[EXPORT] processed {batch_i} batches…")
        x, y = _get_x_y(batch)
        if not isinstance(x, dict):
            continue

        # groups (asset ids)
        g = None
        for k in ("groups", "group_ids", "group_id"):
            if k in x and x[k] is not None:
                g = x[k]
                break
        if g is None:
            continue
        if isinstance(g, (list, tuple)) and len(g) > 0:
            g = g[0]
        if torch.is_tensor(g) and g.ndim > 1 and g.size(-1) == 1:
            g = g.squeeze(-1)
        if not torch.is_tensor(g):
            continue

        # time index (optional)
        t = x.get("decoder_time_idx", None)
        if t is None:
            t = x.get("decoder_relative_idx", None)
        if torch.is_tensor(t) and t.ndim > 1 and t.size(-1) == 1:
            t = t.squeeze(-1)

        # targets from decoder_target if present; else fall back to y
        y_vol_t, y_dir_t = None, None
        dec_t = x.get("decoder_target", None)
        if torch.is_tensor(dec_t):
            yy = dec_t
            if yy.ndim == 3 and yy.size(-1) == 1:
                yy = yy[..., 0]  # [B, n_targets]
            if yy.ndim == 2 and yy.size(1) >= 1:
                y_vol_t = yy[:, 0]
                if yy.size(1) > 1:
                    y_dir_t = yy[:, 1]
        elif isinstance(dec_t, (list, tuple)) and len(dec_t) >= 1:
            y_vol_t = dec_t[0]
            if torch.is_tensor(y_vol_t) and y_vol_t.ndim == 3 and y_vol_t.size(-1) == 1:
                y_vol_t = y_vol_t[..., 0]
            if len(dec_t) > 1 and torch.is_tensor(dec_t[1]):
                y_dir_t = dec_t[1]
                if y_dir_t.ndim == 3 and y_dir_t.size(-1) == 1:
                    y_dir_t = y_dir_t[..., 0]

        if y_vol_t is None and torch.is_tensor(y):
            yy = y
            if yy.ndim == 3 and yy.size(1) == 1:
                yy = yy[:, 0, :]
            if yy.ndim == 2 and yy.size(1) >= 1:
                y_vol_t = yy[:, 0]
                if y_dir_t is None and yy.size(1) > 1:
                    y_dir_t = yy[:, 1]

        # forward
        try:
            out = model(x)
        except Exception:
            continue
        pred = getattr(out, "prediction", out)
        if isinstance(pred, dict) and "prediction" in pred:
            pred = pred["prediction"]

        p_vol, p_dir = _extract_heads(pred)
        if p_vol is None:
            continue

        # shape guards
        L = g.shape[0]
        if y_vol_t is not None:
            y_vol_t = y_vol_t.reshape(-1)[:L]
        p_vol = p_vol.reshape(-1)[:L]

        # decode realised_vol
        y_dec = None
        if y_vol_t is not None:
            y_dec = safe_decode_vol(y_vol_t.unsqueeze(-1), vol_norm, g.unsqueeze(-1)).squeeze(-1)
        p_dec = safe_decode_vol(p_vol.unsqueeze(-1), vol_norm, g.unsqueeze(-1)).squeeze(-1)

        # clamp tiny vols to avoid qlike blow-ups when consuming later
        floor_val = float(globals().get("EVAL_VOL_FLOOR", 1e-6))
        if y_dec is not None:
            y_dec = torch.clamp(y_dec, min=floor_val)
        p_dec = torch.clamp(p_dec, min=floor_val)

        # map asset ids
        assets = [id_to_name.get(int(a), str(int(a))) for a in g.detach().cpu().numpy().tolist()]
        all_assets.extend(assets)

        # time index (optional)
        if torch.is_tensor(t):
            all_tidx.extend(t.detach().cpu().numpy().tolist()[:L])
        else:
            all_tidx.extend([None] * L)

        # store decoded rv
        if y_dec is not None:
            all_y_dec.extend(y_dec.detach().cpu().numpy().tolist())
        else:
            all_y_dec.extend([None] * L)
        all_p_dec.extend(p_dec.detach().cpu().numpy().tolist())

        # optional direction (convert to prob if needed)
        if (y_dir_t is not None) and (p_dir is not None):
            y_dir_t = y_dir_t.reshape(-1)[:L]
            p_dir   = p_dir.reshape(-1)[:L]
            p_prob = p_dir
            try:
                if torch.isfinite(p_prob).any() and (p_prob.min() < 0 or p_prob.max() > 1):
                    p_prob = torch.sigmoid(p_prob)
            except Exception:
                p_prob = torch.sigmoid(p_prob)
            p_prob = torch.clamp(p_prob, 0.0, 1.0)
            all_yd.extend(y_dir_t.detach().cpu().int().numpy().tolist())
            all_pd.extend(p_prob.detach().cpu().numpy().tolist())
        else:
            all_yd.extend([None] * L)
            all_pd.extend([None] * L)

    # build dataframe
    df = pd.DataFrame({
        "asset": all_assets,
        "time_idx": all_tidx,
        "realised_vol": all_y_dec,           # may include None if y not present for this split
        "pred_realised_vol": all_p_dec,
        "direction": all_yd,
        "pred_direction": all_pd,            # probability in [0,1] if available
    })
    return df


@torch.no_grad()
def safe_decode_vol(y: torch.Tensor, normalizer, group_ids: torch.Tensor | None):
    """
    Try normalizer.decode(), then inverse_transform(), else manual fallback.
    y is expected as [B,1].
    """
    try:
        return normalizer.decode(y, group_ids=group_ids)
    except Exception:
        pass
    try:
        return normalizer.inverse_transform(y, group_ids=group_ids)
    except Exception:
        pass
    try:
        return normalizer.inverse_transform(y)
    except Exception:
        pass
    return manual_inverse_transform_groupnorm(normalizer, y, group_ids)


# ---------------- Global QLIKE‑optimal calibration helper ----------------
@torch.no_grad()
def calibrate_vol_predictions(y_true_dec: torch.Tensor, y_pred_dec: torch.Tensor) -> torch.Tensor:
    """
    Global QLIKE‑optimal multiplicative calibration.

    Given decoded targets y and predictions p, choose a single scalar s that minimises
        E[ (σ_y^2 / (s^2 σ_p^2)) - log(σ_y^2 / (s^2 σ_p^2)) - 1 ].
    The optimum satisfies s^2 = E[σ_y^2] / E[σ_p^2], where σ_x^2 ≡ |x|^2.

    Returns p * s with shape preserved. No piece‑wise/tercile scaling.
    """
    # Basic guards
    if y_true_dec is None or y_pred_dec is None:
        return y_pred_dec

    # Flatten to 1D on a working dtype; keep device
    y = y_true_dec.reshape(-1)
    p = y_pred_dec.reshape(-1)
    if y.numel() == 0 or p.numel() == 0:
        return y_pred_dec

    # Work in float32 for numerical stability, keep device
    device = y.device
    y32 = y.to(dtype=torch.float32, device=device)
    p32 = p.to(dtype=torch.float32, device=device)

    # QLIKE‑optimal global scale: s^2 = E[|y|^2] / E[|p|^2]
    eps = 1e-12
    sigma2_y = torch.clamp(y32.abs(), min=eps) ** 2
    sigma2_p = torch.clamp(p32.abs(), min=eps) ** 2
    s2_opt = sigma2_y.mean() / sigma2_p.mean()
    s = torch.sqrt(torch.clamp(s2_opt, min=eps))

    # Apply and restore original shape/dtype
    p_cal32 = p32 * s
    p_cal = p_cal32.to(dtype=y_pred_dec.dtype)
    return p_cal.view_as(y_pred_dec)

if not hasattr(GroupNormalizer, "decode"):
    def _gn_decode(self, y, group_ids=None, **kwargs):
        """
        Alias for `inverse_transform` to keep newer *and* older
        PyTorch‑Forecasting versions compatible with the same call‑site.

        The underlying `inverse_transform` API has changed a few times:
        ▸ Newer versions accept ``group_ids=`` keyword
        ▸ Some legacy variants want ``X=`` instead
        ▸ Very old releases implement the method but raise ``NotImplementedError``  
          (it was a placeholder).

        The cascading fall‑backs below try each signature in turn and, as a
        last resort, simply return the *input* unchanged so downstream code
        can continue without crashing.
        """
        try:
            # 1️⃣  Modern signature (>=0.10): accepts ``group_ids=``
            return self.inverse_transform(y, group_ids=group_ids, **kwargs)
        except (TypeError, NotImplementedError):
            try:
                # 2️⃣  Mid‑vintage signature: expects ``X=None`` instead
                return self.inverse_transform(y, X=None, **kwargs)
            except (TypeError, NotImplementedError):
                try:
                    # 3️⃣  Very early signature: just (y) positional
                    return self.inverse_transform(y)
                except (TypeError, NotImplementedError):
                    # 4️⃣  Ultimate fall‑back – give up on denorm, return y
                    return y

    GroupNormalizer.decode = _gn_decode

# --- Global storage + helpers for a single, leakage-free validation scale ---
GLOBAL_CAL_SCALE = None  # set on validation end; reused for parquet exports
# Per-asset QLIKE-optimal multiplicative calibration scales (computed on validation)
PER_ASSET_CAL_SCALES = None  # dict: {asset_name(str): float}

def _save_val_per_asset_scales(scales: dict):
    """Persist per-asset validation scales to disk for later reuse (e.g., test)."""
    try:
        path = LOCAL_RUN_DIR / f"val_per_asset_scales_e{MAX_EPOCHS}_{RUN_SUFFIX}.json"
        with open(path, "w") as f:
            _json.dump(scales, f, indent=2)
        print(f"✓ Saved per-asset validation scales → {path}")
    except Exception as e:
        print(f"[WARN] Could not save per-asset validation scales: {e}")

def _load_val_per_asset_scales() -> dict | None:
    """Load most recent saved per-asset validation scales if available."""
    try:
        path = LOCAL_RUN_DIR / f"val_per_asset_scales_e{MAX_EPOCHS}_{RUN_SUFFIX}.json"
        if path.exists():
            with open(path, "r") as f:
                d = _json.load(f)
                return {str(k): float(v) for k, v in d.items()}
        files = sorted(LOCAL_RUN_DIR.glob("val_per_asset_scales_*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            with open(files[0], "r") as f:
                d = _json.load(f)
                return {str(k): float(v) for k, v in d.items()}
    except Exception as e:
        print(f"[WARN] Could not load per-asset validation scales: {e}")
    return None
@torch.no_grad()
def compute_global_scale(y_true_dec: torch.Tensor, y_pred_dec: torch.Tensor, eps: float = 1e-12) -> float:
    """Return QLIKE-optimal multiplicative scale s for decoded vols.
    s^2 = E[y^2] / E[p^2]. Works on any device/dtype; returns python float.
    """
    if y_true_dec is None or y_pred_dec is None:
        return 1.0
    y = y_true_dec.reshape(-1)
    p = y_pred_dec.reshape(-1)
    if y.numel() == 0 or p.numel() == 0:
        return 1.0
    y32 = y.to(dtype=torch.float32)
    p32 = p.to(dtype=torch.float32)
    s2 = torch.clamp((torch.clamp(y32.abs(), min=eps) ** 2).mean() /
                     (torch.clamp(p32.abs(), min=eps) ** 2).mean(), min=eps)
    return float(torch.sqrt(s2).item())

import json as _json

def _save_val_cal_scale(scale: float):
    """Persist the validation calibration scale to disk for later reuse (e.g., test)."""
    try:
        path = LOCAL_RUN_DIR / f"val_cal_scale_e{MAX_EPOCHS}_{RUN_SUFFIX}.json"
        with open(path, "w") as f:
            _json.dump({"scale": float(scale)}, f)
        print(f"✓ Saved validation calibration scale s={scale:.6g} → {path}")
    except Exception as e:
        print(f"[WARN] Could not save validation calibration scale: {e}")

def _load_val_cal_scale() -> float | None:
    """Load the most recent saved validation calibration scale if available."""
    try:
        cand = LOCAL_RUN_DIR / f"val_cal_scale_e{MAX_EPOCHS}_{RUN_SUFFIX}.json"
        if cand.exists():
            with open(cand, "r") as f:
                return float(_json.load(f).get("scale", 1.0))
        files = sorted(LOCAL_RUN_DIR.glob("val_cal_scale_*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            with open(files[0], "r") as f:
                return float(_json.load(f).get("scale", 1.0))
    except Exception as e:
        print(f"[WARN] Could not load validation calibration scale: {e}")
    return None


from pytorch_forecasting.metrics import QuantileLoss, MultiLoss

class LabelSmoothedBCE(nn.Module):
    def __init__(self, smoothing: float = 0.1, pos_weight: float = 1.0):
        super().__init__()
        self.smoothing = smoothing
        self.register_buffer("pos_weight", torch.tensor(pos_weight))

    def forward(self, y_pred, target):
        target = target.float()
        target = target * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(
            y_pred.squeeze(-1), target.squeeze(-1),
            pos_weight=self.pos_weight,
        )

class LabelSmoothedBCEWithBrier(nn.Module):
    """
    Direction loss = label-smoothed BCE + lambda * Brier score.
    Brier uses probabilities (sigmoid over logits) and the SAME smoothed
    targets as the BCE term for consistency. Set brier_weight=0 to
    recover plain label-smoothed BCE.
    """
    def __init__(self, smoothing: float = 0.1, pos_weight: float = 1.001, brier_weight: float = 0.15):
        super().__init__()
        self.smoothing = float(smoothing)
        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        self.brier_weight = float(brier_weight)

    def forward(self, y_pred, target):
        target = target.float()
        # label smoothing as in LabelSmoothedBCE
        target_smooth = target * (1.0 - self.smoothing) + 0.5 * self.smoothing

        # BCE on logits
        bce = F.binary_cross_entropy_with_logits(
            y_pred.squeeze(-1), target_smooth.squeeze(-1),
            pos_weight=self.pos_weight,
        )

        if self.brier_weight <= 0.0:
            return bce

        # Brier on probabilities
        probs = torch.sigmoid(y_pred.squeeze(-1))
        brier = ((probs - target_smooth.squeeze(-1)) ** 2).mean()

        return bce + self.brier_weight * brier

import torch
import torch.nn.functional as F


class AsymmetricQuantileLoss(QuantileLoss):
    """
    Quantile loss with:
      • asymmetric underestimation penalty on all quantiles
      • optional mean-bias regulariser on the median (q=0.5)
      • optional tail emphasis (up-weights samples above a quantile threshold)
      • numerically-stable QLIKE term on a scale-free decoded space via log1p()

    Notes
    -----
    - Use with GroupNormalizer(transformation="log1p", center=False, scale_by_group=True)
      so that the y/ŷ ratio is scale-free across assets.
    - QLIKE is stabilized by:
        * clipping encoded values before log1p (med_clip)
        * working in log-variance space with clamped log-ratio (log_ratio_clip)
        * epsilon floors
    """

    def __init__(
        self,
        quantiles,
        underestimation_factor: float = 1.00, #1.1115
        mean_bias_weight: float = 0.0,
        tail_q: float = 0.90,         # ← was 0.85
        tail_weight: float = 0.0,
        qlike_weight: float = 0.0,   # set to 0.0 because we cannot safely decode inside the loss
        eps: float = 1e-8,
        med_clip: float = 3.0,
        log_ratio_clip: float = 12.0,
        **kwargs,
    ):
        super().__init__(quantiles=quantiles, **kwargs)
        self.underestimation_factor = float(underestimation_factor)
        self.mean_bias_weight = float(mean_bias_weight)
        self.tail_q = float(tail_q)
        self.tail_weight = float(tail_weight)
        self.qlike_weight = float(qlike_weight)
        self.eps = float(eps)
        self.med_clip = float(med_clip)
        self.log_ratio_clip = float(log_ratio_clip)

        try:
            self._q50_idx = self.quantiles.index(0.5)
        except Exception:
            self._q50_idx = len(self.quantiles) // 2

    # ----- core quantile component (with optional tail up-weighting) -----
    def loss_per_prediction(self, y_pred, target):
        if isinstance(target, tuple):
            target = target[0]

        diff = target.unsqueeze(-1) - y_pred
        q = y_pred.new_tensor(self.quantiles).view(*([1] * (diff.ndim - 1)), -1)
        alpha = y_pred.new_tensor(self.underestimation_factor)

        loss = torch.where(
            diff >= 0,
            alpha * q * diff,             # under-prediction → amplified
            (1.0 - q) * (-diff),          # over-prediction
        )

        if self.tail_weight and self.tail_weight > 0:
            try:
                thresh = torch.quantile(target.detach(), self.tail_q)
                w = torch.where(target.unsqueeze(-1) > thresh,
                                 1.0 + (self.tail_weight - 1.0),
                                 1.0)
                loss = loss * w
            except Exception:
                pass

        return loss

    def forward(self, y_pred, target):
        base = self.loss_per_prediction(y_pred, target).mean()

        # ---- mean-bias regulariser on the median (q=0.5) ----
        if self.mean_bias_weight > 0:
            try:
                if isinstance(target, tuple):
                    target = target[0]
                med = y_pred[..., self._q50_idx]
                mean_diff = (target - med).mean()
                base = base + (mean_diff ** 2) * self.mean_bias_weight
            except Exception:
                pass

            # ---- Optional surrogate QLIKE (encoded scale; no decode) ----
            # If qlike_weight > 0, add a stable proxy using encoded magnitudes.
            # We approximate sigma by |·| of the encoded target/median prediction.
        if self.qlike_weight and self.qlike_weight > 0:
            try:
                if isinstance(target, tuple):
                    target = target[0]
                # median (q=0.5) prediction on encoded scale
                med = y_pred[..., self._q50_idx]

                # positive "scale" proxies from encoded values
                # clamp to avoid extreme gradients, then floor by eps
                sigma_y = torch.clamp(target.abs(), min=self.eps)
                sigma_p = torch.clamp(med.abs(),    min=self.eps)

                # QLIKE core on scale ratio
                ratio = (sigma_y ** 2) / (sigma_p ** 2)
                # clip log-ratio for numerical stability
                log_ratio = torch.log(torch.clamp(ratio, min=torch.exp(-self.log_ratio_clip), max=torch.exp(self.log_ratio_clip)))
                qlike = ratio - log_ratio - 1.0

                base = base + float(self.qlike_weight) * qlike.mean()
            except Exception:
                # if anything goes sideways, just fall back to quantile-only
                pass

        return torch.nan_to_num(base, nan=0.0, posinf=1e4, neginf=1e4)



class PerAssetMetrics(pl.Callback):
    """Collects per-asset predictions during validation and prints/saves metrics.
    Computes MAE, RMSE, MSE, QLIKE for realised_vol and Accuracy for direction.
    """
    def __init__(self, id_to_name: dict, vol_normalizer, max_print: int = 10):
        super().__init__()
        self.id_to_name = {int(k): str(v) for k, v in id_to_name.items()}
        self.vol_norm = vol_normalizer
        self.max_print = max_print
        self.reset()

    def reset(self):
        # device-resident accumulators (concatenate at epoch end)
        self._g_dev = []   # group ids per sample (device, flattened)
        self._yv_dev = []  # realised vol target (NORMALISED, device)
        self._pv_dev = []  # realised vol pred   (NORMALISED, device)
        self._yd_dev = []  # direction target (device)
        self._pd_dev = []  # direction pred logits/probs (device)
        self._t_dev = []   # decoder time_idx (device) if provided
        # cached final rows/overall from last epoch
        self._last_rows = None
        self._last_overall = None

    @torch.no_grad()
    def on_validation_epoch_start(self, trainer, pl_module):
        self.reset()
        print(f"[VAL HOOK] start epoch {getattr(trainer,'current_epoch',-1)+1}")

    @torch.no_grad()
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0):
        # batch is (x, y, weight) from PF dataloader
        if not isinstance(batch, (list, tuple)):
            return
        x = batch[0]
        if not isinstance(x, dict):
            return
        groups = None
        for k in ("groups", "group_ids", "group_id"):
            if k in x and x[k] is not None:
                groups = x[k]
                break
        dec_t = x.get("decoder_target")

        # optional time index for plotting/joining later
        dec_time = x.get("decoder_time_idx", None)
        if dec_time is None:
            # some PF versions may expose relative index or time via different keys; try a few
            dec_time = x.get("decoder_relative_idx", None)

        # also fetch explicit targets from batch[1] as a fallback
        y_batch = batch[1] if isinstance(batch, (list, tuple)) and len(batch) >= 2 else None
        if groups is None:
            return

        # groups can be a Tensor or a list[Tensor]; take the first if list
        groups_raw = groups[0] if isinstance(groups, (list, tuple)) else groups
        g = groups_raw
        # squeeze trailing singleton dims to get [B]
        while torch.is_tensor(g) and g.ndim > 1 and g.size(-1) == 1:
            g = g.squeeze(-1)
        if not torch.is_tensor(g):
            return

        # --- Extract targets (try decoder_target first, else fall back to batch[1]) ---
        y_vol_t, y_dir_t = None, None
        if dec_t is not None:
            if torch.is_tensor(dec_t):
                y = dec_t
                if y.ndim == 3 and y.size(-1) == 1:
                    y = y[..., 0]  # → [B, n_targets]
                if y.ndim == 2 and y.size(1) >= 1:
                    y_vol_t = y[:, 0]
                    if y.size(1) > 1:
                        y_dir_t = y[:, 1]
            elif isinstance(dec_t, (list, tuple)) and len(dec_t) >= 1:
                y_vol_t = dec_t[0]
                if torch.is_tensor(y_vol_t):
                    if y_vol_t.ndim == 3 and y_vol_t.size(-1) == 1:
                        y_vol_t = y_vol_t[..., 0]
                    if y_vol_t.ndim == 2 and y_vol_t.size(-1) == 1:
                        y_vol_t = y_vol_t[:, 0]
                if len(dec_t) > 1 and torch.is_tensor(dec_t[1]):
                    y_dir_t = dec_t[1]
                    if y_dir_t.ndim == 3 and y_dir_t.size(-1) == 1:
                        y_dir_t = y_dir_t[..., 0]
                    if y_dir_t.ndim == 2 and y_dir_t.size(-1) == 1:
                        y_dir_t = y_dir_t[:, 0]

        # Fallback: PF sometimes provides targets in batch[1] as [B, pred_len, n_targets]
        if (y_vol_t is None or y_dir_t is None) and torch.is_tensor(y_batch):
            yb = y_batch
            if yb.ndim == 3 and yb.size(1) == 1:
                yb = yb[:, 0, :]
            if yb.ndim == 2 and yb.size(1) >= 1:
                if y_vol_t is None:
                    y_vol_t = yb[:, 0]
                if y_dir_t is None and yb.size(1) > 1:
                    y_dir_t = yb[:, 1]

        if y_vol_t is None:
            return

        # Forward pass to get predictions for this batch
        y_hat = pl_module(x)
        pred = getattr(y_hat, "prediction", y_hat)
        if isinstance(pred, dict) and "prediction" in pred:
            pred = pred["prediction"]

        # --- NEW: use shared head extractor ---
        p_vol, p_dir = _extract_heads(pred)
        if p_vol is None:
            return  # nothing usable in this batch

        # Store device tensors; no decode/CPU here
        L = g.shape[0]
        self._g_dev.append(g.reshape(L))
        self._yv_dev.append(y_vol_t.reshape(L))
        self._pv_dev.append(p_vol.reshape(L))

        # capture time index if available and shape-compatible
        if dec_time is not None and torch.is_tensor(dec_time):
            tvec = dec_time
            # squeeze to [B]
            while tvec.ndim > 1 and tvec.size(-1) == 1:
                tvec = tvec.squeeze(-1)
            if tvec.numel() >= L:
                self._t_dev.append(tvec.reshape(-1)[:L])

        if y_dir_t is not None and p_dir is not None:
            y_flat = y_dir_t.reshape(-1)
            p_flat = p_dir.reshape(-1)
            L2 = min(L, y_flat.numel(), p_flat.numel())
            if L2 > 0:
                self._yd_dev.append(y_flat[:L2])
                self._pd_dev.append(p_flat[:L2])

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        global PER_ASSET_CAL_SCALES
        # If nothing collected, exit quietly
        if not self._g_dev:
            return

        # Gather device tensors accumulated during validation
        device = self._g_dev[0].device
        g  = torch.cat(self._g_dev).to(device)            # [N]
        yv = torch.cat(self._yv_dev).to(device)           # realised_vol (encoded)  [N]
        pv = torch.cat(self._pv_dev).to(device)           # realised_vol pred (enc) [N]
        yd = torch.cat(self._yd_dev).to(device) if self._yd_dev else None  # direction labels
        pdir = torch.cat(self._pd_dev).to(device) if self._pd_dev else None  # direction logits/probs

        # --- Decode realised_vol to physical scale (robust to PF version)
        yv_dec = safe_decode_vol(yv.unsqueeze(-1), self.vol_norm, g.unsqueeze(-1)).squeeze(-1)
        pv_dec = safe_decode_vol(pv.unsqueeze(-1), self.vol_norm, g.unsqueeze(-1)).squeeze(-1)

        # Guard against near-zero decoded vols (use global floor if defined)
        floor_val = float(globals().get("EVAL_VOL_FLOOR", 1e-8))
        yv_dec = torch.clamp(yv_dec, min=floor_val)
        pv_dec = torch.clamp(pv_dec, min=floor_val)

        # Debug prints (helpful sanity checks)
        try:
            print("DEBUG transformation:", getattr(self.vol_norm, "transformation", None))
            print("DEBUG mean after decode:", float(yv_dec.mean().item()))
            ratio_dbg = float((yv_dec.mean() / (pv_dec.mean() + 1e-12)).item())
            print("DEBUG: mean(yv_dec)=", float(yv_dec.mean().item()),
                  "mean(pv_dec)=", float(pv_dec.mean().item()),
                  "ratio=", ratio_dbg)
        except Exception:
            pass

        # Move to CPU for metric computation
        y_cpu  = yv_dec.detach().cpu()
        p_cpu  = pv_dec.detach().cpu()
        g_cpu  = g.detach().cpu()
        yd_cpu = yd.detach().cpu() if yd is not None else None
        pdir_cpu = pdir.detach().cpu() if pdir is not None else None
    

        # Compute and persist global validation calibration scale (QLIKE-optimal)
        try:
            s = compute_global_scale(y_cpu, p_cpu)
            global GLOBAL_CAL_SCALE
            GLOBAL_CAL_SCALE = float(s)
            _save_val_cal_scale(GLOBAL_CAL_SCALE)
        except Exception as _e:
            print(f"[WARN] Could not compute/save validation calibration scale: {_e}")

        # Calibration diagnostic (not used in loss)
        try:
            mean_scale = float((y_cpu.mean() / (p_cpu.mean() + 1e-12)).item())
        except Exception:
            mean_scale = float("nan")
        print(f"[CAL DEBUG] mean(y)/mean(p)={mean_scale:.4f} (1==perfect)")
        try:
            trainer.callback_metrics["val_mean_scale"] = torch.tensor(mean_scale)
        except Exception:
            pass

        # --- Compute & save per-asset QLIKE-optimal scales s_a (validation only) ---
        try:
            assets = [self.id_to_name.get(int(i), str(int(i))) for i in g_cpu.tolist()]
            df_pa = pd.DataFrame({"asset": assets, "y": y_cpu.numpy(), "p": p_cpu.numpy()})
            scales = {}
            eps = 1e-12
            for a, gdf in df_pa.groupby("asset", sort=False):
                y_a = torch.tensor(gdf["y"].values, dtype=torch.float32)
                p_a = torch.tensor(gdf["p"].values, dtype=torch.float32)
                if len(gdf) == 0:
                    continue
                s2 = torch.clamp((torch.clamp(y_a.abs(), min=eps) ** 2).mean() /
                                (torch.clamp(p_a.abs(), min=eps) ** 2).mean(), min=eps)
                scales[str(a)] = float(torch.sqrt(s2).item())
            if scales:
                PER_ASSET_CAL_SCALES = scales
                _save_val_per_asset_scales(scales)
        except Exception as e:
            print(f"[WARN] per-asset scale computation failed: {e}")

        # --- Decoded regression metrics (overall) ---
        eps = 1e-8
        diff = (p_cpu - y_cpu)
        overall_mae  = float(diff.abs().mean().item())
        overall_mse  = float((diff ** 2).mean().item())
        overall_rmse = float(overall_mse ** 0.5)

        sigma2_p = torch.clamp(p_cpu.abs(), min=eps) ** 2
        sigma2_y = torch.clamp(y_cpu.abs(), min=eps) ** 2
        ratio    = sigma2_y / sigma2_p
        overall_qlike = float((ratio - torch.log(ratio) - 1.0).mean().item())

        # --- Calibrated regression metrics (overall) ---
        overall_mae_cal = None
        overall_mse_cal = None
        overall_rmse_cal = None
        overall_qlike_cal = None if 'overall_qlike_cal' not in locals() else overall_qlike_cal
        try:
            if 'p_cal' not in locals():
                p_cal = calibrate_vol_predictions(y_cpu, p_cpu)
            diff_cal = (p_cal - y_cpu)
            overall_mae_cal  = float(diff_cal.abs().mean().item())
            overall_mse_cal  = float((diff_cal ** 2).mean().item())
            overall_rmse_cal = float(overall_mse_cal ** 0.5)
            sigma2_pc = torch.clamp(p_cal.abs(), min=eps) ** 2
            ratio_cal = sigma2_y / sigma2_pc
            overall_qlike_cal = float((ratio_cal - torch.log(ratio_cal) - 1.0).mean().item())
        except Exception as _e:
            pass

        # --- Optional direction metrics (overall) ---
        acc = None
        brier = None
        auroc = None
        if yd_cpu is not None and pdir_cpu is not None and yd_cpu.numel() > 0 and pdir_cpu.numel() > 0:
            # Convert logits→probs if needed
            probs = pdir_cpu
            try:
                if torch.isfinite(probs).any() and (probs.min() < 0 or probs.max() > 1):
                    probs = torch.sigmoid(probs)
            except Exception:
                probs = torch.sigmoid(probs)
            probs = torch.clamp(probs, 0.0, 1.0)

            acc = float(((probs >= 0.5).int() == yd_cpu.int()).float().mean().item())
            brier = float(((probs - yd_cpu.float()) ** 2).mean().item())
            try:
                au = BinaryAUROC()
                auroc = float(au(probs, yd_cpu).item())
            except Exception as e:
                print(f"[WARN] AUROC failed: {e}")

            # stash for later printing/saving
            try:
                trainer.callback_metrics["val_brier_overall"] = torch.tensor(brier)
                trainer.callback_metrics["val_auroc_overall"] = torch.tensor(auroc) if auroc is not None else None
                trainer.callback_metrics["val_acc_overall"]   = torch.tensor(acc)
            except Exception:
                pass

        # --- Epoch summary / val loss ---
        N = int(y_cpu.numel())
        try:
            epoch_num = int(getattr(trainer, "current_epoch", -1)) + 1
        except Exception:
            epoch_num = None

        # Composite loss (absolute form for validation)
        val_comp = float(composite_score(overall_mae, overall_rmse, overall_qlike))
        # Preferred monitor key
        trainer.callback_metrics["val_comp_overall"]      = torch.tensor(val_comp)
        # Backward-compat alias (some monitors used this)
        trainer.callback_metrics["val_composite_overall"] = torch.tensor(val_comp)
        trainer.callback_metrics["val_loss_source"]       = "composite(MAE,RMSE,QLIKE)"
        # Lightning's default early-stopping key
        trainer.callback_metrics["val_loss"]              = torch.tensor(val_comp)
        trainer.callback_metrics["val_loss_decoded"]      = torch.tensor(val_comp)
        trainer.callback_metrics["val_mae_overall"]       = torch.tensor(overall_mae)
        trainer.callback_metrics["val_rmse_overall"]      = torch.tensor(overall_rmse)
        trainer.callback_metrics["val_mse_overall"]       = torch.tensor(overall_mse)
        trainer.callback_metrics["val_qlike_overall"]     = torch.tensor(overall_qlike)
        trainer.callback_metrics["val_qlike_cal"]         = torch.tensor(overall_qlike_cal)
        trainer.callback_metrics["val_N_overall"]         = torch.tensor(float(N))
        # Add calibrated metrics to callback_metrics if available
        if overall_mae_cal is not None:
            trainer.callback_metrics['val_mae_cal']   = torch.tensor(overall_mae_cal)
            trainer.callback_metrics['val_mse_cal']   = torch.tensor(overall_mse_cal)
            trainer.callback_metrics['val_rmse_cal']  = torch.tensor(overall_rmse_cal)
        if overall_qlike_cal is not None:
            trainer.callback_metrics['val_qlike_cal'] = torch.tensor(overall_qlike_cal)

        msg = (
            f"[VAL EPOCH {epoch_num}] "
            f"(decoded) MAE={overall_mae:.6f} "
            f"RMSE={overall_rmse:.6f} "
            f"MSE={overall_mse:.6f} "
            f"QLIKE={overall_qlike:.6f} "
            f"CompLoss = {val_comp:.6f} "
            + (f"QLIKE_CAL={overall_qlike_cal:.6f} " if overall_qlike_cal is not None else "")
            + (f" | ACC={acc:.3f}"   if acc   is not None else "")
            + (f" | Brier={brier:.4f}" if brier is not None else "")
            + (f" | AUROC={auroc:.3f}" if auroc is not None else "")
            + f" | N={N}"
        )
        print(msg)
       

        # --- Per-asset metrics table (so on_fit_end can print it) ---
        self._last_rows = []
        try:
            # map group id -> human name
            asset_names = [self.id_to_name.get(int(i), str(int(i))) for i in g_cpu.tolist()]
            # compute per-asset aggregates
            dfm = pd.DataFrame({
                "asset": asset_names,
                "y": y_cpu.numpy(),
                "p": p_cpu.numpy(),
            })
            if yd_cpu is not None and pdir_cpu is not None and yd_cpu.numel() > 0 and pdir_cpu.numel() > 0:
                # ensure probs in [0,1]
                probs = pdir_cpu
                try:
                    if torch.isfinite(probs).any() and (probs.min() < 0 or probs.max() > 1):
                        probs = torch.sigmoid(probs)
                except Exception:
                    probs = torch.sigmoid(probs)
                probs = torch.clamp(probs, 0.0, 1.0)
                dfm["yd"] = yd_cpu.numpy()
                dfm["pd"] = probs.numpy()

            rows = []
            for a, gdf in dfm.groupby("asset", sort=False):
                y_a = torch.tensor(gdf["y"].values)
                p_a = torch.tensor(gdf["p"].values)
                n_a = int(len(gdf))
                mae_a = float((p_a - y_a).abs().mean().item())
                mse_a = float(((p_a - y_a) ** 2).mean().item())
                rmse_a = float(mse_a ** 0.5)

                s2p = torch.clamp(torch.tensor(np.abs(gdf["p"].values)), min=eps) ** 2
                s2y = torch.clamp(torch.tensor(np.abs(gdf["y"].values)), min=eps) ** 2
                ratio_a = s2y / s2p
                qlike_a = float((ratio_a - torch.log(ratio_a) - 1.0).mean().item())

                acc_a = None
                if "yd" in gdf.columns and "pd" in gdf.columns:
                    acc_a = float(((torch.tensor(gdf["pd"].values) >= 0.5).int() ==
                                   torch.tensor(gdf["yd"].values).int()).float().mean().item())

                rows.append((a, mae_a, rmse_a, mse_a, qlike_a, acc_a, n_a))

            # --- Compute per-asset QLIKE-optimal multiplicative scales and persist ---
            try:
                pa_scales = {}
                for a, gdf in dfm.groupby("asset", sort=False):
                    y_a = torch.tensor(gdf["y"].values, dtype=torch.float32)
                    p_a = torch.tensor(gdf["p"].values, dtype=torch.float32)
                    if y_a.numel() == 0 or p_a.numel() == 0:
                        continue
                    s2y = torch.clamp(y_a.abs(), min=eps) ** 2
                    s2p = torch.clamp(p_a.abs(), min=eps) ** 2
                    s2  = s2y.mean() / s2p.mean()
                    s   = float(torch.sqrt(torch.clamp(s2, min=eps)).item())
                    pa_scales[str(a)] = s
                if pa_scales:
                    PER_ASSET_CAL_SCALES = pa_scales
                    _save_val_per_asset_scales(pa_scales)
            except Exception as _e:
                print(f"[WARN] Could not compute/save per-asset scales: {_e}")

            # --- Calibrated per-asset metrics (use per-asset s_a if available, else global s) ---
            rows_cal = []
            try:
                # Load scales computed earlier this epoch or from disk
                s_global = None
                try:
                    s_global = GLOBAL_CAL_SCALE if (GLOBAL_CAL_SCALE is not None) else _load_val_cal_scale()
                except Exception:
                    s_global = None
                try:
                    pa_scales_loaded = PER_ASSET_CAL_SCALES if (PER_ASSET_CAL_SCALES is not None) else _load_val_per_asset_scales()
                except Exception:
                    pa_scales_loaded = None
                if pa_scales_loaded is not None:
                    pa_scales_loaded = {str(k): float(v) for k, v in pa_scales_loaded.items()}

                eps_local = 1e-8
                for a, gdf in dfm.groupby("asset", sort=False):
                    y_a = torch.tensor(gdf["y"].values)
                    p_a = torch.tensor(gdf["p"].values)
                    n_a = int(len(gdf))
                    # Choose scale: per-asset first, else global, else 1.0
                    s_a = 1.0
                    if pa_scales_loaded is not None and str(a) in pa_scales_loaded:
                        s_a = float(pa_scales_loaded[str(a)])
                    elif s_global is not None:
                        s_a = float(s_global)
                    p_ac = p_a * s_a
                    diff_c = (p_ac - y_a)
                    mae_c = float(diff_c.abs().mean().item())
                    mse_c = float((diff_c ** 2).mean().item())
                    rmse_c = float(mse_c ** 0.5)

                    s2pc = torch.clamp(torch.tensor(np.abs(p_ac.numpy())), min=eps_local) ** 2
                    s2yc = torch.clamp(torch.tensor(np.abs(y_a.numpy())),  min=eps_local) ** 2
                    ratio_c = s2yc / s2pc
                    qlike_c = float((ratio_c - torch.log(ratio_c) - 1.0).mean().item())

                    rows_cal.append((a, mae_c, rmse_c, mse_c, qlike_c, n_a))
            except Exception as _e:
                print(f"[WARN] per-asset calibrated metrics failed: {_e}")

            # sort by sample count (desc) so “top by samples” prints nicely
            rows.sort(key=lambda r: r[6], reverse=True)
            # Store both uncalibrated and calibrated for on_fit_end
            self._last_rows = rows
            self._last_rows_cal = rows_cal if 'rows_cal' in locals() else []

            # --- Per-epoch per-asset snapshot (top by samples) ---
            try:
                k = min(5, getattr(self, "max_print", 5))
                if rows:
                    print("Per-asset (epoch snapshot, top by samples):")
                    print("asset | MAE | RMSE | MSE | QLIKE | ACC | N")
                    for r in rows[:k]:
                        acc_str = "-" if r[5] is None else f"{r[5]:.3f}"
                        print(f"{r[0]} | {r[1]:.6f} | {r[2]:.6f} | {r[3]:.6f} | {r[4]:.6f} | {acc_str} | {r[6]}")
            except Exception as _e:
                print(f"[WARN] per-epoch per-asset print failed: {_e}")

            # --- Calibrated per-epoch per-asset snapshot (top by samples) ---
            try:
                if rows_cal:
                    rows_cal.sort(key=lambda r: r[5], reverse=True)
                    k = min(5, getattr(self, "max_print", 5))
                    print("Per-asset (CALIBRATED snapshot, top by samples):")
                    print("asset | MAE | RMSE | MSE | QLIKE | N")
                    for r in rows_cal[:k]:
                        print(f"{r[0]} | {r[1]:.6f} | {r[2]:.6f} | {r[3]:.6f} | {r[4]:.6f} | {r[5]}")
            except Exception as _e:
                print(f"[WARN] per-epoch per-asset calibrated print failed: {_e}")

        except Exception as e:
            print(f"[WARN] per-asset aggregation failed: {e}")
            self._last_rows = []

        # stash overall for on_fit_end
        self._last_overall = {
            "mae": overall_mae,
            "rmse": overall_rmse,
            "mse": overall_mse,
            "qlike": overall_qlike,
            "val_loss": val_comp,
            "dir_bce": brier,   # (kept for backwards compatibility with your saver)
            "yd": yd_cpu,
            "pd": pdir_cpu,
            # calibrated (if available)
            "mae_cal": overall_mae_cal,
            "rmse_cal": overall_rmse_cal,
            "mse_cal": overall_mse_cal,
            "qlike_cal": overall_qlike_cal,
        }

    @torch.no_grad()
    def on_fit_end(self, trainer, pl_module):
        if self._last_rows is None or self._last_overall is None:
            return
        rows = self._last_rows
        overall = self._last_overall
        print("\nOverall decoded metrics (final):")
        print(f"UNCAL — MAE: {overall['mae']:.6f} | RMSE: {overall['rmse']:.6f} | MSE: {overall['mse']:.6f} | QLIKE: {overall['qlike']:.6f}")
        # If calibrated fields were stored in overall dict by on_validation_epoch_end, print them too
        mae_cal  = overall.get('mae_cal', None)
        rmse_cal = overall.get('rmse_cal', None)
        mse_cal  = overall.get('mse_cal', None)
        ql_cal   = overall.get('qlike_cal', None)
        if mae_cal is not None and ql_cal is not None:
            print(f"CALIB — MAE: {mae_cal:.6f} | RMSE: {rmse_cal:.6f} | MSE: {mse_cal:.6f} | QLIKE: {ql_cal:.6f}")

        print("\nPer-asset validation metrics (top by samples):")
        print("asset | MAE | RMSE | MSE | QLIKE | ACC | N")
        for r in rows[: self.max_print]:
            acc_str = "-" if r[5] is None else f"{r[5]:.3f}"
            print(f"{r[0]} | {r[1]:.6f} | {r[2]:.6f} | {r[3]:.6f} | {r[4]:.6f} | {acc_str} | {r[6]}")

        # Calibrated per-asset table at fit end (if available)
        try:
            rows_cal = getattr(self, "_last_rows_cal", [])
            if rows_cal:
                print("\nPer-asset validation metrics (CALIBRATED, top by samples):")
                print("asset | MAE | RMSE | MSE | QLIKE | N")
                for r in rows_cal[: self.max_print]:
                    print(f"{r[0]} | {r[1]:.6f} | {r[2]:.6f} | {r[3]:.6f} | {r[4]:.6f} | {r[5]}")
        except Exception as _e:
            print(f"[WARN] Could not print calibrated per-asset metrics at end: {_e}")

        dir_overall = None
        yd = overall.get("yd", None)
        pd_all = overall.get("pd", None)
        if yd is not None and pd_all is not None and yd.numel() > 0 and pd_all.numel() > 0:
            try:
                L = min(yd.numel(), pd_all.numel())
                yd1 = yd[:L].float()
                pd1 = pd_all[:L]
                try:
                    if torch.isfinite(pd1).any() and (pd1.min() < 0 or pd1.max() > 1):
                        pd1 = torch.sigmoid(pd1)
                except Exception:
                    pd1 = torch.sigmoid(pd1)
                pd1 = torch.clamp(pd1, 0.0, 1.0)
                acc = ((pd1 >= 0.5).int() == yd1.int()).float().mean().item()
                brier = ((pd1 - yd1) ** 2).mean().item()
                auroc = None
                try:
                    from torchmetrics.classification import BinaryAUROC
                    au = BinaryAUROC()
                    auroc = float(au(pd1.detach().cpu(), yd1.detach().cpu()).item())
                except Exception:
                    auroc = None
                dir_overall = {"accuracy": acc, "brier": brier, "auroc": auroc}
                print("\nDirection (final):")
                msg = f"Accuracy: {acc:.3f} | Brier: {brier:.4f}"
                if auroc is not None:
                    msg += f" | AUROC: {auroc:.3f}"
                print(msg)
            except Exception as e:
                print(f"[WARN] Could not compute final direction metrics: {e}")

        try:
            import json
            out = {
                "decoded": True,
                "overall": {k: v for k, v in overall.items() if k in ("mae","rmse","mse","qlike","val_loss","dir_bce")},
                "direction_overall": dir_overall,
                "per_asset": [
                    {"asset": r[0], "mae": r[1], "rmse": r[2], "mse": r[3], "qlike": r[4], "acc": r[5], "n": r[6]}
                    for r in rows
                ],
            }
            path = str(LOCAL_RUN_DIR / f"tft_val_asset_metrics_e{MAX_EPOCHS}_{RUN_SUFFIX}.json")
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"✓ Saved per-asset validation metrics (decoded, final) → {path}")
        except Exception as e:
            print(f"[WARN] Could not save final per-asset metrics: {e}")

        # Optionally save per-sample predictions for plotting
        try:
            import pandas as pd  # ensure pd is bound locally and not shadowed
            # Recompute decoded tensors from the stored device buffers
            g_cpu = torch.cat(self._g_dev).detach().cpu() if self._g_dev else None
            yv_cpu = torch.cat(self._yv_dev).detach().cpu() if self._yv_dev else None
            pv_cpu = torch.cat(self._pv_dev).detach().cpu() if self._pv_dev else None
            yd_cpu = torch.cat(self._yd_dev).detach().cpu() if self._yd_dev else None
            pdir_cpu = torch.cat(self._pd_dev).detach().cpu() if self._pd_dev else None
            # decode vol back to physical scale
            if g_cpu is not None and yv_cpu is not None and pv_cpu is not None:
                yv_dec = self.vol_norm.decode(yv_cpu.unsqueeze(-1), group_ids=g_cpu.unsqueeze(-1)).squeeze(-1)
                pv_dec = self.vol_norm.decode(pv_cpu.unsqueeze(-1), group_ids=g_cpu.unsqueeze(-1)).squeeze(-1)
                # Apply the same calibration used in metrics so saved preds match the plots
                # map group id -> name
                assets = [self.id_to_name.get(int(i), str(int(i))) for i in g_cpu.tolist()]
                # time index (may be missing)
                t_cpu = torch.cat(self._t_dev).detach().cpu() if self._t_dev else None
                df_out = pd.DataFrame({
                    "asset": assets,
                    "time_idx": t_cpu.numpy().tolist() if t_cpu is not None else [None] * len(assets),
                    "y_vol": yv_dec.numpy().tolist(),
                    "y_vol_pred": pv_dec.numpy().tolist(),
                })
                if yd_cpu is not None and pdir_cpu is not None and yd_cpu.numel() > 0 and pdir_cpu.numel() > 0:
                    # ensure pdir is probability
                    pdp = pdir_cpu
                    try:
                        if torch.isfinite(pdp).any() and (pdp.min() < 0 or pdp.max() > 1):
                            pdp = torch.sigmoid(pdp)
                    except Exception:
                        pdp = torch.sigmoid(pdp)
                    pdp = torch.clamp(pdp, 0.0, 1.0)
                    Lm = min(len(df_out), yd_cpu.numel(), pdp.numel())
                    df_out = df_out.iloc[:Lm].copy()
                    df_out["y_dir"] = yd_cpu[:Lm].numpy().tolist()
                    df_out["y_dir_prob"] = pdp[:Lm].numpy().tolist()
                # --- Attach real timestamps from a source DataFrame, if available ---
                try:
                    # Look for a likely source dataframe that contains ['asset','time_idx','Time']
                    candidate_names = ["val_df"]
                    src_df = None
                    for _name in candidate_names:
                        obj = globals().get(_name)
                        if isinstance(obj, pd.DataFrame) and {"asset", "time_idx", "Time"}.issubset(obj.columns):
                            src_df = obj[["asset", "time_idx", "Time"]].copy()
                            break
                    if src_df is not None:
                        # Harmonise dtypes prior to merge
                        src_df["asset"] = src_df["asset"].astype(str)
                        src_df["time_idx"] = pd.to_numeric(src_df["time_idx"], errors="coerce").astype("Int64").astype("int64")
                        df_out["asset"] = df_out["asset"].astype(str)
                        df_out["time_idx"] = pd.to_numeric(df_out["time_idx"], errors="coerce").astype("Int64").astype("int64")

                        # Merge on ['asset','time_idx'] to bring in the real Time column
                        df_out = df_out.merge(src_df, on=["asset", "time_idx"], how="left", validate="m:1")

                        # Coerce Time to timezone-naive pandas datetimes
                        df_out["Time"] = pd.to_datetime(df_out["Time"], errors="coerce")
                        try:
                            # If tz-aware, drop timezone info; if already naive this may raise — ignore
                            df_out["Time"] = df_out["Time"].dt.tz_localize(None)
                        except Exception:
                            pass
                    else:
                        print(
                            "[WARN] Could not locate a source dataframe with ['asset','time_idx','Time'] among candidates: "
                            "raw_df, raw_data, full_df, df (also checked val_df/train_df/test_df)."
                        )
                except Exception as e:
                    print(f"[WARN] Failed to attach real timestamps: {e}")
                pred_path = LOCAL_OUTPUT_DIR / f"tft_val_predictions_e{MAX_EPOCHS}_{RUN_SUFFIX}.parquet"
                df_out.to_parquet(pred_path, index=False)
                print(f"✓ Saved validation predictions (Parquet) → {pred_path}")
                try:
                    upload_file_to_gcs(str(pred_path), f"{GCS_OUTPUT_PREFIX}/{pred_path.name}")
                except Exception as e:
                    print(f"[WARN] Could not upload validation predictions: {e}")
                # Also save a calibrated version (no look-ahead; uses validation-derived scale)
                try:
                    s = GLOBAL_CAL_SCALE if (GLOBAL_CAL_SCALE is not None) else _load_val_cal_scale()
                    if s is not None:
                        df_cal = df_out.copy()
                        # apply scalar to predicted vols only
                        df_cal["y_vol_pred"] = (
                            torch.tensor(df_cal["y_vol_pred"].values) * float(s)
                        ).numpy().tolist()
                        cal_path = LOCAL_OUTPUT_DIR / f"calibrated_validation_predictions_e{MAX_EPOCHS}_{RUN_SUFFIX}.parquet"
                        df_cal.to_parquet(cal_path, index=False)
                        print(f"✓ Saved calibrated validation predictions (Parquet) → {cal_path}")
                        try:
                            upload_file_to_gcs(str(cal_path), f"{GCS_OUTPUT_PREFIX}/{cal_path.name}")
                        except Exception as e:
                            print(f"[WARN] Could not upload calibrated validation predictions: {e}")
                    else:
                        print("[WARN] No validation calibration scale available; skipping calibrated_validation parquet.")
                except Exception as e:
                    print(f"[WARN] Could not save calibrated validation predictions: {e}")
        except Exception as e:
            print(f"[WARN] Could not save validation predictions: {e}")

import lightning.pytorch as pl

class BiasWarmupCallback(pl.Callback):
    """
    Adaptive warm-up with a safety guard.

    • EMA of mean(y)/mean(p) steers underestimation bias (alpha).
    • Warm-ups are frozen if validation worsens for `guard_patience` epochs.
    • QLIKE ramps only when scale is near 1 to avoid fighting calibration.
    scale_ema_alpha (default 0.6) controls how quickly the EMA of mean(y)/mean(p) adapts; higher values react faster.
    """
    def __init__(
        self,
        vol_loss=None,
        target_under: float = 1.00,
        target_mean_bias: float = 0.04,
        warmup_epochs: int = 4,
        qlike_target_weight: float | None = 0.05,
        start_mean_bias: float = 0.0,
        mean_bias_ramp_until: int = 8,
        guard_patience: int = 2,
        guard_tol: float = 0.0,
        alpha_step: float = 0.05,
        scale_ema_alpha: float = 0.99,
    ):
        super().__init__()
        self._vol_loss_hint = vol_loss
        self.target_under = float(target_under)
        self.target_mean_bias = float(target_mean_bias)
        self.qlike_target_weight = None if qlike_target_weight is None else float(qlike_target_weight)
        self.warm = int(max(0, warmup_epochs))
        self.start_mean_bias = float(start_mean_bias)
        self.mean_bias_ramp_until = int(max(mean_bias_ramp_until, self.warm))
        self.guard_patience = int(max(1, guard_patience))
        self.guard_tol = float(guard_tol)
        self.alpha_step = float(alpha_step)
        self.scale_ema_alpha = float(max(1e-6, min(1.0, scale_ema_alpha)))

        self._scale_ema = None
        self._prev_val = None
        self._worse_streak = 0
        self._frozen = False

    def _resolve_vol_loss(self, pl_module):
        import inspect
        def is_vol_loss(obj):
            if obj is None:
                return False
            name = obj.__class__.__name__
            return hasattr(obj, "loss_per_prediction") and hasattr(obj, "quantiles") and (
                name == "AsymmetricQuantileLoss" or hasattr(obj, "qlike_weight")
            )
        cand = self._vol_loss_hint
        if is_vol_loss(cand):
            return cand
        for _, v in inspect.getmembers(pl_module):
            if is_vol_loss(v):
                return v
        return None

    def on_validation_end(self, trainer, pl_module):
        val = trainer.callback_metrics.get("val_loss") or trainer.callback_metrics.get("val_mae_overall")
        try:
            val = float(val.item() if hasattr(val, "item") else val)
        except Exception:
            val = None
        if val is None:
            return
        if self._prev_val is not None and val > self._prev_val + self.guard_tol:
            self._worse_streak += 1
        else:
            self._worse_streak = 0
        self._prev_val = val
        if self._worse_streak >= self.guard_patience:
            self._frozen = True
        if self._worse_streak == 0:  # improved or equal
            self._frozen = False

        # Fast EMA update using the latest validation mean scale (speeds convergence)
        try:
            vms = trainer.callback_metrics.get("val_mean_scale", None)
            if vms is not None:
                s = float(vms.item() if hasattr(vms, "item") else vms)
                a = getattr(self, "scale_ema_alpha", 0.6)
                self._scale_ema = s if (self._scale_ema is None) else (1.0 - a) * self._scale_ema + a * s
        except Exception:
            pass

    def on_train_epoch_start(self, trainer, pl_module):
        # Global CLI switch: if --disable_warmups is true, do nothing
        try:
            if globals().get("ARGS") is not None and getattr(ARGS, "disable_warmups", False):
                print(f"[BIAS] epoch={int(getattr(trainer,'current_epoch',0))} DISABLED via --disable_warmups; skipping")
                return
        except Exception:
            pass
        vol_loss = self._resolve_vol_loss(pl_module)
        if vol_loss is None:
            print("[BIAS] could not resolve vol loss; skipping warm-up tweaks")
            return

        # read last validation mean scale diagnostic
        scale = None
        try:
            val = trainer.callback_metrics.get("val_mean_scale")
            if val is not None:
                scale = float(val.item() if hasattr(val, "item") else val)
        except Exception:
            pass

        if scale is not None:
            a = getattr(self, "scale_ema_alpha", 0.7)
            self._scale_ema = scale if self._scale_ema is None else (1.0 - a) * self._scale_ema + a * scale

        e = int(getattr(trainer, "current_epoch", 0))
        prog = min(1.0, float(e) / float(max(self.warm, 1)))

        # Freeze if getting worse
        if self._frozen:
            vol_loss.underestimation_factor = 1.0
            if hasattr(vol_loss, "qlike_weight"):
                vol_loss.qlike_weight = 0.0
            vol_loss.mean_bias_weight = min(getattr(vol_loss, "mean_bias_weight", 0.0), self.target_mean_bias)
            print(f"[BIAS] epoch={e} FROZEN: alpha=1.0 qlike_w=0.0 mean_bias={vol_loss.mean_bias_weight:.3f}")
            return

        # proportional (small) adjustment to alpha to avoid overshoot
        alpha = 1.0
        if self._scale_ema is not None:
            err = np.log(max(1e-6, self._scale_ema))   # log mean(y)/mean(p)
            step = np.clip(1.0 + self.alpha_step * err, 1.0 - self.alpha_step, 1.0 + self.alpha_step)
            alpha = (1.0 + (self.target_under - 1.0) * prog) * step
            if self._scale_ema < 0.995:  # already over-predicting
                alpha = min(alpha, 1.0)
        vol_loss.underestimation_factor = float(max(1.0, min(alpha, self.target_under)))

        # mean-bias gentle ramp
        if e <= self.mean_bias_ramp_until:
            mb_prog = min(1.0, max(0.0, e / float(max(1, self.mean_bias_ramp_until))))
            vol_loss.mean_bias_weight = self.start_mean_bias + (self.target_mean_bias - self.start_mean_bias) * mb_prog
        else:
            vol_loss.mean_bias_weight = self.target_mean_bias

        # qlike: only when calibration is roughly correct
        if hasattr(vol_loss, "qlike_weight") and self.qlike_target_weight is not None:
            q_target = float(self.qlike_target_weight)
            q_prog = min(1.0, float(e) / float(max(self.warm, 8)))
            near_ok = (self._scale_ema is None) or (0.775 <= self._scale_ema <= 1.2)
            vol_loss.qlike_weight = (q_target * q_prog) if near_ok else 0.0

        try:
            lr0 = trainer.optimizers[0].param_groups[0]["lr"]
        except Exception:
            lr0 = None
        print(
            f"[BIAS] epoch={e} under={vol_loss.underestimation_factor:.3f} "
            f"mean_bias={vol_loss.mean_bias_weight:.3f} "
            f"qlike_w={getattr(vol_loss, 'qlike_weight', 'n/a')} "
            f"scale_ema={self._scale_ema if self._scale_ema is not None else 'n/a'} "
            f"guard={'ON' if self._frozen else 'off'} "
            f"lr={lr0 if lr0 is not None else 'n/a'}"
        )

import sys

class SafeTQDMProgressBar(TQDMProgressBar):
    """Write tqdm to stderr to avoid BrokenPipe on stdout; reduce flush frequency."""
    def __init__(self, refresh_rate: int = 50):
        super().__init__(refresh_rate=refresh_rate)
        if not hasattr(self, "_tqdm_kwargs") or self._tqdm_kwargs is None:
            self._tqdm_kwargs = {}
        self._tqdm_kwargs.update({
            "file": sys.stderr,
            "mininterval": 0.5,      # fewer flushes
            "dynamic_ncols": True,
            "leave": True,
        })



class MedianMSELoss(nn.Module):
    def forward(self, y_hat_quantiles, y_true):
        q50 = y_hat_quantiles[:, 3]  # index of 0.50
        return F.mse_loss(q50, y_true)

class EpochLRDecay(pl.Callback):
    def __init__(self, gamma: float = 1, start_epoch: int = 1):
        """
        gamma: multiplicative decay per epoch (0.95 = -5%/epoch)
        start_epoch: begin decaying after this epoch index (0-based)
        """
        super().__init__()
        self.gamma = float(gamma)
        self.start_epoch = int(start_epoch)

    def on_train_epoch_end(self, trainer, pl_module):
        e = int(getattr(trainer, "current_epoch", 0))
        if e < self.start_epoch:
            return
        # scale all param_group LRs
        try:
            for opt in trainer.optimizers:
                for pg in opt.param_groups:
                    if "lr" in pg and pg["lr"] is not None:
                        pg["lr"] = float(pg["lr"]) * self.gamma
            new_lr = trainer.optimizers[0].param_groups[0]["lr"]
            print(f"[LR] epoch={e} → decayed lr to {new_lr:.6g}")
        except Exception as err:
            print(f"[LR] decay skipped: {err}")
            
# -----------------------------------------------------------------------
# Compute / device configuration (optimised for NVIDIA L4 on GCP)
# -----------------------------------------------------------------------
if torch.cuda.is_available():
    ACCELERATOR = "gpu"
    DEVICES = "auto"          # use all visible GPUs if more than one
    # default to bf16 but fall back to fp16 if unsupported (e.g., T4)
    PRECISION = "32" #bf16-mixed
    try:
        if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
            PRECISION = "16-mixed"
    except Exception:
        try:
            major, _minor = torch.cuda.get_device_capability()
            if major < 8:  # pre-Ampere
                PRECISION = "16-mixed"
        except Exception:
            PRECISION = "16-mixed"
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    ACCELERATOR = "mps"
    DEVICES = 1
    PRECISION = "16-mixed"
else:
    ACCELERATOR = "cpu"
    DEVICES = 1
    PRECISION = 32

# -----------------------------------------------------------------------
# CLI overrides for common CONFIG knobs
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# CLI overrides for common CONFIG knobs
# -----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="TFT training with optional permutation importance", add_help=True)
parser.add_argument("--disable_warmups", type=lambda s: str(s).lower() in ("1","true","t","yes","y","on"),
                    default=False, help="Disable bias/QLIKE warm-ups and tail ramp")
parser.add_argument("--warmup_guard_patience", type=int, default=2,
                    help="Consecutive worsening epochs before freezing warm-ups")
parser.add_argument("--warmup_guard_tol", type=float, default=0.0,
                    help="Minimum delta in val_loss to count as worsening")
parser.add_argument("--max_encoder_length", type=int, default=None, help="Max encoder length")
parser.add_argument("--max_epochs", type=int, default=None, help="Max training epochs")
parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
parser.add_argument("--perm_len", type=int, default=None, help="Permutation block length for importance")
parser.add_argument("--perm_block_size", type=int, default=None, help="Alias for --perm_len (permutation block length)")
parser.add_argument(
    "--enable_perm_importance", "--enable-feature-importance",
    type=lambda s: str(s).lower() in ("1","true","t","yes","y","on"),
    default=None,
    help="Enable permutation feature importance (true/false)"
)
# Cloud paths / storage overrides
parser.add_argument("--gcs_bucket", type=str, default=None, help="GCS bucket name to read/write from")
parser.add_argument("--gcs_data_prefix", type=str, default=None, help="Full GCS prefix for data parquet folder")
parser.add_argument("--gcs_output_prefix", type=str, default=None, help="Full GCS prefix for outputs/checkpoints")
# Performance / input control
parser.add_argument("--data_dir", type=str, default=None, help="Local folder containing universal_*.parquet; if set, prefer local over GCS")
parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers (defaults to CPU count - 1)")
parser.add_argument("--prefetch_factor", type=int, default=8, help="DataLoader prefetch factor (per worker)")
# Performance / input control
parser.add_argument("--check_val_every_n_epoch", type=int, default=1, help="Validate every N epochs")
parser.add_argument("--log_every_n_steps", type=int, default=200, help="How often to log train steps")
parser.add_argument("--learning_rate", type=float, default=None, help="Override model learning rate")
parser.add_argument(
    "--resume",
    type=lambda s: str(s).lower() in ("1","true","t","yes","y","on"),
    default=False,                       # <— make default False
    help="Resume from latest checkpoint if available"
)
# Quick-run subsetting for speed
parser.add_argument("--train_max_rows", type=int, default=None, help="Limit number of rows in TRAIN for fast iterations")
parser.add_argument("--val_max_rows", type=int, default=None, help="Limit number of rows in VAL (optional; default full)")
parser.add_argument(
    "--subset_mode",
    type=str,
    default="per_asset_tail",
    choices=["per_asset_tail", "per_asset_head", "random"],
    help="Strategy for selecting a subset when limiting rows"
)
parser.add_argument("--fi_max_batches", type=int, default=20, help="Max val batches per feature in FI.")
# Parse known args so stray platform args do not crash the script
ARGS, _UNKNOWN = parser.parse_known_args()

# -----------------------------------------------------------------------
# CONFIG – tweak as required (GCS-aware)
# -----------------------------------------------------------------------
GCS_BUCKET = os.environ.get("GCS_BUCKET", "river-ml-bucket")
GCS_DATA_PREFIX = f"gs://{GCS_BUCKET}/Data/CleanedData"
GCS_OUTPUT_PREFIX = f"gs://{GCS_BUCKET}/Dissertation/TFT"

# Apply CLI overrides (if provided)
if getattr(ARGS, "gcs_bucket", None):
    GCS_BUCKET = ARGS.gcs_bucket
    # recompute defaults if specific prefixes are not provided
    if not getattr(ARGS, "gcs_data_prefix", None):
        GCS_DATA_PREFIX = f"gs://{GCS_BUCKET}/CleanedData"
    if not getattr(ARGS, "gcs_output_prefix", None):
        GCS_OUTPUT_PREFIX = f"gs://{GCS_BUCKET}/Dissertation/Feature_Ablation"
if getattr(ARGS, "gcs_data_prefix", None):
    GCS_DATA_PREFIX = ARGS.gcs_data_prefix
if getattr(ARGS, "gcs_output_prefix", None):
    GCS_OUTPUT_PREFIX = ARGS.gcs_output_prefix

# Local ephemerals (good for GCE/Vertex)
LOCAL_DATA_DIR = Path(os.environ.get("LOCAL_DATA_DIR", "/tmp/data/CleanedData"))
LOCAL_OUTPUT_DIR = Path(os.environ.get("LOCAL_OUTPUT_DIR", "/tmp/feature_ablation"))
LOCAL_RUN_DIR = Path(os.environ.get("LOCAL_RUN_DIR", "/tmp/tft_run"))
LOCAL_LOG_DIR = LOCAL_RUN_DIR / "lightning_logs"
LOCAL_CKPT_DIR = LOCAL_RUN_DIR / "checkpoints"
for p in [LOCAL_DATA_DIR, LOCAL_OUTPUT_DIR, LOCAL_CKPT_DIR, LOCAL_LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ---------------- TensorBoard Logger ----------------
logger = TensorBoardLogger(save_dir=str(LOCAL_LOG_DIR.parent), name=LOCAL_LOG_DIR.name)

# Remote checkpoint prefix on GCS
CKPT_GCS_PREFIX = f"{GCS_OUTPUT_PREFIX}/checkpoints"

# ---- Upload helper (GCS-aware) ----
def upload_file_to_gcs(local_path: str, gcs_uri: str):
    if fs is None:
        print(f"[WARN] GCS not available (gcsfs not installed); skipping upload: {gcs_uri}")
        return
    try:
        with fsspec.open(gcs_uri, "wb") as f_out, open(local_path, "rb") as f_in:
            shutil.copyfileobj(f_in, f_out)
        print(f"✓ Uploaded {local_path} → {gcs_uri}")
    except Exception as e:
        print(f"[WARN] Failed to upload {local_path} to {gcs_uri}: {e}")


# Prefer GCS if the files exist there
try:
    fs = fsspec.filesystem("gcs")
except Exception:
    fs = None  # gcsfs not installed / protocol unavailable

def gcs_exists(path: str) -> bool:
    if fs is None:
        return False
    try:
        return fs.exists(path)
    except Exception:
        return False

TRAIN_URI = f"{GCS_DATA_PREFIX}/universal_train.parquet"
VAL_URI   = f"{GCS_DATA_PREFIX}/universal_val.parquet"
TEST_URI  = f"{GCS_DATA_PREFIX}/universal_test.parquet"

# Resolve data paths: prefer explicit --data_dir, then GCS if available, else local default
def _local_cleaned_dir():
    if getattr(ARGS, "data_dir", None):
        return Path(ARGS.data_dir).expanduser().resolve()
    return Path("/Users/riverwest-gomila/Desktop/Data/CleanedData")

LOCAL_TRAIN = _local_cleaned_dir() / "universal_train.parquet"
LOCAL_VAL   = _local_cleaned_dir() / "universal_val.parquet"
LOCAL_TEST  = _local_cleaned_dir() / "universal_test.parquet"

def _all_exists_local(paths):
    try:
        return all(Path(p).exists() for p in paths)
    except Exception:
        return False

def _all_exists_gcs(paths):
    if fs is None:
        return False
    try:
        return all(fs.exists(p) for p in paths)
    except Exception:
        return False

# Choose the source of truth for READ_PATHS
if _all_exists_gcs([TRAIN_URI, VAL_URI, TEST_URI]):
    READ_PATHS = [TRAIN_URI, VAL_URI, TEST_URI]
elif _all_exists_local([LOCAL_TRAIN, LOCAL_VAL, LOCAL_TEST]):
    READ_PATHS = [str(LOCAL_TRAIN), str(LOCAL_VAL), str(LOCAL_TEST)]
else:
    raise FileNotFoundError(
        "Could not locate the dataset. Checked GCS URIs and the local default. "
        "Provide --data_dir to a folder containing universal_train/val/test.parquet, "
        "or ensure GCS is configured (install gcsfs & correct URIs)."
    )

def get_resume_ckpt_path():
    if not RESUME_ENABLED:
        return None

    try:
        local_ckpts = sorted(LOCAL_CKPT_DIR.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if local_ckpts:
            return str(local_ckpts[0])
    except Exception:
        pass
    # Fallback: try GCS "last.ckpt" then the lexicographically latest .ckpt
    try:
        if fs is not None:
            last_uri = f"{CKPT_GCS_PREFIX}/last.ckpt"
            if fs.exists(last_uri):
                dst = LOCAL_CKPT_DIR / "last.ckpt"
                with fsspec.open(last_uri, "rb") as f_in, open(dst, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                return str(dst)
            # else get the latest by name (filenames contain ISO timestamps)
            entries = fs.glob(f"{CKPT_GCS_PREFIX}/*.ckpt") or []
            if entries:
                latest = sorted(entries)[-1]
                dst = LOCAL_CKPT_DIR / Path(latest).name
                with fsspec.open(latest, "rb") as f_in, open(dst, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                return str(dst)
    except Exception as e:
        print(f"[WARN] Could not fetch resume checkpoint from GCS: {e}")
    return None

def mirror_local_ckpts_to_gcs():
    if fs is None:
        print("[WARN] GCS not available; skipping checkpoint upload.")
        return
    try:
        for p in LOCAL_CKPT_DIR.glob("*.ckpt"):
            remote = f"{CKPT_GCS_PREFIX}/{p.name}"
            with fsspec.open(remote, "wb") as f_out, open(p, "rb") as f_in:
                shutil.copyfileobj(f_in, f_out)
            #print(f"✓ Mirrored checkpoint {p} → {remote}")
    except Exception as e:
        print(f"[WARN] Failed to mirror checkpoints: {e}")

class MirrorCheckpoints(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        mirror_local_ckpts_to_gcs()
    def on_exception(self, trainer, pl_module, err):
        mirror_local_ckpts_to_gcs()
    def on_train_end(self, trainer, pl_module):
        mirror_local_ckpts_to_gcs()


class TailWeightRamp(pl.Callback):
    def __init__(
        self,
        vol_loss,
        start: float = 1.0,
        end: float = 1.25,
        ramp_epochs: int = 12,
        gate_by_calibration: bool = True,
        gate_low: float = 0.9,
        gate_high: float = 1.2,
        gate_patience: int = 2,
    ):
        super().__init__()
        self.vol_loss = vol_loss
        self.start = float(start)
        self.end = float(end)
        self.ramp = int(ramp_epochs)
        self.gate = bool(gate_by_calibration)
        self.gate_low = float(gate_low)
        self.gate_high = float(gate_high)
        self.gate_patience = int(gate_patience)
        self._ok_streak = 0
        self._trigger_epoch = None

    def _get_scale_ema(self, trainer):
        for cb in getattr(trainer, "callbacks", []):
            if isinstance(cb, BiasWarmupCallback):
                return getattr(cb, "_scale_ema", None)
        return None

    def on_train_epoch_start(self, trainer, pl_module):
        # Global CLI switch: if --disable_warmups is true, hold tail_weight steady
        try:
            if globals().get("ARGS") is not None and getattr(ARGS, "disable_warmups", False):
                print(f"[TAIL] epoch={int(getattr(trainer,'current_epoch',0))} DISABLED via --disable_warmups; tail_weight stays {self.vol_loss.tail_weight}")
                return
        except Exception:
            pass

        # Freeze if BiasWarmup froze things
        frozen = False
        for cb in getattr(trainer, "callbacks", []):
            if isinstance(cb, BiasWarmupCallback) and getattr(cb, "_frozen", False):
                frozen = True
                break
        e = int(getattr(trainer, "current_epoch", 0))
        if frozen:
            print(f"[TAIL] epoch={e} frozen; tail_weight stays {self.vol_loss.tail_weight}")
            return

        # Optional calibration gate
        if self.gate:
            scale_ema = self._get_scale_ema(trainer)
            if (scale_ema is None) or not (self.gate_low <= float(scale_ema) <= self.gate_high):
                self._ok_streak = 0
                self._trigger_epoch = None
                self.vol_loss.tail_weight = self.start
                print(f"[TAIL] epoch={e} gated (scale_ema={scale_ema}); tail_weight={self.vol_loss.tail_weight}")
                return
            else:
                self._ok_streak = min(self.gate_patience, self._ok_streak + 1)
                if self._ok_streak < self.gate_patience:
                    self.vol_loss.tail_weight = self.start
                    print(f"[TAIL] epoch={e} gating warm-up {self._ok_streak}/{self.gate_patience}; tail_weight={self.vol_loss.tail_weight}")
                    return
                if self._trigger_epoch is None:
                    self._trigger_epoch = e
                    print(f"[TAIL] epoch={e} gate OPEN (scale_ema={scale_ema}); starting ramp")

        # Ramping once gate is open (or immediately if gate disabled)
        eff_end = min(self.end, 1.5)
        eff_ramp = max(self.ramp, 8)
        base = self._trigger_epoch if (self.gate and self._trigger_epoch is not None) else 0
        prog = min(1.0, max(0.0, (e - base + 1) / float(eff_ramp)))
        self.vol_loss.tail_weight = self.start + (eff_end - self.start) * prog
        print(f"[TAIL] epoch={e} tail_weight={self.vol_loss.tail_weight:.4f} (ramp prog={prog:.2f}, gate={'on' if self.gate else 'off'})")



import math

class CosineLR(pl.Callback):
    """
    Cosine decay that ends at a near-zero floor, with an optional final hold
    at the floor. This version updates **per training batch** so the last
    training step lands at the cosine minimum.

    Args:
        start_epoch: Epoch index (0-based) to begin cosine decay. Before this,
            LR stays at the base LR.
        eta_min_ratio: Final LR is base_lr * eta_min_ratio per param group.
        hold_last_epochs: Number of final epochs to hold LR at the floor
            (converted to steps at runtime) to settle in the minima.
        warmup_steps: Optional number of warmup steps (linear 0→1) before cosine.
    """
    def __init__(
        self,
        start_epoch: int = 8,
        eta_min_ratio: float = 1e-5,
        hold_last_epochs: int = 1,
        warmup_steps: int | None = None,
    ):
        super().__init__()
        self.start_epoch = int(start_epoch)
        self.eta_min_ratio = float(eta_min_ratio)
        self.hold_last_epochs = int(max(0, hold_last_epochs))
        self.warmup_steps = None if warmup_steps is None else int(max(0, warmup_steps))

        # resolved at fit start
        self._base_lrs = None
        self._eta_mins = None
        self._total_steps = None
        self._steps_per_epoch = None
        self._start_steps = None
        self._hold_last_steps = None
        self._cosine_span = None

    def on_fit_start(self, trainer, pl_module):
        # Snapshot base LRs and per-group eta_min
        self._base_lrs = []
        for opt in trainer.optimizers:
            for pg in opt.param_groups:
                self._base_lrs.append(float(pg.get("lr", 1e-3)))
        self._eta_mins = [lr * self.eta_min_ratio for lr in self._base_lrs]

        max_epochs = int(getattr(trainer, "max_epochs", 1) or 1)

        # Prefer Lightning’s estimate of total steps
        total_steps = getattr(trainer, "estimated_stepping_batches", None)
        if total_steps is None:
            # Fallback: steps/epoch * epochs
            try:
                steps_per_epoch = int(getattr(trainer, "num_training_batches", 0) or 0)
            except Exception:
                steps_per_epoch = 0
            if steps_per_epoch <= 0:
                steps_per_epoch = 1
            total_steps = steps_per_epoch * max_epochs

        self._total_steps = int(total_steps)
        self._steps_per_epoch = max(1, self._total_steps // max_epochs)

        # Convert epoch-based knobs to steps
        self._start_steps = int(self.start_epoch * self._steps_per_epoch)
        self._hold_last_steps = int(self.hold_last_epochs * self._steps_per_epoch)

        # Warmup default
        if self.warmup_steps is None:
            self.warmup_steps = 0

        # Cosine span (exclude pre-cosine + hold)
        self._cosine_span = max(1, self._total_steps - self._start_steps - self._hold_last_steps)

        print(
            f"[LR Cosine(step)] total_steps={self._total_steps} steps/epoch={self._steps_per_epoch} "
            f"start_epoch={self.start_epoch} (start_steps={self._start_steps}) hold_last_epochs={self.hold_last_epochs} "
            f"(hold_last_steps={self._hold_last_steps}) eta_min_ratio={self.eta_min_ratio} warmup_steps={self.warmup_steps}"
        )

    def _set_lrs(self, trainer, factor: float):
        # factor in [0,1]: 1→base_lr, 0→eta_min
        i = 0
        for opt in trainer.optimizers:
            for pg in opt.param_groups:
                base_lr = self._base_lrs[i]
                eta_min = self._eta_mins[i]
                lr = eta_min + (base_lr - eta_min) * float(max(0.0, min(1.0, factor)))
                pg["lr"] = float(lr)
                i += 1

    def _compute_factor(self, step: int) -> float:
        # Warmup (optional)
        if self.warmup_steps and step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))

        # Before cosine starts
        if step < self._start_steps:
            return 1.0

        # Hold at floor for last K steps
        if step >= self._total_steps - self._hold_last_steps:
            return 0.0

        # Cosine phase
        t = float(step - self._start_steps + 1) / float(max(1, self._cosine_span))
        t = min(max(t, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self._total_steps is None:
            return
        step = int(getattr(trainer, "global_step", 0))
        step = min(max(step, 0), self._total_steps)  # clamp
        factor = self._compute_factor(step)
        self._set_lrs(trainer, factor)
        # Optional sparse logs:
        # if step % max(1, self._steps_per_epoch) == 0:
        #     try:
        #         print(f"[LR Cosine(step)] step={step}/{self._total_steps} factor={factor:.4f} "
        #               f"lr={trainer.optimizers[0].param_groups[0]['lr']:.6g}")
        #     except Exception:
        #         pass

class ReduceLROnPlateauCallback(pl.Callback):
    def __init__(self, monitor="val_comp_overall", factor=0.5, patience=5, min_lr=1e-5, cooldown=0, stop_after_epoch=None):
        self.monitor, self.factor, self.patience, self.min_lr, self.cooldown = monitor, factor, patience, min_lr, cooldown
        self.stop_after_epoch = stop_after_epoch
        self.best, self.bad, self.cool = float("inf"), 0, 0

    def on_validation_end(self, trainer, pl_module):
        e = int(getattr(trainer, "current_epoch", 0))
        if (self.stop_after_epoch is not None) and (e >= int(self.stop_after_epoch)):
            return  # hand over to cosine
        if self.cool:
            self.cool -= 1
            return
        val = trainer.callback_metrics.get(self.monitor)
        if val is None:
            return
        try:
            val = float(val.item())
        except Exception:
            val = float(val)
        if val + 1e-12 < self.best:
            self.best, self.bad = val, 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                for opt in trainer.optimizers:
                    for pg in opt.param_groups:
                        if "lr" in pg and pg["lr"] is not None:
                            pg["lr"] = max(self.min_lr, float(pg["lr"]) * self.factor)
                new_lr = trainer.optimizers[0].param_groups[0]["lr"]
                print(f"[LR Plateau] ↓ lr → {new_lr:.6g}")
                self.bad, self.cool = 0, self.cooldown


VOL_LOSS = AsymmetricQuantileLoss(
    quantiles=VOL_QUANTILES,
    underestimation_factor=1.06,  # managed by BiasWarmupCallback
    mean_bias_weight=0.002,        # small centering on the median for MAE
    tail_q=0.9,
    tail_weight=0,              # will be ramped by TailWeightRamp
    qlike_weight=0.02,             # QLIKE weight is ramped safely in BiasWarmupCallback
    reduction="mean",
)
# ---------------- Callback bundle (bias warm-up, tail ramp, LR control) ----------------
EXTRA_CALLBACKS = [
      BiasWarmupCallback(
          vol_loss=VOL_LOSS,
          target_under=1.12,
          target_mean_bias=0.05,
          warmup_epochs=3,
          qlike_target_weight=0.15,   # keep out of the loss; diagnostics only
          start_mean_bias=0.02,
          mean_bias_ramp_until=10,
          guard_patience=getattr(ARGS, "warmup_guard_patience", 5),
          guard_tol=getattr(ARGS, "warmup_guard_tol", 0.005),
          alpha_step=0.05,
      ),
      TailWeightRamp(
          vol_loss=VOL_LOSS,
          start=1.0,
          end=1.25,
          ramp_epochs=32,
          gate_by_calibration=True,
          gate_low=0.9,
          gate_high=1.2,
          gate_patience=1,
      ),
      ReduceLROnPlateauCallback(
          monitor="val_composite_overall", factor=0.5, patience=5, min_lr=3e-5, cooldown=1, stop_after_epoch=None
      ),
      ModelCheckpoint(
          dirpath=str(LOCAL_CKPT_DIR),
          filename="tft-{epoch:02d}-{val_mae_overall:.4f}",
          monitor="val_qlike_overall",
          mode="min",
          save_top_k=2,
          save_last=True,
      ),
      StochasticWeightAveraging(swa_lrs = 0.00091, swa_epoch_start=max(1, int(0.8 * MAX_EPOCHS))),
      CosineLR(start_epoch=4, eta_min_ratio=5e-6, hold_last_epochs=2, warmup_steps=0),
      ]

class ValLossHistory(pl.Callback):
    """
    Records per-epoch validation metrics to a CSV so you can plot later.
    """
    def __init__(self, out_csv: Path):
        super().__init__()
        self.out_csv = Path(out_csv)
        self.rows = []

    def _as_float(self, x):
        try:
            return float(x.item() if hasattr(x, "item") else x)
        except Exception:
            return float("nan")

    def _current_lr(self, trainer):
        try:
            return float(trainer.optimizers[0].param_groups[0]["lr"])
        except Exception:
            return float("nan")

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        row = {
            "epoch": int(getattr(trainer, "current_epoch", -1)) + 1,
            "val_loss":            self._as_float(m.get("val_loss", float("nan"))),
            "val_qlike_overall":   self._as_float(m.get("val_qlike_overall", float("nan"))),
            "val_qlike_cal":      self._as_float(m.get("val_qlike_cal", float("nan"))),
            "val_mae_overall":     self._as_float(m.get("val_mae_overall", float("nan"))),
            "val_rmse_overall":    self._as_float(m.get("val_rmse_overall", float("nan"))),
            "val_acc_overall":     self._as_float(m.get("val_acc_overall", float("nan"))),
            "val_brier_overall":   self._as_float(m.get("val_brier_overall", float("nan"))),
            "val_auroc_overall":   self._as_float(m.get("val_auroc_overall", float("nan"))),
            "lr":                  self._current_lr(trainer),
        }
        self.rows.append(row)
        try:
            import pandas as _pd
            _pd.DataFrame(self.rows).to_csv(self.out_csv, index=False)
            print(f"[VAL-HIST] wrote {self.out_csv}")
        except Exception as e:
            print(f"[VAL-HIST] write failed: {e}")

GROUP_ID: List[str] = ["asset"]
TIME_COL = "Time"
TARGETS  = ["realised_vol", "direction"]

MAX_ENCODER_LENGTH = 96
MAX_PRED_LENGTH    = 1

EMBEDDING_CARDINALITY = {}

BATCH_SIZE   = 128
MAX_EPOCHS   = 35
EARLY_STOP_PATIENCE = 7
PERM_BLOCK_SIZE = 288

# Artifacts are written locally then uploaded to GCS
from datetime import datetime, timezone, timedelta
RUN_SUFFIX = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
MODEL_SAVE_PATH = (LOCAL_CKPT_DIR / f"tft_realised_vol_e{MAX_EPOCHS}_{RUN_SUFFIX}.ckpt")

SEED = 8
# Full-run determinism for reproducible validation metrics
try:
    seed_everything(SEED, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception:
    pass
WEIGHT_DECAY = 0.0001 #0.00578350719515325     # weight decay for AdamW
GRADIENT_CLIP_VAL = 0.78 #0.78    # gradient clipping value for Trainer
# Feature-importance controls
ENABLE_FEATURE_IMPORTANCE = True   # gate FI so you can toggle it
FI_MAX_BATCHES = 40       # number of val batches to sample for FI

# ---- Apply CLI overrides (only when provided) ----
if ARGS.batch_size is not None:
    BATCH_SIZE = int(ARGS.batch_size)
if ARGS.max_encoder_length is not None:
    MAX_ENCODER_LENGTH = int(ARGS.max_encoder_length)
if ARGS.max_epochs is not None:
    MAX_EPOCHS = int(ARGS.max_epochs)
if ARGS.perm_len is not None:
    PERM_BLOCK_SIZE = int(ARGS.perm_len)
if ARGS.enable_perm_importance is not None:
    ENABLE_FEATURE_IMPORTANCE = bool(ARGS.enable_perm_importance)
if getattr(ARGS, "fi_max_batches", None) is not None:
    FI_MAX_BATCHES = int(ARGS.fi_max_batches)

# ---- Learning rate and resume CLI overrides ----
LR_OVERRIDE = float(ARGS.learning_rate) if getattr(ARGS, "learning_rate", None) is not None else None
RESUME_ENABLED = bool(getattr(ARGS, "resume", True))
RESUME_CKPT = get_resume_ckpt_path() if RESUME_ENABLED else None

# -----------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------
def load_split(path: str) -> pd.DataFrame:
    """
    Load a parquet split and guarantee we have usable `id` and timestamp columns.

    1. Converts `TIME_COL` to pandas datetime.
    2. Ensures an identifier column called `id` (or whatever `GROUP_ID[0]` is).
       If not present, auto-detects common synonyms and renames
    """
    path_str = str(path)
    df = pd.read_parquet(path)
    df = df.reset_index(drop=True).copy()

    # --- convert timestamp ---
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    # --- identifier column handling ---
    if GROUP_ID[0] not in df.columns:
        cand = next(
            (c for c in df.columns if c.lower() in
             {"symbol", "ticker", "asset", "security_id", "instrument"}), None
        )
        if cand is None:
            raise ValueError(
                f"No identifier column '{GROUP_ID[0]}' in {path_str}. Edit GROUP_ID or rename your column."
            )

        df.rename(columns={cand: GROUP_ID[0]}, inplace=True)
        print(f"[INFO] Renamed '{cand}' ➜ '{GROUP_ID[0]}'")

    df[GROUP_ID[0]] = df[GROUP_ID[0]].astype(str)

    # --- target alias handling ------------------------------------------------
    TARGET_ALIASES = {
        "realised_vol": ["Realised_Vol", "rs_sigma", "realized_vol", "rv"],
        "direction":    ["Sign_Label", "sign_label", "Direction", "direction_label"],
    }
    for canonical, aliases in TARGET_ALIASES.items():
        if canonical not in df.columns:
            alias_found = next((a for a in aliases if a in df.columns), None)
            if alias_found:
                df.rename(columns={alias_found: canonical}, inplace=True)
                print(f"[INFO] Renamed '{alias_found}' ➜ '{canonical}'")
            else:
                # Only warn; the downstream code will raise if the column is truly required
                print(f"[WARN] Column '{canonical}' not found in {path_str} and no alias detected.")

    return df


def add_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    """Add monotonically increasing integer time index per asset."""
    df = df.sort_values(GROUP_ID + [TIME_COL])
    df["time_idx"] = (
        df.groupby(GROUP_ID)
          .cumcount()
          .astype("int64")
    )
    return df

# -----------------------------------------------------------------------
# Split sanity: enforce chronological, non-overlapping 80/10/10 per asset
# -----------------------------------------------------------------------
def _chronological_resplit(df: pd.DataFrame, train_frac: float = 0.8, val_frac: float = 0.1):
    """
    Deterministically re-split a combined dataframe into train/val/test by time,
    per asset, using the provided fractions (train, val, test = rest).
    Assumes columns: 'asset', 'Time'.
    Returns: (train_df, val_df, test_df)
    """
    assert 0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1
    df = df.sort_values(["asset", "Time"]).reset_index(drop=True)
    parts = []
    for asset, g in df.groupby("asset", observed=True, sort=False):
        n = len(g)
        if n == 0:
            continue
        tr_end = int(n * train_frac)
        va_end = int(n * (train_frac + val_frac))
        g = g.copy()
        g.loc[g.index[:tr_end], "split"] = "train"
        g.loc[g.index[tr_end:va_end], "split"] = "val"
        g.loc[g.index[va_end:], "split"] = "test"
        parts.append(g)
    out = pd.concat(parts, axis=0, ignore_index=False)
    train_df = out[out["split"] == "train"].drop(columns=["split"]).copy()
    val_df   = out[out["split"] == "val"].drop(columns=["split"]).copy()
    test_df  = out[out["split"] == "test"].drop(columns=["split"]).copy()
    return train_df, val_df, test_df


def assert_and_fix_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Ensure each split is chronologically sorted per asset and there is no time overlap:
      max(train) < min(val) and max(val) < min(test), per asset.
    If violated for any asset, re-split deterministically (80/10/10) from the union.
    Returns possibly corrected (train_df, val_df, test_df).
    """
    # Sort each split per asset/time
    train_df = train_df.sort_values(["asset", "Time"]).copy()
    val_df   = val_df.sort_values(["asset", "Time"]).copy()
    test_df  = test_df.sort_values(["asset", "Time"]).copy()

    def _bounds(df):
        if df.empty:
            return pd.DataFrame(columns=["asset","min","max"]).set_index("asset")
        return df.groupby("asset", observed=True)["Time"].agg(["min","max"])

    tb = _bounds(train_df).rename(columns={"min":"tr_min","max":"tr_max"})
    vb = _bounds(val_df).rename(columns={"min":"va_min","max":"va_max"})
    sb = _bounds(test_df).rename(columns={"min":"te_min","max":"te_max"})

    bounds = tb.join(vb, how="outer").join(sb, how="outer")

    bad_assets = []
    for a, r in bounds.iterrows():
        tr_max = r.get("tr_max", pd.NaT)
        va_min = r.get("va_min", pd.NaT)
        va_max = r.get("va_max", pd.NaT)
        te_min = r.get("te_min", pd.NaT)
        ok = True
        if pd.notna(tr_max) and pd.notna(va_min):
            ok = ok and (tr_max < va_min)
        if pd.notna(va_max) and pd.notna(te_min):
            ok = ok and (va_max < te_min)
        if not ok:
            bad_assets.append(a)

    if bad_assets:
        print(f"[SPLIT] Detected chronological overlap for assets: {bad_assets} — re-splitting 80/10/10 by time.")
        union = pd.concat([
            train_df.assign(_orig_split="train"),
            val_df.assign(_orig_split="val"),
            test_df.assign(_orig_split="test"),
        ], axis=0, ignore_index=False).drop_duplicates().copy()
        union = union.sort_values(["asset","Time"]).drop_duplicates(subset=["asset","Time"], keep="first")
        train_df, val_df, test_df = _chronological_resplit(union, train_frac=0.8, val_frac=0.1)
    else:
        print("[SPLIT] Splits look chronological and non-overlapping.")
    return train_df, val_df, test_df

# -----------------------------------------------------------------------
# Calendar features (help the model learn intraday/weekly seasonality)
# -----------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    minute_of_day = df[TIME_COL].dt.hour * 60 + df[TIME_COL].dt.minute
    df["sin_tod"] = np.sin(2 * np.pi * minute_of_day / 1440.0).astype("float32")
    df["cos_tod"] = np.cos(2 * np.pi * minute_of_day / 1440.0).astype("float32")
    dow = df[TIME_COL].dt.dayofweek
    df["sin_dow"] = np.sin(2 * np.pi * dow / 7.0).astype("float32")
    df["cos_dow"] = np.cos(2 * np.pi * dow / 7.0).astype("float32")
    df["Is_Weekend"] = (dow >= 5).astype("int8")
    return df

# -----------------------------------------------------------------------
# Fast subset helper (keeps temporal structure per asset when possible)
# -----------------------------------------------------------------------
def subset_time_series(df: pd.DataFrame, max_rows: int | None, mode: str = "per_asset_tail") -> pd.DataFrame:
    """Return a subset of *approximately* max_rows, preserving sequence order.
    modes:
      • per_asset_tail: take roughly equal tail slices per asset, then trim
      • per_asset_head: same but from the head
      • random: global random sample (may break sequences; use only for quick smoke tests)
    """
    if max_rows is None or max_rows <= 0 or max_rows >= len(df):
        return df
    df = df.sort_values(GROUP_ID + [TIME_COL]).reset_index(drop=True)
    if mode == "random":
        out = df.sample(n=int(max_rows), random_state=SEED).sort_values(GROUP_ID + [TIME_COL])
        return out.reset_index(drop=True)

    # Per-asset slicing
    groups = list(df.groupby(GROUP_ID[0], observed=True)) if GROUP_ID else [(None, df)]
    n_assets = max(1, len(groups))
    take_each = max(1, int(np.ceil(max_rows / n_assets)))
    parts = []
    for _, gdf in groups:
        if mode == "per_asset_head":
            parts.append(gdf.head(take_each))
        else:  # per_asset_tail
            parts.append(gdf.tail(take_each))
    out = pd.concat(parts, axis=0, ignore_index=True)
    # trim to target size while preserving order (head or tail consistent with mode)
    out = out.sort_values(GROUP_ID + [TIME_COL]).reset_index(drop=True)
    if len(out) > max_rows:
        if mode == "per_asset_head":
            out = out.head(int(max_rows))
        else:
            out = out.tail(int(max_rows))
    return out.reset_index(drop=True)

# -----------------------------------------------------------------------
# Permutation Importance helpers at module scope (decoded metric = MAE + RMSE + 0.05 * DirBCE)
# -----------------------------------------------------------------------


# NOTE: If you explicitly instantiate a TQDMProgressBar in your callbacks, remove it or comment it out.
# For example, if you see:
# callbacks.append(TQDMProgressBar(...))
# or
# callbacks = [TQDMProgressBar(...), ...]
# remove the TQDMProgressBar from the list.

@torch.no_grad()
def _evaluate_decoded_metrics(
    model,
    ds: TimeSeriesDataSet,
    batch_size: int,
    max_batches: int,
    num_workers: int,
    prefetch: int,
    pin_memory: bool,
    vol_norm,                 # GroupNormalizer from TRAIN (for realised_vol)
):
    """
    Evaluate model on a dataset configured with predict=True, computing:
      • MAE, RMSE, QLIKE on decoded realised_vol
      • Brier on direction (if available)
    Returns: (mae, rmse, brier, qlike, N)
    """
    model.eval()
    model_device = next(model.parameters()).device  # <<< NEW

    # dataloader mirrors validation loader; dataset already has predict=True
    loader = ds.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=False,
        prefetch_factor=prefetch if num_workers and num_workers > 0 else None,
    )

    # --- helper to move nested batch dicts to model_device ---
    def _move_to_device(x):
        if torch.is_tensor(x):
            return x.to(model_device, non_blocking=True)
        if isinstance(x, dict):
            return {k: _move_to_device(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            t = [_move_to_device(v) for v in x]
            return type(x)(t) if not isinstance(x, tuple) else tuple(t)
        return x

    g_list, yv_list, pv_list = [], [], []
    yd_list, pd_list = [], []

    batches_seen = 0
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x = batch[0]
            y = batch[1]
        else:
            continue
        if not isinstance(x, dict):
            continue

        # --- MOVE EVERYTHING to the model's device BEFORE forward ---
        x_dev = _move_to_device(x)

        # group ids (taken from x_dev so same device)
        groups = None
        for k in ("groups", "group_ids", "group_id"):
            if k in x_dev and x_dev[k] is not None:
                groups = x_dev[k]
                break
        if groups is None:
            continue
        g = groups[0] if isinstance(groups, (list, tuple)) else groups
        while torch.is_tensor(g) and g.ndim > 1 and g.size(-1) == 1:
            g = g.squeeze(-1)

        # targets from decoder_target (fallback to y)
        dec_t = x_dev.get("decoder_target", None)
        y_vol_t, y_dir_t = None, None
        if torch.is_tensor(dec_t):
            t = dec_t
            if t.ndim == 3 and t.size(-1) == 1:
                t = t[..., 0]
            if t.ndim == 2 and t.size(1) >= 1:
                y_vol_t = t[:, 0]
                if t.size(1) > 1:
                    y_dir_t = t[:, 1]
        elif isinstance(dec_t, (list, tuple)) and len(dec_t) >= 1:
            y_vol_t = dec_t[0]
            if torch.is_tensor(y_vol_t) and y_vol_t.ndim == 3 and y_vol_t.size(-1) == 1:
                y_vol_t = y_vol_t[..., 0]
            if len(dec_t) > 1 and torch.is_tensor(dec_t[1]):
                y_dir_t = dec_t[1]
                if y_dir_t.ndim == 3 and y_dir_t.size(-1) == 1:
                    y_dir_t = y_dir_t[..., 0]

        if (y_vol_t is None) and torch.is_tensor(y):
            yy = y.to(model_device, non_blocking=True)  # <<< ensure same device if used
            if yy.ndim == 3 and yy.size(1) == 1:
                yy = yy[:, 0, :]
            if yy.ndim == 2 and yy.size(1) >= 1:
                y_vol_t = yy[:, 0]
                if yy.size(1) > 1:
                    y_dir_t = yy[:, 1]

        if (y_vol_t is None) or (not torch.is_tensor(g)):
            continue

        # forward pass and head extraction
        y_hat = model(x_dev)  # <<< CUDA-safe now
        pred = getattr(y_hat, "prediction", y_hat)
        if isinstance(pred, dict) and "prediction" in pred:
            pred = pred["prediction"]

        # split heads (same logic as before)
        def _extract_heads(prediction):
            if isinstance(prediction, (list, tuple)):
                vol_q = prediction[0]
                d_log = prediction[1] if len(prediction) > 1 else None
                return vol_q, d_log
            t = prediction
            if torch.is_tensor(t):
                if t.ndim >= 4 and t.size(1) == 1:
                    t = t.squeeze(1)
                if t.ndim == 3 and t.size(1) == 1:
                    t = t[:, 0, :]
                if t.ndim == 2:
                    vol_q = t[:, :-1]
                    d_log = t[:, -1]
                    return vol_q, d_log
            return None, None

        p_vol, p_dir = _extract_heads(pred)
        if p_vol is None:
            continue

        # take median quantile for vol
        def _median_from_quantiles(vol_q: torch.Tensor) -> torch.Tensor:
            vol_q = torch.cummax(vol_q, dim=-1).values
            # assumes 7 quantiles with 0.50 at index 3
            return vol_q[..., 3]

        p_vol_med = _median_from_quantiles(p_vol)

        L = g.shape[0]
        g_list.append(g.reshape(L))
        yv_list.append(y_vol_t.reshape(L))
        pv_list.append(p_vol_med.reshape(L))

        if y_dir_t is not None and p_dir is not None:
            yd = y_dir_t.reshape(-1)
            pd = p_dir.reshape(-1)
            L2 = min(L, yd.numel(), pd.numel())
            if L2 > 0:
                yd_list.append(yd[:L2])
                pd_list.append(pd[:L2])

        batches_seen += 1
        if max_batches is not None and max_batches > 0 and batches_seen >= max_batches:
            break

    if not g_list:
        # empty input → return NaNs-ish but finite
        return 1.0, 1.0, 0.25, 10.0, 0

    device = g_list[0].device
    g_all  = torch.cat(g_list).to(device)
    y_all  = torch.cat(yv_list).to(device)
    p_all  = torch.cat(pv_list).to(device)

    # decode realised_vol
    y_dec = safe_decode_vol(y_all.unsqueeze(-1), vol_norm, g_all.unsqueeze(-1)).squeeze(-1)
    p_dec = safe_decode_vol(p_all.unsqueeze(-1), vol_norm, g_all.unsqueeze(-1)).squeeze(-1)
    floor_val = globals().get("EVAL_VOL_FLOOR", 2e-6)
    y_dec = torch.clamp(y_dec, min=floor_val)
    p_dec = torch.clamp(p_dec, min=floor_val)

    # metrics
    eps = 1e-8
    diff = (p_dec - y_dec)
    mae  = diff.abs().mean().item()
    rmse = (diff.pow(2).mean().sqrt().item())

    sigma2_p = torch.clamp(p_dec.abs(), min=eps) ** 2
    sigma2_y = torch.clamp(y_dec.abs(), min=eps) ** 2
    ratio    = sigma2_y / sigma2_p
    qlike    = (ratio - torch.log(ratio) - 1.0).mean().item()

    brier = None
    if yd_list and pd_list:
        yd = torch.cat(yd_list).to(device).float()
        pd = torch.cat(pd_list).to(device)
        try:
            if torch.isfinite(pd).any() and (pd.min() < 0 or pd.max() > 1):
                pd = torch.sigmoid(pd)
        except Exception:
            pd = torch.sigmoid(pd)
        pd = torch.clamp(pd, 0.0, 1.0)
        brier = ((pd - yd) ** 2).mean().item()

    return float(mae), float(rmse), (float(brier) if brier is not None else float("nan")), float(qlike), int(y_dec.numel())

def _resolve_best_model(trainer, fallback):
    """
    Return the model loaded from the *best* checkpoint as determined by the
    minimum `val_mae_overall` recorded by a ModelCheckpoint callback.

    Preference order:
      1) ModelCheckpoint with monitor == 'val_mae_overall' and a best_model_path
      2) Any ModelCheckpoint that has a best_model_path
      3) Newest local .ckpt in LOCAL_CKPT_DIR
      4) GCS 'last.ckpt' or lexicographically latest in CKPT_GCS_PREFIX
      5) Fallback to the in-memory model
    """
    best_path = None
    # ---- Prefer the val_qlike_overall monitor explicitly ----
    try:
        for cb in getattr(trainer, "callbacks", []):
            if isinstance(cb, ModelCheckpoint):
                mon = getattr(cb, "monitor", None)
                bmp = getattr(cb, "best_model_path", "")
                if mon == "val_qlike_overall" and bmp:
                    best_path = bmp
                    break
        # ---- Otherwise any checkpoint with a best path ----
        if not best_path:
            for cb in getattr(trainer, "callbacks", []):
                if isinstance(cb, ModelCheckpoint) and getattr(cb, "best_model_path", ""):
                    best_path = cb.best_model_path
                    break
    except Exception:
        pass

    # ---- Fallbacks: newest local ckpt, then GCS, else None ----
    if not best_path:
        try:
            ckpts = sorted(LOCAL_CKPT_DIR.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if ckpts:
                best_path = str(ckpts[0])
        except Exception:
            pass
    if not best_path:
        try:
            if fs is not None:
                last_uri = f"{CKPT_GCS_PREFIX}/last.ckpt"
                if fs.exists(last_uri):
                    dst = LOCAL_CKPT_DIR / "last.ckpt"
                    with fsspec.open(last_uri, "rb") as f_in, open(dst, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    best_path = str(dst)
                else:
                    entries = fs.glob(f"{CKPT_GCS_PREFIX}/*.ckpt") or []
                    if entries:
                        latest = sorted(entries)[-1]
                        dst = LOCAL_CKPT_DIR / Path(latest).name
                        with fsspec.open(latest, "rb") as f_in, open(dst, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                        best_path = str(dst)
        except Exception as e:
            print(f"[WARN] Could not fetch resume checkpoint from GCS: {e}")

    if best_path:
        try:
            print(f"Best checkpoint (min val_qlike_overall): {best_path}")
            return TemporalFusionTransformer.load_from_checkpoint(best_path)
        except Exception as e:
            print(f"[WARN] load_from_checkpoint failed: {e}")
    return fallback


# Data preparation
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print(
        f"[CONFIG] batch_size={BATCH_SIZE} | encoder={MAX_ENCODER_LENGTH} | epochs={MAX_EPOCHS} | "
        f"perm_importance={'on' if ENABLE_FEATURE_IMPORTANCE else 'off'} | fi_max_batches={FI_MAX_BATCHES} | "
        f"train_max_rows={getattr(ARGS, 'train_max_rows', None)} | val_max_rows={getattr(ARGS, 'val_max_rows', None)} | subset_mode={getattr(ARGS, 'subset_mode', 'per_asset_tail')} | "
        f"warmups={'off' if getattr(ARGS, 'disable_warmups', False) else 'on'}"
    )
    print("▶ Loading data …")
    train_df = add_time_idx(load_split(READ_PATHS[0]))
    val_df   = add_time_idx(load_split(READ_PATHS[1]))
    test_df  = add_time_idx(load_split(READ_PATHS[2]))
    # -------------------------------------------------------------------
    # Compute per‑asset median realised_vol scale (rv_scale) **once** on the TRAIN split
    # and attach it to every split. This is used only as a **fallback** for manual decode
    # if a normaliser decode is not available.
    # -------------------------------------------------------------------
    asset_scales = (
        train_df.groupby("asset", observed=True)["realised_vol"]
                .median()
                .clip(lower=1e-8)                 # guard against zeros
                .rename("rv_scale")
                .reset_index()
    )

    # Attach rv_scale without using the deprecated/invalid `inplace` kwarg
    asset_scale_map = asset_scales.set_index("asset")["rv_scale"]
    for df in (train_df, val_df, test_df):
        # map() preserves the original row order and keeps dtype float64
        df["rv_scale"] = df["asset"].map(asset_scale_map)
        # If an asset appears only in val/test, fall back to overall median
        df["rv_scale"].fillna(asset_scale_map.median(), inplace=True)
    # Global decoded-scale floor for vol (prevents QLIKE blow-ups on near-zero preds)
    try:
        EVAL_VOL_FLOOR = max(2e-6, float(asset_scales["rv_scale"].median() * 0.02))
    except Exception:
        EVAL_VOL_FLOOR = 2e-6
    print(f"[EVAL] Global vol floor (decoded) set to {EVAL_VOL_FLOOR:.6g}")

    # Add calendar features to all splits
    train_df = add_calendar_features(train_df)
    val_df   = add_calendar_features(val_df)
    test_df  = add_calendar_features(test_df)


    # Optional quick-run subsetting for speed
    _mode = getattr(ARGS, "subset_mode", "per_asset_tail")
    if getattr(ARGS, "train_max_rows", None):
        before = len(train_df)
        train_df = subset_time_series(train_df, int(ARGS.train_max_rows), mode=_mode)
        print(f"[SUBSET] TRAIN: {before} -> {len(train_df)} rows using mode='{_mode}'")
    if getattr(ARGS, "val_max_rows", None):
        before = len(val_df)
        val_df = subset_time_series(val_df, int(ARGS.val_max_rows), mode=_mode)
        print(f"[SUBSET] VAL:   {before} -> {len(val_df)} rows using mode='{_mode}'")

    # Ensure VAL/TEST only contain assets present in TRAIN and have enough history
    _val_df_raw = val_df.copy()
    _test_df_raw = test_df.copy()
    min_required = int(MAX_ENCODER_LENGTH + MAX_PRED_LENGTH)
    _val_before_rows, _val_before_assets = len(val_df), val_df["asset"].nunique()
    _test_before_rows, _test_before_assets = len(test_df), test_df["asset"].nunique()
    train_assets = set(train_df["asset"].unique())

    # Keep only assets seen in TRAIN
    val_df = val_df[val_df["asset"].isin(train_assets)]
    test_df = test_df[test_df["asset"].isin(train_assets)]

    # Drop groups with too few timesteps for at least one window
    val_df = val_df.groupby("asset", observed=True).filter(lambda g: len(g) >= min_required)
    test_df = test_df.groupby("asset", observed=True).filter(lambda g: len(g) >= min_required)

    print(
        f"[FILTER] VAL rows {_val_before_rows}→{len(val_df)} | assets {_val_before_assets}→{val_df['asset'].nunique()} (min_len={min_required})"
    )
    print(
        f"[FILTER] TEST rows {_test_before_rows}→{len(test_df)} | assets {_test_before_assets}→{test_df['asset'].nunique()} (min_len={min_required})"
    )

    # If overly strict and we filtered everything out, relax to overlap-only and let PF handle windows
    if len(val_df) == 0:
        val_df = _val_df_raw[_val_df_raw["asset"].isin(train_assets)]
        print(f"[FILTER] VAL relaxed to overlap-only: {len(val_df)} rows")
    if len(test_df) == 0:
        test_df = _test_df_raw[_test_df_raw["asset"].isin(train_assets)]
        print(f"[FILTER] TEST relaxed to overlap-only: {len(test_df)} rows")


    # -----------------------------------------------------------------------
    # Feature definitions
    # -----------------------------------------------------------------------
    static_categoricals = GROUP_ID
    static_reals: List[str] = []

    base_exclude = set(GROUP_ID + [TIME_COL, "time_idx", "rv_scale"] + TARGETS)

    all_numeric = [c for c, dt in train_df.dtypes.items()
                   if (c not in base_exclude) and pd.api.types.is_numeric_dtype(dt)]

    # Features we want to drop from model
    drop_features = {
        "Parkinson_Vol_2","rv48","MA_cross_12_48","mvmd_z_Log_Range_mode3_energy","Parkinson_Vol_1",
        "mvmd_z_Log_Volume_mode3_sig","rv2","MA_diff_6_12","mvmd_z_Log_Range_mode1_amp","z_Log_Volume",
        "mvmd_z_Log_Range_mode3_freq","mvmd_z_Log_Range_mode4_freq","Log_Range","True_Range",
        "mvmd_z_Log_Range_mode1_entropy","mvmd_z_Log_Range_mode2_amp","z_Log_Close","MA_diff_24_96",
        "mvmd_z_Log_Volume_mode5_entropy","mvmd_z_Log_Volume_mode2_amp","mvmd_z_Log_Close_mode5_sig",
        "mvmd_z_Log_Volume_mode4_freq","mvmd_z_Log_Close_mode1_freq","mvmd_z_Log_Volume_mode3_amp",
        "Corr_DOGE_TRX","mvmd_z_Log_Volume_mode2_freq","mvmd_z_Log_Close_mode2_energy",
        "mvmd_z_Log_Volume_mode5_energy","RS_Vol_2","mvmd_z_Log_Range_mode2_entropy","MA_cross_24_96",
        "Log_Return","Corr_DOGE_XRP","mvmd_z_Log_Close_mode3_sig","mvmd_z_Log_Close_mode3_freq",
        "Corr_BTC_XRP","Parkinson_Var","mvmd_z_Log_Volume_mode1_freq","mvmd_z_Log_Close_mode4_freq",
        "mvmd_z_Log_Range_mode4_energy","mvmd_z_Log_Range_mode1_sig","mvmd_z_Log_Close_mode2_entropy",
        "mvmd_z_Log_Close_mode4_energy","Realised_Var","Corr_BTC_DOGE","rv6","Log_Volume","Corr_DOGE_BTC",
        "mvmd_z_Log_Close_mode5_freq","RQ3","RQ2","Corr_TRX_ETH","Corr_DOGE_XRP_filled","Corr_XRP_DOGE",
        "Corr_XRP_DOGE_filled","Corr_ETH_TRX","Corr_ETH_TRX_filled","Corr_XRP_TRX_filled","Corr_ETH_XRP",
        "Corr_ETH_XRP_filled","Corr_XRP_ETH","Corr_TRX_XRP","Corr_TRX_XRP_filled","Corr_XRP_TRX","mvmd_pos",
        "Corr_TRX_ETH_filled","Corr_TRX_DOGE_filled","Corr_XRP_ETH_filled","Corr_DOGE_TRX_filled",
        "Corr_BTC_DOGE_filled","Corr_DOGE_BTC_filled","Corr_BTC_ETH_filled","Corr_ETH_BTC",
        "Corr_TRX_DOGE","Corr_BTC_TRX_filled","Corr_TRX_BTC","Corr_ETH_BTC_filled","Corr_BTC_XRP_filled",
        "Corr_XRP_BTC","Corr_XRP_BTC_filled","Corr_ETH_DOGE_filled","Corr_DOGE_ETH_filled",
        "Corr_TRX_BTC_filled","Corr_ETH_DOGE"
    }

    # Specify future-known and unknown real features
    calendar_cols = ["sin_tod", "cos_tod", "sin_dow", "cos_dow"]
    time_varying_known_reals = calendar_cols + ["Is_Weekend"]

    time_varying_unknown_reals = [
        c for c in all_numeric
        if c not in (calendar_cols + ["Is_Weekend"]) and c not in drop_features
    ]



    # -----------------------------------------------------------------------
    # TimeSeriesDataSets
    # -----------------------------------------------------------------------
    def build_dataset(df: pd.DataFrame, predict: bool) -> TimeSeriesDataSet:
        return TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target=TARGETS,
            group_ids=GROUP_ID,
            max_encoder_length=MAX_ENCODER_LENGTH,
            max_prediction_length=MAX_PRED_LENGTH,
            # ------------------------------------------------------------------
            # Target normalisation
            #   • realised_vol → GroupNormalizer(asinh, per‑asset scaling)
            #   • direction    → identity (classification logits)
            # ------------------------------------------------------------------
            target_normalizer = MultiNormalizer([
                GroupNormalizer(
                    groups=GROUP_ID,
                    center=False,
                    scale_by_group= True, #True
                    transformation="log1p",
                ),
                TorchNormalizer(method="identity", center=False),   # direction
            ]),
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_reals=time_varying_known_reals,   # known at prediction time
            time_varying_unknown_reals=time_varying_unknown_reals # same set, allows learning lagged targets
            ,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
            predict_mode=predict
        )

    print("▶ Building TimeSeriesDataSets …")

    training_dataset = build_dataset(train_df, predict=False)


    # Build validation/test from TRAIN template so group ID mapping and normalizer stats MATCH
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, val_df, predict=False, stop_randomization=True
    )
    test_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, test_df, predict=False, stop_randomization=True
    )
    vol_normalizer = _extract_norm_from_dataset(training_dataset)  # must be from TRAIN
    # make it available to both the model and the metrics callback
    # (optional) quick alignment check
    try:
        train_vocab = training_dataset.get_parameters()["categorical_encoders"]["asset"].classes_
        val_vocab   = validation_dataset.get_parameters()["categorical_encoders"]["asset"].classes_
        print(f"[ALIGN] val uses train encoders: {np.array_equal(train_vocab, val_vocab)}; "
            f"len(train_vocab)={len(train_vocab)} len(val_vocab)={len(val_vocab)}")
    except Exception:
        pass

    batch_size = min(BATCH_SIZE, len(training_dataset))

    # DataLoader performance knobs — prefer multi-worker, fall back to single-process
    default_workers = max(2, (os.cpu_count() or 4) - 1)
    _cli_workers = int(ARGS.num_workers) if getattr(ARGS, "num_workers", None) is not None else default_workers
    prefetch = int(getattr(ARGS, "prefetch_factor", 8))
    pin = torch.cuda.is_available()

    # Use CLI/default worker count; only disable when explicitly set to 0
    worker_cnt = max(0, _cli_workers)
    use_persist = worker_cnt > 0
    # Only pass prefetch_factor when num_workers > 0
    prefetch_kw = ({"prefetch_factor": prefetch} if worker_cnt > 0 else {})

    test_loader = test_dataset.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=worker_cnt,
        pin_memory=pin,
        persistent_workers=use_persist,
        **prefetch_kw,
    )

    train_dataloader = training_dataset.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=worker_cnt,
        persistent_workers=use_persist,
        pin_memory=pin,
        **prefetch_kw,
    )
    val_dataloader = validation_dataset.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=worker_cnt,
        persistent_workers=use_persist,
        pin_memory=False,
        **prefetch_kw,
    )
    # --- Test dataset ---
    test_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        test_df,
        predict=False,
        stop_randomization=True,
    )

    # use the same loader knobs as train/val to avoid None / tensor-bool pitfalls
    test_dataloader = test_dataset.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=worker_cnt,
        persistent_workers=use_persist,
        pin_memory=False,
        **prefetch_kw,
    )
    # ---- derive id→asset-name mapping for callbacks ----
    asset_vocab = (
        training_dataset.get_parameters()["categorical_encoders"]["asset"].classes_
    )
    rev_asset = {i: lbl for i, lbl in enumerate(asset_vocab)}

    vol_normalizer = _extract_norm_from_dataset(training_dataset)


    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    seed_everything(SEED, workers=True)
    # Loss and output_size for multi-target: realised_vol (quantile regression), direction (classification)
    print("▶ Building model …")
    print(f"[LR] learning_rate={LR_OVERRIDE if LR_OVERRIDE is not None else 0.0017978}")
    
    es_cb = EarlyStopping(
    monitor="val_qlike_overall",
    patience=EARLY_STOP_PATIENCE,
    mode="min",
    min_delta = 1e-3
    )



    metrics_cb = PerAssetMetrics(
        id_to_name=rev_asset,
        vol_normalizer= vol_normalizer
    )

    # If you have a custom checkpoint mirroring callback
    mirror_cb = MirrorCheckpoints()




    from pytorch_forecasting.metrics import MultiLoss
    # one-off in your data prep (TRAIN split)
    counts = train_df["direction"].value_counts()
    n_pos = counts.get(1, 1)
    n_neg = counts.get(0, 1)
    pos_weight = float(n_neg / n_pos)

    # then build the loss with:
    DIR_LOSS = LabelSmoothedBCEWithBrier(smoothing=0.1, pos_weight=pos_weight)


    FIXED_VOL_WEIGHT = 1.0
    FIXED_DIR_WEIGHT = 0.1
 

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        hidden_size=96,
        attention_head_size=4,
        dropout=0.13, #0.0833704625250354,
        hidden_continuous_size=24,
        learning_rate=(LR_OVERRIDE if LR_OVERRIDE is not None else 0.0017978), #0.0019 0017978
        optimizer="AdamW",
        optimizer_params={"weight_decay": WEIGHT_DECAY},
        output_size=[7, 1],  # 7 quantiles + 1 logit
        loss=MultiLoss([VOL_LOSS, DIR_LOSS], weights=[FIXED_VOL_WEIGHT, FIXED_DIR_WEIGHT]),
        logging_metrics=[],
        log_interval=50,
        log_val_interval=10,
    )
    vol_normalizer = _extract_norm_from_dataset(training_dataset)
    tft.vol_norm = vol_normalizer          # if your LightningModule is 'tft'
    metrics_cb = PerAssetMetrics(
        id_to_name=rev_asset,
        vol_normalizer=vol_normalizer,
        max_print=10
    )

    lr_cb = LearningRateMonitor(logging_interval="step")


    val_hist_csv = LOCAL_RUN_DIR / f"tft_val_history_e{MAX_EPOCHS}_{RUN_SUFFIX}.csv"
    val_hist_cb  = ValLossHistory(val_hist_csv)





    # ----------------------------
    # Trainer instance
    # ----------------------------

    trainer = Trainer(
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision=PRECISION,
        max_epochs=MAX_EPOCHS,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        num_sanity_val_steps = 0,
        logger=logger,
        callbacks=[TQDMProgressBar(refresh_rate=50), es_cb, metrics_cb, mirror_cb, lr_cb, val_hist_cb] + EXTRA_CALLBACKS,
        check_val_every_n_epoch=int(ARGS.check_val_every_n_epoch),
        log_every_n_steps=int(ARGS.log_every_n_steps),
        enable_progress_bar=True,
    )


    from types import MethodType

    # Completely skip PF's internal figure creation & TensorBoard logging during validation.
    def _no_log_prediction(self, *args, **kwargs):
        # Intentionally do nothing so BaseModel.create_log won't attempt to log a Matplotlib figure.
        return

    tft.log_prediction = MethodType(_no_log_prediction, tft)

    # (Optional, extra safety) If any PF path calls plot_prediction directly, hand back a blank figure
    # so nothing touches your tensors or bf16 -> NumPy conversion.
    def _blank_plot(self, *args, **kwargs):
        import matplotlib
        matplotlib.use("Agg", force=True)  # headless-safe backend
        import matplotlib.pyplot as plt
        fig = plt.figure()
        return fig

    tft.plot_prediction = MethodType(_blank_plot, tft)
    resume_ckpt = get_resume_ckpt_path() if RESUME_ENABLED else None
    if resume_ckpt:
        print(f"↩️  Resuming from checkpoint: {resume_ckpt}")
    else:
        print("▶ Starting a fresh run (no resume)")

       # Train the model
    trainer.fit(tft, train_dataloader, val_dataloader, ckpt_path=resume_ckpt)

    # --- Evaluate using the best checkpoint (min val_mae_overall) ---
    try:
        trainer.validate(ckpt_path="best", dataloaders=val_dataloader)
    except Exception as _e:
        print(f"[WARN] validate(ckpt_path='best') failed: {_e}; validating current weights instead.")
        trainer.validate(dataloaders=val_dataloader)

    try:
        trainer.test(ckpt_path="best", dataloaders=test_dataloader)
    except Exception as _e:
        print(f"[WARN] test(ckpt_path='best') failed: {_e}; testing current weights instead.")
        trainer.test(dataloaders=test_dataloader)

    # ---------- EXPORT PREDICTIONS FROM BEST CHECKPOINT ----------
    try:
        # Find the PerAssetMetrics callback so we can reuse id->name mapping
        metrics_cb = None
        for cb in getattr(trainer, "callbacks", []):
            if isinstance(cb, PerAssetMetrics):
                metrics_cb = cb
                break
        if metrics_cb is None:
            raise RuntimeError("PerAssetMetrics callback not found; cannot build id->name map/normalizer.")

        # VAL parquet (uncalibrated canonical export)
        val_pred_path = LOCAL_OUTPUT_DIR / f"tft_val_predictions_e{MAX_EPOCHS}_{RUN_SUFFIX}.parquet"
        _save_predictions_from_best(
            trainer,
            val_dataloader,
            "val",
            val_pred_path,
            id_to_name=metrics_cb.id_to_name,
        )
        print(f"✓ Wrote validation parquet → {val_pred_path}")
        try:
            upload_file_to_gcs(str(val_pred_path), f"{GCS_OUTPUT_PREFIX}/{val_pred_path.name}")
        except Exception as e:
            print(f"[WARN] Could not upload validation parquet: {e}")

        # TEST parquet (uncalibrated canonical export)
        test_pred_path = LOCAL_OUTPUT_DIR / f"tft_test_predictions_e{MAX_EPOCHS}_{RUN_SUFFIX}.parquet"
        _save_predictions_from_best(
            trainer,
            test_dataloader,
            "test",
            test_pred_path,
            id_to_name=metrics_cb.id_to_name,
        )
        print(f"✓ Wrote test parquet → {test_pred_path}")
        try:
            upload_file_to_gcs(str(test_pred_path), f"{GCS_OUTPUT_PREFIX}/{test_pred_path.name}")
        except Exception as e:
            print(f"[WARN] Could not upload test parquet: {e}")

    except Exception as e:
        print(f"[WARN] Final export failed: {e}")

    # Resolve the best checkpoint for any downstream analysis (e.g., FI)
    model_for_fi = _resolve_best_model(trainer, fallback=tft)






from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
import fsspec
import pandas as pd

print("✓ Training finished — reloading best checkpoint for test inference")

# --- Resolve best checkpoint path and load ---
best_model_path = None

# Try the explicit callback you created
try:
    if "checkpoint_callback" in globals() and getattr(checkpoint_callback, "best_model_path", ""):
        best_model_path = checkpoint_callback.best_model_path
except Exception:
    pass

# Try any ModelCheckpoint callbacks attached to the trainer
if not best_model_path:
    try:
        from lightning.pytorch.callbacks import ModelCheckpoint
        for cb in getattr(trainer, "callbacks", []):
            if isinstance(cb, ModelCheckpoint) and getattr(cb, "best_model_path", ""):
                best_model_path = cb.best_model_path
                break
    except Exception:
        pass

# Fall back to newest local ckpt
if not best_model_path:
    try:
        ckpts = sorted(LOCAL_CKPT_DIR.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if ckpts:
            best_model_path = str(ckpts[0])
    except Exception:
        pass

# Fall back to GCS (optional)
if not best_model_path and fs is not None:
    try:
        last_uri = f"{CKPT_GCS_PREFIX}/last.ckpt"
        if fs.exists(last_uri):
            dst = LOCAL_CKPT_DIR / "last.ckpt"
            with fsspec.open(last_uri, "rb") as f_in, open(dst, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            best_model_path = str(dst)
    except Exception as e:
        print(f"[WARN] Could not fetch checkpoint from GCS: {e}")

# Load the best checkpoint, or fall back to the in-memory model
if best_model_path:
    print(f"Best checkpoint (local or remote): {best_model_path}")
    try:
        best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    except Exception as e:
        print(f"[WARN] Loading best checkpoint failed ({e}); using in-memory model.")
        best_model = globals().get("tft") or globals().get("model")
else:
    print("[WARN] No checkpoint found; using in-memory model.")
    best_model = globals().get("tft") or globals().get("model")

if best_model is None:
    raise RuntimeError("No model available for testing (checkpoint and in-memory both unavailable).")

print(f"Best checkpoint (local or remote): {best_model_path}")

# --- Save VALIDATION predictions from the BEST checkpoint and compute decoded metrics ---
try:
    ckpt = _get_best_ckpt_path(trainer)
    if ckpt is None:
        raise RuntimeError("Could not resolve best checkpoint path.")
    print(f"Best checkpoint (local or remote): {ckpt}")
    print("Generating validation predictions from best checkpoint …")

    # Load best model (PyTorch 2.6 safe load)
    if _safe_globals_ctx is not None and _PFMultiNormalizer is not None:
        with _safe_globals_ctx([_PFMultiNormalizer]):
            try:
                best_model = TemporalFusionTransformer.load_from_checkpoint(ckpt, map_location="cpu")
            except TypeError:
                best_model = TemporalFusionTransformer.load_from_checkpoint(ckpt, map_location="cpu", weights_only=False)
    else:
        try:
            best_model = TemporalFusionTransformer.load_from_checkpoint(ckpt, map_location="cpu")
        except TypeError:
            best_model = TemporalFusionTransformer.load_from_checkpoint(ckpt, map_location="cpu", weights_only=False)

    # Use a single-worker loader for export (prevents collate/hang issues)
    val_export_loader = make_export_loader(val_dataloader)

    # Get the volatility normalizer from the validation dataset
    vol_norm_val = _extract_norm_from_dataset(val_export_loader.dataset)

    # 1) Collect a DataFrame of validation predictions (UNCALIBRATED)
    df_val_preds = _collect_predictions(
        best_model,
        val_export_loader,
        vol_norm=vol_norm_val,          # <-- use alias the function already accepts
        id_to_name=metrics_cb.id_to_name,
        out_path=None,
    )

    # Pick column names provided by the collector
    y_col = "y_vol" if "y_vol" in df_val_preds.columns else "realised_vol"
    p_col = "y_vol_pred" if "y_vol_pred" in df_val_preds.columns else "pred_realised_vol"

    # 2) Save the uncalibrated parquet (canonical val parquet)
    val_pred_path = LOCAL_OUTPUT_DIR / f"tft_val_predictions_e{MAX_EPOCHS}_{RUN_SUFFIX}.parquet"
    df_val_preds.to_parquet(val_pred_path, index=False)
    print(f"✓ Saved validation predictions (best ckpt, Parquet) → {val_pred_path}")
    try:
        upload_file_to_gcs(str(val_pred_path), f"{GCS_OUTPUT_PREFIX}/{val_pred_path.name}")
    except Exception as e:
        print(f"[WARN] Could not upload validation predictions: {e}")

    # --- Write VALIDATION parquet (uncalibrated, single file) ---
    _save_predictions_from_best(
        trainer,
        val_export_loader,
        "val",
        val_pred_path,
        id_to_name=metrics_cb.id_to_name,
    )
    print(f"✓ Wrote validation parquet → {val_pred_path}")
    try:
        upload_file_to_gcs(str(val_pred_path), f"{GCS_OUTPUT_PREFIX}/{val_pred_path.name}")
    except Exception as e:
        print(f"[WARN] Could not upload validation parquet: {e}")
    # 4) Compute decoded metrics from the saved DF (guard against Nones)
    import torch
    _vy = [v for v in df_val_preds[y_col].tolist() if v is not None]
    _vp = [v for v in df_val_preds[p_col].tolist() if v is not None]
    L = min(len(_vy), len(_vp))
    if L > 0:
        _vy_t = torch.tensor(_vy[:L], dtype=torch.float32)
        _vp_t = torch.tensor(_vp[:L], dtype=torch.float32)
        eps = 1e-8
        v_mae  = torch.mean(torch.abs(_vp_t - _vy_t)).item()
        v_mse  = torch.mean((_vp_t - _vy_t) ** 2).item()
        v_rmse = v_mse ** 0.5
        sigma2_p = torch.clamp(torch.abs(_vp_t), min=eps) ** 2
        sigma2_y = torch.clamp(torch.abs(_vy_t), min=eps) ** 2
        ratio = sigma2_y / sigma2_p
        v_qlike = torch.mean(ratio - torch.log(ratio) - 1.0).item()
        print(f"[VAL (best ckpt)] (decoded) MAE={v_mae:.6f} RMSE={v_rmse:.6f} MSE={v_mse:.6f} QLIKE={v_qlike:.6f}")
    else:
        print("[VAL (best ckpt)] WARNING: could not compute decoded metrics (no valid pairs).")

except Exception as e:
    print(f"[WARN] Failed to save validation predictions from best checkpoint: {e}")


    # --- TEST export (best ckpt) ---
try:
    ckpt = _get_best_ckpt_path(trainer)
    if ckpt is None:
        raise RuntimeError("Could not resolve best checkpoint path.")
    print(f"Loading best checkpoint for test: {ckpt}")

    # Load best model (safe)
    if _safe_globals_ctx is not None and _PFMultiNormalizer is not None:
        with _safe_globals_ctx([_PFMultiNormalizer]):
            try:
                best_model = TemporalFusionTransformer.load_from_checkpoint(ckpt, map_location="cpu")
            except TypeError:
                best_model = TemporalFusionTransformer.load_from_checkpoint(ckpt, map_location="cpu", weights_only=False)
    else:
        try:
            best_model = TemporalFusionTransformer.load_from_checkpoint(ckpt, map_location="cpu")
        except TypeError:
            best_model = TemporalFusionTransformer.load_from_checkpoint(ckpt, map_location="cpu", weights_only=False)

    test_export_loader = make_export_loader(test_dataloader)
    vol_norm_test = _extract_norm_from_dataset(test_export_loader.dataset)

    df_test_preds = _collect_predictions(
        best_model,
        test_export_loader,
        vol_norm=vol_norm_test,         # <-- alias
        id_to_name=metrics_cb.id_to_name,
        out_path=None,
    )

    y_col = "y_vol" if "y_vol" in df_test_preds.columns else "realised_vol"
    p_col = "y_vol_pred" if "y_vol_pred" in df_test_preds.columns else "pred_realised_vol"

    test_pred_path = LOCAL_OUTPUT_DIR / f"tft_test_predictions_e{MAX_EPOCHS}_{RUN_SUFFIX}.parquet"
    df_test_preds.to_parquet(test_pred_path, index=False)
    print(f"✓ Saved TEST predictions (Parquet) → {test_pred_path}")
    try:
        upload_file_to_gcs(str(test_pred_path), f"{GCS_OUTPUT_PREFIX}/{test_pred_path.name}")
    except Exception as e:
        print(f"[WARN] Could not upload test predictions: {e}")

    # Also write calibrated/per-asset calibrated copies beside it
    try:
        _save_predictions_from_best(
            trainer,
            test_export_loader,
            "test",
            test_pred_path,
            id_to_name=metrics_cb.id_to_name,
        )
    except Exception as e:
        print(f"[WARN] Auxiliary calibrated TEST exports failed: {e}")

except Exception as e:
    print(f"[WARN] Failed to collect or save test metrics/predictions: {e}")


# Helpers
def _median_from_quantiles(vol_q: torch.Tensor) -> torch.Tensor:
    vol_q = torch.cummax(vol_q, dim=-1).values
    return vol_q[..., 3]

def _safe_prob(t: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        return t
    try:
        if torch.isfinite(t).any() and (t.min() < 0 or t.max() > 1):
            t = torch.sigmoid(t)
    except Exception:
        t = torch.sigmoid(t)
    return torch.clamp(t, 0.0, 1.0)

@torch.no_grad()
def _collect_predictions(model, dataloader, vol_normalizer=None, vol_norm=None, id_to_name=None, out_path=None, cal_scale=None, per_asset_scales=None, **kwargs):
    # Accept either name for the normalizer
    if vol_normalizer is None:
        vol_normalizer = vol_norm or kwargs.get("vol_norm", None)
    if vol_normalizer is None:
        raise RuntimeError("vol_normalizer/vol_norm must be provided to _collect_predictions")

    import pandas as pd
    model.eval()

    all_g, all_yv, all_pv, all_yd, all_pd, all_t = [], [], [], [], [], []

    for batch in dataloader:
        # ---- Normalise batch shape safely ----
        # PF most commonly yields: (x_dict, y_tensor, weight?)  OR  (x_dict, y_tensor)
        # Some collates produce nested tuples/lists; peel down to get x as dict and y as tensor if present.
        if batch is None:
            continue

        x = None
        yb = None

        try:
            # Case 1: (x_dict, y, *rest)
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                cand = batch[0]
                # Sometimes the first element is itself (x_dict, y)
                if isinstance(cand, (list, tuple)) and len(cand) >= 1 and isinstance(cand[0], dict):
                    x = cand[0]
                    if len(cand) > 1 and torch.is_tensor(cand[1]):
                        yb = cand[1]
                elif isinstance(cand, dict):
                    x = cand
                    if len(batch) > 1 and torch.is_tensor(batch[1]):
                        yb = batch[1]
            # Case 2: batch IS the dict
            if x is None and isinstance(batch, dict):
                x = batch
        except Exception:
            x = None
            yb = None

        if not isinstance(x, dict):
            # Unable to parse this batch; skip defensively
            continue

        # ---- Find group ids ----
        groups = None
        for k in ("groups", "group_ids", "group_id"):
            try:
                if k in x and x[k] is not None:
                    groups = x[k]
                    break
            except Exception:
                pass
        if groups is None:
            continue

        # groups might be a Tensor or a sequence of Tensors
        try:
            g = groups[0] if isinstance(groups, (list, tuple)) and len(groups) > 0 else groups
            while torch.is_tensor(g) and g.ndim > 1 and g.size(-1) == 1:
                g = g.squeeze(-1)
            if not torch.is_tensor(g):
                continue
        except Exception:
            continue

        # ---- Pull decoder targets (vol, dir) from x or fallback yb ----
        y_vol_t, y_dir_t = None, None
        try:
            dec_t = x.get("decoder_target", None)
        except Exception:
            dec_t = None

        if torch.is_tensor(dec_t):
            y = dec_t
            if y.ndim == 3 and y.size(-1) == 1: y = y[..., 0]
            if y.ndim == 2 and y.size(1) >= 1:
                y_vol_t = y[:, 0]
                if y.size(1) > 1: y_dir_t = y[:, 1]
        elif isinstance(dec_t, (list, tuple)) and len(dec_t) >= 1:
            y_vol_t = dec_t[0]
            if torch.is_tensor(y_vol_t):
                if y_vol_t.ndim == 3 and y_vol_t.size(-1) == 1: y_vol_t = y_vol_t[..., 0]
                if y_vol_t.ndim == 2 and y_vol_t.size(-1) == 1: y_vol_t = y_vol_t[:, 0]
            if len(dec_t) > 1 and torch.is_tensor(dec_t[1]):
                y_dir_t = dec_t[1]
                if y_dir_t.ndim == 3 and y_dir_t.size(-1) == 1: y_dir_t = y_dir_t[..., 0]
                if y_dir_t.ndim == 2 and y_dir_t.size(-1) == 1: y_dir_t = y_dir_t[:, 0]

        # Fallback to yb (batch[1]) if needed
        if (y_vol_t is None or y_dir_t is None) and torch.is_tensor(yb):
            yy = yb
            if yy.ndim == 3 and yy.size(1) == 1: yy = yy[:, 0, :]
            if yy.ndim == 2 and yy.size(1) >= 1:
                if y_vol_t is None:
                    y_vol_t = yy[:, 0]
                if y_dir_t is None and yy.size(1) > 1:
                    y_dir_t = yy[:, 1]

        if y_vol_t is None:
            # No realised_vol target → skip this batch
            continue

        # ---- Forward pass and head extraction ----
        try:
            y_hat = model(x)
        except Exception:
            continue
        pred = getattr(y_hat, "prediction", y_hat)
        if isinstance(pred, dict) and "prediction" in pred:
            pred = pred["prediction"]

        p_vol, p_dir = _extract_heads(pred)
        if p_vol is None:
            continue

        # ---- Accumulate on CPU ----
        try:
            L = g.shape[0]
            all_g.append(g.reshape(L).detach().cpu())
            all_yv.append(y_vol_t.reshape(L).detach().cpu())
            all_pv.append(p_vol.reshape(L).detach().cpu())
            if y_dir_t is not None and p_dir is not None:
                y_flat = y_dir_t.reshape(-1)
                p_flat = p_dir.reshape(-1)
                L2 = min(L, y_flat.numel(), p_flat.numel())
                if L2 > 0:
                    all_yd.append(y_flat[:L2].detach().cpu())
                    all_pd.append(p_flat[:L2].detach().cpu())
        except Exception:
            # If shapes are inconsistent, skip this batch
            continue

        # Optional time index
        try:
            dec_time = x.get("decoder_time_idx", None) or x.get("decoder_relative_idx", None)
            if torch.is_tensor(dec_time):
                tvec = dec_time
                while tvec.ndim > 1 and tvec.size(-1) == 1:
                    tvec = tvec.squeeze(-1)
                all_t.append(tvec.reshape(-1)[:L].detach().cpu())
        except Exception:
            pass

    if not all_g:
        raise RuntimeError("No predictions collected — dataloader yielded no usable batches.")

    g_cpu  = torch.cat(all_g)
    yv_cpu = torch.cat(all_yv)
    pv_cpu = torch.cat(all_pv)

    # ---- Decode to physical scale (robust) ----
    yv_dec = safe_decode_vol(yv_cpu.unsqueeze(-1), vol_normalizer, g_cpu.unsqueeze(-1)).squeeze(-1)
    pv_dec = safe_decode_vol(pv_cpu.unsqueeze(-1), vol_normalizer, g_cpu.unsqueeze(-1)).squeeze(-1)

    floor_val = float(globals().get("EVAL_VOL_FLOOR", 1e-6))
    yv_dec = torch.clamp(yv_dec, min=floor_val)
    pv_dec = torch.clamp(pv_dec, min=floor_val)

    # Optional global / per-asset calibration (no look-ahead if from val)
    if cal_scale is None:
        cal_scale = kwargs.get("calibration_scale", None)
    if cal_scale is not None:
        try:
            pv_dec = pv_dec * float(cal_scale)
        except Exception:
            pass

    if per_asset_scales is None:
        per_asset_scales = kwargs.get("per_asset_calibration", None)

    id_to_name = id_to_name or {}
    assets = [id_to_name.get(int(i), str(int(i))) for i in g_cpu.numpy().tolist()]

    if isinstance(per_asset_scales, dict) and len(per_asset_scales) > 0:
        try:
            _map = {str(k): float(v) for k, v in per_asset_scales.items()}
            scales_vec = torch.tensor([_map.get(str(a), 1.0) for a in assets], dtype=pv_dec.dtype)
            pv_dec = pv_dec * scales_vec
        except Exception as _e:
            print(f"[WARN] per-asset calibration failed, skipping: {_e}")

    t_cpu = torch.cat(all_t) if all_t else None

    df = pd.DataFrame({
        "asset": assets,
        "time_idx": t_cpu.numpy().tolist() if t_cpu is not None else [None] * len(assets),
        "y_vol": yv_dec.numpy().tolist(),
        "y_vol_pred": pv_dec.numpy().tolist(),
    })

    if all_yd and all_pd:
        yd_all = torch.cat(all_yd)
        pd_all = torch.cat(all_pd)
        try:
            if torch.isfinite(pd_all).any() and (pd_all.min() < 0 or pd_all.max() > 1):
                pd_all = torch.sigmoid(pd_all)
        except Exception:
            pd_all = torch.sigmoid(pd_all)
        pd_all = torch.clamp(pd_all, 0.0, 1.0)
        Lm = min(len(df), yd_all.numel(), pd_all.numel())
        df = df.iloc[:Lm].copy()
        df["y_dir"] = yd_all[:Lm].numpy().tolist()
        df["y_dir_prob"] = pd_all[:Lm].numpy().tolist()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"✓ Saved {len(df)} predictions → {out_path}")
        return out_path
    return df

# ==== TEST predictions + metrics on decoded scale ====
try:
    df_test_preds = _collect_predictions(
        best_model, test_dataloader,
        id_to_name=metrics_cb.id_to_name,
        vol_normalizer=metrics_cb.vol_norm   # pass as vol_normalizer (collector accepts alias too)
    )

    # Decoded regression metrics
    _y = torch.tensor(df_test_preds["y_vol"].tolist(), dtype=torch.float32)
    _p = torch.tensor(df_test_preds["y_vol_pred"].tolist(), dtype=torch.float32)[: _y.numel()]
    eps = 1e-8
    mae  = torch.mean(torch.abs(_p - _y)).item()
    mse  = torch.mean((_p - _y) ** 2).item()
    rmse = mse ** 0.5
    sigma2_p = torch.clamp(torch.abs(_p), min=eps) ** 2
    sigma2_y = torch.clamp(torch.abs(_y), min=eps) ** 2
    ratio = sigma2_y / sigma2_p
    qlike = torch.mean(ratio - torch.log(ratio) - 1.0).item()

    # Optional direction metrics
    acc_val = brier_val = auroc_val = None
    if "y_dir" in df_test_preds.columns and "y_dir_prob" in df_test_preds.columns:
        yd_t = torch.tensor(df_test_preds["y_dir"].tolist(), dtype=torch.float32)
        pd_t = torch.tensor(df_test_preds["y_dir_prob"].tolist(), dtype=torch.float32)
        # ensure proper probabilities
        if torch.isfinite(pd_t).any() and (pd_t.min() < 0 or pd_t.max() > 1):
            pd_t = torch.sigmoid(pd_t)
        pd_t = torch.clamp(pd_t, 0.0, 1.0)
        acc_val   = float(((pd_t >= 0.5).int() == yd_t.int()).float().mean().item())
        brier_val = float(((pd_t - yd_t) ** 2).mean().item())
        try:
            from torchmetrics.classification import BinaryAUROC
            auroc_val = float(BinaryAUROC()(pd_t, yd_t).item())
        except Exception as e:
            print(f"[WARN] AUROC (test) failed: {e}")

    print(
        f"[TEST] (decoded) MAE={mae:.6f} RMSE={rmse:.6f} MSE={mse:.6f} QLIKE={qlike:.6f}"
        + (f" | ACC={acc_val:.3f}" if acc_val is not None else "")
        + (f" | Brier={brier_val:.4f}" if brier_val is not None else "")
        + (f" | AUROC={auroc_val:.3f}" if auroc_val is not None else "")
    )

    # Save parquet & upload
    pred_path = LOCAL_OUTPUT_DIR / f"tft_test_predictions_e{MAX_EPOCHS}_{RUN_SUFFIX}.parquet"
    df_test_preds.to_parquet(pred_path, index=False)
    print(f"✓ Saved TEST predictions → {pred_path}")

    # Also save “best-ckpt reloaded” exports (uncalibrated + calibrated if available)
    try:
        test_pred_path = LOCAL_OUTPUT_DIR / f"tft_test_predictions_best_e{MAX_EPOCHS}_{RUN_SUFFIX}.parquet"
        _save_predictions_from_best(
            trainer,
            test_dataloader,
            "test",
            test_pred_path,
            id_to_name=metrics_cb.id_to_name,
            vol_norm=metrics_cb.vol_norm
        )
    except Exception as e:
        print(f"[WARN] Failed to save test predictions from best checkpoint: {e}")

except Exception as e:
    print(f"[WARN] Failed to collect or save test metrics/predictions: {e}")