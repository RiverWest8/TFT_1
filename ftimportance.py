#!/usr/bin/env python3
"""
Permutation Feature Importance for TFT — standalone.

Run:
    python3 ftimportance.py
"""

import os
import math
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import fsspec
import shutil


# ----------------------------- GCS config -----------------------------
GCS_BUCKET = os.environ.get("GCS_BUCKET", "river-ml-bucket")
GCS_DATA_PREFIX = os.environ.get("GCS_DATA_PREFIX", f"gs://{GCS_BUCKET}/Data/CleanedData")
GCS_OUTPUT_PREFIX = os.environ.get("GCS_OUTPUT_PREFIX", f"gs://{GCS_BUCKET}/Dissertation/TFT")

# ----------------------------- Helpers -----------------------------

def _extract_norm_from_dataset(ds) -> Optional[object]:
    """Return the GroupNormalizer used for the first target, if present."""
    try:
        params = ds.get_parameters()
        tn = params.get("target_normalizer", None)
        if tn is not None:
            return tn.normalizers[0] if hasattr(tn, "normalizers") else tn
    except Exception:
        pass
    return None


@torch.no_grad()
def _safe_decode_vol(y_B: torch.Tensor, norm, g_B: torch.Tensor) -> torch.Tensor:
    """
    Best-effort decode to physical scale. Tries .decode/.inverse_transform with group_ids;
    falls back to expm1 if transformation='log1p'.
    """
    y = y_B
    g = g_B
    # Ensure shapes [B,1] for APIs that expect that
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    if g.ndim == 1:
        g = g.unsqueeze(-1)

    # Try official APIs
    try:
        return norm.decode(y, group_ids=g).squeeze(-1)
    except Exception:
        pass
    try:
        return norm.inverse_transform(y, group_ids=g).squeeze(-1)
    except Exception:
        pass
    try:
        return norm.inverse_transform(y).squeeze(-1)
    except Exception:
        pass

    # Fallback for log1p
    try:
        tfm = getattr(norm, "transformation", None)
        if tfm == "log1p":
            return torch.expm1(y.squeeze(-1))
    except Exception:
        pass

    # Last resort: return unchanged (will look obviously wrong if still encoded)
    return y.squeeze(-1)


@torch.no_grad()
def _point_from_quantiles(vol_q: torch.Tensor) -> torch.Tensor:
    """
    Take the median (0.50) from a quantile vector with K=7 at index 3.
    Enforce non-decreasing quantiles for safety.
    """
    if vol_q.ndim >= 3 and vol_q.size(1) == 1:
        vol_q = vol_q.squeeze(1)  # [B, K]
    if vol_q.ndim == 2 and vol_q.size(-1) >= 4:
        vol_q = torch.cummax(vol_q, dim=-1).values
        return vol_q[:, 3]
    if vol_q.ndim == 1:
        return vol_q
    return vol_q.reshape(vol_q.size(0), -1)[:, 0]


@torch.no_grad()
def _extract_heads_simple(pred) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Extract (p_vol_encoded_median, p_dir_logit_or_prob) from common TFT outputs.
    """
    import torch
    K = 7

    if isinstance(pred, dict) and "prediction" in pred:
        pred = pred["prediction"]

    # list/tuple => [vol, dir?]
    if isinstance(pred, (list, tuple)) and len(pred) > 0:
        vol = pred[0]
        p_vol = None
        if torch.is_tensor(vol):
            v = vol
            if v.ndim == 4 and v.size(1) == 1: v = v.squeeze(1)   # [B, C, D]
            if v.ndim == 3 and v.size(1) == 1: v = v[:, 0, :]     # [B, D]
            if v.ndim == 3 and v.size(-1) == 1: v = v.squeeze(-1)
            if v.ndim == 2 and v.size(-1) >= K:
                p_vol = _point_from_quantiles(v[:, :K])
            elif v.ndim == 2 and v.size(-1) == 1:
                p_vol = v[:, 0]
        p_dir = None
        if len(pred) > 1 and torch.is_tensor(pred[1]):
            d = pred[1]
            if d.ndim == 3 and d.size(1) == 1: d = d[:, 0, :]
            if d.ndim == 2:
                p_dir = d[:, d.size(-1) // 2]
            elif d.ndim == 1:
                p_dir = d
        return p_vol, p_dir

    # tensor cases
    if torch.is_tensor(pred):
        t = pred
        if t.ndim == 4 and t.size(1) == 1:
            t = t.squeeze(1)
        if t.ndim == 3 and t.size(1) == 1:
            t = t[:, 0, :]
        if t.ndim == 3 and t.size(1) == 2 and t.size(-1) >= K:
            vol_q = t[:, 0, :K]
            return _point_from_quantiles(vol_q), None
        if t.ndim == 2:
            D = t.size(-1)
            if D >= K + 1:
                return _point_from_quantiles(t[:, :K]), t[:, -1]
            if D == K:
                return _point_from_quantiles(t), None
            if D == 1:
                return t[:, 0], None
    return None, None


def _permute_series_inplace(df: pd.DataFrame, col: str, block: int = 1, group_col: str = "asset") -> None:
    """
    In-place, block-wise permutation of a column, grouped by group_col.
    """
    import numpy as np
    for _, gdf in df.groupby(group_col):
        idx = gdf.index.to_numpy()
        vals = gdf[col].to_numpy().copy()
        if block and block > 1:
            n = len(vals)
            cut = n // block
            for i in range(cut):
                seg = slice(i * block, (i + 1) * block)
                np.random.shuffle(vals[seg])
        else:
            np.random.shuffle(vals)
        df.loc[idx, col] = vals


@torch.no_grad()
def _evaluate_decoded_metrics(
    model,
    ds: TimeSeriesDataSet,
    batch_size: int,
    max_batches: Optional[int],
    num_workers: int,
    prefetch: int,
    pin_memory: bool,
    vol_norm=None,
):
    """
    Compute decoded MAE, RMSE, and QLIKE on up to `max_batches`.
    Returns: (mae, rmse, brier=nan, qlike, n) where val_loss == qlike.
    """
    if vol_norm is None:
        vol_norm = _extract_norm_from_dataset(ds)
    assert vol_norm is not None, "Could not resolve target normalizer from dataset."

    dl_kwargs = dict(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    # prefetch/persistent only if workers > 0
    if num_workers > 0:
        dl_kwargs.update(dict(persistent_workers=True, prefetch_factor=prefetch))

    dl = ds.to_dataloader(**dl_kwargs)

    # model device
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")

    model.eval()
    y_all, p_all = [], []

    with torch.no_grad():
        for b_idx, batch in enumerate(dl):
            if max_batches is not None and b_idx >= int(max_batches):
                break

            # Unpack
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                x, y = batch, None
            if not isinstance(x, dict):
                continue

            # Groups → [B]
            g = x.get("groups") if "groups" in x else x.get("group_ids")
            if isinstance(g, (list, tuple)):
                g = g[0] if g else None
            if g is None:
                continue
            while torch.is_tensor(g) and g.ndim > 1 and g.size(-1) == 1:
                g = g.squeeze(-1)
            if not torch.is_tensor(g):
                continue

            # Targets (encoded)
            y_vol = None
            if torch.is_tensor(y):
                t = y
                if t.ndim == 3 and t.size(1) == 1:
                    t = t[:, 0, :]
                if t.ndim == 2 and t.size(1) >= 1:
                    y_vol = t[:, 0]
            else:
                dec_t = x.get("decoder_target")
                if torch.is_tensor(dec_t):
                    t = dec_t
                    if t.ndim == 3 and t.size(1) == 1:
                        t = t[:, 0, :]
                    if t.ndim == 2 and t.size(1) >= 1:
                        y_vol = t[:, 0]
            if not torch.is_tensor(y_vol):
                continue

            # Forward
            x_dev = {
                k: (
                    v.to(device, non_blocking=True) if torch.is_tensor(v)
                    else [vv.to(device, non_blocking=True) if torch.is_tensor(vv) else vv for vv in v]
                    if isinstance(v, (list, tuple)) else v
                )
                for k, v in x.items()
            }
            y_hat = model(x_dev)
            p_vol, _ = _extract_heads_simple(getattr(y_hat, "prediction", y_hat))
            if p_vol is None:
                continue

            # Decode safely
            y_dec = _safe_decode_vol(y_vol.to(device), vol_norm, g.to(device))
            p_dec = _safe_decode_vol(p_vol.to(device), vol_norm, g.to(device))

            # Clamp tiny values for QLIKE stability
            y_dec = torch.clamp(y_dec, min=1e-8)
            p_dec = torch.clamp(p_dec, min=1e-8)

            y_all.append(y_dec.detach().cpu())
            p_all.append(p_dec.detach().cpu())

    if not y_all:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0

    y = torch.cat(y_all)
    p = torch.cat(p_all)

    mae = (p - y).abs().mean().item()
    rmse = torch.sqrt(((p - y) ** 2).mean()).item()

    eps = 1e-8
    sigma2_p = torch.clamp(p.abs(), min=eps) ** 2
    sigma2_y = torch.clamp(y.abs(), min=eps) ** 2
    ratio = sigma2_y / sigma2_p
    qlike = (ratio - torch.log(ratio) - 1.0).mean().item()

    # quick sanity warning if scale still looks encoded
    if mae > 0.01:
        print(f"[FI WARN] Decoded MAE looks large ({mae:.6f}). Check head parsing and decoding.")

    return float(mae), float(rmse), float("nan"), float(qlike), int(y.numel())


def run_permutation_importance(
    model,
    template_ds: TimeSeriesDataSet,
    base_df: pd.DataFrame,
    features: List[str],
    block_size: int,
    batch_size: int,
    max_batches: Optional[int],
    num_workers: int,
    prefetch: int,
    pin_memory: bool,
    vol_norm,
    out_csv_path: str,
) -> None:
    """Compute permutation importance on decoded scale using QLIKE as val_loss."""
    ds_base = TimeSeriesDataSet.from_dataset(
        template_ds, base_df, predict=False, stop_randomization=True
    )

    b_mae, b_rmse, _, b_val, n_base = _evaluate_decoded_metrics(
        model, ds_base, batch_size, max_batches, num_workers, prefetch, pin_memory, vol_norm
    )
    print(f"[FI] Dataset size (samples): {len(base_df)} | batch_size={batch_size}")
    print(f"[FI] Baseline val_loss = {b_val:.6f} (MAE={b_mae:.6f}, RMSE={b_rmse:.6f}) | N={n_base}")

    rows = []
    for feat in features:
        if feat not in base_df.columns:
            print(f"[FI] Skipping missing feature: {feat}")
            continue

        df_perm = base_df.copy()
        _permute_series_inplace(df_perm, feat, block=block_size, group_col="asset")
        ds_perm = TimeSeriesDataSet.from_dataset(
            template_ds, df_perm, predict=False, stop_randomization=True
        )

        p_mae, p_rmse, _, p_val, n_p = _evaluate_decoded_metrics(
            model, ds_perm, batch_size, max_batches, num_workers, prefetch, pin_memory, vol_norm
        )
        delta = p_val - b_val
        print(f"[FI] {feat:>20s} | val_p={p_val:.6f} | Δ={delta:+.6f} | (MAE={p_mae:.6f}, RMSE={p_rmse:.6f})")

        rows.append({
            "feature": feat,
            "baseline_val_loss": float(b_val),
            "permuted_val_loss": float(p_val),
            "delta": float(delta),
            "mae": float(p_mae),
            "rmse": float(p_rmse),
            "n": int(n_p),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv_path, index=False)
    print(f"[FI] wrote {out_csv_path}")

    # Optional: upload to GCS if configured
    try:
        if GCS_OUTPUT_PREFIX:
            gcs_uri = f"{GCS_OUTPUT_PREFIX}/{Path(out_csv_path).name}"
            with fsspec.open(gcs_uri, "wb") as f_out, open(out_csv_path, "rb") as f_in:
                shutil.copyfileobj(f_in, f_out)
            print(f"[FI] uploaded to {gcs_uri}")
    except Exception as e:
        print(f"[FI WARN] GCS upload failed: {e}")


def _reconstruct_features_if_needed(
    df_for_schema: pd.DataFrame,
    GROUP_ID: List[str],
    TIME_COL: str,
    TARGETS: List[str],
) -> List[str]:
    """
    Rebuild `time_varying_unknown_reals` if TFT didn't export it.
    Uses dtypes and drops obvious non-feature columns.
    """
    base_exclude = set(GROUP_ID + [TIME_COL, "time_idx", "rv_scale"] + TARGETS)
    all_numeric = [
        c for c, dt in df_for_schema.dtypes.items()
        if (c not in base_exclude) and pd.api.types.is_numeric_dtype(dt)
    ]
    calendar_cols = ["sin_tod", "cos_tod", "sin_dow", "cos_dow", "Is_Weekend"]
    unknowns = [c for c in all_numeric if c not in calendar_cols]
    return unknowns


# ----------------------------- Main -----------------------------

if __name__ == "__main__":
    # Import training artifacts lazily to avoid triggering training code at import time
    from TFT import (
        get_resume_ckpt_path,
        validation_dataset,
        val_df,
        GROUP_ID,
        TIME_COL,
        TARGETS,
    )

    # Try to bring train_df; if not available, fall back to val_df for schema
    try:
        from TFT import train_df  # type: ignore
        df_for_schema = train_df
    except Exception:
        df_for_schema = val_df

    # Prefer the feature list from TFT if it exists
    try:
        from TFT import time_varying_unknown_reals as _TVUR  # type: ignore
        features = list(_TVUR) if _TVUR is not None else _reconstruct_features_if_needed(
            df_for_schema, GROUP_ID, TIME_COL, TARGETS
        )
    except Exception:
        features = _reconstruct_features_if_needed(df_for_schema, GROUP_ID, TIME_COL, TARGETS)

    # Optional: drop calendar features to focus on learned signals (comment out to keep)
    features = [f for f in features if f not in ("sin_tod", "cos_tod", "sin_dow", "cos_dow", "Is_Weekend")]

    # Load checkpoint
    ckpt_path = get_resume_ckpt_path()
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint found for feature importance run.")

    # Load the model (robust to custom losses)
    try:
        best_model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
    except Exception as e:
        print(f"[WARN] load_from_checkpoint failed ({e}); retrying with loss=None")
        best_model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path, loss=None)

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device).eval()

    # Normalizer from validation dataset
    vol_norm = _extract_norm_from_dataset(validation_dataset)
    if vol_norm is None:
        raise RuntimeError("Could not resolve target normalizer from validation_dataset.")

    # Make sure features exist in val_df
    features = [f for f in features if f in val_df.columns]
    if not features:
        raise RuntimeError("No candidate features found in val_df after filtering.")

    # Run FI
    run_permutation_importance(
        model=best_model,
        template_ds=validation_dataset,
        base_df=val_df,
        features=features,
        block_size=1,
        batch_size=256,
        max_batches=40,   # bump up/down as desired
        num_workers=4,
        prefetch=2,
        pin_memory=True,
        vol_norm=vol_norm,
        out_csv_path="tft_perm_importance.csv",
    )