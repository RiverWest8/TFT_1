import os
import torch
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data.encoders import GroupNormalizer
from TFT import val_df, validation_dataset, time_varying_unknown_reals
"""
python3 ftimportance.py
"""

import pytorch_lightning as pl


# --- Helpers ---
def _extract_norm_from_dataset(ds):
    try:
        tn = ds.get_parameters().get("target_normalizer", None)
        if tn is not None:
            return tn.normalizers[0] if hasattr(tn, "normalizers") else tn
    except Exception:
        pass
    return None


def _permute_series_inplace(df, col, block=1, group_col="asset"):
    import numpy as np
    for _, gdf in df.groupby(group_col):
        idx = gdf.index.to_numpy()
        vals = gdf[col].to_numpy().copy()
        if block > 1:
            n = len(vals)
            cut = n // block
            for i in range(cut):
                seg = slice(i * block, (i + 1) * block)
                np.random.shuffle(vals[seg])
        else:
            np.random.shuffle(vals)
        df.loc[idx, col] = vals


def _evaluate_decoded_metrics(model, ds, batch_size, max_batches,
                              num_workers, prefetch, pin_memory, vol_norm=None):
    if vol_norm is None:
        vol_norm = _extract_norm_from_dataset(ds)

    dl = ds.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch,
        pin_memory=pin_memory,
    )

    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")

    y_all, p_all = [], []
    with torch.no_grad():
        for b_idx, (x, y) in enumerate(dl):
            if max_batches and b_idx >= max_batches:
                break
            if not isinstance(x, dict):
                continue

            g = x.get("groups")
            if g is None:
                g = x.get("group_ids")
            if isinstance(g, (list, tuple)):
                g = g[0] if g else None
            if g is None:
                continue

            y_vol = y[..., 0] if torch.is_tensor(y) else None
            if not torch.is_tensor(y_vol):
                continue

            x_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in x.items()}
            pred = model(x_dev)
            pred = getattr(pred, "prediction", pred)

            if torch.is_tensor(pred) and pred.ndim == 2 and pred.size(-1) >= 7:
                p_vol = pred[:, 3]
            elif isinstance(pred, (list, tuple)):
                p_vol = pred[0][:, 3]
            else:
                continue

            # decode
            y_dec = vol_norm.inverse_transform(y_vol.unsqueeze(-1), g.unsqueeze(-1)).squeeze(-1)
            p_dec = vol_norm.inverse_transform(p_vol.unsqueeze(-1), g.unsqueeze(-1)).squeeze(-1)

            y_all.append(y_dec.detach().cpu())
            p_all.append(p_dec.detach().cpu())

    if not y_all:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0

    y, p = torch.cat(y_all), torch.cat(p_all)
    mae = (p - y).abs().mean().item()
    rmse = torch.sqrt(((p - y) ** 2).mean()).item()

    eps = 1e-8
    sigma2_p = torch.clamp(p.abs(), min=eps) ** 2
    sigma2_y = torch.clamp(y.abs(), min=eps) ** 2
    ratio = sigma2_y / sigma2_p
    qlike = (ratio - torch.log(ratio) - 1.0).mean().item()

    return float(mae), float(rmse), float("nan"), float(qlike), int(y.numel())


def run_permutation_importance(model, template_ds, base_df, features,
                               block_size, batch_size, max_batches,
                               num_workers, prefetch, pin_memory,
                               vol_norm, out_csv_path):
    ds_base = TimeSeriesDataSet.from_dataset(template_ds, base_df,
                                             predict=False,
                                             stop_randomization=True)

    b_mae, b_rmse, _, b_val, n_base = _evaluate_decoded_metrics(
        model, ds_base, batch_size, max_batches,
        num_workers, prefetch, pin_memory, vol_norm
    )

    print(f"[FI] Baseline val_loss={b_val:.6f} (MAE={b_mae:.6f}, RMSE={b_rmse:.6f})")

    rows = []
    for feat in features:
        df_perm = base_df.copy()
        _permute_series_inplace(df_perm, feat, block=block_size, group_col="asset")
        ds_perm = TimeSeriesDataSet.from_dataset(template_ds, df_perm,
                                                 predict=False,
                                                 stop_randomization=True)

        p_mae, p_rmse, _, p_val, _ = _evaluate_decoded_metrics(
            model, ds_perm, batch_size, max_batches,
            num_workers, prefetch, pin_memory, vol_norm
        )
        delta = p_val - b_val

        print(f"[FI] {feat}: Δ={delta:+.6f} | perm_val={p_val:.6f}")
        rows.append({
            "feature": feat,
            "delta": delta,
            "baseline": b_val,
            "permuted": p_val,
            "mae": p_mae,
            "rmse": p_rmse
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv_path, index=False)
    print(f"[FI] wrote {out_csv_path}")


# --- Main Entrypoint ---
if __name__ == "__main__":
    from TFT import get_resume_ckpt_path  # import your function

    ckpt_path = get_resume_ckpt_path()
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint found for feature importance run.")

    best_model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)

    # you must import your val_df and validation_dataset objects here from your training script
    from TFT import val_df, validation_dataset, time_varying_unknown_reals

    vol_norm = _extract_norm_from_dataset(validation_dataset)

    run_permutation_importance(
        model=best_model,
        template_ds=validation_dataset,
        base_df=val_df,
        features=time_varying_unknown_reals,
        block_size=1,
        batch_size=256,
        max_batches=40,
        num_workers=4,
        prefetch=2,
        pin_memory=True,
        vol_norm=vol_norm,
        out_csv_path="tft_perm_importance.csv"
    )