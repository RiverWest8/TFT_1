import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
import numpy as np
import os

"""
source ~/.venvs/tft-mps/bin/activate
python /Users/riverwest-gomila/Desktop/Data/Scripts/extracttft.py
2-Epoch, [1,1.3] punishment direction, 1.5 Underestimation - /tft_predictions_parquet_e2_20250731T103013Z.parquet - Underestimation too large
1-Epoch, [1,1.4], 1.2 underestimation - /tft_predictions_parquet_e1_20250731T120029Z.parquet - Underestimation too large
5-Epoch, Really good volatiltiy lets narrow in on this area underestimate - 1.11484540287644 - tft_predictions_best_parquet_e5_20250804T195439Z.parquet
tft_predictions_best_parquet_e9_20250804T233153Z.parquet" good
/Users/riverwest-gomila//tft_predictions_20trialbest_e20_20250808T224828Z.parquet - Second sweep trial best config - 20epoch

"""
# Paths to your files
path_test = "/Users/riverwest-gomila/Desktop/Data/CleanedData/universal_test.parquet"
path_pred = "/Users/riverwest-gomila/Desktop/CloudResults/tft_val_predictions_e1_20250819T231013Z.parquet"#"/Users/riverwest-gomila/Desktop/CloudResults/tft_predictions_parquet_e1_20250731T120029Z.parquet"

# Load—now that pyarrow/fastparquet is available
df_test = pd.read_parquet(path_test)

# Load predictions flexibly (CSV or Parquet)
if path_pred.lower().endswith(".parquet"):
    df_pred = pd.read_parquet(path_pred)
else:
    df_pred = pd.read_csv(path_pred)

# Ensure required columns exist; rename common variants
rename_map = {}
if 'Time' not in df_pred.columns:
    for alt in ['time', 'timestamp', 'date', 'datetime', 'ds']:
        if alt in df_pred.columns:
            rename_map[alt] = 'Time'
            break
if 'asset' not in df_pred.columns:
    for alt in ['Asset', 'ticker', 'symbol', 'secid']:
        if alt in df_pred.columns:
            rename_map[alt] = 'asset'
            break
if rename_map:
    df_pred = df_pred.rename(columns=rename_map)

# Early debug to see what we actually have before coercions
print("df_pred raw columns:", df_pred.columns.tolist())

# If we have time_idx in both frames, use it to recover Time directly from df_test
if 'Time' not in df_pred.columns and 'time_idx' in df_pred.columns and 'time_idx' in df_test.columns:
    map_cols = ['asset', 'time_idx', 'Time']
    df_pred = df_pred.merge(
        df_test[map_cols].drop_duplicates(),
        on=['asset', 'time_idx'],
        how='left'
    )

# If 'Time' still missing, try to recover it from index or datetime-like columns
if 'Time' not in df_pred.columns:
    # 1) If index looks like a datetime or is named like time, use it
    idx_name = getattr(df_pred.index, 'name', None)
    if pd.api.types.is_datetime64_any_dtype(df_pred.index):
        df_pred = df_pred.reset_index().rename(columns={idx_name if idx_name else 'index': 'Time'})
    elif idx_name in ['Time','time','timestamp','date','datetime','ds']:
        df_pred = df_pred.reset_index().rename(columns={idx_name if idx_name else 'index': 'Time'})
    else:
        # 2) Search for a single best datetime-like column
        candidates = []
        for c in df_pred.columns:
            try:
                converted = pd.to_datetime(df_pred[c], errors='coerce')
                non_na_ratio = converted.notna().mean()
                if non_na_ratio > 0.8:
                    candidates.append((non_na_ratio, c))
            except Exception:
                pass
        if candidates:
            candidates.sort(reverse=True)
            best_col = candidates[0][1]
            df_pred['Time'] = pd.to_datetime(df_pred[best_col], errors='coerce')
        else:
            raise KeyError("Could not find a 'Time' column in predictions. Available columns: " + str(df_pred.columns.tolist()))

# If 'asset' still missing, try to guess a sensible column
if 'asset' not in df_pred.columns:
    for c in df_pred.columns:
        lc = c.lower()
        if any(k in lc for k in ['asset','ticker','symbol','secid','ric','isin']):
            df_pred = df_pred.rename(columns={c: 'asset'})
            break
    if 'asset' not in df_pred.columns:
        raise KeyError("Could not find an 'asset' column in predictions. Available columns: " + str(df_pred.columns.tolist()))

# Normalize prediction column names to what the rest of the script expects
# (be robust to different file schemas)
def _pick_pred_columns(df):
    out = {}
    cols = {c.lower(): c for c in df.columns}

    # ---- volatility (point) prediction ----
    if 'y_vol' in df.columns:
        out['vol'] = 'y_vol'
    elif 'y_vol_pred' in df.columns:
        out['vol'] = 'y_vol_pred'
    else:
        # common quantile column spellings – pick median
        for cand in [
            'pred_vol_p50','vol_p50','p50','q50','vol_q50','pred_q50',
            'vol_p_50','pred_p50','vol_p50_mean','median_vol'
        ]:
            if cand in cols:
                out['vol'] = cols[cand]
                break

    # ---- direction probability/logit ----
    if 'pred_direction' in df.columns:
        out['dir'] = 'pred_direction'
    elif 'y_dir_prob' in df.columns:
        out['dir'] = 'y_dir_prob'
    elif 'direction_prob' in cols:
        out['dir'] = cols['direction_prob']
    elif 'direction_logit' in cols or 'dir_logit' in cols:
        # leave marker; we will convert with sigmoid below
        out['dir_logit'] = cols.get('direction_logit', cols.get('dir_logit'))

    return out

picks = _pick_pred_columns(df_pred)
if 'vol' in picks and picks['vol'] != 'pred_realised_vol':
    df_pred = df_pred.rename(columns={picks['vol']: 'pred_realised_vol'})
if 'dir' in picks and picks['dir'] != 'pred_direction':
    df_pred = df_pred.rename(columns={picks['dir']: 'pred_direction'})
if 'dir_logit' in picks and 'pred_direction' not in df_pred.columns:
    # convert logits → probability
    _sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    df_pred['pred_direction'] = _sigmoid(df_pred[picks['dir_logit']].astype(float))

# final sanity check
if 'pred_realised_vol' not in df_pred.columns:
    raise KeyError(
        "Could not determine the column for predicted volatility. Available: "
        + str(list(df_pred.columns))
    )

# Drop y_vol_pred if present before merge to avoid confusion
df_pred = df_pred.drop(columns=["y_vol_pred"], errors="ignore")

# Ensure keys for merge have matching types and unify formatting
df_test['asset'] = df_test['asset'].astype(str).str.upper()
df_pred['asset'] = df_pred['asset'].astype(str).str.upper()

df_test['Time'] = pd.to_datetime(df_test['Time'])
df_pred['Time'] = pd.to_datetime(df_pred['Time'])

# Drop any timezone info and floor to second resolution for exact key matching
df_test['Time'] = df_test['Time'].dt.tz_localize(None).dt.floor('s')
df_pred['Time'] = df_pred['Time'].dt.tz_localize(None).dt.floor('s')

# Debug: inspect df_pred column names and sample merge keys
print("df_pred columns:", df_pred.columns.tolist())
print(df_pred[['Time', 'asset']].drop_duplicates().head())

# Keep and rename the columns we need from the universal test set (preserve time_idx if present)
base_cols = ['Time', 'asset', 'Direction', 'Realised_Vol']
if 'time_idx' in df_test.columns:
    base_cols.append('time_idx')
df_test = df_test[base_cols].rename(
    columns={'Direction': 'direction', 'Realised_Vol': 'realised_vol'}
)

# Merge on Time and asset
df = pd.merge(df_test, df_pred, on=['Time','asset'], how='inner')

# Drop y_vol_pred if present after merge
df = df.drop(columns=["y_vol_pred"], errors="ignore")

# Debug: ensure merge succeeded
if df.empty:
    print("ERROR: Merged DataFrame is empty!")
    print("Unique keys in test:", df_test[['Time','asset']].drop_duplicates().shape)
    print("Unique keys in pred:", df_pred[['Time','asset']].drop_duplicates().shape)
    raise SystemExit("Empty merge; check key alignment")

# Inspect merge result
print("df_test shape:", df_test.shape, "columns:", df_test.columns.tolist())
print("df_pred shape:", df_pred.shape, "columns:", df_pred.columns.tolist())
print("merged df shape:", df.shape)
print(df.head())

# -------------------------------
# Per-asset linear calibration & tail diagnostics
# -------------------------------
df['pred_vol_cal'] = np.nan

for asset in df['asset'].unique():
    dfa = df[df['asset'] == asset]
    a = dfa['realised_vol'].to_numpy().astype(float)
    p = dfa['pred_realised_vol'].to_numpy().astype(float)
    m = np.isfinite(a) & np.isfinite(p)
    if m.sum() < 10:
        continue

    # ordinary least squares: a ≈ b0 + b1 * p
    X = np.vstack([np.ones(m.sum()), p[m]]).T
    try:
        b, *_ = np.linalg.lstsq(X, a[m], rcond=None)
        b0, b1 = float(b[0]), float(b[1])
    except Exception:
        b0, b1 = 0.0, 1.0

    # apply calibration
    df.loc[df['asset'] == asset, 'pred_vol_cal'] = b0 + b1 * dfa['pred_realised_vol']

    # tail diagnostics (top 10% by actual)
    q = np.nanquantile(a[m], 0.90)
    tail = m & (a >= q)
    if tail.sum() > 0:
        tail_mae_raw = float(np.nanmean(np.abs(a[tail] - p[tail])))
        tail_mae_cal = float(np.nanmean(np.abs(a[tail] - (b0 + b1 * p[tail]))))
    else:
        tail_mae_raw = tail_mae_cal = float('nan')

    print(
        f"[CAL] {asset}: b0={b0:.6g} b1={b1:.6g} | "
        f"tail_MAE raw={tail_mae_raw:.6g} cal={tail_mae_cal:.6g} (q90={q:.6g})"
    )



# df = df.sort_values('Time')

# create binary predictions for direction

if 'pred_direction' in df.columns:
    df['pred_direction_binary'] = (df['pred_direction'] > 0.5).astype(int)
else:
    df['pred_direction_binary'] = np.nan

# compute and report log-loss (cross-entropy) for direction predictions
print("Direction log loss per asset:")
for asset in df['asset'].unique():
    df_a = df[df['asset'] == asset]

    # prepare and sort numpy arrays by timestamp
    time_arr = df_a['Time'].to_numpy()
    idx = np.argsort(time_arr)
    time_arr = time_arr[idx]
    actual = df_a['realised_vol'].to_numpy()[idx]
    pred = df_a['pred_realised_vol'].to_numpy()[idx]
    pred_cal = df_a['pred_vol_cal'].to_numpy()[idx] if 'pred_vol_cal' in df_a.columns else None
    y_true = df_a['direction'].to_numpy()[idx]

    # confusion matrix if we have direction probabilities
    have_dir = 'pred_direction' in df_a.columns and df_a['pred_direction'].notna().any()
    if have_dir:
        y_pred = (df_a['pred_direction'].to_numpy()[idx] > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1], normalize='true')
        cm = np.nan_to_num(cm)
    else:
        cm = None

    # create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # top-left: actual vs predicted scatter
    ax = axes[0, 0]
    ax.scatter(actual, pred, alpha=0.7)
    m = max(np.nanmax(actual), np.nanmax(pred))
    ax.plot([0, m], [0, m], 'k--', linewidth=1)
    ax.set_title(f'Realised Volatility: Actual vs Pred for {asset}')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')

    # highlight tail & optional calibrated series
    q90 = np.nanquantile(actual, 0.90)
    tail_mask = actual >= q90
    if pred_cal is not None and np.isfinite(pred_cal).any():
        ax.scatter(actual[tail_mask], pred_cal[tail_mask], alpha=0.6, marker='x')
    ax.scatter(actual[tail_mask], pred[tail_mask], alpha=0.6, marker='o', edgecolor='k')

    # top-right: time-series overlay
    ax = axes[0, 1]
    ax.plot(time_arr, actual, label='Actual')
    ax.plot(time_arr, pred, label='Predicted')
    if pred_cal is not None and np.isfinite(pred_cal).any():
        ax.plot(time_arr, pred_cal, label='Pred Calibrated', linestyle=':')
    ax.set_title(f'Realised Volatility Over Time for {asset}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Volatility')
    ax.legend()

    # bottom-left: zoomed tail region
    ax = axes[1, 0]
    ax.scatter(actual[tail_mask], pred[tail_mask], alpha=0.7)
    if pred_cal is not None and np.isfinite(pred_cal).any():
        ax.scatter(actual[tail_mask], pred_cal[tail_mask], alpha=0.7, marker='x')
    if tail_mask.any():
        m_tail = max(np.nanmax(actual[tail_mask]), np.nanmax(pred[tail_mask]))
        ax.plot([0, m_tail], [0, m_tail], 'k--', linewidth=1)
    ax.set_title(f'Tail (≥90th pct) Vol: {asset}')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')

    # bottom-right: confusion matrix (optional)
    ax = axes[1, 1]
    if cm is not None:
        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
        fig.colorbar(im, ax=ax)
        ax.set_xticks([0,1]); ax.set_xticklabels(['neg','pos'])
        ax.set_yticks([0,1]); ax.set_yticklabels(['neg','pos'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i,j]:.2f}', ha='center', va='center', color='white' if cm[i,j] > 0.5 else 'black')
        ax.set_title(f'Direction Confusion for {asset}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    else:
        ax.axis('off')

    # finalize layout and show
    fig.tight_layout()
    plt.show()

df_pred = df_pred[df_pred["time_idx"].isin(df_test["time_idx"])]
