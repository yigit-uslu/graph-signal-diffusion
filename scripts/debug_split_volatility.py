"""
Diagnostic: volatility of z-scored DailyLogReturn per dataset split.

Uses the raw values.csv and computes per-stock z-scores with training-period
statistics, then reports the pooled std for each split.

Split boundaries (from compare_baselines log):
  total samples = 2328,  past_window=20,  future_window=5
  train     : sample indices   0 – 1861  → target date range relative to sorted dates
  val       : sample indices 1862 – 2094
  test      : sample indices 2095 – 2327
  train-val : sample indices   0 –  231   (first 232 of train; quick proxy)
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")

PAST_W  = 20
FUTURE_W = 5
ROOT = "data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.6_sector_bonus_0.05/raw"

# ── Load raw data ──────────────────────────────────────────────────────────────
print("Loading values.csv ...")
df = pd.read_csv(f"{ROOT}/values.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "ticker"] if "ticker" in df.columns else ["Date"]).reset_index(drop=True)

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

# Determine the stock column
stock_col = "ticker" if "ticker" in df.columns else df.columns[df.columns.str.lower().str.contains("ticker|symbol|stock")].tolist()[0]
print("Stock column:", stock_col)

# ── Pivot to [dates × stocks] for DailyLogReturn ───────────────────────────────
pivot = df.pivot(index="Date", columns=stock_col, values="DailyLogReturn")
pivot = pivot.sort_index()
dates = pivot.index
T = len(dates)
print(f"Total trading dates: {T}, stocks: {pivot.shape[1]}")

# ── Sample index → target date mapping ────────────────────────────────────────
# Sample i → target dates [i+PAST_W : i+PAST_W+FUTURE_W]
# The sample range [i_start, i_end] covers target dates
#   [i_start+PAST_W : i_end+PAST_W+FUTURE_W]
N_SAMPLES = T - PAST_W - FUTURE_W + 1   # should be ~2328
print(f"Total samples: {N_SAMPLES}")

splits = {
    "train":     (0,    1862),
    "train-val": (0,     232),
    "val":       (1862, 2095),
    "test":      (2095, 2328),
}

# ── Compute per-stock z-score stats from TRAINING period ──────────────────────
train_s, train_e = splits["train"]
# Target date indices for train: [train_s+PAST_W : train_e+PAST_W+FUTURE_W]
train_dates = dates[train_s + PAST_W : train_e + PAST_W + FUTURE_W]
train_vals  = pivot.loc[train_dates].values  # [T_train, N]

train_mean = np.nanmean(train_vals, axis=0)   # [N]
train_std  = np.nanstd(train_vals,  axis=0)   # [N]
train_std  = np.where(train_std < 1e-8, 1.0, train_std)   # guard zeros

# ── Report per-split z-scored std ─────────────────────────────────────────────
print(f"\n{'Split':<12} {'zscored_mean':>14} {'zscored_std':>12} {'n_obs':>12}  raw_std (pre-zscore)")
print("-" * 72)
for name, (s, e) in splits.items():
    d = dates[s + PAST_W : e + PAST_W + FUTURE_W]
    raw = pivot.loc[d].values               # [T_split, N]
    z   = (raw - train_mean) / train_std    # z-scored with TRAIN stats
    z_flat = z[~np.isnan(z)]
    raw_flat = raw[~np.isnan(raw)]
    print(f"{name:<12} {np.mean(z_flat):>14.5f} {np.std(z_flat):>12.5f} {len(z_flat):>12,}  raw_std={np.std(raw_flat):.5f}")

