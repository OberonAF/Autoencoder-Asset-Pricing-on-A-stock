"""
Quick diagnostic: is the training factor returns series persistent,
mean-reverting, or white noise? Compares equal-mean vs EMA forecasts.
"""
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Data import read_data, risk_free_rate
from Autoencoder import (
    AutoencoderAssetPricing, prepare_panel_data, train_autoencoder,
    predict_returns, compute_rank_ic, compute_r2_oos,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ---- Load & split ----
print("Loading data...")
_, _, char_data = read_data()
rf = risk_free_rate()
panel_daily = prepare_panel_data(char_data, return_column="return_adj",
                                 freq="D", risk_free_rate=rf)
daily_dates = sorted(panel_daily.keys())
split = int(len(daily_dates) * 0.8)
train_daily = {k: panel_daily[k] for k in daily_dates[:split]}
val_daily   = {k: panel_daily[k] for k in daily_dates[split:]}
print(f"Train {len(train_daily)} periods, Val {len(val_daily)} periods")

input_dim = next(iter(panel_daily.values()))[0].shape[1]

# ---- Train K=4 ----
print("Training K=4...")
model = AutoencoderAssetPricing(
    input_dim=input_dim, hidden_dims=[32, 16], latent_dim=4,
    dropout=0.1, decoder_type="linear", beta_nonneg=True,
)
train_autoencoder(model, train_daily, n_epochs=200, batch_size=256,
                  lr=1e-3, device=DEVICE, val_panel=val_daily, verbose=False)

# ---- Extract factor return time series ----
dates = sorted(model.factor_returns_.keys())
K = model.latent_dim
F = np.stack([model.factor_returns_[t].numpy() for t in dates])  # (T, K)
print(f"\nFactor returns: {F.shape[0]} periods × {K} factors")

# ---- Autocorrelation ----
def acf(x, max_lag=30):
    """Manual ACF for lags 1..max_lag."""
    x = x - x.mean()
    denom = np.sum(x ** 2) + 1e-12
    return [np.sum(x[lag:] * x[:-lag]) / denom for lag in range(1, max_lag + 1)]

fig, axes = plt.subplots(K, 2, figsize=(14, 2.3 * K))
for k in range(K):
    f_k = F[:, k]
    # ACF
    lags = min(30, len(f_k) // 5)
    ac = acf(f_k, lags)
    axes[k, 0].bar(range(1, lags + 1), ac, color="steelblue", alpha=0.8)
    axes[k, 0].axhline(y=0, color="black", linewidth=0.5)
    # 95% CI
    ci = 1.96 / np.sqrt(len(f_k))
    axes[k, 0].axhline(y=ci, color="gray", linestyle="--", linewidth=0.5)
    axes[k, 0].axhline(y=-ci, color="gray", linestyle="--", linewidth=0.5)
    axes[k, 0].set_title(f"Factor {k+1} ACF")
    axes[k, 0].set_xlabel("Lag")
    axes[k, 0].set_ylabel("Autocorrelation")

    # Factor returns + rolling means
    T = len(f_k)
    s = pd.Series(f_k)
    ema22  = s.ewm(span=22).mean().values   # ~1 month
    ema252 = s.ewm(span=252).mean().values  # ~1 year
    eq_mean = np.full(T, f_k.mean())
    axes[k, 1].plot(f_k, alpha=0.25, linewidth=0.4, color="gray")
    axes[k, 1].plot(ema22, linewidth=1.0, label="EMA(22d)")
    axes[k, 1].plot(ema252, linewidth=1.0, label="EMA(252d)")
    axes[k, 1].plot(eq_mean, color="black", linestyle="--", linewidth=0.8, label="equal mean")
    axes[k, 1].legend(fontsize=7)
    axes[k, 1].set_title(f"Factor {k+1} — Rolling Means")

fig.suptitle("Factor Return Persistence Diagnostic", fontsize=14, fontweight="bold")
fig.tight_layout()
plt.savefig("factor_autocorr_check.png", dpi=150)
plt.close()
print("Saved: factor_autocorr_check.png")

# ---- Stats ----
print(f"\n{'Factor':<8} {'Mean':>10} {'Std':>10} {'AC(1)':>8} {'AC(5)':>8} {'AC(22)':>8}")
print("-" * 54)
for k in range(K):
    f = F[:, k]
    ac1 = np.corrcoef(f[1:], f[:-1])[0, 1]
    ac5 = np.corrcoef(f[5:], f[:-5])[0, 1]
    ac22 = np.corrcoef(f[22:], f[:-22])[0, 1]
    print(f"Factor {k+1}  {f.mean():+10.6f} {f.std():10.6f} {ac1:+8.4f} {ac5:+8.4f} {ac22:+8.4f}")

# ---- Forecast method comparison ----
print(f"\n{'Forecast method':<22} {'Val R²':>10} {'Val Rank IC':>12}")
print("-" * 44)

F_mean = F.mean(axis=0)

for label, f_forecast in [
    ("equal mean", F_mean),
    ("EMA(22d) last", pd.DataFrame(F).ewm(span=22).mean().iloc[-1].values),
    ("EMA(120d) last", pd.DataFrame(F).ewm(span=120).mean().iloc[-1].values),
    ("EMA(252d) last", pd.DataFrame(F).ewm(span=252).mean().iloc[-1].values),
]:
    model.factor_forecast_ = torch.tensor(f_forecast, dtype=torch.float32)
    preds = predict_returns(model, val_daily, device=DEVICE)
    r2 = compute_r2_oos(preds)
    ic = compute_rank_ic(preds)
    print(f"  {label:<20}  {r2:+.4f}     {ic:+.4f}")

print("\nDone.")
