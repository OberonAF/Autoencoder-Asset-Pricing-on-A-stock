#%%
import torch
from pathlib import Path
from Data import read_data, risk_free_rate
from Autoencoder import (
    AutoencoderAssetPricing,
    prepare_panel_data, train_autoencoder, predict_returns,
    extract_betas, compute_r2_oos, compute_rank_ic, rolling_window_predict,
    shuffle_panel_returns
)
from Plot import (
    plot_beta_heatmap, plot_training_history, plot_decile_portfolio_returns,
    plot_factor_returns, plot_r2_by_factor_count,
    plot_predictive_r2_comparison, plot_predicted_vs_actual
)


# ---- Configuration ----
FIG_DIR = Path(__file__).parent / "figures"
DIR1 = FIG_DIR / "task1"
DIR2 = FIG_DIR / "task2"
DIR3 = FIG_DIR / "task3"
DIR4 = FIG_DIR / "task4"
for d in [DIR1, DIR2, DIR3, DIR4]:
    d.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ---- Load & prepare data ----
print("Loading data...")
stock_code, date, char_data = read_data()
rf = risk_free_rate()

print("Preparing daily panel...")
panel_daily = prepare_panel_data(char_data, return_column="return_adj", freq="D", risk_free_rate=rf)
daily_dates = sorted(panel_daily.keys())
print(f"Prepared {len(daily_dates)} daily periods.")

print("Preparing monthly panel...")
panel_monthly = prepare_panel_data(char_data, return_column="return_adj", freq="M", risk_free_rate=rf)
monthly_dates = sorted(panel_monthly.keys())
print(f"Prepared {len(monthly_dates)} monthly periods.")

sample_X = next(iter(panel_daily.values()))[0]
input_dim = sample_X.shape[1]
hidden_dims = [32, 16]
print(f"Input dimension (P): {input_dim}")

# ---- Train/Val/Test splits ----
# Daily: 80% train / 20% validation
daily_split = int(len(daily_dates) * 0.8)
train_daily_dates = daily_dates[:daily_split]
val_daily_dates   = daily_dates[daily_split:]
train_daily = {k: panel_daily[k] for k in train_daily_dates}
val_daily   = {k: panel_daily[k] for k in val_daily_dates}
print(f"Daily split: train {len(train_daily)} periods, val {len(val_daily)} periods")

# Monthly: 50% train / 50% test (rolling window OOS)
monthly_split = int(len(monthly_dates) * 0.5)
train_monthly_dates = monthly_dates[:monthly_split]
test_monthly_dates  = monthly_dates[monthly_split:]
train_monthly = {k: panel_monthly[k] for k in train_monthly_dates}
test_monthly  = {k: panel_monthly[k] for k in test_monthly_dates}
print(f"Monthly split: train {len(train_monthly)} periods, test {len(test_monthly)} periods")

#%%
# ================================================================
# 1. Train on daily data, validate on daily hold-out
# ================================================================
print("\n" + "=" * 60)
print("Task 1: Daily train → Daily validation (K=4)")
print("=" * 60)

model_k4 = AutoencoderAssetPricing(
    input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=4,
    dropout=0.1, decoder_type="linear", beta_nonneg=True
)
history = train_autoencoder(model_k4, train_daily, n_epochs=200, batch_size=256,
                            lr=1e-3, device=DEVICE, val_panel=val_daily)

# In-sample predictions (training periods)
preds_train = predict_returns(model_k4, train_daily, device=DEVICE)
# Out-of-sample predictions (validation periods)
preds_val = predict_returns(model_k4, val_daily, device=DEVICE)

r2_train = compute_r2_oos(preds_train)
r2_val = compute_r2_oos(preds_val)
ic_train = compute_rank_ic(preds_train)
ic_val = compute_rank_ic(preds_val)
print(f"\nTrain R²: {r2_train:.4f}  |  Rank IC: {ic_train:.4f}")
print(f"Val   R²: {r2_val:.4f}  |  Rank IC: {ic_val:.4f}")

betas_df = extract_betas(model_k4, train_daily, device=DEVICE)

# ---- Figure 1: Training loss curve ----
print("Plotting training history...")
plot_training_history(history, title="Training Loss — Autoencoder (K=4)",
                        save_path=str(DIR1 / "01_training_loss.png"))

# ---- Figure 2: Decile portfolio (validation set) ----
print("Plotting decile portfolio performance (val)...")
market_bench_val = preds_val.groupby("time")["actual"].mean()
plot_decile_portfolio_returns(preds_val, n_deciles=10,
                                title="Decile Portfolio — Autoencoder (K=4) — Val",
                                save_path=str(DIR1 / "02_decile_portfolios_val.png"),
                                benchmark_returns=market_bench_val)

# ---- Figure 3: Factor returns time series ----
print("Plotting factor returns...")
plot_factor_returns(model_k4, train_daily,
                    title="Learned Factor Returns — Autoencoder (K=4)",
                    save_path=str(DIR1 / "03_factor_returns.png"))

# ---- Figure 4: Beta heatmap ----
print("Plotting beta heatmap...")
plot_beta_heatmap(betas_df, n_stocks=50,
                    title="Factor Loadings Heatmap — Autoencoder (K=4)",
                    save_path=str(DIR1 / "04_beta_heatmap.png"))

# ---- Figure 5: Predicted vs Actual scatter (validation) ----
print("Plotting predicted vs actual (val)...")
plot_predicted_vs_actual(preds_val,
                         title="Predicted vs Actual Returns — Val (Strict Forecast)",
                         save_path=str(DIR1 / "05_predicted_vs_actual_val.png"))

#%%
# ================================================================
# 2. Return permutation test: shuffle returns within each date
# ================================================================
print("\n" + "=" * 60)
print("Task 2: Within-date return permutation test (K=4)")
print("=" * 60)

print("Shuffling training returns...")
train_daily_shuffled = shuffle_panel_returns(train_daily, seed=42)

print("Training on shuffled data...")
model_perm = AutoencoderAssetPricing(
    input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=6,
    dropout=0.1, decoder_type="linear", beta_nonneg=True
)
history_perm = train_autoencoder(model_perm, train_daily_shuffled, n_epochs=200,
                                 batch_size=256, lr=1e-3, device=DEVICE,
                                 val_panel=val_daily, verbose=False)
preds_perm = predict_returns(model_perm, val_daily, device=DEVICE)
r2_perm = compute_r2_oos(preds_perm)
ic_perm = compute_rank_ic(preds_perm)

print(f"\nReal     Val R²: {r2_val:.4f}  |  Rank IC: {ic_val:.4f}")
print(f"Shuffled Val R²: {r2_perm:.4f}  |  Rank IC: {ic_perm:.4f}")
print(f"→ Rank IC drop = {ic_val - ic_perm:.4f}  "
      f"({'real signal exists' if ic_val > ic_perm + 0.01 else 'WARNING: signal may be spurious'})")

# ---- Figure 5: Permutation test comparison ----
print("Plotting permutation test comparison...")
plot_training_history(history_perm,
                      title="Training History — Shuffled Returns (K=4)",
                      save_path=str(DIR2 / "permutation_training.png"))

#%%
# ================================================================
# 3. K = 2..7 comparison on daily data
# ================================================================
print("\n" + "=" * 60)
print("Task 3: K = 2..7 comparison (daily)")
print("=" * 60)

r2_by_k = {}
for K in range(2, 8):
    print(f"\n--- Training model K={K} ---")
    m = AutoencoderAssetPricing(
        input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=K,
        dropout=0.1, decoder_type="linear", beta_nonneg=True
    )
    train_autoencoder(m, train_daily, n_epochs=150, batch_size=256,
                      lr=1e-3, device=DEVICE, verbose=False)
    p = predict_returns(m, train_daily, device=DEVICE)
    r2_by_k[K] = compute_r2_oos(p)
    print(f"  K={K}: R² = {r2_by_k[K]:.4f}")

# ---- Figure 6: R² by factor count ----
print("\nPlotting R² by factor count...")
plot_r2_by_factor_count(r2_by_k,
                        title="Predictive R² vs Number of Factors",
                        save_path=str(DIR3 / "01_r2_by_k.png"))

# ---- Figure 7: R² comparison bar chart ----
print("Plotting R² comparison...")
plot_predictive_r2_comparison(
    r2_results={"Autoencoder": r2_by_k},
    title="Predictive R² — Autoencoder Across Factor Counts",
    save_path=str(DIR3 / "02_r2_comparison.png"),
)

#%%
# ================================================================
# 4. Rolling window strict OOS on monthly data (60% train → 40% test)
# ================================================================
print("\n" + "=" * 60)
print("Task 4: Rolling window strict OOS on monthly data (K=4)")
print("=" * 60)

oos_template = AutoencoderAssetPricing(
    input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=4,
    dropout=0.1, decoder_type="linear", beta_nonneg=True
)
preds_oos, r2_oos = rolling_window_predict(
    oos_template, panel_monthly,
    min_train_size=monthly_split,   # first 50% as minimum training size
    n_epochs=150, batch_size=256, lr=1e-3, device=DEVICE,
)
print(f"Rolling window OOS R²: {r2_oos:.4f}  |  Rank IC: {compute_rank_ic(preds_oos):.4f}")

market_bench_oos = preds_oos.groupby("time")["actual"].mean()

print("Plotting rolling window results...")
plot_decile_portfolio_returns(preds_oos, n_deciles=10,
                                title="Decile Portfolio — Rolling Window OOS (K=4)",
                                save_path=str(DIR4 / "rolling_decile.png"),
                                benchmark_returns=market_bench_oos)

#%%
# ================================================================
# Summary
# ================================================================
print(f"\nAll figures saved to: {FIG_DIR}")
print(f"Daily: train R²={r2_train:.4f} IC={ic_train:.4f} | val R²={r2_val:.4f} IC={ic_val:.4f}")
print(f"Best K = {max(r2_by_k, key=r2_by_k.get)}, Val R² = {r2_by_k[max(r2_by_k, key=r2_by_k.get)]:.4f}")
print(f"Monthly rolling OOS: R²={r2_oos:.4f} IC={compute_rank_ic(preds_oos):.4f}")
