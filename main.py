#%%
import torch
from pathlib import Path
from Data import read_data, risk_free_rate
from Autoencoder import (
    AutoencoderAssetPricing,
    prepare_panel_data, train_autoencoder, predict_returns,
    extract_betas, compute_r2_oos, rolling_window_predict,
)
from Plot import (
    plot_beta_heatmap, plot_training_history, plot_decile_portfolio_returns,
    plot_factor_returns, plot_r2_by_factor_count,
    plot_predictive_r2_comparison
)



# ---- Configuration ----
FIG_DIR = Path(__file__).parent / "figures"
DIR1 = FIG_DIR / "task1_baseline"
DIR2 = FIG_DIR / "task2_k_comparison"
DIR3 = FIG_DIR / "task3_rolling_window"
for d in [DIR1, DIR2, DIR3]:
    d.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ---- Load & prepare data ----
print("Loading data...")
stock_code, date, char_data = read_data()
rf = risk_free_rate()

print("Preparing daily panel...")
panel_daily = prepare_panel_data(char_data, return_column="return_adj", freq="D", risk_free_rate=rf)
print(f"Prepared {len(panel_daily)} daily periods.")

print("Preparing monthly panel...")
panel_monthly = prepare_panel_data(char_data, return_column="return_adj", freq="M", risk_free_rate=rf)
print(f"Prepared {len(panel_monthly)} monthly periods.")

sample_X = next(iter(panel_daily.values()))[0]
input_dim = sample_X.shape[1]
hidden_dims = [32, 16]
print(f"Input dimension (P): {input_dim}")

#%%
# ================================================================
# 1. Train a single model & generate all standard plots
# ================================================================
print("\n" + "=" * 60)
print("Training baseline model (K=4) — daily data")
print("=" * 60)

model_k4 = AutoencoderAssetPricing(
    input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=4,
    dropout=0.1, decoder_type="linear", beta_nonneg=True
)
history = train_autoencoder(model_k4, panel_daily, n_epochs=200, batch_size=256,
                            lr=1e-3, device=DEVICE)

preds = predict_returns(model_k4, panel_daily, device=DEVICE)
betas_df = extract_betas(model_k4, panel_daily, device=DEVICE)
r2_full = compute_r2_oos(preds)
print(f"\nFull-sample R²: {r2_full:.4f}")

# Equal-weighted market benchmark (all stocks avg return per period)
market_bench = preds.groupby("time")["actual"].mean()

# ---- Figure 1: Training loss curve ----
print("Plotting training history...")
plot_training_history(history, title="Training Loss — Autoencoder (K=4)",
                        save_path=str(DIR1 / "01_training_loss.png"))

# ---- Figure 2: Decile portfolio bar + cumulative returns ----
print("Plotting decile portfolio performance...")
plot_decile_portfolio_returns(preds, n_deciles=10,
                                title="Decile Portfolio — Autoencoder (K=4)",
                                save_path=str(DIR1 / "02_decile_portfolios.png"),
                                benchmark_returns=market_bench)

# ---- Figure 3: Factor returns time series ----
print("Plotting factor returns...")
plot_factor_returns(model_k4, panel_daily,
                    title="Learned Factor Returns — Autoencoder (K=4)",
                    save_path=str(DIR1 / "03_factor_returns.png"))

# ---- Figure 4: Beta heatmap ----
print("Plotting beta heatmap...")
plot_beta_heatmap(betas_df, n_stocks=50,
                    title="Factor Loadings Heatmap — Autoencoder (K=4)",
                    save_path=str(DIR1 / "04_beta_heatmap.png"))

#%%
# ================================================================
# 2. Train across K = 1..6 and compare R² vs factor count
# ================================================================
print("\n" + "=" * 60)
print("Training models for K = 1, 2, 3, 4, 5, 6")
print("=" * 60)

r2_by_k = {}
for K in range(1, 7):
    print(f"\n--- Training model K={K} ---")
    m = AutoencoderAssetPricing(
        input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=K,
        dropout=0.1, decoder_type="linear", beta_nonneg=True
    )
    train_autoencoder(m, panel_daily, n_epochs=150, batch_size=256,
                        lr=1e-3, device=DEVICE, verbose=False)
    p = predict_returns(m, panel_daily, device=DEVICE)
    r2_by_k[K] = compute_r2_oos(p)
    print(f"  K={K}: Predictive R² = {r2_by_k[K]:.4f}")

# ---- Figure 6: R² by factor count ----
print("\nPlotting R² by factor count...")
plot_r2_by_factor_count(r2_by_k,
                        title="Predictive R² vs Number of Factors",
                        save_path=str(DIR2 / "06_r2_by_k.png"))

# ---- Figure 7: R² comparison bar chart ----
print("Plotting R² comparison...")
plot_predictive_r2_comparison(
    r2_results={"Autoencoder": r2_by_k},
    title="Predictive R² — Autoencoder Across Factor Counts",
    save_path=str(DIR2 / "07_r2_comparison.png"),
)

#%%
# ================================================================
# 3. Rolling window (expanding) out-of-sample prediction
# ================================================================
print("\n" + "=" * 60)
print("Rolling window OOS prediction (K=4)")
print("=" * 60)

oos_template = AutoencoderAssetPricing(
    input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=4,
    dropout=0.1, decoder_type="linear", beta_nonneg=True
)
preds_oos, r2_oos = rolling_window_predict(
    oos_template, panel_monthly,
    min_train_size=int(len(panel_monthly) * 0.6),
    n_epochs=150, batch_size=256, lr=1e-3, device=DEVICE,
)

market_bench_oos = preds_oos.groupby("time")["actual"].mean()

print("Plotting rolling window results...")
plot_decile_portfolio_returns(preds_oos, n_deciles=10,
                                title="Decile Portfolio — Rolling Window OOS (K=4)",
                                save_path=str(DIR3 / "08_rolling_decile.png"),
                                benchmark_returns=market_bench_oos)

#%%
# ================================================================
# Summary
# ================================================================
print(f"\nAll figures saved to: {FIG_DIR}")
print(f"Best K = {max(r2_by_k, key=r2_by_k.get)}, R² = {r2_by_k[max(r2_by_k, key=r2_by_k.get)]:.4f}")