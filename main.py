import torch
from pathlib import Path
from Autoencoder import AutoencoderAssetPricing
from Autoencoder import prepare_panel_data, train_autoencoder, predict_returns, extract_betas, compute_r2_oos
from Plot import (
    plot_beta_heatmap, plot_training_history, plot_decile_portfolio_returns,
    plot_comprehensive_report, plot_factor_returns, plot_r2_by_factor_count,
    plot_predictive_r2_comparison
)



if __name__ == "__main__":
    from character_data import read_character

    # ---- Configuration ----
    OUTPUT_DIR = Path(__file__).parent / "figures"
    OUTPUT_DIR.mkdir(exist_ok=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # ---- Load & prepare data ----
    print("Loading data...")
    char_data = read_character()
    print(f"Loaded {len(char_data)} features.")

    print("Preparing panel data...")
    panel = prepare_panel_data(char_data, return_column="return_adj")
    print(f"Prepared {len(panel)} time periods.")

    sample_X = next(iter(panel.values()))[0]
    input_dim = sample_X.shape[1]
    print(f"Input dimension (P): {input_dim}")

    # ================================================================
    # 1. Train a single model & generate all standard plots
    # ================================================================
    print("\n" + "=" * 60)
    print("Training baseline model (K=4)")
    print("=" * 60)

    model_k4 = AutoencoderAssetPricing(
        input_dim=input_dim, hidden_dims=[64, 32], latent_dim=4,
        dropout=0.1, decoder_type="linear",
    )
    history = train_autoencoder(model_k4, panel, n_epochs=200, batch_size=256,
                                lr=1e-3, device=DEVICE)

    preds = predict_returns(model_k4, panel, device=DEVICE)
    betas_df = extract_betas(model_k4, panel, device=DEVICE)
    r2_full = compute_r2_oos(preds)
    print(f"\nFull-sample R²: {r2_full:.4f}")

    # ---- Figure 1: Training loss curve ----
    print("Plotting training history...")
    plot_training_history(history, title="Training Loss — Autoencoder (K=4)",
                          save_path=str(OUTPUT_DIR / "01_training_loss.png"))

    # ---- Figure 2: Decile portfolio bar + cumulative returns ----
    print("Plotting decile portfolio performance...")
    plot_decile_portfolio_returns(preds, n_deciles=10,
                                  title="Decile Portfolio — Autoencoder (K=4)",
                                  save_path=str(OUTPUT_DIR / "02_decile_portfolios.png"))

    # ---- Figure 3: Comprehensive 2×2 report ----
    print("Plotting comprehensive report...")
    plot_comprehensive_report(preds, history, model=model_k4, panel_data=panel,
                              title_prefix="Autoencoder Asset Pricing (K=4)",
                              save_dir=str(OUTPUT_DIR))

    # ---- Figure 4: Factor returns time series ----
    print("Plotting factor returns...")
    plot_factor_returns(model_k4, panel,
                        title="Learned Factor Returns — Autoencoder (K=4)",
                        save_path=str(OUTPUT_DIR / "04_factor_returns.png"))

    # ---- Figure 5: Beta heatmap ----
    print("Plotting beta heatmap...")
    plot_beta_heatmap(betas_df, n_stocks=50,
                      title="Factor Loadings Heatmap — Autoencoder (K=4)",
                      save_path=str(OUTPUT_DIR / "05_beta_heatmap.png"))

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
            input_dim=input_dim, hidden_dims=[64, 32], latent_dim=K,
            dropout=0.1, decoder_type="linear",
        )
        train_autoencoder(m, panel, n_epochs=150, batch_size=256,
                         lr=1e-3, device=DEVICE, verbose=False)
        p = predict_returns(m, panel, device=DEVICE)
        r2_by_k[K] = compute_r2_oos(p)
        print(f"  K={K}: Predictive R² = {r2_by_k[K]:.4f}")

    # ---- Figure 6: R² by factor count ----
    print("\nPlotting R² by factor count...")
    plot_r2_by_factor_count(r2_by_k,
                            title="Predictive R² vs Number of Factors",
                            save_path=str(OUTPUT_DIR / "06_r2_by_k.png"))

    # ---- Figure 7: R² comparison bar chart ----
    print("Plotting R² comparison...")
    plot_predictive_r2_comparison(
        r2_results={"Autoencoder": r2_by_k},
        title="Predictive R² — Autoencoder Across Factor Counts",
        save_path=str(OUTPUT_DIR / "07_r2_comparison.png"),
    )

    # ================================================================
    # 3. Summary
    # ================================================================
    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print(f"Best K = {max(r2_by_k, key=r2_by_k.get)}, R² = {r2_by_k[max(r2_by_k, key=r2_by_k.get)]:.4f}")
