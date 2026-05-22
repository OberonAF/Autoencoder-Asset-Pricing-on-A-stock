import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from typing import Dict, Optional
from Autoencoder import AutoencoderAssetPricing

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})


def plot_predictive_r2_comparison(
    r2_results: Dict[str, Dict[int, float]],
    baseline_models: Optional[Dict[str, float]] = None,
    title: str = "Predictive R$^2$ Comparison",
    save_path: Optional[str] = None,
):
    """
    Bar chart comparing predictive R² across model types and factor counts.

    Parameters
    ----------
    r2_results : dict
        Nested dict: model_name → {K: r²_value}.
        Example: {"CA": {1: 0.015, 2: 0.022, ..., 6: 0.030},
                   "A":  {1: 0.012, 2: 0.018, ..., 6: 0.028}}
    baseline_models : dict, optional
        Flat dict of baseline-model R²: {"FF3": 0.008, "FF5": 0.010, "IPCA": 0.020}
    title : str
        Plot title.
    save_path : str, optional
        If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Collect model names and their K values
    model_names = list(r2_results.keys())
    palette = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    x_offset = 0
    all_labels = []
    all_x = []
    bar_width = 0.13

    # Plot baselines first (single bars)
    if baseline_models:
        for name, r2 in baseline_models.items():
            ax.bar(x_offset, r2 * 100, bar_width * 3, color="gray", alpha=0.7,
                   edgecolor="black", linewidth=0.5, zorder=3)
            all_labels.append(name)
            all_x.append(x_offset)
            x_offset += 1.0

        # Separator
        x_offset += 0.3

    # Plot autoencoder models grouped by K
    all_K = sorted(set(k for d in r2_results.values() for k in d))
    for ki, K in enumerate(all_K):
        for mi, mname in enumerate(model_names):
            if K in r2_results[mname]:
                r2 = r2_results[mname][K]
                x = x_offset + ki * (len(model_names) + 0.3) * bar_width * 3 + mi * bar_width
                ax.bar(x, r2 * 100, bar_width, color=palette[mi], alpha=0.85,
                       edgecolor="white", linewidth=0.3, zorder=3)
                if ki == len(all_K) - 1:
                    all_labels.append(f"{mname}")
                    all_x.append(x)
        if ki == 0:
            all_labels.append("")
            all_x.append(x_offset - bar_width)

    # K-group labels
    for ki, K in enumerate(all_K):
        mid_x = x_offset + ki * (len(model_names) + 0.3) * bar_width * 3 + (len(model_names) - 1) * bar_width / 2
        ax.text(mid_x, -0.8, f"K={K}", ha="center", va="top", fontsize=9, fontweight="bold")

    ax.set_ylabel("Predictive R$^2$ (%)")
    ax.set_title(title)
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Legend
    handles = []
    if baseline_models:
        handles.append(plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.7, ec="black", lw=0.5))
    for mi, mname in enumerate(model_names):
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=palette[mi], alpha=0.85, ec="white", lw=0.3))
    labels = (list(baseline_models.keys()) if baseline_models else []) + model_names
    ax.legend(handles, labels, loc="upper left", framealpha=0.9, ncol=min(6, len(labels)))

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.show()


def plot_decile_portfolio_returns(
    predictions: pd.DataFrame,
    n_deciles: int = 10,
    title: str = "Decile Portfolio Performance",
    save_path: Optional[str] = None,
    benchmark_returns: Optional[pd.Series] = None,
    benchmark_label: str = "Market",
):
    """
    Left: bar chart of avg return per decile. Right: cumulative returns.
    If benchmark_returns is provided, it is plotted as a dashed line.
    """
    # Sort each period into deciles
    df = predictions.copy()
    decile_frames = []
    for _, grp in df.groupby("time"):
        if len(grp) < n_deciles:
            continue
        grp = grp.copy()
        grp["decile"] = pd.qcut(grp["predicted"], q=n_deciles, labels=False,
                                duplicates="drop")
        decile_frames.append(grp)
    df_deciles = pd.concat(decile_frames)

    # Average return per decile (pooled)
    decile_avg = df_deciles.groupby("decile")["actual"].mean()

    # Cumulative returns per decile over time
    time_decile = df_deciles.groupby(["time", "decile"])["actual"].mean().unstack()
    time_decile = time_decile.sort_index()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    # --- Left: Bar chart ---
    colors = plt.cm.RdYlGn(np.linspace(0.05, 0.95, n_deciles))
    ax1.bar(range(n_deciles), decile_avg.values * 100, color=colors,
            edgecolor="black", linewidth=0.5, zorder=3)
    ax1.set_xlabel("Predicted Return Decile")
    ax1.set_ylabel("Average Actual Return (%)")
    ax1.set_title("Average Return by Decile")
    ax1.set_xticks(range(n_deciles))
    ax1.set_xticklabels([f"D{d+1}" for d in range(n_deciles)], rotation=45)
    ax1.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    # Label high/low
    ax1.annotate("Low", xy=(0, decile_avg.iloc[0] * 100), xytext=(0, decile_avg.iloc[0] * 100 + 0.02),
                ha="center", fontsize=8, color="red")
    ax1.annotate("High", xy=(n_deciles - 1, decile_avg.iloc[-1] * 100),
                xytext=(n_deciles - 1, decile_avg.iloc[-1] * 100 + 0.02),
                ha="center", fontsize=8, color="green")

    # --- Center: Cumulative returns ---
    cum_ret = (1 + time_decile).cumprod()
    for d in range(n_deciles):
        ax2.plot(cum_ret.index, cum_ret[d].values, color=colors[d],
                 linewidth=1.2, label=f"D{d+1}" if d in [0, n_deciles - 1] else "",
                 alpha=0.7 if d not in [0, n_deciles - 1] else 1.0)
    if benchmark_returns is not None:
        bench_cum = (1 + benchmark_returns.sort_index()).cumprod()
        ax2.plot(bench_cum.index, bench_cum.values, color="black",
                 linewidth=1.5, linestyle="--", label=benchmark_label)
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.set_xlabel("Time Period")
    ax2.set_ylabel("Cumulative Return")
    ax2.set_title("Cumulative Decile Returns")
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(6))

    # --- Right: D10 excess over benchmark (long-only with short-sale constraint) ---
    if benchmark_returns is not None:
        bench_cum = (1 + benchmark_returns.sort_index()).cumprod()
        d10_excess = cum_ret[n_deciles - 1] / bench_cum - 1
        color_excess = "green" if d10_excess.iloc[-1] > 0 else "red"
        ax3.fill_between(range(len(d10_excess)), 0, d10_excess.values,
                         color=color_excess, alpha=0.3)
        ax3.plot(range(len(d10_excess)), d10_excess.values,
                 color="darkgreen" if d10_excess.iloc[-1] > 0 else "darkred", linewidth=1.5)
        ax3.set_title(f"D{n_deciles} vs {benchmark_label}\n(Long-Only Excess)")
    else:
        ax3.text(0.5, 0.5, "Need benchmark data", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=12, color="gray")
    ax3.set_xlabel("Time Period")
    ax3.set_ylabel("Cumulative Excess Return")
    ax3.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax3.grid(alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.show()


def plot_factor_returns(
    model: AutoencoderAssetPricing,
    panel_data: Dict,
    title: str = "Learned Factor Returns",
    save_path: Optional[str] = None,
):
    """Time-series plot of the learned factor returns F_t for each factor."""
    if not hasattr(model, "factor_returns_"):
        raise ValueError("Model has no stored factor_returns_. Train with linear decoder first.")

    time_periods = sorted(panel_data.keys())
    K = model.latent_dim

    # Collect factor returns
    F_matrix = np.zeros((len(time_periods), K))
    for i, t in enumerate(time_periods):
        if t in model.factor_returns_:
            F_matrix[i] = model.factor_returns_[t].numpy()

    # Cumulative factor returns
    F_cum = np.cumsum(F_matrix, axis=0)

    fig, axes = plt.subplots(K, 2, figsize=(14, 2.5 * K))
    if K == 1:
        axes = axes.reshape(1, -1)

    colors = plt.cm.tab10(np.linspace(0, 1, K))
    for k in range(K):
        # Raw factor returns
        ax1 = axes[k, 0]
        ax1.bar(range(len(time_periods)), F_matrix[:, k], color=colors[k], alpha=0.7,
                edgecolor="white", linewidth=0.2)
        ax1.axhline(y=0, color="black", linewidth=0.5)
        ax1.set_ylabel(f"Factor {k+1}")
        ax1.set_title(f"Factor {k+1} — Raw Returns" if k == 0 else f"Factor {k+1}")
        ax1.grid(alpha=0.3)

        # Cumulative
        ax2 = axes[k, 1]
        ax2.plot(range(len(time_periods)), F_cum[:, k], color=colors[k], linewidth=1.5)
        ax2.axhline(y=0, color="black", linewidth=0.5)
        ax2.set_ylabel(f"Factor {k+1}")
        ax2.set_title(f"Factor {k+1} — Cumulative" if k == 0 else f"Factor {k+1}")
        ax2.grid(alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.show()


def plot_beta_heatmap(
    betas_df: pd.DataFrame,
    n_stocks: int = 50,
    title: str = "Factor Loadings Heatmap",
    save_path: Optional[str] = None,
):
    """
    Heatmap of factor betas for a subset of stocks at the last time period.
    """
    # Get last time period's betas
    last_t = betas_df["time"].max()
    sub = betas_df[betas_df["time"] == last_t].head(n_stocks)
    beta_cols = [c for c in sub.columns if c.startswith("beta_")]
    K = len(beta_cols)

    beta_mat = sub[beta_cols].values.T  # K × N

    fig, ax = plt.subplots(figsize=(max(8, n_stocks * 0.25), max(3, K * 0.6)))
    im = ax.imshow(beta_mat, aspect="auto", cmap="RdBu_r", interpolation="nearest",
                   norm=Normalize(vmin=-2, vmax=2))
    ax.set_xlabel("Stock")
    ax.set_ylabel("Factor")
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"Factor {k+1}" for k in range(K)])
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Loading")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.show()


def plot_training_history(
    history: Dict[str, list],
    title: str = "Training History",
    save_path: Optional[str] = None,
):
    """Plot MSE loss curve over epochs."""
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(history["loss"]) + 1)
    ax.plot(epochs, history["loss"], color="steelblue", linewidth=1.5, label="Training Loss")
    ax.plot(epochs, history["mse"], color="darkorange", linewidth=1.0, alpha=0.6, label="MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.show()


def plot_r2_by_factor_count(
    model_results: Dict[int, float],
    title: str = "Predictive R$^2$ by Number of Factors",
    save_path: Optional[str] = None,
):
    """
    Line plot of predictive R² vs number of latent factors K.

    Parameters
    ----------
    model_results : dict
        {K: r²_value}, e.g. {1: 0.012, 2: 0.018, 3: 0.022, 4: 0.025, 5: 0.027, 6: 0.028}
    """
    K_values = sorted(model_results.keys())
    r2_values = [model_results[k] * 100 for k in K_values]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(K_values, r2_values, "o-", color="steelblue", linewidth=2, markersize=8,
            markerfacecolor="white", markeredgewidth=1.5, zorder=3)
    ax.set_xlabel("Number of Factors (K)")
    ax.set_ylabel("Predictive R$^2$ (%)")
    ax.set_title(title)
    ax.set_xticks(K_values)
    ax.grid(alpha=0.3, zorder=0)

    # Annotate each point
    for k, r2 in zip(K_values, r2_values):
        ax.annotate(f"{r2:.3f}%", (k, r2), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color="gray")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.show()