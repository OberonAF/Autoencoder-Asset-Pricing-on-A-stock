"""Controlled runner for the latest GitHub autoencoder code.

The upstream main.py assumes each feather frame is stock x date, while the
bundled A-share feather files are date x stock.  This runner keeps the latest
model code intact, fixes the orientation at the runner boundary, and uses a
bounded date/stock subset so the result is reproducible on a desktop machine.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from Autoencoder import (
    AutoencoderAssetPricing,
    compute_portfolio_performance,
    compute_r2_oos,
    predict_returns,
    prepare_panel_data,
    train_autoencoder,
)


DEFAULT_FEATURES = [
    "pb_ratio",
    "pe_ratio",
    "ps_ratio",
    "pcf_ratio",
    "turnover_ratio",
    "market_cap",
    "circulating_market_cap",
    "volume",
    "money",
    "vwap",
    "close",
    "high",
    "low",
    "open",
    "basic_eps",
    "net_profit",
    "operating_revenue",
    "total_assets",
]


def repo_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return "unknown"


def load_feature_frames(base_dir: Path, feature_names: List[str]) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for name in ["return_adj", *feature_names]:
        path = base_dir / f"{name}.feather"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_feather(path)
        frame.index = frame.index.astype(str)
        frame.columns = frame.columns.astype(str)
        frame = frame.replace([np.inf, -np.inf], np.nan).astype("float32")
        frames[name] = frame
    return frames


def choose_dates(returns: pd.DataFrame, start: str, end: str, n_dates: int) -> List[str]:
    dates = pd.Index(returns.index.astype(str))
    selected = dates[(dates >= start) & (dates <= end)]
    if len(selected) == 0:
        raise ValueError(f"No dates found between {start} and {end}")
    if n_dates > 0:
        selected = selected[-n_dates:]
    return selected.tolist()


def choose_stocks(frames: Dict[str, pd.DataFrame], dates: List[str], n_stocks: int) -> List[str]:
    completeness = []
    for frame in frames.values():
        completeness.append(frame.loc[dates].notna().sum(axis=0))
    score = pd.concat(completeness, axis=1).min(axis=1)
    selected = score.sort_values(ascending=False)
    if n_stocks > 0:
        selected = selected.head(n_stocks)
    if selected.empty:
        raise ValueError("No eligible stocks after filtering")
    return selected.index.astype(str).tolist()


def subset_panel(
    frames: Dict[str, pd.DataFrame],
    dates: List[str],
    stocks: List[str],
) -> Dict[str, pd.DataFrame]:
    """Return date x stock frames expected by the latest GitHub code."""

    return {
        name: frame.loc[dates, stocks]
        for name, frame in frames.items()
    }


def rank_ic_by_period(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for t, group in predictions.groupby("time"):
        if len(group) < 10:
            continue
        rank_ic = group["predicted"].rank().corr(group["actual"].rank())
        ic = group["predicted"].corr(group["actual"])
        records.append({"time": t, "ic": ic, "rank_ic": rank_ic})
    return pd.DataFrame(records)


def long_short_stats(portfolios: pd.DataFrame) -> dict:
    wide = portfolios.pivot(index="time", columns="decile", values="avg_actual_return")
    if wide.empty:
        return {"long_short_mean": None, "long_short_ir": None}
    low = wide.columns.min()
    high = wide.columns.max()
    ls = wide[high] - wide[low]
    mean = float(ls.mean())
    std = float(ls.std(ddof=1))
    ir = float(np.sqrt(252.0) * mean / std) if std > 0 else None
    return {"long_short_mean": mean, "long_short_ir": ir}


def save_plots(out_dir: Path, history: dict, portfolios: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["loss"], label="loss", linewidth=2)
    ax.plot(history["mse"], label="mse", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Scaled MSE")
    ax.set_title("Controlled Autoencoder Training")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_loss.png", dpi=160)
    plt.close(fig)

    decile_avg = portfolios.groupby("decile")["avg_actual_return"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(decile_avg.index.astype(int), decile_avg.values)
    ax.set_xlabel("Predicted-return decile")
    ax.set_ylabel("Average actual return")
    ax.set_title("Decile Portfolio Realized Returns")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "decile_returns.png", dpi=160)
    plt.close(fig)

    wide = portfolios.pivot(index="time", columns="decile", values="avg_actual_return")
    if not wide.empty:
        ls = wide[wide.columns.max()] - wide[wide.columns.min()]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot((1.0 + ls.fillna(0.0)).cumprod() - 1.0, linewidth=2)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative return")
        ax.set_title("Top-minus-bottom Decile Cumulative Return")
        ax.grid(alpha=0.25)
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()
        fig.savefig(out_dir / "long_short_cumulative.png", dpi=160)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2020-12-31")
    parser.add_argument("--n-dates", type=int, default=80)
    parser.add_argument("--n-stocks", type=int, default=300, help="Use 0 to keep all stocks")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out-dir", default="outputs/latest_controlled")
    parser.add_argument("--features", nargs="*", default=DEFAULT_FEATURES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    root = Path(__file__).parent
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    print(f"Commit: {repo_commit()}", flush=True)
    print(f"Device: {device}", flush=True)
    print("Loading selected feather files...", flush=True)
    frames = load_feature_frames(root / "fire_data", args.features)

    dates = choose_dates(frames["return_adj"], args.start, args.end, args.n_dates)
    stocks = choose_stocks(frames, dates, args.n_stocks)
    print(f"Selected dates: {dates[0]} to {dates[-1]} ({len(dates)} days)", flush=True)
    print(f"Selected stocks: {len(stocks)}", flush=True)
    print(f"Selected features: {len(args.features)}", flush=True)

    panel_input = subset_panel(frames, dates, stocks)
    panel = prepare_panel_data(panel_input, return_column="return_adj", time_periods=dates)
    if not panel:
        raise RuntimeError("No usable panel periods after missing-value filtering")

    input_dim = next(iter(panel.values()))[0].shape[1]
    print(f"Prepared panel periods: {len(panel)}, input dim: {input_dim}", flush=True)

    model = AutoencoderAssetPricing(
        input_dim=input_dim,
        hidden_dims=[64, 32],
        latent_dim=args.latent_dim,
        dropout=0.1,
        decoder_type="linear",
    )
    history = train_autoencoder(
        model,
        panel,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        verbose=True,
    )

    predictions = predict_returns(model, panel, device=device)
    portfolios = compute_portfolio_performance(predictions, n_deciles=10)
    ic = rank_ic_by_period(predictions)
    ls = long_short_stats(portfolios)
    r2 = float(compute_r2_oos(predictions))

    predictions.to_csv(out_dir / "predictions.csv", index=False)
    portfolios.to_csv(out_dir / "decile_portfolios.csv", index=False)
    ic.to_csv(out_dir / "period_ic.csv", index=False)
    save_plots(out_dir, history, portfolios)

    summary = {
        "commit": repo_commit(),
        "device": device,
        "start": dates[0],
        "end": dates[-1],
        "n_dates_requested": args.n_dates,
        "n_dates_used": len(panel),
        "n_stocks_requested": args.n_stocks,
        "n_stocks_used": len(stocks),
        "n_features": len(args.features),
        "features": args.features,
        "latent_dim": args.latent_dim,
        "epochs": args.epochs,
        "r2_full_sample": r2,
        "ic_mean": None if ic.empty else float(ic["ic"].mean()),
        "rank_ic_mean": None if ic.empty else float(ic["rank_ic"].mean()),
        "rank_ic_ir_daily": None
        if ic.empty or ic["rank_ic"].std(ddof=1) == 0
        else float(np.sqrt(252.0) * ic["rank_ic"].mean() / ic["rank_ic"].std(ddof=1)),
        **ls,
        "elapsed_seconds": round(time.time() - started, 3),
        "note": (
            "This is a controlled full-sample reconstruction run using the latest "
            "GitHub model code. The upstream code now expects local date-by-stock "
            "feather matrices, so this runner only subsets dates/stocks before "
            "calling prepare_panel_data. It is not yet a strict out-of-sample "
            "replication of Gu-Kelly-Xiu."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved outputs to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
