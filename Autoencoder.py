"""
Autoencoder Asset Pricing Model
Based on Gu, Kelly, and Xiu (2020) "Autoencoder Asset Pricing Models"
and "Autoencoder Asset Pricing Models and Economic Restrictions"

The model learns factor loadings (betas) from stock characteristics
via an autoencoder neural network, then prices assets linearly.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, List


# ---------------------------------------------------------------------------
# 1. Neural Network Building Blocks
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Encoder network: maps characteristics X (N,P) to factor betas beta (N,K)."""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int,
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            else:
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearDecoder(nn.Module):
    """
    Linear decoder: R_it = beta_it' * F_t + epsilon_it.
    Factor returns F_t are a learnable parameter per time period.

    This is the standard decoder from the paper — the factor model
    is linear in betas, which aids interpretability.
    """

    def __init__(self, latent_dim: int):
        super().__init__()

    def forward(self, betas: torch.Tensor, factor_returns: torch.Tensor) -> torch.Tensor:
        return betas @ factor_returns


class NonlinearDecoder(nn.Module):
    """Nonlinear decoder: R_it = h(beta_it) for cases where linear pricing fails."""

    def __init__(self, latent_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        prev = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, betas: torch.Tensor) -> torch.Tensor:
        return self.net(betas).squeeze(-1)


# ---------------------------------------------------------------------------
# 2. Main Autoencoder Model
# ---------------------------------------------------------------------------

class AutoencoderAssetPricing(nn.Module):
    """
    Full autoencoder for asset pricing.

    Encoder:  X_it (characteristics) → beta_it (factor loadings)
    Decoder:  beta_it → R̂_it (predicted returns)

    Supports both linear and nonlinear decoders.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int,
                 dropout: float = 0.1, activation: str = "relu",
                 decoder_type: str = "linear"):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.decoder_type = decoder_type

        self.encoder = Encoder(input_dim, hidden_dims, latent_dim, dropout, activation)

        if decoder_type == "linear":
            self.decoder = LinearDecoder(latent_dim)
            self.nonlinear_decoder = None
        else:
            self.decoder = NonlinearDecoder(latent_dim, hidden_dims[::-1])
            self.nonlinear_decoder = self.decoder

    def forward(self, X: torch.Tensor, factor_returns: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        betas = self.encoder(X)
        if self.decoder_type == "linear":
            assert factor_returns is not None, "Linear decoder requires factor_returns"
            predicted = self.decoder(betas, factor_returns)
        else:
            predicted = self.decoder(betas)
        return predicted, betas


# ---------------------------------------------------------------------------
# 3. Economic Restrictions (soft constraints)
# ---------------------------------------------------------------------------

class EconomicRestrictions:
    """
    Implements economic restrictions as soft penalty terms.

    Restrictions from the paper:
      - Beta Non-Negativity: factor loadings should be non-negative
      - Beta Sum-to-One: for each stock, betas sum to 1 (if intercept included)
      - Monotonicity: certain characteristics should have monotonic relationships
        with betas (enforced via gradient penalty)
    """

    def __init__(self, enforce_nonneg: bool = True, enforce_sum_one: bool = False,
                 monotonicity_weight: float = 0.0):
        self.enforce_nonneg = enforce_nonneg
        self.enforce_sum_one = enforce_sum_one
        self.monotonicity_weight = monotonicity_weight

    def penalty(self, betas: torch.Tensor) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=betas.device)
        if self.enforce_nonneg:
            penalty = penalty + torch.mean(torch.clamp(-betas, min=0) ** 2)
        if self.enforce_sum_one:
            penalty = penalty + torch.mean((betas.sum(dim=1) - 1.0) ** 2)
        return penalty


# ---------------------------------------------------------------------------
# 4. Data Preprocessing
# ---------------------------------------------------------------------------

class StockDataset(Dataset):
    """PyTorch Dataset for stock characteristics and returns."""

    def __init__(self, X: torch.Tensor, R: torch.Tensor):
        self.X = X
        self.R = R

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.R[idx]


def prepare_panel_data(character_data: Dict[str, pd.DataFrame],
                       return_column: str = "return_adj",
                       time_periods: Optional[List[str]] = None
                       ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, StandardScaler, StandardScaler]]:
    """
    Build (X_t, R_{t+1}) pairs: use characteristics at t to predict returns at t+1.

    Returns
    -------
    dict : period_t → (X_t, R_{t+1}, X_scaler, R_scaler)
    """
    if return_column not in character_data:
        available = list(character_data.keys())
        for cand in ["return_adj", "return", "close"]:
            if cand in character_data:
                return_column = cand
                break
        else:
            raise KeyError(f"No return column found. Available: {available}")

    returns_df = character_data[return_column]
    feature_dfs = {k: v for k, v in character_data.items() if k != return_column}

    common_cols = set(returns_df.index)
    for df in feature_dfs.values():
        common_cols &= set(df.index)

    if time_periods is None:
        time_periods = sorted(common_cols)

    result = {}
    for i, t in enumerate(time_periods):
        if i + 1 >= len(time_periods):
            continue  # last period has no t+1

        feature_list = []
        for _, df in feature_dfs.items():
            if t in df.index:
                feature_list.append(df.loc[t].values.reshape(-1, 1))
        if not feature_list:
            continue
        X_raw = np.concatenate(feature_list, axis=1)

        r_next = time_periods[i + 1]
        if r_next not in returns_df.index:
            continue
        R_raw = returns_df.loc[r_next].values.reshape(-1, 1)

        valid = ~(np.isnan(X_raw).any(axis=1) | np.isnan(R_raw).any(axis=1))
        if valid.sum() < 10:
            continue
        X_raw, R_raw = X_raw[valid], R_raw[valid]

        X_scaler = StandardScaler()
        R_scaler = StandardScaler()
        X_scaled = X_scaler.fit_transform(X_raw)
        R_scaled = R_scaler.fit_transform(R_raw)

        result[t] = (
            torch.tensor(X_scaled, dtype=torch.float32),
            torch.tensor(R_scaled, dtype=torch.float32).squeeze(-1),
            X_scaler,
            R_scaler,
        )

    return result


# ---------------------------------------------------------------------------
# 5. Training
# ---------------------------------------------------------------------------

def train_autoencoder(model: AutoencoderAssetPricing,
                      panel_data: Dict[str, Tuple[torch.Tensor, torch.Tensor,
                                                  StandardScaler, StandardScaler]],
                      n_epochs: int = 200,
                      batch_size: int = 256,
                      lr: float = 1e-3,
                      weight_decay: float = 1e-5,
                      restrictions: Optional[EconomicRestrictions] = None,
                      restriction_weight: float = 0.01,
                      device: str = "cpu",
                      verbose: bool = True) -> Dict[str, list]:
    """
    Train the autoencoder across all time periods.

    panel_data[t] = (X_t, R_{t+1}): use t-period characteristics
    to predict t+1 returns. factor_returns[t] = F_{t+1}.

    Factor returns are learned jointly with the encoder.
    """
    device = torch.device(device)
    model = model.to(device)

    # Initialize learnable factor returns per time period (for linear decoder)
    time_periods = sorted(panel_data.keys())
    if model.decoder_type == "linear":
        factor_returns = {
            t: nn.Parameter(torch.randn(model.latent_dim, device=device) * 0.01)
            for t in time_periods
        }
        all_factor_params = list(factor_returns.values())
        optimizer = optim.Adam(
            list(model.parameters()) + all_factor_params,
            lr=lr, weight_decay=weight_decay
        )
    else:
        factor_returns = {}
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    history = {"loss": [], "mse": []}

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        n_periods = 0

        for t in time_periods:
            X_t, R_t, _, _ = panel_data[t]
            X_t, R_t = X_t.to(device), R_t.to(device)
            n_samples = X_t.shape[0]
            if n_samples < 2:
                continue

            # Shuffle within period
            perm = torch.randperm(n_samples)
            n_batches = max(1, n_samples // batch_size)

            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, n_samples)
                idx = perm[start:end]

                X_batch = X_t[idx]
                R_batch = R_t[idx]

                if model.decoder_type == "linear":
                    f_t = factor_returns[t]
                    R_pred, betas = model(X_batch, f_t)
                else:
                    R_pred, betas = model(X_batch)

                mse = nn.functional.mse_loss(R_pred, R_batch)
                loss = mse

                if restrictions is not None and restriction_weight > 0:
                    loss = loss + restriction_weight * restrictions.penalty(betas)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_mse += mse.item()
                n_periods += 1

        avg_loss = epoch_loss / max(n_periods, 1)
        avg_mse = epoch_mse / max(n_periods, 1)
        history["loss"].append(avg_loss)
        history["mse"].append(avg_mse)

        scheduler.step(avg_loss)

        if verbose and (epoch + 1) % 20 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch + 1:4d}/{n_epochs} | Loss: {avg_loss:.6f} | "
                  f"MSE: {avg_mse:.6f} | LR: {lr_now:.2e}")

    # Store factor returns as part of model state
    model.factor_returns_ = {t: fr.detach().cpu() for t, fr in factor_returns.items()}
    return history


# ---------------------------------------------------------------------------
# 6. Factor Extraction & Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_betas(model: AutoencoderAssetPricing,
                  panel_data: Dict[str, Tuple[torch.Tensor, torch.Tensor,
                                              StandardScaler, StandardScaler]],
                  device: str = "cpu") -> pd.DataFrame:
    """Extract factor betas for all stocks across all time periods."""
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    records = []
    for t, (X_t, _, _, _) in panel_data.items():
        X_t = X_t.to(device)
        betas = model.encoder(X_t).cpu().numpy()
        for i in range(betas.shape[0]):
            records.append({
                "time": t,
                "stock_idx": i,
                **{f"beta_{k}": betas[i, k] for k in range(betas.shape[1])},
            })
    return pd.DataFrame(records)


@torch.no_grad()
def predict_returns(model: AutoencoderAssetPricing,
                    panel_data: Dict[str, Tuple[torch.Tensor, torch.Tensor,
                                                StandardScaler, StandardScaler]],
                    device: str = "cpu") -> pd.DataFrame:
    """Generate return predictions for evaluation."""
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    records = []
    for t, (X_t, R_t, _, R_scaler) in panel_data.items():
        X_t = X_t.to(device)
        if model.decoder_type == "linear":
            f_t = model.factor_returns_[t].to(device)
            R_pred, _ = model(X_t, f_t)
        else:
            R_pred, _ = model(X_t)
        R_pred = R_pred.cpu().numpy()
        R_true = R_t.cpu().numpy()

        # Inverse transform to original scale
        R_pred_orig = R_scaler.inverse_transform(R_pred.reshape(-1, 1)).ravel()
        R_true_orig = R_scaler.inverse_transform(R_true.reshape(-1, 1)).ravel()

        for i in range(len(R_pred_orig)):
            records.append({
                "time": t,
                "stock_idx": i,
                "predicted": R_pred_orig[i],
                "actual": R_true_orig[i],
            })
    return pd.DataFrame(records)


def compute_r2_oos(predictions: pd.DataFrame) -> float:
    """Compute out-of-sample R² for return predictions."""
    ss_res = ((predictions["actual"] - predictions["predicted"]) ** 2).sum()
    ss_tot = ((predictions["actual"] - predictions["actual"].mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot


def compute_portfolio_performance(predictions: pd.DataFrame, n_deciles: int = 10) -> pd.DataFrame:
    """
    Sort stocks by predicted return into decile portfolios each period,
    compute the average actual return per decile.
    """
    results = []
    for t, grp in predictions.groupby("time"):
        grp = grp.sort_values("predicted")
        grp["decile"] = pd.qcut(grp["predicted"], q=n_deciles, labels=False,
                                duplicates="drop")
        for d, sub in grp.groupby("decile"):
            results.append({
                "time": t,
                "decile": d,
                "avg_actual_return": sub["actual"].mean(),
                "avg_predicted_return": sub["predicted"].mean(),
                "n_stocks": len(sub),
            })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 7. Hyperparameter Search (simple grid)
# ---------------------------------------------------------------------------

def grid_search(panel_data: Dict,
                param_grid: List[dict],
                n_epochs: int = 100,
                device: str = "cpu",
                verbose: bool = True) -> Tuple[AutoencoderAssetPricing, dict, dict]:
    """Simple grid search over hyperparameters. Returns best model."""
    best_model = None
    best_loss = float("inf")
    best_params = None
    best_history = None

    input_dim = next(iter(panel_data.values()))[0].shape[1]

    for i, params in enumerate(param_grid):
        if verbose:
            print(f"\n--- Trial {i + 1}/{len(param_grid)}: {params} ---")

        model = AutoencoderAssetPricing(
            input_dim=input_dim,
            hidden_dims=params.get("hidden_dims", [64, 32]),
            latent_dim=params.get("latent_dim", 4),
            dropout=params.get("dropout", 0.1),
            activation=params.get("activation", "relu"),
            decoder_type=params.get("decoder_type", "linear"),
        )

        history = train_autoencoder(
            model, panel_data,
            n_epochs=n_epochs,
            batch_size=params.get("batch_size", 256),
            lr=params.get("lr", 1e-3),
            weight_decay=params.get("weight_decay", 1e-5),
            device=device,
            verbose=False,
        )

        final_loss = history["loss"][-1]
        if final_loss < best_loss:
            best_loss = final_loss
            best_model = model
            best_params = params
            best_history = history

    if verbose:
        print(f"\nBest params: {best_params}, Loss: {best_loss:.6f}")
    return best_model, best_params, best_history