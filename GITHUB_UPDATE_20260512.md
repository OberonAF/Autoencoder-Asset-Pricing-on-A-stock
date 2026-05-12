# GitHub update sync notes

Date: 2026-05-12

## Sync status

- Repository: https://github.com/OberonAF/Autoencoder-Asset-Pricing-on-A-stock.git
- Previous local HEAD: `2a98c31`
- Latest remote HEAD: `7ab14ea9161905b13bc29eabd302e85f5958a889`
- Local `zcw` HEAD after sync: `7ab14ea`
- New upstream commits:
  - `f1e80d5 change Autoencoder`
  - `7ab14ea add rolling window predict procces`
- Upstream modified files:
  - `Autoencoder.py`
  - `main.py`

Local untracked experiment files were kept:

- `run_latest_controlled.py`
- `RUN_LATEST_RESULTS.md`
- `outputs/`

## What changed upstream

- `prepare_panel_data()` now uses the DataFrame index as the time axis, matching the local feather layout of date x stock.
- Labels are now shifted: characteristics at `t` are paired with returns at `t+1`.
- `AutoencoderAssetPricing` now stores `hidden_dims`, enabling model cloning.
- A new `rolling_window_predict()` function was added.
- `main.py` now adds a rolling-window OOS section after the full-sample and K-search runs.

## Updated controlled run

The local controlled runner was updated to match the new upstream data direction.

Command:

```powershell
D:\anaconda\envs\wcz\python.exe -u run_latest_controlled.py --start 2019-01-01 --end 2020-12-31 --n-dates 0 --n-stocks 500 --epochs 30 --latent-dim 4 --batch-size 128 --out-dir outputs/github_updated_controlled_2019_2020_500s_e30
```

Output directory:

```text
C:\Users\Administrator\Documents\New project 3\zcw\outputs\github_updated_controlled_2019_2020_500s_e30
```

Metrics:

| Metric | Value |
|---|---:|
| Commit | `7ab14ea` |
| Dates used | 486 |
| Stocks requested | 500 |
| Features | 18 |
| Latent factors | 4 |
| Epochs | 30 |
| Full-sample R2 | 0.3220 |
| Mean IC | 0.2470 |
| Mean Rank IC | 0.2236 |
| Long-short mean return | 0.0196 |
| Long-short IR | 28.5666 |

## Local fix applied after sync

After checking the upstream rolling-window implementation, I patched the local
`zcw` copy so rolling OOS prediction no longer fails on unseen test periods.

Changed files:

- `Autoencoder.py`
- `main.py`
- `run_latest_controlled.py`

Implementation:

- `predict_returns()` now accepts an optional `factor_returns` argument.
- `forecast_factor_returns()` was added with three simple forecast modes:
  - `mean`: average learned training-window factor returns
  - `last`: last learned training-window factor return
  - `zero`: zero-factor baseline
- `rolling_window_predict()` now forecasts a test-period factor from the
  training window instead of looking for `model.factor_returns_[test_key]`.
- `main.py` calls `rolling_window_predict(..., factor_forecast="mean")`.
- `run_latest_controlled.py` now accepts `--n-stocks 0` to keep all stocks
  for a later full-universe controlled run.

Smoke test:

```text
mean (3421, 4) 0.266268
last (3421, 4) 0.266411
zero (3421, 4) 0.266261
```

This confirms that the previous `KeyError` is fixed.

## Remaining issues

1. The new rolling-window forecast is intentionally simple.

   The current local fix uses a historical factor-return forecast such as the
   training-window mean.  This avoids using test-period realized returns, but it
   is still a basic baseline.  Stronger options include:

   - learn a factor-return predictor/prior,
   - estimate factors from a contemporaneous train-stock cross-section and apply to held-out stocks,
   - or evaluate only reconstruction/pricing rather than forecast.

2. `main.py` is still computationally unrealistic for the full data.

   It runs:

   - baseline K=4 for 200 epochs,
   - K=1..6 search for 150 epochs each,
   - then rolling-window retraining with 100 epochs per window.

   With hundreds or thousands of periods, this is far too slow for normal use.

3. Full-sample metrics remain leaky.

   The baseline and K-search train date-specific factor returns on all evaluation dates, so the resulting R2/IC should be interpreted as reconstruction/pricing fit, not strict out-of-sample prediction.

4. Per-date return scaling can leak target-date distribution information.

   `prepare_panel_data()` fits `R_scaler` separately on each `R_{t+1}` cross-section, including evaluation dates.  This is fine for some normalized reconstruction diagnostics, but it is not a clean forecast protocol.

5. Default feature usage is too broad.

   `main.py` uses all 52 non-return matrices.  Because some features are sparse, the usable panel shrinks from 2509 raw dates to 1332 prepared periods.  A curated feature list is better for controlled experiments.

## Recommended next fix

The best next step is to add a dedicated full-run script instead of using
`main.py`.  It should expose:

- curated feature list,
- date range,
- stock cap or full universe,
- epochs,
- factor forecast method,
- and whether to run full-sample reconstruction or rolling OOS forecast.

## Full-universe controlled run

After the rolling fix and the `--n-stocks 0` full-universe option, I ran one
complete controlled full-sample test on the latest GitHub code.

Command:

```powershell
D:\anaconda\envs\wcz\python.exe -u run_latest_controlled.py --start 2015-01-05 --end 2025-04-30 --n-dates 0 --n-stocks 0 --epochs 30 --latent-dim 4 --batch-size 8192 --out-dir outputs\github_updated_full_controlled_2015_2025_allstocks_e30
```

Output directory:

```text
C:\Users\Administrator\Documents\New project 3\zcw\outputs\github_updated_full_controlled_2015_2025_allstocks_e30
```

Metrics:

| Metric | Value |
|---|---:|
| Git commit | `7ab14ea` |
| Device | `cuda` |
| Dates used | 2508 |
| Stocks used | 5363 |
| Features | 18 |
| Latent factors | 4 |
| Epochs | 30 |
| Elapsed seconds | 201.984 |
| Full-sample R2 | 0.2962 |
| Mean IC | 0.0914 |
| Mean Rank IC | 0.0930 |
| Rank IC IR | 9.4373 |
| Long-short mean return | 0.0087 |
| Long-short IR | 9.9265 |

Artifacts:

- `predictions.csv` (388,839,554 bytes)
- `period_ic.csv`
- `decile_portfolios.csv`
- `training_loss.png`
- `decile_returns.png`
- `long_short_cumulative.png`
- `summary.json`

Notes:

- The Python run wrote `commit: unknown` in `summary.json`, because the
  subprocess could not resolve Git from that environment.  Running
  `git rev-parse --short HEAD` in the same repository returns `7ab14ea`.
- This remains a full-sample reconstruction/pricing-fit run, not a strict
  out-of-sample replication.

## Full-feature controlled run

I also ran the same controlled full-sample setup with all available candidate
feature matrices.  This excludes only `return_adj` as the label and
`index`/`columns` as auxiliary files.

Command:

```powershell
D:\anaconda\envs\wcz\python.exe -u run_latest_controlled.py --start 2015-01-05 --end 2025-04-30 --n-dates 0 --n-stocks 0 --epochs 30 --latent-dim 4 --batch-size 8192 --out-dir outputs\github_updated_allfeatures_controlled_2015_2025_allstocks_e30 --features <52 candidate features>
```

Output directory:

```text
C:\Users\Administrator\Documents\New project 3\zcw\outputs\github_updated_allfeatures_controlled_2015_2025_allstocks_e30
```

Metrics:

| Metric | 18-feature run | 52-feature run |
|---|---:|---:|
| Git commit | `7ab14ea` | `7ab14ea` |
| Device | `cuda` | `cuda` |
| Dates used | 2508 | 1332 |
| Stocks used | 5363 | 5363 |
| Features | 18 | 52 |
| Latent factors | 4 | 4 |
| Epochs | 30 | 30 |
| Elapsed seconds | 201.984 | 133.266 |
| Prediction rows | 9,781,366 | 895,615 |
| Full-sample R2 | 0.2962 | 0.3212 |
| Mean IC | 0.0914 | 0.2355 |
| Mean Rank IC | 0.0930 | 0.2357 |
| Rank IC IR | 9.4373 | 27.1129 |
| Long-short mean return | 0.0087 | 0.0189 |
| Long-short IR | 9.9265 | 22.6816 |

Artifacts:

- `predictions.csv` (34,941,419 bytes)
- `period_ic.csv`
- `decile_portfolios.csv`
- `training_loss.png`
- `decile_returns.png`
- `long_short_cumulative.png`
- `actual_vs_predicted_sample.png`
- `summary.json`

Interpretation:

- The full-feature run has better full-sample fit and cross-sectional ranking
  metrics than the 18-feature run.
- The improvement is not apples-to-apples, because the all-feature missing
  data filter reduces the usable panel from 2508 dates and 9.78 million rows to
  1332 dates and 0.90 million rows.
- This is still a full-sample reconstruction/pricing-fit result, not strict
  out-of-sample prediction.
