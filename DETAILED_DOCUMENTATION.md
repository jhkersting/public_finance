# Oil VaR Project: Detailed Technical Documentation

## 1. Project Overview

This package is a research-grade, deterministic Value-at-Risk (VaR) framework for oil prices. It supports:

- Data ingestion from CSV with required columns `date` and `price`
- Log-return computation and scaling returns by 100 (percent units)
- Rolling one-day VaR backtesting for:
  - EWMA
  - GARCH(1,1)
  - EGARCH(1,1,1)
- Distribution assumptions:
  - Normal
  - Student-t
- VaR diagnostic tests:
  - Kupiec unconditional coverage
  - Christoffersen independence
  - Conditional coverage
- Forecast quality diagnostics:
  - MSE, RMSE, MAE for variance
  - QLIKE
  - Mean hit rate
- Residual diagnostics:
  - Ljung-Box
  - ARCH LM
- Machine-learning quantile VaR using gradient boosting with rolling retraining
- Multi-day (X-day) VaR forecasting

The design emphasizes numerical stability, robust fallbacks, and predictable behavior in small samples or model convergence failures.

## 2. Folder Structure and Responsibilities

- `oil_var_model.py`
  - Core parametric VaR pipeline
  - Data preparation, rolling backtest, VaR tests, diagnostics, X-day VaR
- `ml_var_model.py`
  - Quantile gradient boosting VaR model with optional hybrid adjustment
- `utils.py`
  - Shared statistical and cleaning utilities
- `run_example.py`
  - Minimal runnable script to execute the full workflow
- `sample_oil_prices.csv`
  - Synthetic 1500-observation daily oil price series
- `requirements.txt`
  - Package dependencies
- `README.md`
  - High-level quick start
- `DETAILED_DOCUMENTATION.md`
  - This in-depth technical reference

## 3. End-to-End Data and Model Flow

1. Load raw price data (`date`, `price`).
2. Clean dates and prices, sort by date, remove invalid rows.
3. Compute log returns and convert to percent units (`return_pct`).
4. Compute realized variance proxy (`return_pct^2`).
5. Run rolling backtest for parametric models.
6. Optionally build ML features and run rolling quantile backtest.
7. Aggregate all model outputs.
8. Run VaR coverage tests and validation metrics.
9. Run residual diagnostics.
10. Export CSV outputs and print concise summaries.

## 4. Core Conventions and Units

### 4.1 Return Scale

- Raw log return: `log(price_t / price_{t-1})`
- Model return unit: `return_pct = 100 * log_return`

Reasoning:
- ARCH optimizers often behave better when return magnitudes are in percent units.
- Improves numerical conditioning in recursive variance equations.

### 4.2 VaR Sign Convention

- Lower-tail alpha (example: `alpha = 0.01`) is used.
- Forecast VaR is typically negative in return space.
- A hit (exception) occurs when:
  - `realized_return_pct < forecast_var_pct`

### 4.3 Realized Variance Proxy

- `realized_var_proxy = realized_return_pct^2`

Reasoning:
- Standard high-level proxy when realized intraday variance is unavailable.
- Needed for variance forecast loss calculations (MSE/MAE/QLIKE).

## 5. Failure Handling and Determinism

### 5.1 Determinism

- Random seeds are set to `42` in:
  - `OilRiskPipeline.__init__`
  - `MLVaRModel.__init__`
  - `run_example.main`

### 5.2 Optimizer Failure Handling

- If `arch` is unavailable or model fit fails, the code falls back to stable recursive settings derived from EWMA-style initialization.
- If sample size is too small for ARCH fitting (`< 80`), fallback is used.

### 5.3 Small-Sample Safeguards

- Rolling routines return empty DataFrames with full schema when sample is insufficient.
- Quantiles and losses are clipped or guarded against invalid values.
- Variances are floored (e.g., `1e-10`) to avoid division-by-zero or log issues.

## 6. File-Level Technical Reference

---

## 6A. `utils.py`

### `clean_numeric_series(values)`

Purpose:
- Convert arbitrary input into a finite numeric pandas Series.

Inputs:
- `values`: iterable or Series

Key variables:
- `series`: raw Series view
- `numeric`: numeric-converted and cleaned Series

Dependencies:
- `pandas.to_numeric`
- `numpy` finite checks via replacement

Output:
- `pd.Series[float]` with NaN/inf removed

Reasoning:
- Many downstream stats require finite float arrays; this centralizes cleaning.

---

### `safe_quantile(values, q)`

Purpose:
- Compute robust quantile with clipping and empty-input protection.

Inputs:
- `values`: iterable or Series
- `q`: target quantile in `[0, 1]`

Key variables:
- `clean`: cleaned numeric series
- `q`: clipped quantile

Dependencies:
- `clean_numeric_series`
- `numpy.nanquantile`

Output:
- `float` quantile, or `nan` if no valid data

Reasoning:
- Ensures quantile calls never crash on dirty inputs.

---

### `rolling_volatility(returns, window, min_periods=None)`

Purpose:
- Compute rolling sample standard deviation.

Inputs:
- `returns`: return series
- `window`: rolling window length
- `min_periods`: minimum observations required

Key variables:
- `series`: returns as Series
- auto-derived `min_periods`

Dependencies:
- `pandas.Series.rolling(...).std(ddof=1)`

Output:
- `pd.Series` rolling volatility

Reasoning:
- Provides standard rolling vol feature for ML model engineering.

---

### `calculate_qlike(realized_variance, forecast_variance, eps=1e-10)`

Purpose:
- Compute numerically stable QLIKE loss.

Inputs:
- `realized_variance`
- `forecast_variance`
- `eps`: lower floor for stability

Key variables:
- `aligned`: non-missing aligned realized/forecast DataFrame
- `rv`, `fv`: clipped variance arrays
- `ratio`: `rv/fv`

Dependencies:
- `numpy` vector math
- `pandas` alignment

Output:
- `pd.Series` of QLIKE values

Reasoning:
- QLIKE is robust for volatility evaluation and penalizes under-forecasting variance.

---

### `diebold_mariano_test(loss_1, loss_2, horizon=1)`

Purpose:
- Test equal predictive accuracy for two loss sequences.

Inputs:
- `loss_1`, `loss_2`: aligned loss candidates
- `horizon`: forecast horizon for HAC-like long-run variance adjustment

Key variables:
- `diff`: loss differential
- `mean_diff`
- `long_run_var`: weighted autocovariance estimate
- `dm_stat`, `p_value`

Dependencies:
- `clean_numeric_series`
- `scipy.stats.norm`

Output:
- Tuple `(dm_statistic, two_sided_p_value)`

Reasoning:
- Formal comparison of competing forecast models beyond average loss ranking.

---

## 6B. `oil_var_model.py`

### Dataclass `_ArchState`

Purpose:
- Carry fitted/recurrent state for recursive one-step forecasts between refits.

Fields:
- `model`: `GARCH` or `EGARCH`
- `dist`: `normal` or `t`
- `mu`, `omega`, `alpha`, `beta`, `gamma`: model parameters
- `sigma2_prev`: latest conditional variance
- `last_return`: latest observed return in percent units
- `nu`: Student-t degrees of freedom (optional)
- `source`: `arch` or `fallback`

Reasoning:
- Avoids full refit every day; supports `refit_every` scheduling.

---

### `OilRiskPipeline.__init__(csv_path, random_state=42)`

Purpose:
- Set pipeline configuration and deterministic seed.

Inputs:
- `csv_path`: source CSV path
- `random_state`: reproducibility seed

Key variables:
- `self.csv_path`
- `self.random_state`
- `self.data`: prepared dataset cache

Dependencies:
- `numpy.random.seed`

Output:
- Initialized object

Reasoning:
- Keeps pipeline stateful and reusable across methods.

---

### `_required_columns()`

Purpose:
- Central source-of-truth for expected input columns.

Output:
- `['date', 'price']`

Reasoning:
- Avoid hardcoding schema in multiple places.

---

### `_coerce_dist(dist)`

Purpose:
- Normalize distribution aliases to canonical names (`t`, `normal`).

Inputs:
- `dist`: user string

Output:
- normalized string

Reasoning:
- Ensures consistent distribution handling throughout model logic.

---

### `_safe_positive(value, floor=1e-10)`

Purpose:
- Enforce positive finite values for variance-like terms.

Output:
- finite positive float

Reasoning:
- Prevents invalid operations (`sqrt`, `log`, divide-by-zero).

---

### `_dist_quantile(alpha, dist, nu=None)`

Purpose:
- Return left-tail quantile for chosen distribution.

Inputs:
- `alpha`: tail probability
- `dist`: `normal` or `t`
- `nu`: degrees of freedom for Student-t

Dependencies:
- `scipy.stats.norm.ppf`
- `scipy.stats.t.ppf`

Output:
- quantile `float`

Reasoning:
- Centralizes quantile logic used in VaR calculations.

---

### `prepare_data()`

Purpose:
- Load CSV and build model-ready return dataset.

Inputs:
- Reads `self.csv_path`

Key variables:
- `df['date']`: parsed datetime
- `df['price']`: numeric positive prices
- `df['log_return']`: log diff
- `df['return_pct']`: scaled returns
- `df['realized_var_proxy']`: squared returns

Dependencies:
- `_required_columns`
- `pandas` cleaning and sorting

Output:
- Prepared DataFrame indexed by date

Reasoning:
- Standardized preprocessing ensures consistent downstream metrics.

---

### `_ewma_variance_last(returns, ewma_lambda=0.94)`

Purpose:
- Compute latest EWMA variance estimate from history.

Inputs:
- `returns`: percent return series
- `ewma_lambda`: decay factor

Key variables:
- `values`: cleaned numeric array
- `variance`: recursively updated EWMA variance

Dependencies:
- `numpy` recursion

Output:
- one-step variance forecast `float`

Reasoning:
- EWMA is stable, lightweight, and useful as both model and fallback initializer.

---

### `_fallback_state(train_returns, model, dist)`

Purpose:
- Build conservative fallback ARCH-like state when fitting is unavailable/failed.

Inputs:
- training returns and requested model/distribution

Key variables:
- `sigma2`, `mu`, fixed persistence parameters

Dependencies:
- `_ewma_variance_last`
- `_ArchState`

Output:
- `_ArchState`

Reasoning:
- Keeps pipeline operational in constrained environments and during optimizer failures.

---

### `_fit_arch_state(train_returns, model, dist)`

Purpose:
- Fit GARCH/EGARCH model and return reusable state.

Inputs:
- training returns, model kind, distribution

Key variables:
- fitted `params` (`mu`, `omega`, `alpha`, `beta`, `gamma`, `nu`)
- `sigma2_prev`, `last_return`

Dependencies:
- `arch.arch_model` (if available)
- `_coerce_dist`
- `_fallback_state`

Output:
- `_ArchState` from either fit or fallback

Reasoning:
- Encapsulates robust fitting behavior and parameter extraction in one place.

---

### `_forecast_one_step_from_state(state, alpha)`

Purpose:
- Produce one-day mean/vol/variance/VaR forecast from current state.

Inputs:
- `_ArchState`
- `alpha`

Key variables:
- `eps_prev`
- `sigma2` recursion (GARCH or EGARCH)
- `q`: distribution quantile

Dependencies:
- `_safe_positive`
- `_dist_quantile`

Output:
- dict with `mean`, `vol`, `variance`, `var`

Reasoning:
- Supports fast daily recursion between refits.

---

### `_compute_hit_loss(realized_return, forecast_var, hit)`

Purpose:
- Return non-zero tail-loss only on exceptions.

Output:
- `0` if no hit, else `max(forecast_var - realized_return, 0)`

Reasoning:
- Quantifies exception severity while preserving directionality.

---

### `rolling_backtest(train_size, alpha, refit_every, ewma_lambda, garch_dist, egarch_dist)`

Purpose:
- Run rolling one-day VaR backtest for EWMA, GARCH, and EGARCH.

Inputs:
- `train_size`: initial training observations
- `alpha`: VaR tail level
- `refit_every`: ARCH refit frequency
- `ewma_lambda`: EWMA decay
- `garch_dist`, `egarch_dist`: distribution assumptions

Key variables:
- `rows`: output record list
- `states`: recursive ARCH states by model
- `realized_return`, `realized_var`
- per-model forecasts and errors

Dependencies:
- `prepare_data`
- `_ewma_variance_last`
- `_fit_arch_state`
- `_forecast_one_step_from_state`
- `calculate_qlike`

Output:
- DataFrame columns:
  - `date`
  - `model`
  - `realized_return_pct`
  - `forecast_mean_pct`
  - `forecast_vol_pct`
  - `forecast_variance_pct2`
  - `forecast_var_pct`
  - `hit`
  - `hit_loss`
  - `realized_var_proxy`
  - `sq_error_var`
  - `abs_error_var`
  - `qlike`

Reasoning:
- Core evaluation engine for model comparison in realistic out-of-sample conditions.

---

### `_kupiec_test(hits, alpha)`

Purpose:
- Test unconditional VaR exception rate accuracy.

Inputs:
- binary hit series
- expected exception rate `alpha`

Key variables:
- `x`, `n`, `phat`
- `LR_uc`, `p_value_uc`

Dependencies:
- `scipy.stats.chi2`

Output:
- dict with exceptions, rates, LR statistic, p-value

Reasoning:
- Ensures observed exception frequency aligns with target confidence level.

---

### `_christoffersen_independence_test(hits)`

Purpose:
- Test whether exceptions are independent over time.

Inputs:
- binary hit series

Key variables:
- transition counts `n00, n01, n10, n11`
- transition probabilities `pi, pi01, pi11`
- `LR_ind`, `p_value_ind`

Dependencies:
- `scipy.stats.chi2`

Output:
- dict with independence LR statistic and p-value

Reasoning:
- Detects exception clustering (important in stressed regimes).

---

### `var_test_summary(backtest_df, alpha)`

Purpose:
- Compute per-model Kupiec, independence, and conditional coverage summaries.

Inputs:
- merged backtest DataFrame
- alpha

Key variables:
- grouped `hits`
- `LR_cc = LR_uc + LR_ind`

Dependencies:
- `_kupiec_test`
- `_christoffersen_independence_test`

Output:
- DataFrame with:
  - `model`
  - `exceptions`
  - `expected_rate`
  - `actual_rate`
  - `LR_uc`
  - `p_value_uc`
  - `LR_ind`
  - `p_value_ind`
  - `LR_cc`
  - `p_value_cc`

Reasoning:
- Consolidates statistical VaR validity checks in one model-comparison table.

---

### `validation_summary(backtest_df)`

Purpose:
- Aggregate variance forecast and hit-rate metrics per model.

Inputs:
- backtest DataFrame

Key variables:
- forecast variance error vector
- MSE/RMSE/MAE
- mean QLIKE
- expected shortfall proxy from hit observations

Dependencies:
- `numpy` aggregations

Output:
- DataFrame sorted by `qlike` with:
  - `model`
  - `mse_variance`
  - `rmse_variance`
  - `mae_variance`
  - `qlike`
  - `mean_hit_rate`
  - `expected_shortfall_pct`

Reasoning:
- Provides ranking and diagnostics for volatility forecast quality and tail behavior.

---

### `residual_diagnostics(backtest_df, lags=10)`

Purpose:
- Run residual serial-correlation and remaining ARCH-effect diagnostics.

Inputs:
- backtest DataFrame
- `lags`

Key variables:
- standardized residuals: `(realized - mean_forecast) / vol_forecast`

Dependencies:
- `statsmodels.acorr_ljungbox`
- `statsmodels.het_arch`

Output:
- DataFrame with:
  - `model`
  - `lb_stat`, `lb_pvalue`
  - `arch_lm_stat`, `arch_lm_pvalue`

Reasoning:
- Checks if volatility model captured dynamics adequately.

---

### `_arch_multi_step_forecast(train_returns, model, dist, horizon_days)`

Purpose:
- Build cumulative multi-day mean/variance forecast for ARCH-family models.

Inputs:
- training returns
- model kind
- distribution
- horizon days

Key variables:
- fitted multi-horizon paths (`mean_path`, `var_path`)
- cumulative sums (`mean_cum`, `variance_cum`)

Dependencies:
- `arch_model(...).forecast(horizon=h)` (if available)
- `_ewma_variance_last` fallback

Output:
- dict with:
  - `mean_cum`
  - `variance_cum`
  - `dist`
  - `nu`

Reasoning:
- Uses model-consistent horizon forecasts rather than naive scaling.

---

### `forecast_var_x_days(horizon_days, alpha, ewma_lambda, garch_dist, egarch_dist)`

Purpose:
- Compute cumulative X-day VaR for EWMA, GARCH, EGARCH.

Inputs:
- `horizon_days`
- `alpha`
- distribution and EWMA settings

Key variables:
- EWMA cumulative variance via `h * sigma^2`
- ARCH-family cumulative moments from `_arch_multi_step_forecast`

Dependencies:
- `_ewma_variance_last`
- `_arch_multi_step_forecast`
- `_dist_quantile`

Output:
- DataFrame with:
  - `model`
  - `horizon_days`
  - `alpha`
  - `forecast_mean_pct`
  - `forecast_vol_pct`
  - `forecast_variance_pct2`
  - `forecast_var_pct`

Reasoning:
- Supports risk management horizon analysis (e.g., 5-day VaR).

---

## 6C. `ml_var_model.py`

### Dataclass `_ModelConfig`

Purpose:
- Encapsulate gradient boosting hyperparameters.

Fields:
- `n_estimators`
- `learning_rate`
- `max_depth`
- `min_samples_leaf`

Reasoning:
- Keeps model configuration explicit and easy to override.

---

### `MLVaRModel.__init__(alpha, lookback, random_state, include_garch_feature, hybrid_weight, config)`

Purpose:
- Initialize rolling quantile VaR model settings.

Inputs:
- `alpha`: quantile target
- `lookback`: max rolling train window
- `include_garch_feature`: include optional GARCH vol feature
- `hybrid_weight`: weight on parametric VaR adjustment
- `config`: optional `_ModelConfig`

Key variables:
- clipped alpha/hybrid bounds
- deterministic seed

Output:
- initialized model object

Reasoning:
- Makes ML VaR behavior configurable without globals.

---

### `_ewma_volatility_series(returns, lam=0.94)`

Purpose:
- Build full-series EWMA volatility feature.

Inputs:
- return series
- lambda

Key variables:
- recursive `var`
- output `out`

Dependencies:
- `numpy` recursion

Output:
- `pd.Series` EWMA volatility

Reasoning:
- Adds robust volatility context as an ML feature.

---

### `_future_cumulative_returns(returns, horizon_days)`

Purpose:
- Construct target variable for direct multi-day quantile prediction.

Inputs:
- return series
- horizon length

Key variables:
- cumulative shifted sum

Dependencies:
- pandas shifts

Output:
- `pd.Series` future cumulative returns

Reasoning:
- Enables direct horizon VaR modeling (not only one-day).

---

### `_build_feature_frame(data, horizon_days=1, garch_vol_feature=None)`

Purpose:
- Engineer supervised feature/target matrix.

Inputs:
- data containing `return_pct`
- optional `garch_vol_feature`

Features created:
- `lag_ret_1`, `lag_ret_2`, `lag_ret_5`
- `roll_vol_5`, `roll_vol_20`
- `ewma_vol`
- optional `garch_vol` (fallback-filled by `ewma_vol`)

Target:
- `target_return_pct` via `_future_cumulative_returns`

Dependencies:
- `rolling_volatility`
- `_ewma_volatility_series`
- `_future_cumulative_returns`

Output:
- cleaned feature frame with no missing values

Reasoning:
- Uses compact, interpretable predictors tied to volatility clustering.

---

### `rolling_backtest(data, train_size, refit_every, horizon_days, garch_vol_feature, parametric_var_feature)`

Purpose:
- Run rolling quantile model backtest and output VaR-compatible schema.

Inputs:
- prepared data
- training and refit controls
- optional parametric features for hybrid adjustment

Key variables:
- `X`, `y`: supervised matrix/target
- `model`: `GradientBoostingRegressor(loss='quantile')`
- `fallback_q`: robust fallback quantile when fit/predict fails
- `q_ml`, `q_final`: raw and hybrid-adjusted quantiles
- sigma proxy from quantile: `abs(q_final / z_alpha)`

Dependencies:
- `_build_feature_frame`
- `safe_quantile`
- `calculate_qlike`
- `GradientBoostingRegressor`

Output:
- DataFrame columns:
  - `date`
  - `model` (`ML_QUANTILE`)
  - `realized_return_pct`
  - `forecast_mean_pct`
  - `forecast_vol_pct`
  - `forecast_variance_pct2`
  - `forecast_var_pct`
  - `hit`
  - `hit_loss`
  - `realized_var_proxy`
  - `sq_error_var`
  - `abs_error_var`
  - `qlike`
  - `horizon_days`

Reasoning:
- Provides nonparametric VaR benchmark while preserving output compatibility with parametric pipeline diagnostics.

---

## 6D. `run_example.py`

### `main()`

Purpose:
- Demonstrate complete workflow with minimal runnable script.

Workflow steps:
1. Set deterministic seed.
2. Load sample data through `OilRiskPipeline.prepare_data`.
3. Run parametric rolling backtest.
4. Build optional GARCH-derived features.
5. Run ML rolling quantile backtest.
6. Merge outputs.
7. Compute VaR tests, validation summary, residual diagnostics.
8. Run Diebold-Mariano test (GARCH vs ML QLIKE losses).
9. Produce 5-day VaR forecast.
10. Export result CSV files.
11. Print concise summaries.

Dependencies:
- `OilRiskPipeline`
- `MLVaRModel`
- `diebold_mariano_test`

Outputs:
- Console summaries
- CSV files:
  - `backtest_results.csv`
  - `validation_summary.csv`
  - `var_test_summary.csv`

Reasoning:
- Serves as executable baseline and template for custom research scripts.

## 7. Output Schemas

### 7.1 Backtest Results (`backtest_results.csv`)

Primary fields:
- Identification: `date`, `model`
- Realized values: `realized_return_pct`, `realized_var_proxy`
- Forecast moments: `forecast_mean_pct`, `forecast_vol_pct`, `forecast_variance_pct2`
- Risk metric: `forecast_var_pct`
- Exception diagnostics: `hit`, `hit_loss`
- Variance loss diagnostics: `sq_error_var`, `abs_error_var`, `qlike`
- ML-only field: `horizon_days`

### 7.2 Validation Summary (`validation_summary.csv`)

- `mse_variance`, `rmse_variance`, `mae_variance`
- `qlike`
- `mean_hit_rate`
- `expected_shortfall_pct`

### 7.3 VaR Test Summary (`var_test_summary.csv`)

- Exception counts and rates
- `LR_uc`, `p_value_uc`
- `LR_ind`, `p_value_ind`
- `LR_cc`, `p_value_cc`

## 8. Practical Usage Examples

### 8.1 Baseline run

```bash
cd /Users/jhkersting/Desktop/oil_var_project
python run_example.py
```

### 8.2 Parametric backtest only

```python
from oil_var_model import OilRiskPipeline

pipe = OilRiskPipeline("sample_oil_prices.csv", random_state=42)
pipe.prepare_data()
res = pipe.rolling_backtest(
    train_size=900,
    alpha=0.01,
    refit_every=20,
    ewma_lambda=0.94,
    garch_dist="t",
    egarch_dist="t",
)

print(res.head())
print(pipe.var_test_summary(res, alpha=0.01))
```

### 8.3 Multi-alpha comparison (1%, 2.5%, 5%)

```python
from oil_var_model import OilRiskPipeline

pipe = OilRiskPipeline("sample_oil_prices.csv", random_state=42)
pipe.prepare_data()

for alpha in [0.01, 0.025, 0.05]:
    bt = pipe.rolling_backtest(train_size=900, alpha=alpha, refit_every=20)
    summary = pipe.var_test_summary(bt, alpha=alpha)
    print(f"alpha={alpha}")
    print(summary[["model", "actual_rate", "p_value_uc", "p_value_cc"]])
```

### 8.4 X-day VaR forecast

```python
from oil_var_model import OilRiskPipeline

pipe = OilRiskPipeline("sample_oil_prices.csv", random_state=42)
pipe.prepare_data()

var_5d = pipe.forecast_var_x_days(horizon_days=5, alpha=0.01)
var_10d = pipe.forecast_var_x_days(horizon_days=10, alpha=0.01)

print(var_5d)
print(var_10d)
```

### 8.5 ML quantile model with hybrid adjustment

```python
from ml_var_model import MLVaRModel
from oil_var_model import OilRiskPipeline

pipe = OilRiskPipeline("sample_oil_prices.csv", random_state=42)
data = pipe.prepare_data()

param = pipe.rolling_backtest(train_size=900, alpha=0.01, refit_every=20)
garch = param[param["model"] == "GARCH"].set_index("date")

ml = MLVaRModel(alpha=0.01, lookback=600, random_state=42, hybrid_weight=0.25)
ml_bt = ml.rolling_backtest(
    data=data,
    train_size=900,
    refit_every=20,
    horizon_days=1,
    garch_vol_feature=garch["forecast_vol_pct"],
    parametric_var_feature=garch["forecast_var_pct"],
)

print(ml_bt.head())
```

## 9. Interpretation Guidelines

### 9.1 VaR tests

- Good unconditional coverage: `actual_rate` close to `alpha`, high `p_value_uc`.
- Good independence: high `p_value_ind`.
- Stronger overall acceptance: high `p_value_cc`.

Low p-values suggest potential misspecification or changing market regimes.

### 9.2 Forecast quality metrics

- Lower `qlike` is generally better for variance forecasts.
- Lower MSE/RMSE/MAE indicates tighter variance prediction.
- `mean_hit_rate` should be near alpha for calibrated one-day VaR.

### 9.3 Residual diagnostics

- High Ljung-Box p-values: less serial dependence in standardized residuals.
- High ARCH LM p-values: less remaining ARCH structure.

## 10. Numerical Stability Notes

Implemented safeguards include:

- Clipping of probabilities and quantiles
- Floors on variances and long-run variances
- Explicit replacement/removal of NaN/inf
- Fallback states for failed ARCH fits
- Distribution fallback to normal when Student-t parameters are invalid

These controls prevent brittle behavior in real-world noisy data.

## 11. Performance and Scaling Notes

- Main cost driver: repeated ARCH fitting in rolling loops.
- Increase `refit_every` to reduce compute cost.
- EWMA is fastest and can be used as baseline or fallback.
- ML runtime depends on `n_estimators`, `lookback`, and refit frequency.

## 12. Known Limitations and Extension Ideas

Current limitations:

- Realized variance proxy uses squared close-to-close returns (no intraday RV).
- ML variance proxy from quantile inversion is approximate.
- Multi-day ML quantile forecasts are direct-sum target based, not path-simulated.

Potential extensions:

- Add Expected Shortfall forecasts/tests
- Add bootstrap confidence intervals for VaR diagnostics
- Add richer features (term structure, macro variables, seasonality)
- Add model versioning and experiment tracking
- Add visualization module (VaR bands, hits, rolling hit-rate)

## 13. Reproducibility Checklist

To reproduce consistent results:

1. Use identical dependency versions from `requirements.txt`.
2. Keep `random_state=42` settings.
3. Use the included synthetic dataset unchanged.
4. Run from project root:
   - `python run_example.py`

## 14. Quick Troubleshooting

### `ModuleNotFoundError: No module named 'arch'`

- Install dependencies with `pip install -r requirements.txt`.
- The package still runs via internal fallback logic, but ARCH-specific behavior is approximated.

### Empty ML output

- Ensure enough rows after feature engineering.
- Reduce `train_size` or increase data length.

### VaR tests look poor

- Try Student-t distribution and shorter `refit_every`.
- Re-evaluate alpha choice and training window size.
- Check for regime changes requiring model adaptation.

## 15. Suggested Research Workflow

1. Run baseline script and verify outputs.
2. Compare parametric model diagnostics.
3. Add ML model and compare via QLIKE + DM test.
4. Evaluate sensitivity across alphas and horizons.
5. Inspect residual diagnostics for misspecification.
6. Decide production candidate based on calibration + stability + performance.

