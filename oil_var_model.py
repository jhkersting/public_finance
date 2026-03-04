"""Core parametric VaR pipeline for oil prices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, t
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from utils import calculate_qlike

try:
    from arch import arch_model
except Exception:  # pragma: no cover - handled via fallback path
    arch_model = None


@dataclass
class _ArchState:
    """Container for recursive volatility updates between refits."""

    model: str
    dist: str
    mu: float
    omega: float
    alpha: float
    beta: float
    gamma: float
    sigma2_prev: float
    last_return: float
    nu: Optional[float] = None
    source: str = "arch"


class OilRiskPipeline:
    """Institutional-style VaR research pipeline for oil prices."""

    def __init__(
        self,
        csv_path: str | Path,
        random_state: int = 42,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.random_state = int(random_state)
        np.random.seed(self.random_state)

        self.data: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def _required_columns() -> List[str]:
        return ["date", "price"]

    @staticmethod
    def _coerce_dist(dist: str) -> str:
        dist_key = (dist or "normal").strip().lower()
        if dist_key in {"t", "student-t", "student_t", "student"}:
            return "t"
        return "normal"

    @staticmethod
    def _safe_positive(value: float, floor: float = 1e-10) -> float:
        if not np.isfinite(value):
            return floor
        return float(max(value, floor))

    @staticmethod
    def _dist_quantile(alpha: float, dist: str, nu: Optional[float] = None) -> float:
        alpha = float(np.clip(alpha, 1e-6, 1 - 1e-6))
        if dist == "t" and nu is not None and np.isfinite(nu) and nu > 2.05:
            return float(t.ppf(alpha, df=nu))
        return float(norm.ppf(alpha))

    def prepare_data(self) -> pd.DataFrame:
        """Load and prepare price/return dataset."""
        df = pd.read_csv(self.csv_path)
        missing = [c for c in self._required_columns() if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df[["date", "price"]].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = (
            df.dropna(subset=["date", "price"])
            .sort_values("date")
            .drop_duplicates(subset="date", keep="last")
        )
        df = df[df["price"] > 0.0]
        if df.empty:
            raise ValueError("No valid observations after cleaning input data.")

        df["log_return"] = np.log(df["price"]).diff()
        df["return_pct"] = 100.0 * df["log_return"]
        df["realized_var_proxy"] = df["return_pct"].pow(2)

        df = df.dropna(subset=["return_pct"]).set_index("date")
        if df.empty:
            raise ValueError("Not enough observations to compute returns.")

        self.data = df
        return self.data.copy()

    @staticmethod
    def _ewma_variance_last(returns: pd.Series, ewma_lambda: float = 0.94) -> float:
        clean = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            return 1e-4

        ewma_lambda = float(np.clip(ewma_lambda, 0.80, 0.999))
        values = clean.to_numpy(dtype=float)

        if values.size < 3:
            return float(np.var(values, ddof=1) if values.size > 1 else max(values[0] ** 2, 1e-4))

        start_n = min(30, values.size)
        variance = float(np.var(values[:start_n], ddof=1))
        variance = max(variance, 1e-8)
        for r in values:
            variance = ewma_lambda * variance + (1.0 - ewma_lambda) * (r * r)
        return float(max(variance, 1e-10))

    def _fallback_state(self, train_returns: pd.Series, model: str, dist: str) -> _ArchState:
        sigma2 = self._ewma_variance_last(train_returns)
        last_return = float(train_returns.iloc[-1]) if not train_returns.empty else 0.0
        mu = float(train_returns.mean()) if not train_returns.empty else 0.0
        # Conservative persistence fallback
        omega = sigma2 * 0.01
        alpha = 0.08
        beta = 0.90
        gamma = -0.02 if model.upper() == "EGARCH" else 0.0
        nu = 8.0 if dist == "t" else None
        return _ArchState(
            model=model.upper(),
            dist=dist,
            mu=mu,
            omega=omega,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            sigma2_prev=max(sigma2, 1e-10),
            last_return=last_return,
            nu=nu,
            source="fallback",
        )

    def _fit_arch_state(self, train_returns: pd.Series, model: str, dist: str) -> _ArchState:
        model = model.upper()
        dist = self._coerce_dist(dist)
        clean = pd.to_numeric(train_returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

        if arch_model is None:
            return self._fallback_state(clean, model, dist)

        if clean.size < 80:
            return self._fallback_state(clean, model, dist)

        try:
            if model == "GARCH":
                spec = arch_model(
                    clean,
                    mean="Constant",
                    vol="GARCH",
                    p=1,
                    q=1,
                    dist=dist,
                    rescale=False,
                )
            elif model == "EGARCH":
                spec = arch_model(
                    clean,
                    mean="Constant",
                    vol="EGARCH",
                    p=1,
                    o=1,
                    q=1,
                    dist=dist,
                    rescale=False,
                )
            else:
                raise ValueError(f"Unsupported model: {model}")

            result = spec.fit(disp="off", show_warning=False, options={"maxiter": 2000})
            params = result.params
            mu = float(params.get("mu", params.get("Const", 0.0)))
            omega = float(params.get("omega", np.nan))
            alpha = float(params.get("alpha[1]", np.nan))
            beta = float(params.get("beta[1]", np.nan))
            gamma = float(params.get("gamma[1]", 0.0))
            nu = float(params.get("nu", np.nan)) if dist == "t" else None

            sigma2_prev = float(result.conditional_volatility.iloc[-1] ** 2)
            last_return = float(clean.iloc[-1])

            values_ok = all(np.isfinite(x) for x in [mu, omega, alpha, beta, gamma, sigma2_prev])
            if (not values_ok) or sigma2_prev <= 0:
                raise FloatingPointError("Non-finite ARCH fit output")

            return _ArchState(
                model=model,
                dist=dist,
                mu=mu,
                omega=max(omega, 1e-12),
                alpha=max(alpha, 0.0),
                beta=max(beta, 0.0),
                gamma=gamma,
                sigma2_prev=max(sigma2_prev, 1e-10),
                last_return=last_return,
                nu=nu if (nu is not None and np.isfinite(nu)) else None,
                source="arch",
            )
        except Exception:
            return self._fallback_state(clean, model, dist)

    def _forecast_one_step_from_state(self, state: _ArchState, alpha: float) -> Dict[str, float]:
        sigma2_prev = self._safe_positive(state.sigma2_prev)
        eps_prev = state.last_return - state.mu

        if state.model == "GARCH":
            sigma2 = state.omega + state.alpha * (eps_prev**2) + state.beta * sigma2_prev
        elif state.model == "EGARCH":
            z_prev = eps_prev / np.sqrt(sigma2_prev)
            expected_abs_z = np.sqrt(2.0 / np.pi)
            log_sigma2 = (
                state.omega
                + state.beta * np.log(sigma2_prev)
                + state.alpha * (abs(z_prev) - expected_abs_z)
                + state.gamma * z_prev
            )
            sigma2 = np.exp(log_sigma2)
        else:
            sigma2 = sigma2_prev

        sigma2 = self._safe_positive(float(sigma2))
        vol = float(np.sqrt(sigma2))
        q = self._dist_quantile(alpha=alpha, dist=state.dist, nu=state.nu)
        var = float(state.mu + vol * q)

        return {
            "mean": float(state.mu),
            "vol": vol,
            "variance": sigma2,
            "var": var,
        }

    @staticmethod
    def _compute_hit_loss(realized_return: float, forecast_var: float, hit: int) -> float:
        if hit == 0:
            return 0.0
        return float(max(forecast_var - realized_return, 0.0))

    def rolling_backtest(
        self,
        train_size: int,
        alpha: float = 0.01,
        refit_every: int = 20,
        ewma_lambda: float = 0.94,
        garch_dist: str = "t",
        egarch_dist: str = "t",
    ) -> pd.DataFrame:
        """Run one-day rolling VaR backtest for EWMA, GARCH, and EGARCH."""
        if self.data.empty:
            self.prepare_data()

        df = self.data
        train_size = int(max(train_size, 50))
        refit_every = int(max(refit_every, 1))

        columns = [
            "date",
            "model",
            "realized_return_pct",
            "forecast_mean_pct",
            "forecast_vol_pct",
            "forecast_variance_pct2",
            "forecast_var_pct",
            "hit",
            "hit_loss",
            "realized_var_proxy",
            "sq_error_var",
            "abs_error_var",
            "qlike",
        ]

        if len(df) <= train_size:
            return pd.DataFrame(columns=columns)

        rows: List[Dict[str, float | int | str | pd.Timestamp]] = []
        states: Dict[str, _ArchState] = {}

        for t_idx in range(train_size, len(df)):
            date = df.index[t_idx]
            train_returns = df["return_pct"].iloc[:t_idx]
            realized_return = float(df["return_pct"].iloc[t_idx])
            realized_var = float(df["realized_var_proxy"].iloc[t_idx])
            step = t_idx - train_size

            # EWMA
            ewma_sigma2 = self._ewma_variance_last(train_returns, ewma_lambda=ewma_lambda)
            ewma_mean = float(train_returns.mean()) if not train_returns.empty else 0.0
            ewma_vol = float(np.sqrt(ewma_sigma2))
            ewma_q = self._dist_quantile(alpha=alpha, dist="normal")
            ewma_var = float(ewma_mean + ewma_vol * ewma_q)
            ewma_hit = int(realized_return < ewma_var)
            ewma_sq_err = (ewma_sigma2 - realized_var) ** 2
            ewma_abs_err = abs(ewma_sigma2 - realized_var)
            ewma_qlike = float(calculate_qlike([realized_var], [ewma_sigma2]).iloc[0])

            rows.append(
                {
                    "date": date,
                    "model": "EWMA",
                    "realized_return_pct": realized_return,
                    "forecast_mean_pct": ewma_mean,
                    "forecast_vol_pct": ewma_vol,
                    "forecast_variance_pct2": ewma_sigma2,
                    "forecast_var_pct": ewma_var,
                    "hit": ewma_hit,
                    "hit_loss": self._compute_hit_loss(realized_return, ewma_var, ewma_hit),
                    "realized_var_proxy": realized_var,
                    "sq_error_var": ewma_sq_err,
                    "abs_error_var": ewma_abs_err,
                    "qlike": ewma_qlike,
                }
            )

            # GARCH and EGARCH
            for model_name, dist in (("GARCH", garch_dist), ("EGARCH", egarch_dist)):
                if model_name not in states or step % refit_every == 0:
                    states[model_name] = self._fit_arch_state(train_returns, model=model_name, dist=dist)

                state = states[model_name]
                fcst = self._forecast_one_step_from_state(state, alpha=alpha)
                hit = int(realized_return < fcst["var"])

                sq_error = (fcst["variance"] - realized_var) ** 2
                abs_error = abs(fcst["variance"] - realized_var)
                qlike = float(calculate_qlike([realized_var], [fcst["variance"]]).iloc[0])

                rows.append(
                    {
                        "date": date,
                        "model": model_name,
                        "realized_return_pct": realized_return,
                        "forecast_mean_pct": fcst["mean"],
                        "forecast_vol_pct": fcst["vol"],
                        "forecast_variance_pct2": fcst["variance"],
                        "forecast_var_pct": fcst["var"],
                        "hit": hit,
                        "hit_loss": self._compute_hit_loss(realized_return, fcst["var"], hit),
                        "realized_var_proxy": realized_var,
                        "sq_error_var": sq_error,
                        "abs_error_var": abs_error,
                        "qlike": qlike,
                    }
                )

                # Advance state one step after seeing the realized return
                states[model_name] = _ArchState(
                    model=state.model,
                    dist=state.dist,
                    mu=state.mu,
                    omega=state.omega,
                    alpha=state.alpha,
                    beta=state.beta,
                    gamma=state.gamma,
                    sigma2_prev=max(fcst["variance"], 1e-10),
                    last_return=realized_return,
                    nu=state.nu,
                    source=state.source,
                )

        out = pd.DataFrame(rows, columns=columns)
        out["date"] = pd.to_datetime(out["date"])
        return out

    @staticmethod
    def _kupiec_test(hits: pd.Series, alpha: float) -> Dict[str, float]:
        x = int(hits.sum())
        n = int(hits.shape[0])
        p = float(np.clip(alpha, 1e-8, 1 - 1e-8))

        if n == 0:
            return {"exceptions": 0, "expected_rate": p, "actual_rate": np.nan, "LR_uc": np.nan, "p_value_uc": np.nan}

        phat = np.clip(x / n, 1e-8, 1 - 1e-8)
        log_l_null = (n - x) * np.log(1 - p) + x * np.log(p)
        log_l_alt = (n - x) * np.log(1 - phat) + x * np.log(phat)
        lr_uc = -2.0 * (log_l_null - log_l_alt)
        p_value = 1.0 - chi2.cdf(lr_uc, df=1)

        return {
            "exceptions": x,
            "expected_rate": p,
            "actual_rate": x / n,
            "LR_uc": float(max(lr_uc, 0.0)),
            "p_value_uc": float(np.clip(p_value, 0.0, 1.0)),
        }

    @staticmethod
    def _christoffersen_independence_test(hits: pd.Series) -> Dict[str, float]:
        h = hits.astype(int).to_numpy()
        if h.size < 3:
            return {"LR_ind": np.nan, "p_value_ind": np.nan}

        n00 = n01 = n10 = n11 = 0
        for prev, curr in zip(h[:-1], h[1:]):
            if prev == 0 and curr == 0:
                n00 += 1
            elif prev == 0 and curr == 1:
                n01 += 1
            elif prev == 1 and curr == 0:
                n10 += 1
            elif prev == 1 and curr == 1:
                n11 += 1

        eps = 1e-12
        pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
        pi01 = n01 / max(n00 + n01, 1)
        pi11 = n11 / max(n10 + n11, 1)

        pi = np.clip(pi, eps, 1 - eps)
        pi01 = np.clip(pi01, eps, 1 - eps)
        pi11 = np.clip(pi11, eps, 1 - eps)

        log_l0 = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
        log_l1 = n00 * np.log(1 - pi01) + n01 * np.log(pi01) + n10 * np.log(1 - pi11) + n11 * np.log(pi11)

        lr_ind = -2.0 * (log_l0 - log_l1)
        p_value = 1.0 - chi2.cdf(lr_ind, df=1)

        return {
            "LR_ind": float(max(lr_ind, 0.0)),
            "p_value_ind": float(np.clip(p_value, 0.0, 1.0)),
        }

    def var_test_summary(self, backtest_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
        """Run Kupiec, Christoffersen independence, and conditional coverage tests."""
        if backtest_df.empty:
            return pd.DataFrame(
                columns=[
                    "model",
                    "exceptions",
                    "expected_rate",
                    "actual_rate",
                    "LR_uc",
                    "p_value_uc",
                    "LR_ind",
                    "p_value_ind",
                    "LR_cc",
                    "p_value_cc",
                ]
            )

        rows: List[Dict[str, float | str]] = []
        for model, grp in backtest_df.groupby("model"):
            hits = pd.to_numeric(grp["hit"], errors="coerce").fillna(0).astype(int)
            uc = self._kupiec_test(hits, alpha=alpha)
            ind = self._christoffersen_independence_test(hits)

            lr_cc = np.nan
            p_cc = np.nan
            if np.isfinite(uc["LR_uc"]) and np.isfinite(ind["LR_ind"]):
                lr_cc = float(uc["LR_uc"] + ind["LR_ind"])
                p_cc = float(1.0 - chi2.cdf(lr_cc, df=2))

            rows.append(
                {
                    "model": str(model),
                    "exceptions": uc["exceptions"],
                    "expected_rate": uc["expected_rate"],
                    "actual_rate": uc["actual_rate"],
                    "LR_uc": uc["LR_uc"],
                    "p_value_uc": uc["p_value_uc"],
                    "LR_ind": ind["LR_ind"],
                    "p_value_ind": ind["p_value_ind"],
                    "LR_cc": lr_cc,
                    "p_value_cc": p_cc,
                }
            )

        return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)

    @staticmethod
    def validation_summary(backtest_df: pd.DataFrame) -> pd.DataFrame:
        """Compute variance forecast quality metrics and mean hit rate."""
        if backtest_df.empty:
            return pd.DataFrame(
                columns=[
                    "model",
                    "mse_variance",
                    "rmse_variance",
                    "mae_variance",
                    "qlike",
                    "mean_hit_rate",
                    "expected_shortfall_pct",
                ]
            )

        rows: List[Dict[str, float | str]] = []
        for model, grp in backtest_df.groupby("model"):
            err = grp["forecast_variance_pct2"] - grp["realized_var_proxy"]
            mse = float(np.nanmean(np.square(err)))
            rmse = float(np.sqrt(mse)) if np.isfinite(mse) else np.nan
            mae = float(np.nanmean(np.abs(err)))
            qlike = float(np.nanmean(grp["qlike"]))
            hit_rate = float(np.nanmean(grp["hit"]))

            hits_grp = grp.loc[grp["hit"] == 1, "realized_return_pct"]
            expected_shortfall = float(hits_grp.mean()) if not hits_grp.empty else np.nan

            rows.append(
                {
                    "model": str(model),
                    "mse_variance": mse,
                    "rmse_variance": rmse,
                    "mae_variance": mae,
                    "qlike": qlike,
                    "mean_hit_rate": hit_rate,
                    "expected_shortfall_pct": expected_shortfall,
                }
            )

        return pd.DataFrame(rows).sort_values("qlike").reset_index(drop=True)

    @staticmethod
    def residual_diagnostics(backtest_df: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
        """Residual diagnostics using Ljung-Box and ARCH LM tests."""
        if backtest_df.empty:
            return pd.DataFrame(columns=["model", "lb_stat", "lb_pvalue", "arch_lm_stat", "arch_lm_pvalue"])

        rows: List[Dict[str, float | str]] = []
        for model, grp in backtest_df.groupby("model"):
            resid = grp["realized_return_pct"] - grp["forecast_mean_pct"]
            denom = grp["forecast_vol_pct"].clip(lower=1e-6)
            std_resid = (resid / denom).replace([np.inf, -np.inf], np.nan).dropna()

            if std_resid.size < max(15, lags + 5):
                rows.append(
                    {
                        "model": str(model),
                        "lb_stat": np.nan,
                        "lb_pvalue": np.nan,
                        "arch_lm_stat": np.nan,
                        "arch_lm_pvalue": np.nan,
                    }
                )
                continue

            lb = acorr_ljungbox(std_resid, lags=[lags], return_df=True)
            lb_stat = float(lb["lb_stat"].iloc[-1])
            lb_p = float(lb["lb_pvalue"].iloc[-1])

            lm_stat, lm_pvalue, _, _ = het_arch(std_resid, nlags=lags)

            rows.append(
                {
                    "model": str(model),
                    "lb_stat": lb_stat,
                    "lb_pvalue": lb_p,
                    "arch_lm_stat": float(lm_stat),
                    "arch_lm_pvalue": float(lm_pvalue),
                }
            )

        return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)

    def _arch_multi_step_forecast(
        self,
        train_returns: pd.Series,
        model: str,
        dist: str,
        horizon_days: int,
    ) -> Dict[str, float]:
        clean = pd.to_numeric(train_returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        horizon_days = int(max(1, horizon_days))

        if arch_model is None:
            sigma2 = self._ewma_variance_last(clean)
            mean = float(clean.mean()) if not clean.empty else 0.0
            return {
                "mean_cum": mean * horizon_days,
                "variance_cum": sigma2 * horizon_days,
                "dist": "normal",
                "nu": None,
            }

        if clean.size < 80:
            sigma2 = self._ewma_variance_last(clean)
            mean = float(clean.mean())
            return {
                "mean_cum": mean * horizon_days,
                "variance_cum": sigma2 * horizon_days,
                "dist": "normal",
                "nu": None,
            }

        dist = self._coerce_dist(dist)

        try:
            if model.upper() == "GARCH":
                spec = arch_model(clean, mean="Constant", vol="GARCH", p=1, q=1, dist=dist, rescale=False)
            else:
                spec = arch_model(clean, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist=dist, rescale=False)

            result = spec.fit(disp="off", show_warning=False, options={"maxiter": 2000})
            fcast = result.forecast(horizon=horizon_days, reindex=False)

            mean_path = fcast.mean.iloc[-1].to_numpy(dtype=float)
            var_path = fcast.variance.iloc[-1].to_numpy(dtype=float)

            mean_cum = float(np.nansum(mean_path))
            variance_cum = float(np.nansum(np.clip(var_path, 1e-12, None)))
            nu = float(result.params.get("nu", np.nan)) if dist == "t" else None

            return {
                "mean_cum": mean_cum,
                "variance_cum": max(variance_cum, 1e-12),
                "dist": dist,
                "nu": nu if (nu is not None and np.isfinite(nu)) else None,
            }
        except Exception:
            sigma2 = self._ewma_variance_last(clean)
            mean = float(clean.mean())
            return {
                "mean_cum": mean * horizon_days,
                "variance_cum": sigma2 * horizon_days,
                "dist": "normal",
                "nu": None,
            }

    def forecast_var_x_days(
        self,
        horizon_days: int,
        alpha: float = 0.01,
        ewma_lambda: float = 0.94,
        garch_dist: str = "t",
        egarch_dist: str = "t",
    ) -> pd.DataFrame:
        """Forecast cumulative horizon VaR for EWMA, GARCH, and EGARCH."""
        if self.data.empty:
            self.prepare_data()

        horizon_days = int(max(1, horizon_days))
        returns = self.data["return_pct"]

        rows: List[Dict[str, float | int | str]] = []

        # EWMA: sqrt(h) scaling
        ewma_var1 = self._ewma_variance_last(returns, ewma_lambda=ewma_lambda)
        ewma_var_cum = float(ewma_var1 * horizon_days)
        ewma_mean_cum = float(returns.mean() * horizon_days)
        ewma_vol_cum = float(np.sqrt(ewma_var_cum))
        ewma_q = self._dist_quantile(alpha=alpha, dist="normal")
        rows.append(
            {
                "model": "EWMA",
                "horizon_days": horizon_days,
                "alpha": alpha,
                "forecast_mean_pct": ewma_mean_cum,
                "forecast_vol_pct": ewma_vol_cum,
                "forecast_variance_pct2": ewma_var_cum,
                "forecast_var_pct": float(ewma_mean_cum + ewma_vol_cum * ewma_q),
            }
        )

        # GARCH / EGARCH: direct multi-step forecast from model
        for model_name, dist in (("GARCH", garch_dist), ("EGARCH", egarch_dist)):
            fcst = self._arch_multi_step_forecast(
                train_returns=returns,
                model=model_name,
                dist=dist,
                horizon_days=horizon_days,
            )
            vol_cum = float(np.sqrt(fcst["variance_cum"]))
            q = self._dist_quantile(alpha=alpha, dist=str(fcst["dist"]), nu=fcst["nu"])

            rows.append(
                {
                    "model": model_name,
                    "horizon_days": horizon_days,
                    "alpha": alpha,
                    "forecast_mean_pct": float(fcst["mean_cum"]),
                    "forecast_vol_pct": vol_cum,
                    "forecast_variance_pct2": float(fcst["variance_cum"]),
                    "forecast_var_pct": float(fcst["mean_cum"] + vol_cum * q),
                }
            )

        return pd.DataFrame(rows)

    def build_var_horizon_surface(
        self,
        horizon_days: Sequence[int],
        alphas: Sequence[float],
        ewma_lambda: float = 0.94,
        garch_dist: str = "t",
        egarch_dist: str = "t",
    ) -> pd.DataFrame:
        """
        Generate full model x horizon x alpha VaR surface.

        Returns one row per (model, horizon_days, alpha).
        """
        if self.data.empty:
            self.prepare_data()

        clean_horizons = sorted({int(max(1, h)) for h in horizon_days})
        clean_alphas = sorted({float(np.clip(a, 1e-6, 1 - 1e-6)) for a in alphas})

        rows: List[pd.DataFrame] = []
        for h in clean_horizons:
            for alpha in clean_alphas:
                out = self.forecast_var_x_days(
                    horizon_days=h,
                    alpha=alpha,
                    ewma_lambda=ewma_lambda,
                    garch_dist=garch_dist,
                    egarch_dist=egarch_dist,
                ).copy()
                out["forecast_var_abs_pct"] = out["forecast_var_pct"].abs()
                rows.append(out)

        if not rows:
            return pd.DataFrame(
                columns=[
                    "model",
                    "horizon_days",
                    "alpha",
                    "forecast_mean_pct",
                    "forecast_vol_pct",
                    "forecast_variance_pct2",
                    "forecast_var_pct",
                    "forecast_var_abs_pct",
                ]
            )

        surface = pd.concat(rows, ignore_index=True)
        surface = surface.sort_values(["model", "alpha", "horizon_days"]).reset_index(drop=True)
        return surface

    def integrate_var_horizon_surface(
        self,
        surface_df: pd.DataFrame,
        value_col: str = "forecast_var_pct",
        use_absolute: bool = True,
        integration_start: float = 0.0,
        notional: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Compute cumulative area-under-VaR curve for each model and alpha.

        Integration is done across horizon_days with trapezoid rule only.
        """
        required = {"model", "alpha", "horizon_days", value_col}
        missing = sorted(required.difference(surface_df.columns))
        if missing:
            raise ValueError(f"surface_df is missing required columns: {missing}")

        df = surface_df.copy()
        df["horizon_days"] = pd.to_numeric(df["horizon_days"], errors="coerce")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=["model", "alpha", "horizon_days", value_col]).copy()
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "model",
                    "alpha",
                    "horizon_days",
                    value_col,
                    "integration_value_pct",
                    "auc_trapezoid_pct_days",
                    "avg_integration_value_pct",
                    "auc_trapezoid_dollar_days",
                    "avg_borrow_dollars",
                ]
            )

        out_rows: List[Dict[str, float | str]] = []
        start = float(integration_start)

        for (model, alpha), grp in df.groupby(["model", "alpha"], sort=True):
            sub = grp.sort_values("horizon_days").copy()
            x = sub["horizon_days"].to_numpy(dtype=float)
            y_raw = sub[value_col].to_numpy(dtype=float)
            y = np.abs(y_raw) if use_absolute else y_raw

            # Trapezoid cumulative area from integration_start to each horizon.
            x_aug = np.concatenate(([start], x))
            y0 = y[0] if y.size > 0 else 0.0
            y_aug = np.concatenate(([y0], y))
            auc = np.zeros_like(x_aug, dtype=float)
            for i in range(1, x_aug.size):
                dx = x_aug[i] - x_aug[i - 1]
                auc[i] = auc[i - 1] + 0.5 * dx * (y_aug[i] + y_aug[i - 1])

            auc_to_h = auc[1:]
            span = np.maximum(x - start, 1e-12)
            avg_to_h = auc_to_h / span

            for i in range(sub.shape[0]):
                row = {
                    "model": str(model),
                    "alpha": float(alpha),
                    "horizon_days": float(x[i]),
                    value_col: float(y_raw[i]),
                    "integration_value_pct": float(y[i]),
                    "auc_trapezoid_pct_days": float(auc_to_h[i]),
                    "avg_integration_value_pct": float(avg_to_h[i]),
                }

                if notional is not None and np.isfinite(notional):
                    # Convert percent-days into dollar-days: pct/100 * notional.
                    auc_dollar_days = float(auc_to_h[i] / 100.0 * float(notional))
                    row["auc_trapezoid_dollar_days"] = auc_dollar_days
                    row["avg_borrow_dollars"] = float(auc_dollar_days / span[i])
                else:
                    row["auc_trapezoid_dollar_days"] = np.nan
                    row["avg_borrow_dollars"] = np.nan

                out_rows.append(row)

        out = pd.DataFrame(out_rows)
        out = out.sort_values(["model", "alpha", "horizon_days"]).reset_index(drop=True)
        return out

    @staticmethod
    def pivot_surface_by_model(
        surface_df: pd.DataFrame,
        value_col: str = "forecast_var_pct",
    ) -> Dict[str, pd.DataFrame]:
        """Create per-model grids with rows=horizon_days and columns=alpha."""
        required = {"model", "horizon_days", "alpha", value_col}
        missing = sorted(required.difference(surface_df.columns))
        if missing:
            raise ValueError(f"surface_df is missing required columns: {missing}")

        grids: Dict[str, pd.DataFrame] = {}
        for model, sub in surface_df.groupby("model"):
            grid = (
                sub.pivot_table(
                    index="horizon_days",
                    columns="alpha",
                    values=value_col,
                    aggfunc="first",
                )
                .sort_index()
                .sort_index(axis=1)
            )
            grids[str(model)] = grid
        return grids
