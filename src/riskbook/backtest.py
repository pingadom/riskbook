"""Strictly causal walk-forward backtest engine with SQL persistence and VaR calibration."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from riskbook.config import BacktestConfig
from riskbook.covariance import ledoit_wolf_covariance, sample_covariance
from riskbook.db import RiskBookDB
from riskbook.optimisers import cvar_optimise, equal_weight, hrp_weights, min_variance
from riskbook.risk import (
    historical_var_cvar,
    kupiec_and_christoffersen_tests,
)
from scipy.stats import norm

STRATEGIES: tuple[str, ...] = (
    "equal_weight",
    "mv_sample",
    "mv_ledoit_wolf",
    "hrp",
    "cvar_opt",
)


@dataclass
class StrategyResult:
    strategy: str
    daily: pd.DataFrame
    weights_by_date: dict[pd.Timestamp, pd.Series]
    snapshot: dict[str, object]


def _allocate(
    strategy: str,
    window: pd.DataFrame,
    config: BacktestConfig,
    prev_weights: pd.Series | None,
) -> pd.Series:
    assets = list(window.columns)
    if strategy == "equal_weight":
        return equal_weight(assets)
    if strategy == "mv_sample":
        cov = sample_covariance(window)
        return min_variance(
            cov,
            max_weight=config.max_position_weight,
            sectors=config.universe,
            max_sector_weight=config.max_sector_weight,
            prev_weights=prev_weights,
            turnover_penalty=0.0,
        )
    if strategy == "mv_ledoit_wolf":
        cov, _ = ledoit_wolf_covariance(window)
        return min_variance(
            cov,
            max_weight=config.max_position_weight,
            sectors=config.universe,
            max_sector_weight=config.max_sector_weight,
            prev_weights=prev_weights,
            turnover_penalty=config.turnover_penalty,
        )
    if strategy == "hrp":
        cov, _ = ledoit_wolf_covariance(window)
        return hrp_weights(
            cov,
            max_weight=config.max_position_weight,
            sectors=config.universe,
            max_sector_weight=config.max_sector_weight,
        )
    if strategy == "cvar_opt":
        return cvar_optimise(
            window,
            alpha=config.var_alpha_95,
            max_weight=config.max_position_weight,
            sectors=config.universe,
            max_sector_weight=config.max_sector_weight,
            prev_weights=prev_weights,
            turnover_penalty=config.turnover_penalty,
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def _rebalance_indices(dates: pd.DatetimeIndex, lookback: int) -> set[int]:
    """Identify month-end trading indices strictly at or after ``lookback - 1``."""
    rebal: set[int] = set()
    # First eligible rebalance happens at lookback - 1 so next day (lookback) is first OOS day
    rebal.add(lookback - 1)
    months = dates.to_period("M")
    for idx in range(lookback - 1, len(dates) - 1):
        if months[idx] != months[idx + 1]:
            rebal.add(idx)
    return rebal


def run_strategy_walkforward(
    strategy: str,
    asset_returns: pd.DataFrame,
    config: BacktestConfig,
) -> StrategyResult:
    """Run a single strategy strictly causally: weights decided at close t apply to day t+1."""
    n_days = len(asset_returns)
    lookback = config.lookback_days
    if n_days <= lookback + 5:
        raise ValueError(f"Need > {lookback + 5} return rows, got {n_days}.")

    dates = asset_returns.index
    assets = list(asset_returns.columns)
    ret_mat = asset_returns.to_numpy(dtype=float)
    rebal_set = _rebalance_indices(dates, lookback)
    # Precompute normalised EWMA weights once for O(T) 1-day-ahead parametric VaR
    ewma_exponents = np.arange(lookback - 1, -1, -1, dtype=float)
    ewma_w = (1.0 - config.ewma_lambda) * (config.ewma_lambda ** ewma_exponents)
    ewma_w /= ewma_w.sum()
    z_95 = float(norm.ppf(config.var_alpha_95))

    weights_by_date: dict[pd.Timestamp, pd.Series] = {}
    records: list[dict[str, object]] = []

    current_w: np.ndarray | None = None
    pending_turnover = 0.0
    rebal_turnovers: list[float] = []

    for idx in range(lookback - 1, n_days - 1):
        # 1) Rebalance at close of day `idx` using strictly past window [idx - lookback + 1 .. idx]
        if idx in rebal_set:
            window = asset_returns.iloc[idx - lookback + 1 : idx + 1]
            prev_series = pd.Series(current_w, index=assets) if current_w is not None else None
            target_series = _allocate(strategy, window, config, prev_series)
            target_w = target_series.reindex(assets).to_numpy(dtype=float)

            if current_w is None:
                # Initial portfolio funding
                turnover = 0.0
            else:
                turnover = 0.5 * float(np.sum(np.abs(target_w - current_w)))
                rebal_turnovers.append(turnover)

            current_w = target_w.copy()
            pending_turnover = turnover
            weights_by_date[pd.Timestamp(dates[idx])] = pd.Series(current_w, index=assets)

        assert current_w is not None

        # 2) Compute 1-day-ahead causal VaR/CVaR for day `idx + 1` using window up to `idx`
        hist_window = ret_mat[idx - lookback + 1 : idx + 1]
        hist_port_r = hist_window @ current_w
        var_95_hist, cvar_95_hist = historical_var_cvar(hist_port_r, alpha=config.var_alpha_95)
        var_99_hist, _ = historical_var_cvar(hist_port_r, alpha=config.var_alpha_99)

        # Parametric EWMA VaR (algebraically identical to w^T Sigma_EWMA w, O(T) per day)
        port_mu = float(hist_port_r.mean())
        centered_port = hist_port_r - port_mu
        port_ewma_vol = float(np.sqrt(max(float(np.dot(ewma_w, centered_port * centered_port)), 0.0)))
        var_95_param = max(-(port_mu + z_95 * port_ewma_vol), 0.0)

        # 3) Realise day `idx + 1` return and transaction cost
        next_r_vec = ret_mat[idx + 1]
        gross_ret = float(current_w @ next_r_vec)
        cost_frac = pending_turnover * (config.tc_bps / 10_000.0)
        net_ret = gross_ret - cost_frac

        breach_95 = int(net_ret < -var_95_hist)
        breach_99 = int(net_ret < -var_99_hist)

        records.append(
            {
                "trade_date": pd.Timestamp(dates[idx + 1]),
                "gross_return": gross_ret,
                "net_return": net_ret,
                "turnover": pending_turnover,
                "cost_bps": cost_frac * 10_000.0,
                "var_95_hist": var_95_hist,
                "var_99_hist": var_99_hist,
                "cvar_95_hist": cvar_95_hist,
                "var_95_param": var_95_param,
                "breach_95": breach_95,
                "breach_99": breach_99,
            }
        )

        # 4) Drift weights through day `idx + 1`
        pending_turnover = 0.0
        drifted = current_w * (1.0 + next_r_vec)
        total = float(drifted.sum())
        current_w = drifted / total if total > 0 else np.full(len(assets), 1.0 / len(assets))

    daily_df = pd.DataFrame(records).set_index("trade_date")

    # Summary snapshot
    nav = (1.0 + daily_df["net_return"]).cumprod()
    n_oos = len(daily_df)
    ann_return = float(nav.iloc[-1] ** (252.0 / n_oos) - 1.0)
    ann_vol = float(daily_df["net_return"].std(ddof=1) * np.sqrt(252.0))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    max_dd = float((nav / nav.cummax() - 1.0).min())
    avg_monthly_turnover = float(np.mean(rebal_turnovers)) if rebal_turnovers else 0.0

    var_bt = kupiec_and_christoffersen_tests(daily_df["breach_95"], alpha=config.var_alpha_95)

    latest_date = max(weights_by_date.keys())
    latest_w = weights_by_date[latest_date]
    effective_n = float(1.0 / np.sum(latest_w.to_numpy(dtype=float) ** 2))
    sector_series = latest_w.groupby(lambda t: config.universe.get(t, "Other")).sum()
    max_sector_weight = float(sector_series.max())

    snapshot = {
        "as_of_date": pd.Timestamp(daily_df.index[-1]).strftime("%Y-%m-%d"),
        "strategy": strategy,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "avg_monthly_turnover": avg_monthly_turnover,
        "var_95_breach_rate": var_bt.observed_rate,
        "kupiec_p_value": var_bt.kupiec_p_value,
        "christoffersen_p_value": var_bt.christoffersen_cc_p_value,
        "effective_n": effective_n,
        "max_sector_weight": max_sector_weight,
    }

    return StrategyResult(
        strategy=strategy,
        daily=daily_df,
        weights_by_date=weights_by_date,
        snapshot=snapshot,
    )


def run_all_strategies(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    config: BacktestConfig,
    db: RiskBookDB,
    strategies: tuple[str, ...] = STRATEGIES,
) -> dict[str, StrategyResult]:
    """Run all configured strategies and persist full ledger to SQLite."""
    db.init_schema()
    db.upsert_securities(config.universe, config.benchmark)
    db.write_daily_prices(prices, returns)

    asset_cols = [t for t in sorted(config.universe.keys()) if t in returns.columns]
    asset_returns = returns[asset_cols]

    results: dict[str, StrategyResult] = {}
    for strat in strategies:
        res = run_strategy_walkforward(strat, asset_returns, config)
        db.write_rebalance_weights(strat, res.weights_by_date)
        db.write_daily_portfolio_returns(strat, res.daily)
        db.write_risk_snapshot(res.snapshot)
        results[strat] = res

    return results
