"""Risk estimation, Euler risk decomposition, VaR backtesting, and historical stress testing.

Conventions:
- VaR and CVaR are returned as positive loss magnitudes (e.g. 0.018 means a 1.8% 1-day loss).
- A VaR breach occurs on day t when realised portfolio return r_t < -VaR_t.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


@dataclass(frozen=True)
class VaRBacktestResult:
    n_obs: int
    n_breaches: int
    expected_rate: float
    observed_rate: float
    kupiec_lr: float
    kupiec_p_value: float
    christoffersen_ind_lr: float
    christoffersen_ind_p_value: float
    christoffersen_cc_lr: float
    christoffersen_cc_p_value: float


def historical_var_cvar(portfolio_returns: np.ndarray | pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    """Historical (empirical) 1-day VaR and CVaR as positive loss fractions."""
    r = np.asarray(portfolio_returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return float("nan"), float("nan")
    q = float(np.quantile(r, alpha))
    tail = r[r <= q]
    var = max(-q, 0.0)
    cvar = max(-float(tail.mean()) if len(tail) > 0 else var, var)
    return var, cvar


def parametric_var_cvar(
    mean: np.ndarray | pd.Series,
    cov: np.ndarray | pd.DataFrame,
    weights: np.ndarray | pd.Series,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Gaussian parametric 1-day VaR and CVaR as positive loss fractions."""
    mu = np.asarray(mean, dtype=float)
    sigma = np.asarray(cov, dtype=float)
    w = np.asarray(weights, dtype=float)

    port_mu = float(w @ mu)
    port_vol = float(np.sqrt(max(float(w @ sigma @ w), 0.0)))
    if port_vol == 0.0:
        return max(-port_mu, 0.0), max(-port_mu, 0.0)

    z = float(norm.ppf(alpha))  # negative for alpha < 0.5
    var = -(port_mu + z * port_vol)
    cvar = -(port_mu - port_vol * float(norm.pdf(z)) / alpha)
    return max(var, 0.0), max(cvar, var)


def monte_carlo_var_cvar(
    mean: np.ndarray | pd.Series,
    cov: np.ndarray | pd.DataFrame,
    weights: np.ndarray | pd.Series,
    alpha: float = 0.05,
    n_sims: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Multivariate Gaussian Monte Carlo 1-day VaR and CVaR."""
    mu = np.asarray(mean, dtype=float)
    sigma = np.asarray(cov, dtype=float)
    w = np.asarray(weights, dtype=float)

    rng = np.random.default_rng(seed)
    # Regularise slightly for Cholesky stability
    sym = 0.5 * (sigma + sigma.T) + 1e-12 * np.eye(len(mu))
    chol = np.linalg.cholesky(sym)
    z = rng.standard_normal(size=(n_sims, len(mu)))
    sim_asset_returns = mu[None, :] + z @ chol.T
    sim_port_returns = sim_asset_returns @ w
    return historical_var_cvar(sim_port_returns, alpha=alpha)


def euler_risk_contributions(cov: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Exact Euler decomposition of portfolio volatility.

    Satisfies:
        sum_i component_risk_i == portfolio_vol
        sum_i pct_risk_i == 1.0
    """
    assets = list(cov.columns)
    w = weights.reindex(assets).fillna(0.0).to_numpy(dtype=float)
    sigma = cov.to_numpy(dtype=float)

    port_var = float(w @ sigma @ w)
    port_vol = float(np.sqrt(max(port_var, 1e-18)))
    mrc = (sigma @ w) / port_vol
    rc = w * mrc
    pct_rc = rc / port_vol

    return pd.DataFrame(
        {
            "weight": w,
            "marginal_risk": mrc,
            "component_risk": rc,
            "pct_risk": pct_rc,
        },
        index=assets,
    )


def _safe_log_term(count: int, prob: float) -> float:
    if count == 0:
        return 0.0
    p = min(max(prob, 1e-15), 1.0 - 1e-15)
    return count * float(np.log(p))


def kupiec_and_christoffersen_tests(
    breaches: np.ndarray | pd.Series,
    alpha: float = 0.05,
) -> VaRBacktestResult:
    """Kupiec (1995) POF unconditional coverage + Christoffersen (1998) independence & CC tests."""
    b = np.asarray(breaches, dtype=int)
    t_obs = len(b)
    x = int(b.sum())
    p_hat = x / t_obs if t_obs > 0 else 0.0

    # Kupiec POF: LR_pof = -2 * ln( L(alpha) / L(p_hat) )
    ll_null = _safe_log_term(t_obs - x, 1.0 - alpha) + _safe_log_term(x, alpha)
    ll_alt = _safe_log_term(t_obs - x, 1.0 - p_hat) + _safe_log_term(x, p_hat)
    lr_pof = max(-2.0 * (ll_null - ll_alt), 0.0)
    p_pof = float(1.0 - chi2.cdf(lr_pof, df=1))

    # Christoffersen Markov independence test on consecutive transitions
    if t_obs >= 2:
        prev = b[:-1]
        curr = b[1:]
        n00 = int(np.sum((prev == 0) & (curr == 0)))
        n01 = int(np.sum((prev == 0) & (curr == 1)))
        n10 = int(np.sum((prev == 1) & (curr == 0)))
        n11 = int(np.sum((prev == 1) & (curr == 1)))

        pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
        pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
        pi = (n01 + n11) / (n00 + n01 + n10 + n11)

        ll_ind_null = _safe_log_term(n00 + n10, 1.0 - pi) + _safe_log_term(n01 + n11, pi)
        ll_ind_alt = (
            _safe_log_term(n00, 1.0 - pi0)
            + _safe_log_term(n01, pi0)
            + _safe_log_term(n10, 1.0 - pi1)
            + _safe_log_term(n11, pi1)
        )
        lr_ind = max(-2.0 * (ll_ind_null - ll_ind_alt), 0.0)
    else:
        lr_ind = 0.0

    p_ind = float(1.0 - chi2.cdf(lr_ind, df=1))
    lr_cc = lr_pof + lr_ind
    p_cc = float(1.0 - chi2.cdf(lr_cc, df=2))

    return VaRBacktestResult(
        n_obs=t_obs,
        n_breaches=x,
        expected_rate=alpha,
        observed_rate=p_hat,
        kupiec_lr=lr_pof,
        kupiec_p_value=p_pof,
        christoffersen_ind_lr=lr_ind,
        christoffersen_ind_p_value=p_ind,
        christoffersen_cc_lr=lr_cc,
        christoffersen_cc_p_value=p_cc,
    )


def evaluate_stress_scenarios(
    returns: pd.DataFrame,
    weights: pd.Series,
    scenarios: dict[str, tuple[str, str]],
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Apply fixed portfolio weights over historical stress episodes."""
    assets = list(weights.index)
    w = weights.to_numpy(dtype=float)
    rows = []

    for name, (start, end) in scenarios.items():
        window = returns.loc[start:end, assets]
        if window.empty:
            continue
        port_r = pd.Series(window.to_numpy(dtype=float) @ w, index=window.index)
        nav = (1.0 + port_r).cumprod()
        cum_ret = float(nav.iloc[-1] - 1.0)
        dd = float((nav / nav.cummax() - 1.0).min())
        worst_day = float(port_r.min())

        bench_cum = float("nan")
        if benchmark_returns is not None:
            b_win = benchmark_returns.loc[start:end]
            if not b_win.empty:
                bench_cum = float((1.0 + b_win).prod() - 1.0)

        rows.append(
            {
                "scenario": name,
                "start": start,
                "end": end,
                "n_days": len(port_r),
                "cum_return": cum_ret,
                "max_drawdown": dd,
                "worst_day": worst_day,
                "benchmark_return": bench_cum,
                "relative_return": cum_ret - bench_cum if not np.isnan(bench_cum) else float("nan"),
            }
        )
    return pd.DataFrame(rows)
