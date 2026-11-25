"""Portfolio allocators with realistic position, sector, and turnover constraints.

Includes:
- Equal Weight (1/N baseline)
- Constrained Minimum Variance (Sample or Ledoit-Wolf covariance, optional L2 turnover penalty)
- Hierarchical Risk Parity (Lopez de Prado, 2016) with constraint projection
- Exact CVaR (Expected Shortfall) Minimisation via Rockafellar-Uryasev (2000) linear program
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import squareform


def _sector_groups(assets: list[str], sectors: dict[str, str] | None) -> dict[str, list[int]]:
    if not sectors:
        return {}
    groups: dict[str, list[int]] = {}
    for idx, ticker in enumerate(assets):
        sec = sectors.get(ticker, "Other")
        groups.setdefault(sec, []).append(idx)
    return groups


def _project_weights(
    raw_weights: np.ndarray,
    assets: list[str],
    max_weight: float = 1.0,
    sectors: dict[str, str] | None = None,
    max_sector_weight: float = 1.0,
) -> np.ndarray:
    """Euclidean projection onto the simplex with box and sector-cap constraints."""
    n = len(assets)
    w0 = np.asarray(raw_weights, dtype=float)
    w0 = np.maximum(w0, 0.0)
    if w0.sum() > 0:
        w0 = w0 / w0.sum()
    else:
        w0 = np.full(n, 1.0 / n)

    groups = _sector_groups(assets, sectors)
    # Fast path if already feasible
    if np.all(w0 <= max_weight + 1e-9) and all(
        w0[idxs].sum() <= max_sector_weight + 1e-9 for idxs in groups.values()
    ):
        return w0

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    for idxs in groups.values():
        constraints.append(
            {"type": "ineq", "fun": lambda w, i=idxs: max_sector_weight - np.sum(w[i])}
        )

    res = minimize(
        lambda w: 0.5 * float(np.sum((w - w0) ** 2)),
        x0=np.full(n, 1.0 / n),
        jac=lambda w: w - w0,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500},
    )
    w = np.clip(res.x, 0.0, max_weight)
    return w / w.sum()


def equal_weight(assets: list[str]) -> pd.Series:
    """1/N equal-weight benchmark allocation."""
    n = len(assets)
    return pd.Series(np.full(n, 1.0 / n), index=assets, dtype=float)


def min_variance(
    cov: pd.DataFrame,
    max_weight: float = 1.0,
    sectors: dict[str, str] | None = None,
    max_sector_weight: float = 1.0,
    prev_weights: pd.Series | None = None,
    turnover_penalty: float = 0.0,
) -> pd.Series:
    """Long-only minimum-variance portfolio with box, sector, and optional turnover penalty.

    Minimises:
        0.5 * w^T Sigma w + 0.5 * lambda_tc * ||w - w_prev||_2^2
    subject to:
        sum(w) = 1, 0 <= w_i <= max_weight, sum_{i in s} w_i <= max_sector_weight.
    """
    assets = list(cov.columns)
    n = len(assets)
    sigma = cov.to_numpy(dtype=float)

    if prev_weights is not None and turnover_penalty > 0.0:
        w_prev = prev_weights.reindex(assets).fillna(0.0).to_numpy(dtype=float)
    else:
        w_prev = None

    def obj(w: np.ndarray) -> float:
        val = 0.5 * float(w @ sigma @ w)
        if w_prev is not None and turnover_penalty > 0.0:
            diff = w - w_prev
            val += 0.5 * turnover_penalty * float(diff @ diff)
        return val

    def grad(w: np.ndarray) -> np.ndarray:
        g = sigma @ w
        if w_prev is not None and turnover_penalty > 0.0:
            g = g + turnover_penalty * (w - w_prev)
        return g

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    for idxs in _sector_groups(assets, sectors).values():
        constraints.append(
            {"type": "ineq", "fun": lambda w, i=idxs: max_sector_weight - np.sum(w[i])}
        )

    x0 = w_prev if w_prev is not None else np.full(n, 1.0 / n)
    res = minimize(
        obj,
        x0=x0,
        jac=grad,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 600},
    )
    if not res.success:
        w = _project_weights(np.full(n, 1.0 / n), assets, max_weight, sectors, max_sector_weight)
    else:
        w = np.clip(res.x, 0.0, max_weight)
        w = w / w.sum()
    return pd.Series(w, index=assets, dtype=float)


def _cluster_variance(cov: np.ndarray, indices: list[int]) -> float:
    sub_cov = cov[np.ix_(indices, indices)]
    diag = np.maximum(np.diag(sub_cov), 1e-12)
    ivp = 1.0 / diag
    ivp /= ivp.sum()
    return float(ivp @ sub_cov @ ivp)


def hrp_weights(
    cov: pd.DataFrame,
    max_weight: float = 1.0,
    sectors: dict[str, str] | None = None,
    max_sector_weight: float = 1.0,
) -> pd.Series:
    """Hierarchical Risk Parity (Lopez de Prado, 2016) with optional constraint projection."""
    assets = list(cov.columns)
    n = len(assets)
    sigma = cov.to_numpy(dtype=float)

    std = np.sqrt(np.maximum(np.diag(sigma), 1e-12))
    corr = sigma / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    dist = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")
    sort_ix = list(leaves_list(link))

    weights = np.ones(n, dtype=float)
    clusters: list[list[int]] = [sort_ix]

    while clusters:
        next_clusters: list[list[int]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            left = cluster[:split]
            right = cluster[split:]
            var_l = _cluster_variance(sigma, left)
            var_r = _cluster_variance(sigma, right)
            alpha = 1.0 - var_l / (var_l + var_r)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            if len(left) > 1:
                next_clusters.append(left)
            if len(right) > 1:
                next_clusters.append(right)
        clusters = next_clusters

    weights = weights / weights.sum()
    weights = _project_weights(weights, assets, max_weight, sectors, max_sector_weight)
    return pd.Series(weights, index=assets, dtype=float)


def cvar_optimise(
    returns: pd.DataFrame,
    alpha: float = 0.05,
    max_weight: float = 1.0,
    sectors: dict[str, str] | None = None,
    max_sector_weight: float = 1.0,
    prev_weights: pd.Series | None = None,
    turnover_penalty: float = 0.0,
) -> pd.Series:
    """Exact CVaR (Expected Shortfall) minimisation via Rockafellar-Uryasev (2000) LP.

    Decision variables:
        w in R^N (portfolio weights)
        zeta in R (VaR threshold)
        u in R^T_+ (tail loss excesses: u_t >= -r_t^T w - zeta)
        [optional] d in R^N_+ (L1 turnover slacks: d_i >= w_i - w_prev_i, d_i >= w_prev_i - w_i)

    Minimises:
        zeta + (1 / (alpha * T)) * sum_t u_t + turnover_penalty * sum_i d_i
    """
    assets = list(returns.columns)
    r = returns.to_numpy(dtype=float)
    t_obs, n = r.shape
    use_tc = prev_weights is not None and turnover_penalty > 0.0

    # Variable layout: [w (n), zeta (1), u (t_obs), d (n if use_tc else 0)]
    n_vars = n + 1 + t_obs + (n if use_tc else 0)
    c = np.zeros(n_vars, dtype=float)
    c[n] = 1.0
    c[n + 1 : n + 1 + t_obs] = 1.0 / (alpha * t_obs)
    if use_tc:
        c[n + 1 + t_obs :] = turnover_penalty

    # Inequality constraints: A_ub @ x <= b_ub
    ub_rows: list[np.ndarray] = []
    ub_rhs: list[float] = []

    # 1) Tail loss constraints: -r_t^T w - zeta - u_t <= 0
    a_tail = np.zeros((t_obs, n_vars), dtype=float)
    a_tail[:, :n] = -r
    a_tail[:, n] = -1.0
    a_tail[np.arange(t_obs), n + 1 + np.arange(t_obs)] = -1.0
    b_tail = np.zeros(t_obs, dtype=float)

    # 2) Sector caps: sum_{i in s} w_i <= max_sector_weight
    for idxs in _sector_groups(assets, sectors).values():
        row = np.zeros(n_vars, dtype=float)
        row[idxs] = 1.0
        ub_rows.append(row)
        ub_rhs.append(max_sector_weight)

    # 3) Optional L1 turnover slacks: w_i - d_i <= w_prev_i and -w_i - d_i <= -w_prev_i
    if use_tc:
        assert prev_weights is not None
        w_prev = prev_weights.reindex(assets).fillna(0.0).to_numpy(dtype=float)
        d_offset = n + 1 + t_obs
        for i in range(n):
            row_pos = np.zeros(n_vars, dtype=float)
            row_pos[i] = 1.0
            row_pos[d_offset + i] = -1.0
            ub_rows.append(row_pos)
            ub_rhs.append(float(w_prev[i]))

            row_neg = np.zeros(n_vars, dtype=float)
            row_neg[i] = -1.0
            row_neg[d_offset + i] = -1.0
            ub_rows.append(row_neg)
            ub_rhs.append(float(-w_prev[i]))

    if ub_rows:
        a_ub = np.vstack([a_tail, np.asarray(ub_rows, dtype=float)])
        b_ub = np.concatenate([b_tail, np.asarray(ub_rhs, dtype=float)])
    else:
        a_ub = a_tail
        b_ub = b_tail

    # Equality constraint: sum(w) = 1
    a_eq = np.zeros((1, n_vars), dtype=float)
    a_eq[0, :n] = 1.0
    b_eq = np.array([1.0], dtype=float)

    bounds: list[tuple[float | None, float | None]] = (
        [(0.0, max_weight)] * n
        + [(None, None)]  # zeta unrestricted
        + [(0.0, None)] * t_obs  # u >= 0
        + ([(0.0, None)] * n if use_tc else [])
    )

    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success or res.x is None:
        w = _project_weights(np.full(n, 1.0 / n), assets, max_weight, sectors, max_sector_weight)
    else:
        w = np.clip(res.x[:n], 0.0, max_weight)
        w = w / w.sum()
    return pd.Series(w, index=assets, dtype=float)
