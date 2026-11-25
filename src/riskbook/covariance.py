"""Covariance estimators: Sample, analytical Ledoit-Wolf (2004) shrinkage, and EWMA."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ensure_psd(cov: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    """Symmetrise and clip tiny negative numerical eigenvalues."""
    sym = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals, floor)
    return (eigvecs * eigvals) @ eigvecs.T


def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Unbiased sample covariance matrix (ddof=1)."""
    cov = np.cov(returns.to_numpy(dtype=float), rowvar=False, ddof=1)
    cov = _ensure_psd(np.atleast_2d(cov))
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def ledoit_wolf_covariance(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Analytical Ledoit & Wolf (2004) linear shrinkage toward scaled identity mu * I.

    Implemented from first principles (matches sklearn.covariance.LedoitWolf):
        Sigma_hat = (1 - delta) * S + delta * mu * I
    where S = (1/T) X^T X on demeaned returns X.

    Returns
    -------
    (shrunk_cov_df, shrinkage_intensity)
    """
    x = returns.to_numpy(dtype=float)
    t_obs, n_assets = x.shape
    if t_obs < 2:
        raise ValueError("Need at least 2 observations to estimate covariance.")

    x_centered = x - x.mean(axis=0, keepdims=True)
    # Sample covariance with 1/T scaling per Ledoit & Wolf (2004)
    sample = (x_centered.T @ x_centered) / t_obs
    mu = float(np.trace(sample)) / n_assets
    target = mu * np.eye(n_assets)

    # Squared Frobenius distance between sample cov and target: ||S - mu I||_F^2
    delta_sq = float(np.sum((sample - target) ** 2))
    if delta_sq <= 1e-18:
        return pd.DataFrame(target, index=returns.columns, columns=returns.columns), 1.0

    # Estimate beta^2 = (1 / T^2) sum_t ||x_t x_t^T - S||_F^2
    x2 = x_centered ** 2
    # sum_t ||x_t x_t^T||_F^2 = sum_t (x_t^T x_t)^2
    row_norm_sq = np.sum(x2, axis=1)
    sum_outer_sq = float(np.sum(row_norm_sq ** 2))
    # ||S||_F^2
    sample_frob_sq = float(np.sum(sample ** 2))
    beta_bar_sq = (sum_outer_sq / t_obs - sample_frob_sq) / t_obs

    beta_sq = min(max(beta_bar_sq, 0.0), delta_sq)
    shrinkage = float(np.clip(beta_sq / delta_sq, 0.0, 1.0))

    shrunk = (1.0 - shrinkage) * sample + shrinkage * target
    shrunk = _ensure_psd(shrunk)
    return pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns), shrinkage


def ewma_covariance(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """RiskMetrics-style exponentially weighted moving average covariance."""
    if not (0.0 < lam < 1.0):
        raise ValueError("lam must lie strictly in (0, 1).")

    x = returns.to_numpy(dtype=float)
    t_obs = x.shape[0]
    x_centered = x - x.mean(axis=0, keepdims=True)

    # Most recent observation gets the highest weight
    exponents = np.arange(t_obs - 1, -1, -1, dtype=float)
    raw_weights = (1.0 - lam) * (lam ** exponents)
    weights = raw_weights / raw_weights.sum()

    weighted_x = x_centered * weights[:, None]
    cov = weighted_x.T @ x_centered
    cov = _ensure_psd(cov)
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
