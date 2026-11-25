"""Market data ingestion with local CSV caching and deterministic synthetic generator for tests."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from riskbook.config import BacktestConfig


def fetch_equities_prices(
    config: BacktestConfig,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download adjusted close prices for the configured equity universe + benchmark via yfinance.

    Caches locally as CSV under ``config.data_dir / 'prices.csv'`` for reproducible reruns.
    """
    cache_path = Path(config.data_dir) / "prices.csv"
    tickers = sorted(set(config.universe.keys()) | {config.benchmark})

    if use_cache and cache_path.exists():
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if set(tickers).issubset(set(cached.columns)):
            return cached.loc[config.start_date : config.end_date, tickers].dropna(how="any")

    import yfinance as yf

    raw = yf.download(
        tickers=tickers,
        start=config.start_date,
        end=config.end_date,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()
    # Forward-fill single-day exchange holidays if any, then drop leading NaNs
    prices = prices.ffill().dropna(how="any")
    prices = prices[tickers]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache_path)
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns from adjusted close prices."""
    return prices.pct_change().dropna(how="any")


def generate_synthetic_prices(
    tickers: list[str],
    sectors: dict[str, str],
    n_days: int = 800,
    seed: int = 42,
    start_date: str = "2020-01-02",
) -> pd.DataFrame:
    """Generate a reproducible factor-structured equity price panel for unit tests.

    Includes a common market factor, sector factors, idiosyncratic noise, and a
    volatility-clustering shock regime so covariance & VaR tests exercise realistic structure.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    unique_sectors = sorted(set(sectors[t] for t in tickers))
    sector_to_idx = {s: i for i, s in enumerate(unique_sectors)}

    # Market factor with a 40-day high-vol stress regime in the middle
    vol_regime = np.ones(n_days)
    vol_regime[n_days // 2 : n_days // 2 + 40] = 2.6
    mkt = rng.normal(0.0003, 0.009, size=n_days) * vol_regime
    mkt[n_days // 2 : n_days // 2 + 15] -= 0.012

    sec_factors = rng.normal(0.0, 0.006, size=(n_days, len(unique_sectors))) * vol_regime[:, None]

    returns = np.zeros((n_days, len(tickers)))
    for j, t in enumerate(tickers):
        beta = rng.uniform(0.6, 1.4)
        sec_beta = rng.uniform(0.4, 0.9)
        idio_vol = rng.uniform(0.006, 0.014)
        sec_idx = sector_to_idx[sectors[t]]
        returns[:, j] = (
            beta * mkt
            + sec_beta * sec_factors[:, sec_idx]
            + rng.normal(0.0, idio_vol, size=n_days) * vol_regime
        )

    prices = 100.0 * np.cumprod(1.0 + returns, axis=0)
    return pd.DataFrame(prices, index=dates, columns=tickers)
