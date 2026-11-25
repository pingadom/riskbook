"""Configuration, equities universe, and constraint defaults for riskbook."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# 30 liquid US large-cap equities spanning all 11 GICS sectors.
# Selected for continuous daily history from 2018-01-01 onward.
EQUITY_UNIVERSE: dict[str, str] = {
    # Information Technology
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "AVGO": "Technology",
    # Health Care
    "JNJ": "Health Care",
    "UNH": "Health Care",
    "LLY": "Health Care",
    # Financials
    "JPM": "Financials",
    "BAC": "Financials",
    "BRK-B": "Financials",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    # Communication Services
    "GOOGL": "Communication Services",
    "META": "Communication Services",
    "DIS": "Communication Services",
    # Industrials
    "CAT": "Industrials",
    "HON": "Industrials",
    "UNP": "Industrials",
    # Consumer Staples
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "WMT": "Consumer Staples",
    # Energy
    "XOM": "Energy",
    "CVX": "Energy",
    # Utilities
    "NEE": "Utilities",
    "DUK": "Utilities",
    # Real Estate
    "AMT": "Real Estate",
    "PLD": "Real Estate",
    # Materials
    "LIN": "Materials",
    "SHW": "Materials",
}

BENCHMARK_TICKER = "SPY"

# Historical stress episodes for scenario analysis
STRESS_SCENARIOS: dict[str, tuple[str, str]] = {
    "COVID-19 Liquidity Shock (2020)": ("2020-02-19", "2020-03-23"),
    "Fed Rate-Hike Drawdown (2022)": ("2022-01-03", "2022-10-12"),
    "SVB Regional Banking Stress (2023)": ("2023-03-08", "2023-03-24"),
    "Aug 2024 Yen-Carry Vol Spike": ("2024-07-31", "2024-08-05"),
}


@dataclass(frozen=True)
class BacktestConfig:
    start_date: str = "2018-01-01"
    end_date: str = "2025-12-31"
    lookback_days: int = 252
    rebalance_freq: str = "ME"  # Month-end rebalance
    max_position_weight: float = 0.12
    max_sector_weight: float = 0.28
    tc_bps: float = 5.0  # One-way transaction cost in basis points
    turnover_penalty: float = 0.0015
    ewma_lambda: float = 0.94
    var_alpha_95: float = 0.05
    var_alpha_99: float = 0.01
    mc_simulations: int = 10_000
    seed: int = 42
    universe: dict[str, str] = field(default_factory=lambda: dict(EQUITY_UNIVERSE))
    benchmark: str = BENCHMARK_TICKER
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")


@dataclass(frozen=True)
class RiskLimits:
    max_single_weight: float = 0.12
    max_sector_weight: float = 0.28
    min_effective_n: float = 10.0
    max_var_95_breach_rate: float = 0.075
    min_kupiec_p_value: float = 0.05
    max_20d_breaches: int = 3
    max_drawdown_alert: float = -0.20
