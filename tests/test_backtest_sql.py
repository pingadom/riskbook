import numpy as np
import pandas as pd

from riskbook.backtest import run_all_strategies, run_strategy_walkforward
from riskbook.config import BacktestConfig
from riskbook.data import compute_returns, generate_synthetic_prices
from riskbook.db import RiskBookDB


def _mini_universe():
    universe = {f"EQ{i}": f"Sector{i % 4}" for i in range(12)}
    return universe


def test_walkforward_is_strictly_causal_no_lookahead():
    universe = _mini_universe()
    tickers = sorted(universe.keys())
    prices = generate_synthetic_prices(tickers, universe, n_days=420, seed=11)
    returns = compute_returns(prices)

    cfg = BacktestConfig(
        lookback_days=120,
        max_position_weight=0.20,
        max_sector_weight=0.40,
        universe=universe,
    )

    res_orig = run_strategy_walkforward("mv_ledoit_wolf", returns, cfg)

    # Perturb returns strictly AFTER the 3rd rebalance date
    rebal_dates = sorted(res_orig.weights_by_date.keys())
    cutoff = rebal_dates[2]
    perturbed = returns.copy()
    perturbed.loc[perturbed.index > cutoff] *= -3.5

    res_pert = run_strategy_walkforward("mv_ledoit_wolf", perturbed, cfg)

    for dt in rebal_dates[:3]:
        assert np.allclose(
            res_orig.weights_by_date[dt].to_numpy(),
            res_pert.weights_by_date[dt].to_numpy(),
            atol=1e-12,
        )


def test_sql_window_analytics_match_pandas():
    universe = _mini_universe()
    tickers = sorted(universe.keys()) + ["SPY"]
    sectors = {**universe, "SPY": "Benchmark"}
    prices = generate_synthetic_prices(tickers, sectors, n_days=350, seed=21)
    returns = compute_returns(prices)

    cfg = BacktestConfig(
        lookback_days=120,
        max_position_weight=0.20,
        max_sector_weight=0.40,
        universe=universe,
        benchmark="SPY",
    )

    with RiskBookDB(":memory:") as db:
        results = run_all_strategies(
            prices,
            returns,
            cfg,
            db,
            strategies=("equal_weight", "mv_ledoit_wolf", "hrp"),
        )
        dd_sql = db.run_named_query("rolling_drawdowns")
        sec_sql = db.run_named_query("sector_exposures")
        conc_sql = db.run_named_query("concentration_by_rebalance")

    assert not sec_sql.empty
    assert not conc_sql.empty

    # Verify SQL rolling drawdown matches pandas cummax calculation
    for strat in ("equal_weight", "mv_ledoit_wolf", "hrp"):
        nav = (1.0 + results[strat].daily["net_return"]).cumprod()
        pd_dd = (nav / nav.cummax() - 1.0).min()
        sql_dd = dd_sql.loc[dd_sql["strategy"] == strat, "drawdown"].min()
        assert np.isclose(sql_dd, pd_dd, atol=1e-5)
