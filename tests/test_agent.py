from riskbook.agent import RiskAuditAgent, ToolLedger, verify_memo_grounding
from riskbook.backtest import run_all_strategies
from riskbook.config import BacktestConfig, RiskLimits
from riskbook.data import compute_returns, generate_synthetic_prices
from riskbook.db import RiskBookDB


def test_agent_audit_detects_breaches_and_verifies_grounding():
    universe = {f"EQ{i}": f"Sector{i % 4}" for i in range(12)}
    tickers = sorted(universe.keys()) + ["SPY"]
    sectors = {**universe, "SPY": "Benchmark"}
    prices = generate_synthetic_prices(tickers, sectors, n_days=360, seed=31)
    returns = compute_returns(prices)

    cfg = BacktestConfig(
        lookback_days=120,
        max_position_weight=0.20,
        max_sector_weight=0.40,
        universe=universe,
        benchmark="SPY",
    )
    # Intentionally tight limits so at least one finding is triggered
    tight_limits = RiskLimits(
        max_sector_weight=0.22,
        min_effective_n=11.5,
        max_20d_breaches=1,
        max_drawdown_alert=-0.05,
    )

    with RiskBookDB(":memory:") as db:
        run_all_strategies(prices, returns, cfg, db, strategies=("equal_weight", "mv_ledoit_wolf"))
        agent = RiskAuditAgent(db=db, returns=returns, config=cfg, limits=tight_limits)
        report = agent.run()
        persisted = db.query_df("SELECT * FROM audit_findings")

    assert report.grounded is True
    assert report.ungrounded_claims == []
    assert len(report.findings) > 0
    assert len(persisted) == len(report.findings)
    assert len(report.tool_trace) >= 3


def test_verifier_rejects_hallucinated_numbers():
    ledger = ToolLedger()
    ledger.record("query_risk_snapshots", {}, [{"sharpe": 1.15, "ann_vol": 0.142}])

    good_memo = "Portfolio Sharpe is [metric: 1.15] with annualised vol [metric: 14.2%]."
    ok, bad_claims = verify_memo_grounding(good_memo, ledger)
    assert ok is True
    assert bad_claims == []

    hallucinated_memo = "Portfolio Sharpe is [metric: 3.85] with annualised vol [metric: 14.2%]."
    ok2, bad_claims2 = verify_memo_grounding(hallucinated_memo, ledger)
    assert ok2 is False
    assert "[metric: 3.85]" in bad_claims2
