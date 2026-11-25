"""Command-line entry point for running the riskbook pipeline and grounded risk audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from riskbook.agent import RiskAuditAgent
from riskbook.backtest import run_all_strategies
from riskbook.config import BacktestConfig, RiskLimits
from riskbook.data import compute_returns, fetch_equities_prices
from riskbook.db import RiskBookDB


def run_pipeline(config: BacktestConfig, use_cache: bool = True) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    db_path = config.output_dir / "riskbook.sqlite"

    prices = fetch_equities_prices(config, use_cache=use_cache)
    returns = compute_returns(prices)

    with RiskBookDB(db_path) as db:
        results = run_all_strategies(prices, returns, config, db)
        agent = RiskAuditAgent(db=db, returns=returns, config=config, limits=RiskLimits())
        report = agent.run()

        snapshots_df = db.query_df("SELECT * FROM risk_snapshots ORDER BY sharpe DESC")
        sector_df = db.run_named_query("sector_exposures")

    memo_path = config.output_dir / "risk_audit_memo.md"
    memo_path.write_text(report.memo_markdown, encoding="utf-8")

    summary = {
        "as_of_date": report.as_of_date,
        "n_equities": len(config.universe),
        "n_trading_days": int(len(returns)),
        "grounded": report.grounded,
        "ungrounded_claims": report.ungrounded_claims,
        "n_findings": len(report.findings),
        "n_tool_calls": len(report.tool_trace),
        "snapshots": {k: v.snapshot for k, v in results.items()},
    }
    summary_bytes = json.dumps(summary, indent=2, sort_keys=True).encode("utf-8")
    summary["artifact_sha256"] = hashlib.sha256(summary_bytes).hexdigest()[:16]

    summary_path = config.output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== Walk-Forward Out-of-Sample Summary ===")
    print(snapshots_df.to_string(index=False))
    print(f"\nAudit memo written to: {memo_path} (grounded={report.grounded}, tool_calls={len(report.tool_trace)}, findings={len(report.findings)})")
    print(f"SQLite ledger written to: {db_path}")
    print(f"Top sector exposures rows: {len(sector_df)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the riskbook equities walk-forward & audit pipeline.")
    parser.add_argument("--no-cache", action="store_true", help="Force fresh market data download.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory for DB and reports.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory for cached market data.")
    args = parser.parse_args()

    cfg = BacktestConfig(data_dir=args.data_dir, output_dir=args.output_dir)
    run_pipeline(cfg, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
