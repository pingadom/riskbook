"""SQLite ledger management and reusable SQL analytics queries."""

from __future__ import annotations

from pathlib import Path
import re
import sqlite3
from typing import Any

import pandas as pd


def _default_sql_dir() -> Path:
    # Project root / sql
    return Path(__file__).resolve().parents[2] / "sql"


def load_named_queries(analytics_path: Path | None = None) -> dict[str, str]:
    """Parse ``-- name: <query_name>`` blocks from sql/analytics.sql."""
    path = analytics_path or (_default_sql_dir() / "analytics.sql")
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"--\s*name:\s*([a-zA-Z0-9_]+)\s*\n(.*?)(?=(?:--\s*name:)|\Z)", re.DOTALL)
    queries: dict[str, str] = {}
    for match in pattern.finditer(text):
        queries[match.group(1).strip()] = match.group(2).strip()
    return queries


class RiskBookDB:
    """Thin, explicit SQLite ledger for prices, rebalance weights, risk snapshots, and audit findings."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        if str(db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._queries = load_named_queries()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "RiskBookDB":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def init_schema(self, schema_path: Path | None = None) -> None:
        path = schema_path or (_default_sql_dir() / "schema.sql")
        self.conn.executescript(path.read_text(encoding="utf-8"))
        self.conn.commit()

    def upsert_securities(self, universe: dict[str, str], benchmark: str) -> None:
        rows = [(ticker, sector, 0) for ticker, sector in universe.items()]
        rows.append((benchmark, "Benchmark", 1))
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO securities (ticker, sector, is_benchmark)
            VALUES (?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def write_daily_prices(self, prices: pd.DataFrame, returns: pd.DataFrame) -> None:
        rows = []
        for dt in prices.index:
            dt_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
            has_ret = dt in returns.index
            for ticker in prices.columns:
                px = float(prices.loc[dt, ticker])
                ret = float(returns.loc[dt, ticker]) if has_ret else None
                rows.append((dt_str, ticker, px, ret))
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO daily_prices (trade_date, ticker, adj_close, daily_return)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def write_rebalance_weights(
        self,
        strategy: str,
        weights_by_date: dict[pd.Timestamp, pd.Series],
    ) -> None:
        rows = []
        for dt, weights in weights_by_date.items():
            dt_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
            for ticker, w in weights.items():
                rows.append((dt_str, strategy, str(ticker), float(w)))
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO rebalance_weights (rebalance_date, strategy, ticker, weight)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def write_daily_portfolio_returns(self, strategy: str, df: pd.DataFrame) -> None:
        rows = []
        for dt, row in df.iterrows():
            rows.append(
                (
                    pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    strategy,
                    float(row["gross_return"]),
                    float(row["net_return"]),
                    float(row["turnover"]),
                    float(row["cost_bps"]),
                    None if pd.isna(row["var_95_hist"]) else float(row["var_95_hist"]),
                    None if pd.isna(row["var_99_hist"]) else float(row["var_99_hist"]),
                    None if pd.isna(row["cvar_95_hist"]) else float(row["cvar_95_hist"]),
                    None if pd.isna(row["var_95_param"]) else float(row["var_95_param"]),
                    int(row["breach_95"]),
                    int(row["breach_99"]),
                )
            )
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO daily_portfolio_returns (
                trade_date, strategy, gross_return, net_return, turnover, cost_bps,
                var_95_hist, var_99_hist, cvar_95_hist, var_95_param,
                breach_95, breach_99
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def write_risk_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO risk_snapshots (
                as_of_date, strategy, ann_return, ann_vol, sharpe, max_drawdown,
                avg_monthly_turnover, var_95_breach_rate, kupiec_p_value,
                christoffersen_p_value, effective_n, max_sector_weight
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot["as_of_date"],
                snapshot["strategy"],
                float(snapshot["ann_return"]),
                float(snapshot["ann_vol"]),
                float(snapshot["sharpe"]),
                float(snapshot["max_drawdown"]),
                float(snapshot["avg_monthly_turnover"]),
                float(snapshot["var_95_breach_rate"]),
                float(snapshot["kupiec_p_value"]),
                float(snapshot["christoffersen_p_value"]),
                float(snapshot["effective_n"]),
                float(snapshot["max_sector_weight"]),
            ),
        )
        self.conn.commit()

    def write_audit_findings(self, findings: list[dict[str, Any]]) -> None:
        rows = [
            (
                f["as_of_date"],
                f["strategy"],
                f["severity"],
                f["check_name"],
                float(f["metric_value"]),
                float(f["threshold_value"]),
                f["detail"],
                f["evidence_hash"],
            )
            for f in findings
        ]
        self.conn.executemany(
            """
            INSERT INTO audit_findings (
                as_of_date, strategy, severity, check_name,
                metric_value, threshold_value, detail, evidence_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def run_named_query(self, name: str) -> pd.DataFrame:
        if name not in self._queries:
            raise KeyError(f"Unknown SQL analytics query: {name}")
        return pd.read_sql_query(self._queries[name], self.conn)

    def query_df(self, sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.conn, params=params)
