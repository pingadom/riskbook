"""Grounded agentic risk-audit workflow with deterministic numeric verification.

Architecture:
1. RiskAuditTools: explicit read-only SQL + analytics tool surface over the SQLite ledger.
2. ToolLedger: records every tool invocation, JSON payload, and SHA-256 evidence hash.
3. RiskAuditAgent: multi-step tool-calling investigation loop that triages risk-limit
   breaches, drills into SQL breach clusters / Euler attribution / historical stress
   scenarios, persists findings to SQL, and drafts a grounded risk memo.
4. verify_memo_grounding: deterministic post-check ensuring every cited numeric value
   in a risk memo matches an observed tool output (rejecting hallucinated numbers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Callable

import pandas as pd

from riskbook.config import BacktestConfig, RiskLimits, STRESS_SCENARIOS
from riskbook.covariance import ledoit_wolf_covariance
from riskbook.db import RiskBookDB
from riskbook.risk import euler_risk_contributions, evaluate_stress_scenarios


@dataclass(frozen=True)
class ToolCallRecord:
    tool_name: str
    arguments: dict[str, Any]
    result: Any
    evidence_hash: str


@dataclass
class AuditReport:
    as_of_date: str
    findings: list[dict[str, Any]]
    tool_trace: list[ToolCallRecord]
    memo_markdown: str
    grounded: bool
    ungrounded_claims: list[str] = field(default_factory=list)


def _hash_payload(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:16]


class ToolLedger:
    """Immutable log of tool calls and their exact numeric values for grounding checks."""

    def __init__(self) -> None:
        self.records: list[ToolCallRecord] = []
        self._observed_numbers: set[float] = set()

    def record(self, tool_name: str, arguments: dict[str, Any], result: Any) -> ToolCallRecord:
        ev_hash = _hash_payload({"tool": tool_name, "args": arguments, "result": result})
        rec = ToolCallRecord(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            evidence_hash=ev_hash,
        )
        self.records.append(rec)
        self._harvest_numbers(result)
        return rec

    def _harvest_numbers(self, obj: Any) -> None:
        if isinstance(obj, bool):
            return
        if isinstance(obj, (int, float)):
            val = float(obj)
            if not pd.isna(val):
                self._observed_numbers.add(val)
                # Also allow percentage representation (e.g. 0.052 -> 5.2)
                self._observed_numbers.add(val * 100.0)
        elif isinstance(obj, dict):
            for v in obj.values():
                self._harvest_numbers(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                self._harvest_numbers(v)

    def has_number(self, value: float, abs_tol: float = 0.06, rel_tol: float = 0.015) -> bool:
        for obs in self._observed_numbers:
            if abs(value - obs) <= max(abs_tol, rel_tol * abs(obs)):
                return True
        return False


class RiskAuditTools:
    """Explicit tool surface callable by either the deterministic planner or an LLM."""

    def __init__(
        self,
        db: RiskBookDB,
        returns: pd.DataFrame,
        config: BacktestConfig,
        ledger: ToolLedger,
    ) -> None:
        self.db = db
        self.returns = returns
        self.config = config
        self.ledger = ledger

    def query_risk_snapshots(self) -> tuple[list[dict[str, Any]], str]:
        df = self.db.query_df("SELECT * FROM risk_snapshots ORDER BY strategy")
        rows = df.to_dict(orient="records")
        rec = self.ledger.record("query_risk_snapshots", {}, rows)
        return rows, rec.evidence_hash

    def query_sector_exposures(self, strategy: str) -> tuple[list[dict[str, Any]], str]:
        df = self.db.run_named_query("sector_exposures")
        sub = df[df["strategy"] == strategy]
        rows = sub.to_dict(orient="records")
        rec = self.ledger.record("query_sector_exposures", {"strategy": strategy}, rows)
        return rows, rec.evidence_hash

    def query_var_breach_clusters(self, strategy: str) -> tuple[dict[str, Any], str]:
        df = self.db.run_named_query("var_breach_clusters")
        sub = df[df["strategy"] == strategy]
        summary = {
            "strategy": strategy,
            "total_breaches": int(len(sub)),
            "consecutive_breaches": int(sub["consecutive_breach"].sum()) if not sub.empty else 0,
            "max_20d_breaches": int(sub["breaches_20d"].max()) if not sub.empty else 0,
            "worst_breach_date": str(sub.loc[sub["net_return"].idxmin(), "trade_date"])
            if not sub.empty
            else None,
            "worst_breach_return": float(sub["net_return"].min()) if not sub.empty else 0.0,
        }
        rec = self.ledger.record("query_var_breach_clusters", {"strategy": strategy}, summary)
        return summary, rec.evidence_hash

    def compute_euler_attribution(self, strategy: str) -> tuple[list[dict[str, Any]], str]:
        w_df = self.db.query_df(
            """
            SELECT ticker, weight
            FROM rebalance_weights
            WHERE strategy = ?
              AND rebalance_date = (
                  SELECT MAX(rebalance_date) FROM rebalance_weights WHERE strategy = ?
              )
            """,
            (strategy, strategy),
        )
        weights = pd.Series(w_df["weight"].to_numpy(), index=w_df["ticker"].to_list())
        asset_cols = [c for c in weights.index if c in self.returns.columns]
        window = self.returns[asset_cols].tail(self.config.lookback_days)
        cov, _ = ledoit_wolf_covariance(window)
        euler = euler_risk_contributions(cov, weights.reindex(asset_cols))
        top5 = euler.sort_values("pct_risk", ascending=False).head(5).reset_index()
        top5 = top5.rename(columns={"index": "ticker"})
        rows = top5.to_dict(orient="records")
        rec = self.ledger.record("compute_euler_attribution", {"strategy": strategy}, rows)
        return rows, rec.evidence_hash

    def run_stress_test(self, strategy: str) -> tuple[list[dict[str, Any]], str]:
        w_df = self.db.query_df(
            """
            SELECT ticker, weight
            FROM rebalance_weights
            WHERE strategy = ?
              AND rebalance_date = (
                  SELECT MAX(rebalance_date) FROM rebalance_weights WHERE strategy = ?
              )
            """,
            (strategy, strategy),
        )
        weights = pd.Series(w_df["weight"].to_numpy(), index=w_df["ticker"].to_list())
        bench = (
            self.returns[self.config.benchmark]
            if self.config.benchmark in self.returns.columns
            else None
        )
        stress_df = evaluate_stress_scenarios(self.returns, weights, STRESS_SCENARIOS, bench)
        rows = stress_df.to_dict(orient="records")
        rec = self.ledger.record("run_stress_test", {"strategy": strategy}, rows)
        return rows, rec.evidence_hash


# Matches explicit tagged metric tokens like `[metric: 12.34%]` or `[metric: 0.042]`
_METRIC_TAG_RE = re.compile(r"\[metric:\s*(-?\d+(?:\.\d+)?)\s*%?\]")


def verify_memo_grounding(memo: str, ledger: ToolLedger) -> tuple[bool, list[str]]:
    """Verify every ``[metric: X]`` claim in a risk memo against the ToolLedger.

    Returns ``(is_grounded, list_of_ungrounded_claims)``.
    """
    ungrounded: list[str] = []
    for match in _METRIC_TAG_RE.finditer(memo):
        raw = match.group(0)
        val = float(match.group(1))
        if not ledger.has_number(val):
            ungrounded.append(raw)
    return len(ungrounded) == 0, ungrounded


class RiskAuditAgent:
    """Multi-step tool-calling risk audit workflow with deterministic verification."""

    def __init__(
        self,
        db: RiskBookDB,
        returns: pd.DataFrame,
        config: BacktestConfig,
        limits: RiskLimits | None = None,
        memo_synthesizer: Callable[[list[dict[str, Any]], list[ToolCallRecord]], str] | None = None,
    ) -> None:
        self.db = db
        self.returns = returns
        self.config = config
        self.limits = limits or RiskLimits()
        self.memo_synthesizer = memo_synthesizer

    def run(self) -> AuditReport:
        ledger = ToolLedger()
        tools = RiskAuditTools(self.db, self.returns, self.config, ledger)

        # Step 1: Query portfolio-wide risk snapshots via SQL
        snapshots, snap_hash = tools.query_risk_snapshots()
        if not snapshots:
            return AuditReport(
                as_of_date="",
                findings=[],
                tool_trace=ledger.records,
                memo_markdown="No risk snapshots available in ledger.",
                grounded=True,
            )

        as_of_date = str(snapshots[0]["as_of_date"])
        findings: list[dict[str, Any]] = []
        flagged_strategies: set[str] = set()

        # Step 2: Triage limit checks across all strategies
        for snap in snapshots:
            strat = str(snap["strategy"])

            if float(snap["var_95_breach_rate"]) > self.limits.max_var_95_breach_rate:
                flagged_strategies.add(strat)
                findings.append(
                    {
                        "as_of_date": as_of_date,
                        "strategy": strat,
                        "severity": "HIGH",
                        "check_name": "var_95_breach_rate",
                        "metric_value": float(snap["var_95_breach_rate"]),
                        "threshold_value": self.limits.max_var_95_breach_rate,
                        "detail": f"95% VaR breach rate {snap['var_95_breach_rate']:.2%} exceeds {self.limits.max_var_95_breach_rate:.2%} cap.",
                        "evidence_hash": snap_hash,
                    }
                )

            if float(snap["kupiec_p_value"]) < self.limits.min_kupiec_p_value:
                flagged_strategies.add(strat)
                findings.append(
                    {
                        "as_of_date": as_of_date,
                        "strategy": strat,
                        "severity": "MEDIUM",
                        "check_name": "kupiec_calibration",
                        "metric_value": float(snap["kupiec_p_value"]),
                        "threshold_value": self.limits.min_kupiec_p_value,
                        "detail": f"Kupiec POF p-value {snap['kupiec_p_value']:.4f} < {self.limits.min_kupiec_p_value:.2f} (unconditional VaR miscalibration).",
                        "evidence_hash": snap_hash,
                    }
                )

            if float(snap["effective_n"]) < self.limits.min_effective_n:
                flagged_strategies.add(strat)
                findings.append(
                    {
                        "as_of_date": as_of_date,
                        "strategy": strat,
                        "severity": "MEDIUM",
                        "check_name": "concentration_effective_n",
                        "metric_value": float(snap["effective_n"]),
                        "threshold_value": self.limits.min_effective_n,
                        "detail": f"Effective N {snap['effective_n']:.2f} below minimum diversification floor {self.limits.min_effective_n:.1f}.",
                        "evidence_hash": snap_hash,
                    }
                )

            if float(snap["max_sector_weight"]) > self.limits.max_sector_weight + 1e-4:
                flagged_strategies.add(strat)
                findings.append(
                    {
                        "as_of_date": as_of_date,
                        "strategy": strat,
                        "severity": "HIGH",
                        "check_name": "sector_cap_breach",
                        "metric_value": float(snap["max_sector_weight"]),
                        "threshold_value": self.limits.max_sector_weight,
                        "detail": f"Max sector exposure {snap['max_sector_weight']:.2%} breaches {self.limits.max_sector_weight:.2%} limit.",
                        "evidence_hash": snap_hash,
                    }
                )

            if float(snap["max_drawdown"]) < self.limits.max_drawdown_alert:
                flagged_strategies.add(strat)
                findings.append(
                    {
                        "as_of_date": as_of_date,
                        "strategy": strat,
                        "severity": "MEDIUM",
                        "check_name": "max_drawdown_alert",
                        "metric_value": float(snap["max_drawdown"]),
                        "threshold_value": self.limits.max_drawdown_alert,
                        "detail": f"Realised max drawdown {snap['max_drawdown']:.2%} exceeded alert level {self.limits.max_drawdown_alert:.2%}.",
                        "evidence_hash": snap_hash,
                    }
                )

        # Step 3: Targeted drill-down tool calls for every strategy (breach clusters)
        # and deeper attribution/stress tests for flagged strategies (plus best Sharpe)
        best_strat = max(snapshots, key=lambda s: float(s["sharpe"]))["strategy"]
        drill_targets = sorted(flagged_strategies | {str(best_strat)})

        drill_context: dict[str, dict[str, Any]] = {}
        for snap in snapshots:
            strat = str(snap["strategy"])
            clusters, cl_hash = tools.query_var_breach_clusters(strat)
            if clusters["max_20d_breaches"] > self.limits.max_20d_breaches:
                findings.append(
                    {
                        "as_of_date": as_of_date,
                        "strategy": strat,
                        "severity": "HIGH",
                        "check_name": "var_breach_clustering",
                        "metric_value": float(clusters["max_20d_breaches"]),
                        "threshold_value": float(self.limits.max_20d_breaches),
                        "detail": f"Peak 20-day VaR breach cluster of {clusters['max_20d_breaches']} breaches (limit {self.limits.max_20d_breaches}).",
                        "evidence_hash": cl_hash,
                    }
                )
            if strat in drill_targets:
                sectors, _ = tools.query_sector_exposures(strat)
                euler, _ = tools.compute_euler_attribution(strat)
                stress, _ = tools.run_stress_test(strat)
                drill_context[strat] = {
                    "clusters": clusters,
                    "sectors": sectors[:3],
                    "top_risk_contributors": euler[:3],
                    "stress": stress,
                }

        # Step 4: Persist audit findings to SQLite
        if findings:
            self.db.write_audit_findings(findings)

        # Step 5: Draft grounded memo (custom LLM synthesizer or deterministic structured memo)
        if self.memo_synthesizer is not None:
            memo = self.memo_synthesizer(findings, ledger.records)
        else:
            memo = self._render_deterministic_memo(as_of_date, snapshots, findings, drill_context)

        # Step 6: Deterministic grounding check
        grounded, ungrounded = verify_memo_grounding(memo, ledger)
        return AuditReport(
            as_of_date=as_of_date,
            findings=findings,
            tool_trace=ledger.records,
            memo_markdown=memo,
            grounded=grounded,
            ungrounded_claims=ungrounded,
        )

    @staticmethod
    def _render_deterministic_memo(
        as_of_date: str,
        snapshots: list[dict[str, Any]],
        findings: list[dict[str, Any]],
        drill_context: dict[str, dict[str, Any]],
    ) -> str:
        lines = [
            f"# RiskBook Audit Memo ({as_of_date})",
            "",
            "## 1. Walk-Forward Risk & Performance Summary",
            "| Strategy | Ann. Return | Ann. Vol | Sharpe | Max DD | 95% VaR Breach | Kupiec p | Eff. N |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for s in snapshots:
            lines.append(
                f"| `{s['strategy']}` "
                f"| [metric: {s['ann_return']*100:.2f}%] "
                f"| [metric: {s['ann_vol']*100:.2f}%] "
                f"| [metric: {s['sharpe']:.2f}] "
                f"| [metric: {s['max_drawdown']*100:.2f}%] "
                f"| [metric: {s['var_95_breach_rate']*100:.2f}%] "
                f"| [metric: {s['kupiec_p_value']:.3f}] "
                f"| [metric: {s['effective_n']:.1f}] |"
            )

        lines.extend(["", f"## 2. Limit & Calibration Findings ({len(findings)} triggered)"])
        if not findings:
            lines.append("- No risk-limit or VaR-calibration breaches triggered.")
        else:
            for f in findings:
                lines.append(
                    f"- **[{f['severity']}]** `{f['strategy']}` / `{f['check_name']}` "
                    f"(observed `[metric: {f['metric_value']:.4f}]`, evidence `{f['evidence_hash']}`): {f['detail']}"
                )

        lines.extend(["", "## 3. Targeted Attribution & Stress Drill-Down"])
        for strat, ctx in drill_context.items():
            lines.append(f"### `{strat}`")
            top_rc = ctx["top_risk_contributors"]
            if top_rc:
                rc_str = ", ".join(
                    f"{r['ticker']} ([metric: {r['pct_risk']*100:.1f}%] of vol)"
                    for r in top_rc
                )
                lines.append(f"- **Top Euler risk contributors:** {rc_str}")
            cl = ctx["clusters"]
            lines.append(
                f"- **VaR breach clustering:** [metric: {cl['total_breaches']}] total breaches, "
                f"[metric: {cl['consecutive_breaches']}] consecutive, "
                f"peak 20-day cluster [metric: {cl['max_20d_breaches']}]."
            )
            if ctx["stress"]:
                worst = min(ctx["stress"], key=lambda x: x["cum_return"])
                lines.append(
                    f"- **Worst historical stress episode:** {worst['scenario']} "
                    f"([metric: {worst['cum_return']*100:.2f}%] cumulative return vs "
                    f"SPY [metric: {worst['benchmark_return']*100:.2f}%])."
                )
        return "\n".join(lines) + "\n"
