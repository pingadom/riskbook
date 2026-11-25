# riskbook

Equities allocation, covariance estimation, tail-risk calibration, and verified risk-monitoring ledger in Python and SQL.

## What It Does

`riskbook` runs a strictly causal monthly walk-forward backtest across 30 liquid US large-cap equities spanning all 11 GICS sectors (2018–2025, 1,757 out-of-sample trading days after a 252-day estimation window), persists the full book and daily risk ledger to SQLite, and runs a tool-calling risk-audit workflow whose numeric claims are verified against SHA-256 tool hashes.

### Implemented Components
- **Covariance Estimation (`src/riskbook/covariance.py`):** Sample covariance, analytical Ledoit–Wolf (2004) linear shrinkage implemented from scratch and cross-checked against `sklearn.covariance.LedoitWolf`, and RiskMetrics EWMA covariance.
- **Constrained Allocators (`src/riskbook/optimisers.py`):** Equal-weight (`1/N`), constrained Minimum Variance (sample vs. Ledoit–Wolf with turnover penalty), Hierarchical Risk Parity (López de Prado, 2016) with constraint projection, and exact CVaR minimisation via the Rockafellar–Uryasev (2000) linear program (`scipy.optimize.linprog` / HiGHS) with position caps, sector caps, and L1 turnover costs.
- **Risk & Calibration (`src/riskbook/risk.py`):** Historical, Gaussian parametric, and Cholesky Monte Carlo 1-day VaR/CVaR (95% & 99%), exact Euler marginal/component risk decomposition ($\sum_i \text{RC}_i = \sigma_p$), Kupiec (1995) POF unconditional coverage tests, Christoffersen (1998) independence/conditional coverage tests, and historical stress replay (2020 COVID shock, 2022 rate drawdown, 2023 SVB, Aug 2024 vol spike).
- **SQL Ledger & Window Analytics (`sql/schema.sql`, `sql/analytics.sql`):** Normalised tables (`securities`, `daily_prices`, `rebalance_weights`, `daily_portfolio_returns`, `risk_snapshots`, `audit_findings`) with reusable CTE and window-function queries for rolling drawdowns, sector exposures, effective-$N$ concentration, and 20-day VaR breach clustering.
- **Grounded Agentic Risk Audit (`src/riskbook/agent.py`):** Multi-step tool-calling investigation loop over SQL and risk-attribution tools that triages limit breaches, drills into Euler attribution and historical stress episodes, persists findings with SHA-256 evidence hashes, and runs a deterministic post-check (`verify_memo_grounding`) rejecting any ungrounded or hallucinated numeric claim.

## Out-of-Sample Results (2019–2025, 5 bps one-way cost)

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max DD | Monthly Turnover | 95% VaR Breach | Kupiec $p$ | Christoffersen CC $p$ | Eff. $N$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `equal_weight` | 21.72% | 18.61% | 1.17 | -32.70% | 2.37% | 4.67% | 0.517 | 0.001 | 30.0 |
| `hrp` | 17.19% | 16.89% | 1.02 | -31.41% | 8.06% | 4.84% | 0.754 | <0.001 | 23.4 |
| `cvar_opt` | 15.84% | 15.73% | 1.01 | -29.01% | 6.30% | 6.60% | 0.003 | <0.001 | 10.3 |
| `mv_sample` | 15.04% | 15.76% | 0.95 | -29.95% | 9.41% | 5.35% | 0.505 | 0.014 | 9.9 |
| `mv_ledoit_wolf` | 15.27% | 16.36% | 0.93 | -34.02% | 3.15% | 5.81% | 0.131 | <0.001 | 12.3 |

**Key empirical takeaways:**
1. **Shrinkage cuts turnover by 3x:** `mv_ledoit_wolf` reduces average monthly turnover from 9.41% (`mv_sample`) to 3.15% while maintaining higher diversification (Effective $N$ = 12.3 vs. 9.9).
2. **In-sample tail overfitting shows up in VaR backtests:** `cvar_opt` achieves the lowest realised vol (15.73%) and max drawdown (-29.01%), yet fails Kupiec unconditional coverage ($p = 0.003$, 6.60% realised breach rate vs. 5% target) because empirical tail weights overfit the trailing 252-day sample.
3. **Static historical VaR fails independence across all strategies:** Every strategy rejects Christoffersen conditional coverage ($p \le 0.014$), with 12–13 breaches clustering in 20-day windows during volatility regime shifts—illustrating why risk desks pair historical VaR with EWMA scaling and breach-cluster monitoring.

## Quickstart

```bash
# Run unit, analytical-identity, anti-leakage, SQL, and agent-verification tests
pytest -v

# Run full S&P 500 walk-forward pipeline + SQL ledger + grounded audit memo
PYTHONPATH=src python -m riskbook
```
