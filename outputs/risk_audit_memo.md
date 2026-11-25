# RiskBook Audit Memo (2025-12-30)

## 1. Walk-Forward Risk & Performance Summary
| Strategy | Ann. Return | Ann. Vol | Sharpe | Max DD | 95% VaR Breach | Kupiec p | Eff. N |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cvar_opt` | [metric: 15.84%] | [metric: 15.73%] | [metric: 1.01] | [metric: -29.01%] | [metric: 6.60%] | [metric: 0.003] | [metric: 10.3] |
| `equal_weight` | [metric: 21.72%] | [metric: 18.61%] | [metric: 1.17] | [metric: -32.70%] | [metric: 4.67%] | [metric: 0.517] | [metric: 30.0] |
| `hrp` | [metric: 17.19%] | [metric: 16.89%] | [metric: 1.02] | [metric: -31.41%] | [metric: 4.84%] | [metric: 0.754] | [metric: 23.4] |
| `mv_ledoit_wolf` | [metric: 15.27%] | [metric: 16.36%] | [metric: 0.93] | [metric: -34.02%] | [metric: 5.81%] | [metric: 0.131] | [metric: 12.3] |
| `mv_sample` | [metric: 15.04%] | [metric: 15.76%] | [metric: 0.95] | [metric: -29.95%] | [metric: 5.35%] | [metric: 0.505] | [metric: 9.9] |

## 2. Limit & Calibration Findings (12 triggered)
- **[MEDIUM]** `cvar_opt` / `kupiec_calibration` (observed `[metric: 0.0032]`, evidence `b7e7671f489dc354`): Kupiec POF p-value 0.0032 < 0.05 (unconditional VaR miscalibration).
- **[MEDIUM]** `cvar_opt` / `max_drawdown_alert` (observed `[metric: -0.2901]`, evidence `b7e7671f489dc354`): Realised max drawdown -29.01% exceeded alert level -20.00%.
- **[MEDIUM]** `equal_weight` / `max_drawdown_alert` (observed `[metric: -0.3270]`, evidence `b7e7671f489dc354`): Realised max drawdown -32.70% exceeded alert level -20.00%.
- **[MEDIUM]** `hrp` / `max_drawdown_alert` (observed `[metric: -0.3141]`, evidence `b7e7671f489dc354`): Realised max drawdown -31.41% exceeded alert level -20.00%.
- **[MEDIUM]** `mv_ledoit_wolf` / `max_drawdown_alert` (observed `[metric: -0.3402]`, evidence `b7e7671f489dc354`): Realised max drawdown -34.02% exceeded alert level -20.00%.
- **[MEDIUM]** `mv_sample` / `concentration_effective_n` (observed `[metric: 9.8665]`, evidence `b7e7671f489dc354`): Effective N 9.87 below minimum diversification floor 10.0.
- **[MEDIUM]** `mv_sample` / `max_drawdown_alert` (observed `[metric: -0.2995]`, evidence `b7e7671f489dc354`): Realised max drawdown -29.95% exceeded alert level -20.00%.
- **[HIGH]** `cvar_opt` / `var_breach_clustering` (observed `[metric: 12.0000]`, evidence `3f8d9bdd4f4c87b9`): Peak 20-day VaR breach cluster of 12 breaches (limit 3).
- **[HIGH]** `equal_weight` / `var_breach_clustering` (observed `[metric: 12.0000]`, evidence `130a8d0c11f98a90`): Peak 20-day VaR breach cluster of 12 breaches (limit 3).
- **[HIGH]** `hrp` / `var_breach_clustering` (observed `[metric: 12.0000]`, evidence `91b316e515414c91`): Peak 20-day VaR breach cluster of 12 breaches (limit 3).
- **[HIGH]** `mv_ledoit_wolf` / `var_breach_clustering` (observed `[metric: 13.0000]`, evidence `e3127d0de345db96`): Peak 20-day VaR breach cluster of 13 breaches (limit 3).
- **[HIGH]** `mv_sample` / `var_breach_clustering` (observed `[metric: 13.0000]`, evidence `2b13e9019bb750f3`): Peak 20-day VaR breach cluster of 13 breaches (limit 3).

## 3. Targeted Attribution & Stress Drill-Down
### `cvar_opt`
- **Top Euler risk contributors:** HD ([metric: 12.6%] of vol), MCD ([metric: 12.1%] of vol), WMT ([metric: 11.2%] of vol)
- **VaR breach clustering:** [metric: 116] total breaches, [metric: 19] consecutive, peak 20-day cluster [metric: 12].
- **Worst historical stress episode:** COVID-19 Liquidity Shock (2020) ([metric: -31.43%] cumulative return vs SPY [metric: -33.40%]).
### `equal_weight`
- **Top Euler risk contributors:** AVGO ([metric: 6.1%] of vol), NVDA ([metric: 6.0%] of vol), AAPL ([metric: 5.2%] of vol)
- **VaR breach clustering:** [metric: 82] total breaches, [metric: 12] consecutive, peak 20-day cluster [metric: 12].
- **Worst historical stress episode:** COVID-19 Liquidity Shock (2020) ([metric: -32.32%] cumulative return vs SPY [metric: -33.40%]).
### `hrp`
- **Top Euler risk contributors:** MCD ([metric: 7.0%] of vol), HON ([metric: 5.1%] of vol), UNP ([metric: 4.9%] of vol)
- **VaR breach clustering:** [metric: 85] total breaches, [metric: 14] consecutive, peak 20-day cluster [metric: 12].
- **Worst historical stress episode:** COVID-19 Liquidity Shock (2020) ([metric: -32.32%] cumulative return vs SPY [metric: -33.40%]).
### `mv_ledoit_wolf`
- **Top Euler risk contributors:** JNJ ([metric: 12.1%] of vol), DUK ([metric: 10.4%] of vol), KO ([metric: 10.2%] of vol)
- **VaR breach clustering:** [metric: 102] total breaches, [metric: 18] consecutive, peak 20-day cluster [metric: 13].
- **Worst historical stress episode:** COVID-19 Liquidity Shock (2020) ([metric: -31.11%] cumulative return vs SPY [metric: -33.40%]).
### `mv_sample`
- **Top Euler risk contributors:** MCD ([metric: 13.2%] of vol), PG ([metric: 12.3%] of vol), JNJ ([metric: 12.1%] of vol)
- **VaR breach clustering:** [metric: 94] total breaches, [metric: 12] consecutive, peak 20-day cluster [metric: 13].
- **Worst historical stress episode:** COVID-19 Liquidity Shock (2020) ([metric: -32.05%] cumulative return vs SPY [metric: -33.40%]).
