-- Reusable SQL analytics queries using CTEs and window functions

-- name: sector_exposures
WITH latest_rebalance AS (
    SELECT strategy, MAX(rebalance_date) AS max_date
    FROM rebalance_weights
    GROUP BY strategy
)
SELECT
    w.strategy,
    w.rebalance_date,
    s.sector,
    ROUND(SUM(w.weight), 6) AS sector_weight,
    COUNT(*) AS n_names,
    ROUND(MAX(w.weight), 6) AS max_single_name
FROM rebalance_weights w
JOIN latest_rebalance lr
    ON w.strategy = lr.strategy AND w.rebalance_date = lr.max_date
JOIN securities s
    ON w.ticker = s.ticker
GROUP BY w.strategy, w.rebalance_date, s.sector
ORDER BY w.strategy, sector_weight DESC;

-- name: rolling_drawdowns
WITH equity_curve AS (
    SELECT
        strategy,
        trade_date,
        net_return,
        EXP(SUM(LN(1.0 + net_return)) OVER (
            PARTITION BY strategy
            ORDER BY trade_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )) AS nav
    FROM daily_portfolio_returns
),
peaks AS (
    SELECT
        strategy,
        trade_date,
        net_return,
        nav,
        MAX(nav) OVER (
            PARTITION BY strategy
            ORDER BY trade_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS running_peak
    FROM equity_curve
)
SELECT
    strategy,
    trade_date,
    ROUND(nav, 6) AS nav,
    ROUND((nav / running_peak) - 1.0, 6) AS drawdown
FROM peaks
ORDER BY strategy, trade_date;

-- name: var_breach_clusters
WITH lagged AS (
    SELECT
        strategy,
        trade_date,
        net_return,
        var_95_hist,
        breach_95,
        LAG(breach_95, 1, 0) OVER (
            PARTITION BY strategy ORDER BY trade_date
        ) AS prev_breach_95,
        SUM(breach_95) OVER (
            PARTITION BY strategy
            ORDER BY trade_date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS breaches_20d
    FROM daily_portfolio_returns
)
SELECT
    strategy,
    trade_date,
    ROUND(net_return, 6) AS net_return,
    ROUND(var_95_hist, 6) AS var_95_hist,
    breach_95,
    CASE WHEN breach_95 = 1 AND prev_breach_95 = 1 THEN 1 ELSE 0 END AS consecutive_breach,
    breaches_20d
FROM lagged
WHERE breach_95 = 1
ORDER BY strategy, trade_date;

-- name: concentration_by_rebalance
SELECT
    strategy,
    rebalance_date,
    ROUND(1.0 / SUM(weight * weight), 4) AS effective_n,
    ROUND(MAX(weight), 6) AS max_position_weight,
    SUM(CASE WHEN weight >= 0.005 THEN 1 ELSE 0 END) AS active_positions
FROM rebalance_weights
GROUP BY strategy, rebalance_date
ORDER BY strategy, rebalance_date;
