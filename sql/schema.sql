-- Normalized SQLite / ANSI-compatible schema for equities riskbook ledger

CREATE TABLE IF NOT EXISTS securities (
    ticker TEXT PRIMARY KEY,
    sector TEXT NOT NULL,
    is_benchmark INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS daily_prices (
    trade_date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    adj_close REAL NOT NULL,
    daily_return REAL,
    PRIMARY KEY (trade_date, ticker),
    FOREIGN KEY (ticker) REFERENCES securities(ticker)
);

CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker_date
    ON daily_prices(ticker, trade_date);

CREATE TABLE IF NOT EXISTS rebalance_weights (
    rebalance_date TEXT NOT NULL,
    strategy TEXT NOT NULL,
    ticker TEXT NOT NULL,
    weight REAL NOT NULL,
    PRIMARY KEY (rebalance_date, strategy, ticker),
    FOREIGN KEY (ticker) REFERENCES securities(ticker)
);

CREATE INDEX IF NOT EXISTS idx_rebalance_strategy_date
    ON rebalance_weights(strategy, rebalance_date);

CREATE TABLE IF NOT EXISTS daily_portfolio_returns (
    trade_date TEXT NOT NULL,
    strategy TEXT NOT NULL,
    gross_return REAL NOT NULL,
    net_return REAL NOT NULL,
    turnover REAL NOT NULL,
    cost_bps REAL NOT NULL,
    var_95_hist REAL,
    var_99_hist REAL,
    cvar_95_hist REAL,
    var_95_param REAL,
    breach_95 INTEGER NOT NULL DEFAULT 0,
    breach_99 INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (trade_date, strategy)
);

CREATE INDEX IF NOT EXISTS idx_portfolio_returns_strategy_date
    ON daily_portfolio_returns(strategy, trade_date);

CREATE TABLE IF NOT EXISTS risk_snapshots (
    as_of_date TEXT NOT NULL,
    strategy TEXT NOT NULL,
    ann_return REAL NOT NULL,
    ann_vol REAL NOT NULL,
    sharpe REAL NOT NULL,
    max_drawdown REAL NOT NULL,
    avg_monthly_turnover REAL NOT NULL,
    var_95_breach_rate REAL NOT NULL,
    kupiec_p_value REAL NOT NULL,
    christoffersen_p_value REAL NOT NULL,
    effective_n REAL NOT NULL,
    max_sector_weight REAL NOT NULL,
    PRIMARY KEY (as_of_date, strategy)
);

CREATE TABLE IF NOT EXISTS audit_findings (
    finding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    as_of_date TEXT NOT NULL,
    strategy TEXT NOT NULL,
    severity TEXT NOT NULL,
    check_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    threshold_value REAL NOT NULL,
    detail TEXT NOT NULL,
    evidence_hash TEXT NOT NULL
);
