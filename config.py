"""
config.py
=========
Central configuration for the mid-cap tech alpha system: universe,
date range, rebalancing, cost assumptions, and model hyperparameters.

Edit this file to change tickers, dates, or strategy parameters without
touching pipeline logic elsewhere.
"""

from datetime import date

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
# A static basket of historically mid-cap-to-small-large-cap technology /
# tech-adjacent names with long trading histories. This is a SIMPLIFICATION:
# a production system should reconstruct the mid-cap universe dynamically
# at each rebalance date (e.g. via a point-in-time market-cap screen against
# an index such as the S&P MidCap 400 Information Technology constituents),
# since names migrate between cap buckets over a 10-year window. Using a
# static list picked today introduces a mild survivorship / hindsight bias
# because every ticker below is known to have survived to 2026.
UNIVERSE = [
    "AKAM",  # Akamai Technologies
    "FFIV",  # F5, Inc.
    "JNPR",  # Juniper Networks
    "NTAP",  # NetApp
    "TER",   # Teradyne
    "CIEN",  # Ciena
    "ZBRA",  # Zebra Technologies
    "SYNA",  # Synaptics
    "CDW",   # CDW Corporation
    "FLEX",  # Flex Ltd.
    "JBL",   # Jabil Inc.
    "YELP",  # Yelp Inc.
    "PEGA",  # Pegasystems
    "MANH",  # Manhattan Associates
    "CSGS",  # CSG Systems International
]

# Benchmark ticker (S&P 500 proxy)
BENCHMARK = "SPY"

# ---------------------------------------------------------------------------
# Backtest window
# ---------------------------------------------------------------------------
START_DATE = date(2016, 8, 4)
END_DATE = date(2026, 8, 4)   # 10-year window ending "today"

# ---------------------------------------------------------------------------
# Rebalancing & portfolio construction
# ---------------------------------------------------------------------------
REBALANCE_FREQ = "W-FRI"        # weekly, rebalance Fridays
TOP_QUANTILE = 0.20             # go long top 20% of predicted-score universe
MIN_HOLDINGS = 3                # floor on number of names held at once
MAX_WEIGHT_PER_NAME = 0.35      # position concentration cap

# ---------------------------------------------------------------------------
# Trading frictions
# ---------------------------------------------------------------------------
COMMISSION_BPS = 5.0    # one-way commission, in basis points of notional
SLIPPAGE_BPS = 7.0      # one-way slippage / market impact estimate, bps

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
MOMENTUM_WINDOWS = [5, 10, 20, 60]
VOL_WINDOW = 20
RSI_WINDOW = 14
SENTIMENT_LOOKBACKS = [5, 20]
FUNDAMENTAL_REPORT_LAG_DAYS = 45  # realistic lag before fundamentals are
                                   # public/usable, to avoid look-ahead bias

# Forward-return label horizon (trading days) the model tries to predict
LABEL_HORIZON = 5

# ---------------------------------------------------------------------------
# Walk-forward model training
# ---------------------------------------------------------------------------
INITIAL_TRAIN_DAYS = 504     # ~2 years before the model starts predicting
RETRAIN_EVERY_DAYS = 63      # retrain roughly quarterly
PURGE_DAYS = LABEL_HORIZON   # gap between train end and test start to
                              # prevent label leakage across the boundary

MODEL_PARAMS = dict(
    max_iter=200,
    max_depth=4,
    learning_rate=0.05,
    l2_regularization=1.0,
    random_state=42,
)

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE_ANNUAL = 0.02  # approximate, for Sharpe/Sortino calculations
