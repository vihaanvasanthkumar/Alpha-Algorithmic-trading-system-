"""
features.py
============
Builds the daily, cross-sectionally standardized feature panel used by the
model: technical/momentum features from price, plus alt-data features from
sentiment and the supply-chain proxy.

All rolling/lookback windows use only data available up to and including
date t (no centering, no future leakage). Cross-sectional z-scoring is done
per date, across whatever tickers are active that day, so the model learns
relative ranking rather than absolute price levels -- this makes features
far more stationary across a 10-year sample than raw prices would be.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig


def build_technical_features(prices: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    feats = {}
    for w in config.MOMENTUM_WINDOWS:
        feats[f"mom_{w}d"] = pd.concat(
            {t: prices[t]["adj_close"].pct_change(w) for t in tickers}, axis=1
        )

    feats["vol_20d"] = pd.concat(
        {t: prices[t]["adj_close"].pct_change().rolling(config.VOL_WINDOW).std() for t in tickers},
        axis=1,
    )

    feats["rsi_14"] = pd.concat(
        {t: _rsi(prices[t]["adj_close"], config.RSI_WINDOW) for t in tickers}, axis=1
    )

    feats["macd_hist"] = pd.concat(
        {t: _macd_hist(prices[t]["adj_close"]) for t in tickers}, axis=1
    )

    feats["px_vs_ma50"] = pd.concat(
        {
            t: prices[t]["adj_close"] / prices[t]["adj_close"].rolling(50).mean() - 1
            for t in tickers
        },
        axis=1,
    )
    feats["px_vs_ma200"] = pd.concat(
        {
            t: prices[t]["adj_close"] / prices[t]["adj_close"].rolling(200).mean() - 1
            for t in tickers
        },
        axis=1,
    )

    feats["vol_chg_20d"] = pd.concat(
        {
            t: prices[t]["volume"].rolling(20).mean().pct_change(20)
            for t in tickers
        },
        axis=1,
    )
    return feats


def build_sentiment_features(sentiment: pd.DataFrame) -> dict[str, pd.DataFrame]:
    feats = {}
    for lb in config.SENTIMENT_LOOKBACKS:
        feats[f"sent_avg_{lb}d"] = sentiment.rolling(lb).mean()
    feats["sent_chg_5d"] = sentiment - sentiment.shift(5)
    return feats


def build_supply_chain_features(supply: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    ccc = pd.concat({t: supply[t]["cash_conversion_cycle"] for t in tickers}, axis=1)
    inv_growth = pd.concat({t: supply[t]["inventory_growth"] for t in tickers}, axis=1)
    feats = {
        "ccc_level": ccc,
        "ccc_chg_63d": ccc - ccc.shift(63),   # ~1 fiscal quarter of trading days
        "inventory_growth": inv_growth,
    }
    return feats


def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each row (date) across tickers, robust to a handful of
    missing columns per row."""
    mu = df.mean(axis=1)
    sigma = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sigma, axis=0)


def build_feature_panel(prices, sentiment, supply, tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns a long-format panel: index = (date, ticker), columns = features.
    Also returns the ordered list of feature column names.

    `sentiment` may be None (e.g. real-data mode with no sentiment source
    configured) -- in that case sentiment features are simply omitted
    rather than raising an error, so the pipeline still runs on technical
    + supply-chain features alone.
    """
    all_feats = {}
    all_feats.update(build_technical_features(prices, tickers))
    if sentiment is not None:
        all_feats.update(build_sentiment_features(sentiment[tickers]))
    all_feats.update(build_supply_chain_features(supply, tickers))

    zscored = {name: cross_sectional_zscore(df) for name, df in all_feats.items()}

    long_frames = []
    for name, df in zscored.items():
        s = df.stack(future_stack=True)
        s.name = name
        long_frames.append(s)

    panel = pd.concat(long_frames, axis=1)
    panel.index.names = ["date", "ticker"]
    feature_cols = list(zscored.keys())
    return panel, feature_cols


def build_labels(prices: pd.DataFrame, tickers: list[str], horizon: int) -> pd.Series:
    """
    Forward `horizon`-day return in excess of the cross-sectional mean that
    day -- i.e. the model predicts relative outperformance within the
    universe, not absolute direction. This is generally an easier, more
    stationary target for cross-sectional equity ML and naturally supports
    a long-top-quantile portfolio construction.
    """
    fwd_ret = pd.concat(
        {t: prices[t]["adj_close"].pct_change(horizon).shift(-horizon) for t in tickers},
        axis=1,
    )
    excess = fwd_ret.sub(fwd_ret.mean(axis=1), axis=0)
    s = excess.stack(future_stack=True)
    s.index.names = ["date", "ticker"]
    s.name = "fwd_excess_return"
    return s
