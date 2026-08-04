"""
model.py
=========
Walk-forward training of a gradient-boosted tree regressor to predict
forward excess return (relative to the universe), plus a couple of naive
baselines used for comparison in the backtest report.

Walk-forward discipline:
  - the model is trained only on data with a label date strictly before
    the test window, with a purge gap (config.PURGE_DAYS) so that no
    training label window overlaps the first test date -- this prevents
    the classic leakage bug where overlapping forward-return windows let
    the model "see" test-period information through the training labels.
  - it is retrained periodically (config.RETRAIN_EVERY_DAYS) on an
    expanding window, mimicking how a live desk would periodically refit
    rather than fit once on the full history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

import config


def walk_forward_predict(panel: pd.DataFrame, feature_cols: list[str], label_col: str) -> pd.DataFrame:
    """
    Returns a DataFrame indexed like `panel` with columns:
        pred_ml       -- gradient boosting prediction
        pred_linear   -- ridge regression baseline prediction
    Rows before the model has ever been trained are NaN (no signal yet,
    handled as "stay in cash" by the backtest engine).
    """
    dates = panel.index.get_level_values("date").unique().sort_values()
    n_dates = len(dates)

    preds_ml = pd.Series(index=panel.index, dtype=float)
    preds_lin = pd.Series(index=panel.index, dtype=float)

    first_train_end_idx = config.INITIAL_TRAIN_DAYS
    if first_train_end_idx >= n_dates:
        raise ValueError("Not enough history for the configured INITIAL_TRAIN_DAYS.")

    ml_model = None
    lin_model = None
    next_retrain_idx = first_train_end_idx

    for test_start_idx in range(first_train_end_idx, n_dates, 1):
        test_date = dates[test_start_idx]

        if test_start_idx >= next_retrain_idx:
            train_end_idx = max(test_start_idx - config.PURGE_DAYS, 1)
            train_dates = dates[:train_end_idx]
            train_mask = panel.index.get_level_values("date").isin(train_dates)
            train_df = panel.loc[train_mask].dropna(subset=feature_cols + [label_col])

            if len(train_df) >= 200:
                X_train = train_df[feature_cols].values
                y_train = train_df[label_col].values

                ml_model = HistGradientBoostingRegressor(**config.MODEL_PARAMS)
                ml_model.fit(X_train, y_train)

                lin_model = Ridge(alpha=5.0)
                lin_model.fit(X_train, y_train)

            next_retrain_idx = test_start_idx + config.RETRAIN_EVERY_DAYS

        if ml_model is None:
            continue

        test_mask = panel.index.get_level_values("date") == test_date
        test_df = panel.loc[test_mask]
        valid = test_df[feature_cols].notna().all(axis=1)
        if valid.any():
            X_test = test_df.loc[valid, feature_cols].values
            preds_ml.loc[test_df.index[valid]] = ml_model.predict(X_test)
            preds_lin.loc[test_df.index[valid]] = lin_model.predict(X_test)

    out = pd.DataFrame({"pred_ml": preds_ml, "pred_linear": preds_lin})
    return out


def naive_momentum_score(panel: pd.DataFrame) -> pd.Series:
    """A dead-simple non-ML baseline: rank purely by 20-day price momentum
    (already present as a cross-sectionally z-scored feature)."""
    return panel["mom_20d"]


TECHNICAL_FEATURES = [
    "mom_5d", "mom_10d", "mom_20d", "mom_60d", "vol_20d", "rsi_14",
    "macd_hist", "px_vs_ma50", "px_vs_ma200", "vol_chg_20d",
]
ALT_DATA_FEATURES = [
    "sent_avg_5d", "sent_avg_20d", "sent_chg_5d",
    "ccc_level", "ccc_chg_63d", "inventory_growth",
]


def feature_group_ic(panel: pd.DataFrame, label_col: str, train_frac: float = 0.6) -> dict:
    """
    Diagnostic (not part of the traded strategy): fits a simple Ridge model
    separately on technical-only, alt-data-only, and all features on the
    first `train_frac` of history, and reports each group's out-of-sample
    rank IC on the remainder. Used to check whether the alternative-data
    features carry genuine incremental signal beyond price/volume alone.
    """
    from sklearn.linear_model import Ridge

    tech_cols = [c for c in TECHNICAL_FEATURES if c in panel.columns]
    alt_cols = [c for c in ALT_DATA_FEATURES if c in panel.columns]
    cols_needed = tech_cols + alt_cols + [label_col]
    df = panel.dropna(subset=cols_needed).sort_index(level="date")
    split = int(len(df) * train_frac)
    train, test = df.iloc[:split], df.iloc[split:]

    results = {}
    for name, cols in [
        ("technical_only", tech_cols),
        ("alt_data_only", alt_cols),
        ("all_features", tech_cols + alt_cols),
    ]:
        if not cols:
            results[name] = float("nan")
            continue
        m = Ridge(alpha=5.0).fit(train[cols], train[label_col])
        pred = pd.Series(m.predict(test[cols]), index=test.index)
        results[name] = information_coefficient(pred, test[label_col])
    return results


def information_coefficient(preds: pd.Series, labels: pd.Series) -> float:
    """Spearman-style rank correlation between predictions and realized
    forward excess returns -- the standard metric for how much genuine
    predictive signal a cross-sectional equity model has found."""
    df = pd.concat([preds, labels], axis=1).dropna()
    if len(df) < 30:
        return float("nan")
    return df.iloc[:, 0].corr(df.iloc[:, 1], method="spearman")
