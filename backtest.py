"""
backtest.py
============
Event-driven-ish (but vectorized for speed) backtest engine:
  - rebalances on a fixed schedule (weekly by default),
  - at each rebalance, ranks the universe by the supplied score and goes
    long the top quantile, equal-weighted subject to a per-name cap,
  - charges commission + slippage on turnover,
  - holds cash (zero return) whenever there's no signal yet or too few
    names pass the ranking (e.g. warm-up period),
  - computes standard risk-adjusted performance metrics vs a benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def build_target_weights(score_panel: pd.DataFrame, rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    score_panel: index=(date,ticker), single column of scores (higher=better)
    Returns a DataFrame indexed by rebalance_dates, columns=tickers, weights
    summing to <=1 (remainder sits in cash).
    """
    tickers = score_panel.index.get_level_values("ticker").unique()
    weights = pd.DataFrame(0.0, index=rebalance_dates, columns=tickers)

    for d in rebalance_dates:
        if d not in score_panel.index.get_level_values("date"):
            continue
        day_scores = score_panel.xs(d, level="date").iloc[:, 0].dropna()
        if len(day_scores) == 0:
            continue
        n_names = max(config.MIN_HOLDINGS, int(np.ceil(len(day_scores) * config.TOP_QUANTILE)))
        top = day_scores.sort_values(ascending=False).head(n_names)
        if len(top) == 0:
            continue
        w = 1.0 / len(top)
        w = min(w, config.MAX_WEIGHT_PER_NAME)
        weights.loc[d, top.index] = w
        # renormalize residual so weights don't exceed 1 due to the cap
        total = weights.loc[d].sum()
        if total > 1.0:
            weights.loc[d] = weights.loc[d] / total
    return weights


def run_backtest(
    prices: pd.DataFrame,
    tickers: list[str],
    target_weights: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """
    Simulates holding target_weights between rebalance dates, applying
    costs on each rebalance's turnover. Returns a daily DataFrame with
    columns: gross_return, net_return, equity, turnover, n_holdings.
    """
    daily_index = prices[tickers[0]].index
    daily_rets = pd.concat(
        {t: prices[t]["adj_close"].pct_change() for t in tickers}, axis=1
    ).reindex(daily_index).fillna(0.0)

    current_weights = pd.Series(0.0, index=tickers)
    holdings_over_time = pd.DataFrame(0.0, index=daily_index, columns=tickers)

    # IMPORTANT: a rebalance "decided" using date d's closing price cannot
    # possibly capture date d's own return -- that return is already baked
    # into the close used to compute the score. Execution is therefore
    # pushed to the next trading day. Skipping this lag is a classic,
    # easy-to-miss look-ahead bug that mechanically inflates backtested
    # performance (confirmed here: a purely random score with same-day
    # execution still showed a spurious edge before this fix).
    pending_weights = {}
    rebalance_list = sorted(target_weights.index)
    exec_map = {}
    for d in rebalance_list:
        pos = daily_index.searchsorted(d)
        if pos + 1 < len(daily_index):
            exec_date = daily_index[pos + 1]
            exec_map[exec_date] = target_weights.loc[d].reindex(tickers).fillna(0.0)

    cost_rate = (config.COMMISSION_BPS + config.SLIPPAGE_BPS) / 1e4

    net_rets = []
    gross_rets = []
    turnovers = []
    n_holdings = []

    for d in daily_index:
        if d in exec_map:
            new_weights = exec_map[d]
            turnover = (new_weights - current_weights).abs().sum()
            cost_today = turnover * cost_rate
            current_weights = new_weights
        else:
            turnover = 0.0
            cost_today = 0.0

        holdings_over_time.loc[d] = current_weights
        day_gross_ret = (current_weights * daily_rets.loc[d]).sum()
        day_net_ret = day_gross_ret - cost_today

        gross_rets.append(day_gross_ret)
        net_rets.append(day_net_ret)
        turnovers.append(turnover)
        n_holdings.append((current_weights > 1e-9).sum())

        # weights drift with returns between rebalances (cash portion,
        # 1 - sum(weights), earns zero and simply stays out of the sum)
        invested = current_weights.sum()
        if invested > 0:
            grossed = current_weights * (1 + daily_rets.loc[d])
            new_invested = grossed.sum()
            if new_invested > 0:
                current_weights = grossed * (invested / new_invested)

    result = pd.DataFrame({
        "gross_return": gross_rets,
        "net_return": net_rets,
        "turnover": turnovers,
        "n_holdings": n_holdings,
    }, index=daily_index)
    result["equity"] = 100 * (1 + result["net_return"]).cumprod()
    result.attrs["label"] = label
    return result


def buy_and_hold(price_series: pd.Series, label: str) -> pd.DataFrame:
    rets = price_series.pct_change().fillna(0.0)
    equity = 100 * (1 + rets).cumprod()
    df = pd.DataFrame({"gross_return": rets, "net_return": rets, "equity": equity})
    df.attrs["label"] = label
    return df


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def performance_summary(returns: pd.Series, benchmark_returns: pd.Series | None = None) -> dict:
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return {}
    ann_factor = config.TRADING_DAYS_PER_YEAR
    rf_daily = config.RISK_FREE_RATE_ANNUAL / ann_factor

    total_return = (1 + r).prod() - 1
    years = n / ann_factor
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
    ann_vol = r.std() * np.sqrt(ann_factor)

    excess = r - rf_daily
    sharpe = excess.mean() / r.std() * np.sqrt(ann_factor) if r.std() > 0 else np.nan

    downside = r[r < rf_daily]
    downside_dev = downside.std() * np.sqrt(ann_factor) if len(downside) > 1 else np.nan
    sortino = (r.mean() * ann_factor - config.RISK_FREE_RATE_ANNUAL) / downside_dev if downside_dev and downside_dev > 0 else np.nan

    equity = (1 + r).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    out = {
        "CAGR": cagr,
        "Ann. Volatility": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
        "Total Return": total_return,
    }

    if benchmark_returns is not None:
        b = benchmark_returns.reindex(r.index).dropna()
        common = r.index.intersection(b.index)
        r_c, b_c = r.loc[common], b.loc[common]
        if len(common) > 30 and b_c.std() > 0:
            cov = np.cov(r_c, b_c)
            beta = cov[0, 1] / cov[1, 1]
            alpha_daily = r_c.mean() - beta * b_c.mean()
            alpha_ann = alpha_daily * ann_factor
            active = r_c - b_c
            info_ratio = active.mean() / active.std() * np.sqrt(ann_factor) if active.std() > 0 else np.nan
            out.update({
                "Beta vs Benchmark": beta,
                "Alpha (ann.)": alpha_ann,
                "Information Ratio": info_ratio,
            })

    return out
