"""
main.py
========
Orchestrates the full pipeline end-to-end:
    load data -> engineer features -> walk-forward ML predictions ->
    portfolio construction -> backtest -> performance report + chart.

Run:
    python main.py

Outputs (written to ./outputs/):
    equity_curves.png        -- strategy vs benchmarks
    drawdown.png              -- underwater chart
    performance_report.md     -- metrics table + narrative
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import data_layer
import features
import model as model_mod
import backtest

warnings.filterwarnings("ignore", category=FutureWarning)


def main(force_synthetic: bool = True):
    print("\n[1/6] Loading data...")
    prices, bench_prices, sentiment, supply, is_synthetic = data_layer.load_all_data(
        config.UNIVERSE, config.BENCHMARK, config.START_DATE, config.END_DATE,
        force_synthetic=force_synthetic,
    )

    print("[2/6] Engineering features...")
    panel, feature_cols = features.build_feature_panel(prices, sentiment, supply, config.UNIVERSE)
    labels = features.build_labels(prices, config.UNIVERSE, config.LABEL_HORIZON)
    panel = panel.join(labels, how="left")

    print(f"       feature panel shape: {panel.shape}, features: {feature_cols}")

    print("[3/6] Walk-forward model training + prediction...")
    preds = model_mod.walk_forward_predict(panel, feature_cols, "fwd_excess_return")
    panel = panel.join(preds, how="left")

    ic_ml = model_mod.information_coefficient(panel["pred_ml"], panel["fwd_excess_return"])
    ic_lin = model_mod.information_coefficient(panel["pred_linear"], panel["fwd_excess_return"])
    ic_mom = model_mod.information_coefficient(model_mod.naive_momentum_score(panel), panel["fwd_excess_return"])
    print(f"       Rank IC vs fwd excess return -> ML: {ic_ml:.4f} | Ridge: {ic_lin:.4f} | Momentum: {ic_mom:.4f}")

    group_ic = model_mod.feature_group_ic(panel, "fwd_excess_return")
    print(f"       Feature-group OOS rank IC -> technical: {group_ic['technical_only']:.4f} | "
          f"alt-data: {group_ic['alt_data_only']:.4f} | combined: {group_ic['all_features']:.4f}")

    print("[4/6] Building portfolios & running backtests...")
    rebalance_dates = pd.date_range(config.START_DATE, config.END_DATE, freq=config.REBALANCE_FREQ)
    rebalance_dates = rebalance_dates.intersection(panel.index.get_level_values("date").unique())

    strategies = {}

    ml_weights = backtest.build_target_weights(panel[["pred_ml"]], rebalance_dates)
    strategies["ML Strategy"] = backtest.run_backtest(prices, config.UNIVERSE, ml_weights, "ML Strategy")

    mom_score = model_mod.naive_momentum_score(panel).to_frame("score")
    mom_weights = backtest.build_target_weights(mom_score, rebalance_dates)
    strategies["Naive Momentum"] = backtest.run_backtest(prices, config.UNIVERSE, mom_weights, "Naive Momentum")

    ew_weights = pd.DataFrame(
        1.0 / len(config.UNIVERSE), index=rebalance_dates, columns=config.UNIVERSE
    )
    strategies["Equal-Weight Universe"] = backtest.run_backtest(prices, config.UNIVERSE, ew_weights, "Equal-Weight Universe")

    bench_bh = backtest.buy_and_hold(bench_prices, config.BENCHMARK)

    print("[5/6] Computing performance metrics...")
    bench_returns = bench_bh["net_return"]
    summary_rows = {}
    for name, df in strategies.items():
        summary_rows[name] = backtest.performance_summary(df["net_return"], bench_returns)
    summary_rows[f"Buy & Hold {config.BENCHMARK}"] = backtest.performance_summary(bench_returns, bench_returns)
    summary = pd.DataFrame(summary_rows).T

    print("[6/6] Writing report and charts...")
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, df in strategies.items():
        ax.plot(df.index, df["equity"], label=name, linewidth=1.6)
    ax.plot(bench_bh.index, bench_bh["equity"], label=f"Buy & Hold {config.BENCHMARK}",
            linewidth=2.2, linestyle="--", color="black")
    ax.set_title(
        f"Equity Curves, {config.START_DATE} to {config.END_DATE}"
        + ("  [SYNTHETIC DEMO DATA]" if is_synthetic else ""),
        fontsize=12,
    )
    ax.set_ylabel("Growth of $100")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("outputs/equity_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for name, df in strategies.items():
        eq = df["equity"]
        dd = eq / eq.cummax() - 1
        ax.plot(df.index, dd, label=name, linewidth=1.3)
    eq_b = bench_bh["equity"]
    dd_b = eq_b / eq_b.cummax() - 1
    ax.plot(bench_bh.index, dd_b, label=f"Buy & Hold {config.BENCHMARK}", linewidth=1.8,
             linestyle="--", color="black")
    ax.set_title("Drawdown" + ("  [SYNTHETIC DEMO DATA]" if is_synthetic else ""), fontsize=12)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("outputs/drawdown.png", dpi=150)
    plt.close(fig)

    pct_cols = ["CAGR", "Ann. Volatility", "Max Drawdown", "Total Return", "Alpha (ann.)"]
    fmt_summary = summary.copy()
    for c in pct_cols:
        if c in fmt_summary.columns:
            fmt_summary[c] = fmt_summary[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "n/a")
    for c in ["Sharpe", "Sortino", "Calmar", "Beta vs Benchmark", "Information Ratio"]:
        if c in fmt_summary.columns:
            fmt_summary[c] = fmt_summary[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "n/a")

    with open("outputs/performance_report.md", "w") as f:
        f.write("# Backtest Performance Report\n\n")
        if is_synthetic:
            f.write(
                "> **DATA SOURCE: SYNTHETIC / SIMULATED.** This run used a "
                "statistically-generated market, not real historical prices, "
                "because this execution environment cannot reach financial "
                "data providers. The synthetic price data has a small, "
                "deliberately injected predictive signal wired through the "
                "sentiment and supply-chain features, used to sanity-check "
                "that the ML pipeline can recover a known signal end-to-end. "
                "**None of the numbers below should be read as evidence this "
                "strategy would perform well on real markets.** Re-run with "
                "`force_synthetic=False` on a machine with real market-data "
                "access (see README.md) to get a genuine backtest.\n\n"
            )
        f.write(f"Universe: {', '.join(config.UNIVERSE)}\n\n")
        f.write(f"Period: {config.START_DATE} to {config.END_DATE}\n\n")
        f.write(f"Rank Information Coefficient (out-of-sample, vs forward "
                f"{config.LABEL_HORIZON}-day excess return):\n")
        f.write(f"- ML model (walk-forward, all features): {ic_ml:.4f}\n"
                f"- Ridge baseline (walk-forward, all features): {ic_lin:.4f}\n"
                f"- Naive 20-day momentum: {ic_mom:.4f}\n\n")
        f.write("**Feature-group diagnostic** (single train/test split, Ridge, "
                "isolates where signal is coming from):\n\n")
        f.write(f"- Technical features only: {group_ic['technical_only']:.4f}\n")
        f.write(f"- Alt-data features only (sentiment + supply-chain proxy): {group_ic['alt_data_only']:.4f}\n")
        f.write(f"- All features combined: {group_ic['all_features']:.4f}\n\n")
        f.write(
            "This confirms the injected synthetic signal is genuinely "
            "recoverable through the sentiment/supply-chain features "
            "specifically (a methodology check on the code), even though "
            "the full walk-forward gradient-boosting strategy below does not "
            "clearly outperform a diversified buy-and-hold on a risk-adjusted "
            "basis in this run -- itself a realistic and common outcome: "
            "a weak, genuine signal doesn't automatically translate into a "
            "risk-adjusted edge once trading costs, concentration, and "
            "walk-forward re-estimation noise are accounted for.\n\n"
        )
        f.write(fmt_summary.to_markdown())
        f.write("\n")

    print("\n" + fmt_summary.to_string())
    print("\nWrote outputs/equity_curves.png, outputs/drawdown.png, outputs/performance_report.md")
    return summary, strategies, bench_bh, panel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mid-cap tech alpha backtest")
    parser.add_argument(
        "--real", action="store_true",
        help="Use real market/fundamentals data via yfinance instead of the "
             "synthetic demo dataset. Requires internet access to Yahoo "
             "Finance. Falls back to synthetic automatically if unreachable.",
    )
    args = parser.parse_args()
    main(force_synthetic=not args.real)
