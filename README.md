# Mid-Cap Tech Alpha

A research project for backtesting a machine-learning trading strategy on mid-cap tech stocks, using a mix of technical indicators, sentiment, and a supply-chain-health signal pulled from company financials.

I built this as a self-contained pipeline you can actually run and inspect, not a black box — every stage (data, features, model, backtest) is a separate file so you can swap pieces out or poke at intermediate results.

## What it does

1. Pulls daily price data for a basket of mid-cap tech names + the S&P 500 as a benchmark
2. Builds a feature set: momentum, RSI, MACD, volatility, plus a sentiment signal and a supply-chain proxy (cash conversion cycle, derived from quarterly financials)
3. Trains a gradient-boosted model to predict which stocks will outperform the group over the next few days, retraining periodically as it walks forward through history (no peeking at the future)
4. Builds a long-only portfolio from the model's picks, rebalanced weekly, with trading costs factored in
5. Compares it against a couple of simple baselines (naive momentum, equal-weighting the whole basket) and against just holding the S&P 500

## Before you run it — the honest version

I don't have a free, historical source for real social-media sentiment data (that stuff is usually paid — RavenPack, StockTwits' enterprise tier, etc.), so out of the box this runs on synthetic sentiment. The price data and the supply-chain numbers are real (via yfinance) once you use the `--real` flag; sentiment isn't, and the code says so explicitly when it runs.

Also, worth saying plainly: **nothing here is a guarantee this beats the market.** While building this I hit a backtest that showed a 200%+ annual return, which was obviously a bug (a look-ahead bias where the model was accidentally trading on same-day information), not a genius model. I fixed it, and after the fix the "smart" ML strategy didn't even clearly beat a plain equal-weighted basket of the same stocks in my own test run — a simple momentum rule did better. That's a pretty normal outcome in quant research, and it's exactly the kind of thing this project is set up to reveal rather than hide. Don't trust any backtest, mine or yours, that looks too good on the first try.

## Quickstart

```bash
git clone <this-repo>
cd mts-alpha
pip install -r requirements.txt
python main.py --real
```

`--real` pulls live data from Yahoo Finance for the last 10 years. Drop the flag and it'll run on synthetic data instead — useful for testing the pipeline without needing a network connection, or just to see how it behaves.

Results land in `outputs/`:
- `equity_curves.png` — strategy vs. baselines vs. S&P 500, growth of $100
- `drawdown.png` — how far underwater each strategy got, and when
- `performance_report.md` — the actual numbers: CAGR, Sharpe, max drawdown, alpha/beta vs. benchmark, etc.

## Project layout

```
config.py       tickers, date range, costs, model settings — start here to tweak things
data_layer.py   data fetching, with real and synthetic versions of each data source
features.py     turns raw prices/sentiment/fundamentals into model inputs
model.py        the walk-forward training loop + a couple of baseline models
backtest.py     turns predictions into a portfolio and simulates trading it
main.py         runs the whole thing end to end
```

Prices and the supply-chain data have real implementations behind `yfinance`. Sentiment doesn't — there's an interface for it (`SentimentProvider` in `data_layer.py`) so you can plug in a real feed if you have access to one; until then it falls back to a synthetic placeholder and the pipeline just runs without a sentiment signal in real-data mode.

## Things worth knowing before you take this seriously

- **The stock list is static and hand-picked.** A more rigorous setup would reconstruct the mid-cap universe at each point in time, since companies move in and out of that size bracket over a decade. Picking a list of names that are all still around today is a mild form of survivorship bias.
- **It's long-only.** No shorting, so no borrow costs to worry about, but also no way to bet against a stock.
- **The trading-cost assumption is a flat estimate**, not a real market-impact model. Fine for testing an idea, not fine for judging how much money you could actually allocate to this.
- **Ten years of daily data across ~15 stocks isn't a huge dataset** for a model with a dozen-plus features. Overfitting is a real risk, and my own test run is a live example: the fancier model didn't beat simple momentum.
- Markets change. Whatever edge you might find in a historical backtest can shrink or vanish once enough people are trading on it, or the market regime shifts.

## License

MIT — see `LICENSE`. Do whatever you want with it, just don't blame me if you lose money trading on it.

## Not investment advice

This is a coding/research project, not a recommendation to buy or sell anything. Backtested performance — real or synthetic — doesn't predict future results. If you're going to trade on anything built from this, that's your call and your risk.
