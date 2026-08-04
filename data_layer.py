"""
data_layer.py
==============
All external data access is isolated behind three abstract interfaces:

    MarketDataProvider    -> daily OHLCV price data
    SentimentProvider      -> daily sentiment score per ticker
    SupplyChainProvider    -> periodic supply-chain / working-capital health

Each interface has:
  (a) a REAL implementation that hits a genuine data source, and
  (b) a SYNTHETIC implementation used only when real data is unavailable
      (e.g. no network access), clearly labeled as such.

IMPORTANT HONESTY NOTE
-----------------------
There is no free, historically-complete API for "alternative data" like
premium social-media sentiment feeds (e.g. RavenPack, StockTwits Enterprise)
or true supply-chain tracking (e.g. satellite/shipping data from Thinknum,
Craft.co, or Bloomberg SPLC). The SimulatedSentimentProvider and the
synthetic supply-chain path exist to keep the *pipeline architecture*
correct and pluggable -- swap them for a real vendor by implementing the
same interface -- but they do not represent real markets. The
FundamentalSupplyChainProvider below IS real: it derives a genuine
supply-chain-health proxy (the cash conversion cycle) from public
financial statements, which is a legitimate technique used by real
quant desks that lack expensive tracking data.
"""

from __future__ import annotations

import abc
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config


# =============================================================================
# 1. MARKET DATA
# =============================================================================

class MarketDataProvider(abc.ABC):
    @abc.abstractmethod
    def get_prices(self, tickers: list[str], start, end) -> pd.DataFrame:
        """Return a DataFrame indexed by date, columns = MultiIndex
        (ticker, field) where field in {open, high, low, close, adj_close,
        volume}."""
        raise NotImplementedError


class YFinanceMarketDataProvider(MarketDataProvider):
    """
    REAL data source. Requires internet access to Yahoo Finance, which is
    NOT available in this sandbox (financial data hosts are blocked here),
    but works fine on a normal machine with `pip install yfinance`.

    NOTE: this code path could not be executed/tested in this sandbox.
    yfinance's exact return schema has shifted across versions; if column
    names differ on your installed version, adjust the `rename` map below.
    """

    def get_prices(self, tickers: list[str], start, end) -> pd.DataFrame:
        import yfinance as yf

        raw = yf.download(
            tickers, start=start, end=end, auto_adjust=False, group_by="ticker",
            progress=False, threads=True,
        )
        frames = {}
        for t in tickers:
            try:
                df = raw[t].rename(columns={
                    "Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
                })
                frames[t] = df[["open", "high", "low", "close", "adj_close", "volume"]]
            except KeyError:
                warnings.warn(f"No data returned for {t}; skipping.")
        if not frames:
            raise RuntimeError(
                "yfinance returned no usable data. Check your internet "
                "connection and ticker list."
            )
        return pd.concat(frames, axis=1)


@dataclass
class _TickerParams:
    beta: float
    idio_vol: float
    alpha_loading: float
    base_ccc: float  # base cash conversion cycle, days


class SyntheticMarketDataProvider(MarketDataProvider):
    """
    *** SYNTHETIC DATA -- NOT REAL MARKET HISTORY ***

    Generates a statistically-plausible daily OHLCV panel using:
      - a common "market factor" (mimics S&P 500 systematic risk) with a
        simple volatility-clustering process,
      - per-ticker beta exposure to that factor,
      - a persistent, mean-reverting *latent alpha* process per ticker that
        represents "true" (unobservable in real life) near-term fundamental
        momentum,
      - idiosyncratic noise.

    Critically, the latent alpha process is *also* what drives the synthetic
    sentiment and supply-chain signals produced elsewhere in this module
    (with realistic lag and noise). This means the synthetic alt-data
    features contain a genuine, injected, but noisy predictive signal about
    forward returns -- by construction, not by accident. That lets us test
    whether the ML pipeline can actually recover a known signal. It is a
    methodology check on the CODE, not evidence that real alt data predicts
    real markets.

    This class also exposes `latent_alpha_` so features.py and other
    providers can derive consistent sentiment/supply-chain series.
    """

    def __init__(self, seed: int = config.RANDOM_SEED):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.latent_alpha_: pd.DataFrame | None = None
        self.market_factor_: pd.Series | None = None
        self.dates_: pd.DatetimeIndex | None = None
        self._ticker_params: dict[str, _TickerParams] = {}

    def _simulate_market_factor(self, dates: pd.DatetimeIndex) -> pd.Series:
        n = len(dates)
        # simple GARCH(1,1)-like volatility clustering
        omega, a, b = 1e-6, 0.08, 0.90
        var = np.empty(n)
        var[0] = omega / (1 - a - b)
        shocks = self.rng.standard_t(df=5, size=n)  # fat tails
        rets = np.empty(n)
        daily_drift = 0.0003  # ~ +8%/yr systematic drift, roughly SPY-like
        for i in range(n):
            if i > 0:
                var[i] = omega + a * rets[i - 1] ** 2 + b * var[i - 1]
            rets[i] = daily_drift + np.sqrt(var[i]) * shocks[i] * 0.6
        return pd.Series(rets, index=dates, name="market_factor")

    def _simulate_latent_alpha(self, dates: pd.DatetimeIndex, ticker: str) -> pd.Series:
        n = len(dates)
        rho = 0.90  # persistence -> ~6-7 trading day half-life
        innov_std = 0.012
        z = self.rng.normal(0, innov_std, size=n)
        a = np.empty(n)
        a[0] = z[0]
        for i in range(1, n):
            a[i] = rho * a[i - 1] + z[i]
        return pd.Series(a, index=dates, name=ticker)

    def build(self, tickers: list[str], start, end) -> pd.DataFrame:
        dates = pd.bdate_range(start, end)
        self.dates_ = dates
        self.market_factor_ = self._simulate_market_factor(dates)

        latent = {}
        frames = {}
        for t in tickers:
            beta = float(np.clip(self.rng.normal(1.0, 0.18), 0.55, 1.65))
            idio_vol = float(self.rng.uniform(0.016, 0.032))  # ~25%-51% ann.
            # Deliberately WEAK alpha loading: real-world exploitable signals
            # are small and noisy relative to idiosyncratic risk. This range
            # is calibrated (by trial) to produce an out-of-sample rank IC in
            # roughly the 0.02-0.06 band, which is a realistic magnitude for
            # a genuine-but-modest cross-sectional equity signal.
            alpha_loading = float(self.rng.uniform(0.055, 0.095))
            base_ccc = float(self.rng.uniform(35, 95))
            self._ticker_params[t] = _TickerParams(beta, idio_vol, alpha_loading, base_ccc)

            la = self._simulate_latent_alpha(dates, t)
            latent[t] = la

            idio_noise = self.rng.normal(0, idio_vol, size=len(dates))
            # latent alpha at t-1 feeds return at t (predictive, avoids
            # same-day leakage)
            la_lag = la.shift(1).fillna(0.0).values
            daily_ret = (
                beta * self.market_factor_.values
                + alpha_loading * la_lag
                + idio_noise
            )
            price = 50 * np.exp(np.cumsum(daily_ret))  # arbitrary start=$50
            close = pd.Series(price, index=dates)
            # simple synthetic OHLV around close
            noise_range = close * self.rng.uniform(0.004, 0.012, size=len(dates))
            open_ = close.shift(1).fillna(close.iloc[0]) + self.rng.normal(0, 1, len(dates)) * noise_range * 0.1
            high = np.maximum(open_, close) + noise_range
            low = np.minimum(open_, close) - noise_range
            volume = pd.Series(
                self.rng.lognormal(mean=13.5, sigma=0.5, size=len(dates)), index=dates
            ).round()

            frames[t] = pd.DataFrame({
                "open": open_, "high": high, "low": low, "close": close,
                "adj_close": close, "volume": volume,
            })

        self.latent_alpha_ = pd.DataFrame(latent)
        return pd.concat(frames, axis=1)

    def get_prices(self, tickers: list[str], start, end) -> pd.DataFrame:
        return self.build(tickers, start, end)

    def build_benchmark(self, start, end) -> pd.Series:
        """Synthetic S&P-500-like benchmark: same market factor, beta=1,
        low idio vol, no injected alpha (it's the market, not a stock-picking
        target)."""
        if self.market_factor_ is None:
            dates = pd.bdate_range(start, end)
            self.market_factor_ = self._simulate_market_factor(dates)
        idio = self.rng.normal(0, 0.004, size=len(self.market_factor_))
        ret = self.market_factor_.values + idio
        price = 200 * np.exp(np.cumsum(ret))
        return pd.Series(price, index=self.market_factor_.index, name=config.BENCHMARK)


# =============================================================================
# 2. SENTIMENT DATA
# =============================================================================

class SentimentProvider(abc.ABC):
    @abc.abstractmethod
    def get_sentiment(self, tickers: list[str], start, end) -> pd.DataFrame:
        """Return DataFrame indexed by date, columns = tickers, values in
        roughly [-1, 1]."""
        raise NotImplementedError


class SimulatedSentimentProvider(SentimentProvider):
    """
    *** SYNTHETIC PROXY -- NOT REAL SOCIAL DATA ***

    Stands in for a real feed (e.g. StockTwits API, X/Twitter API v2 filtered
    stream + a finetuned classifier, or a paid vendor like RavenPack). To
    plug in a real source, implement `get_sentiment` with the same signature:
    pull raw posts/headlines per ticker per day, run them through a sentiment
    classifier (e.g. a finetuned transformer or VADER as a cheap baseline),
    and aggregate to a daily score.

    Here, the score is a noisy, lagged read on the same `latent_alpha`
    process used to generate prices -- i.e. it is a deliberately imperfect
    window onto the true predictive signal, similar to how real sentiment
    is a noisy proxy for actual forward fundamentals.
    """

    def __init__(self, market_provider: SyntheticMarketDataProvider):
        self.mp = market_provider

    def get_sentiment(self, tickers: list[str], start, end) -> pd.DataFrame:
        if self.mp.latent_alpha_ is None:
            raise RuntimeError("Call market_provider.build(...) before requesting sentiment.")
        rng = np.random.default_rng(config.RANDOM_SEED + 1)
        la = self.mp.latent_alpha_[tickers]
        # Sentiment is a NOISY, weak proxy of the latent signal -- real
        # social sentiment is a diluted, delayed echo of fundamentals, not
        # a clean readout of them.
        noise = rng.normal(0, la.std().mean() * 4.0, size=la.shape)
        raw = 0.6 * la.values + noise
        scaled = np.tanh(raw / (np.nanstd(raw) + 1e-9))  # squash to [-1, 1]
        return pd.DataFrame(scaled, index=la.index, columns=tickers)


# =============================================================================
# 3. SUPPLY-CHAIN / WORKING-CAPITAL DATA
# =============================================================================

class SupplyChainProvider(abc.ABC):
    @abc.abstractmethod
    def get_supply_chain_metrics(self, tickers: list[str], start, end) -> pd.DataFrame:
        """Return DataFrame indexed by date, MultiIndex columns
        (ticker, metric) with metric in {cash_conversion_cycle,
        inventory_growth}. Values should already be forward-filled and
        lagged to their real-world public-availability date."""
        raise NotImplementedError


class FundamentalSupplyChainProvider(SupplyChainProvider):
    """
    REAL methodology: derives a supply-chain-health proxy from public
    financial statements, since genuine tracking data (satellite imagery,
    shipment/customs records, IoT sensor feeds from vendors like Thinknum
    or Craft.co) is an expensive commercial product this system does not
    have access to.

    Computes the Cash Conversion Cycle (CCC) each fiscal quarter:
        DIO = 365 * Inventory / COGS
        DSO = 365 * Accounts Receivable / Revenue
        DPO = 365 * Accounts Payable / COGS
        CCC = DIO + DSO - DPO
    A shortening CCC generally indicates a tightening, healthier supply
    chain / working-capital cycle; a lengthening CCC often signals
    inventory buildup or collection trouble.

    NOTE: like the real price provider, this hits yfinance and could not be
    executed in this sandbox. Field names in yfinance's financial-statement
    tables have changed across versions -- verify against your installed
    version (`Ticker.quarterly_balance_sheet.index` /
    `Ticker.quarterly_income_stmt.index`) before relying on this in
    production.
    """

    def get_supply_chain_metrics(self, tickers: list[str], start, end) -> pd.DataFrame:
        import yfinance as yf

        frames = {}
        for t in tickers:
            try:
                tk = yf.Ticker(t)
                bs = tk.quarterly_balance_sheet
                inc = tk.quarterly_income_stmt

                inventory = bs.loc["Inventory"]
                receivables = bs.loc["Receivables"]
                payables = bs.loc["Payables"]
                cogs = inc.loc["Cost Of Revenue"]
                revenue = inc.loc["Total Revenue"]

                dio = 365 * inventory / cogs
                dso = 365 * receivables / revenue
                dpo = 365 * payables / cogs
                ccc = (dio + dso - dpo).sort_index()
                inv_growth = inventory.sort_index().pct_change()

                df = pd.DataFrame({
                    "cash_conversion_cycle": ccc,
                    "inventory_growth": inv_growth,
                })
                # apply realistic public-availability lag, then forward-fill
                # to daily so a rebalance on any given day only ever sees
                # data that would genuinely have been public by then
                df.index = pd.to_datetime(df.index) + pd.Timedelta(
                    days=config.FUNDAMENTAL_REPORT_LAG_DAYS
                )
                daily_index = pd.bdate_range(start, end)
                df = df.reindex(daily_index, method="ffill")
                frames[t] = df
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"Supply-chain fetch failed for {t}: {e}")
        if not frames:
            raise RuntimeError("No supply-chain data retrieved for any ticker.")
        return pd.concat(frames, axis=1)


class SimulatedSupplyChainProvider(SupplyChainProvider):
    """
    *** SYNTHETIC PROXY -- NOT REAL FUNDAMENTALS ***

    Generates a quarterly cash-conversion-cycle-like series per ticker that
    is a noisy function of the same latent alpha driving returns (improving
    "true" fundamentals -> shortening CCC), reported with a realistic lag
    and forward-filled daily, mirroring how real fundamental data actually
    reaches the market in discrete, lagged steps rather than continuously.
    """

    def __init__(self, market_provider: SyntheticMarketDataProvider):
        self.mp = market_provider

    def get_supply_chain_metrics(self, tickers: list[str], start, end) -> pd.DataFrame:
        if self.mp.latent_alpha_ is None:
            raise RuntimeError("Call market_provider.build(...) before requesting supply-chain data.")
        rng = np.random.default_rng(config.RANDOM_SEED + 2)
        daily_index = self.mp.latent_alpha_.index
        quarter_ends = pd.date_range(start, end, freq="QE")

        frames = {}
        for t in tickers:
            params = self.mp._ticker_params[t]
            la_daily = self.mp.latent_alpha_[t]
            rows = []
            for qe in quarter_ends:
                window = la_daily[(la_daily.index > qe - pd.Timedelta(days=91)) & (la_daily.index <= qe)]
                mean_alpha = window.mean() if len(window) else 0.0
                # Also a weak, noisy proxy -- quarterly fundamentals only
                # partially reflect the true near-term signal.
                ccc = params.base_ccc - 130 * mean_alpha + rng.normal(0, 9.0)
                inv_growth = -0.5 * mean_alpha + rng.normal(0, 0.045)
                rows.append({"date": qe, "cash_conversion_cycle": ccc, "inventory_growth": inv_growth})
            qdf = pd.DataFrame(rows).set_index("date")
            qdf.index = qdf.index + pd.Timedelta(days=config.FUNDAMENTAL_REPORT_LAG_DAYS)
            qdf = qdf.reindex(daily_index, method="ffill").bfill()
            frames[t] = qdf
        return pd.concat(frames, axis=1)


# =============================================================================
# Loader with automatic real -> synthetic fallback
# =============================================================================

def load_all_data(tickers: list[str], benchmark: str, start, end, force_synthetic: bool = False):
    """
    Attempts real data first (yfinance): real prices + real fundamentals-
    derived supply-chain proxy. There is no free historical sentiment
    source, so in real mode `sentiment` comes back as None and
    features.py automatically skips sentiment features rather than
    crashing -- you still get a genuine backtest using real prices and a
    real alt-data proxy (supply chain), just without sentiment until you
    wire in your own SentimentProvider (see README).

    Falls back to clearly-labeled fully-synthetic data (including
    simulated sentiment) if real data is unreachable (e.g. this sandbox)
    or on any fetch error.

    Returns (prices, benchmark_prices, sentiment_or_None, supply_chain, is_synthetic).
    """
    if not force_synthetic:
        try:
            mp = YFinanceMarketDataProvider()
            prices = mp.get_prices(tickers, start, end)
            bench = mp.get_prices([benchmark], start, end)[benchmark]["adj_close"]
            sp = FundamentalSupplyChainProvider()
            supply = sp.get_supply_chain_metrics(tickers, start, end)
            print("[data_layer] Real prices + real fundamentals-based supply-chain "
                  "data loaded. No real sentiment source is configured, so "
                  "sentiment features will be skipped this run (see README to add one).")
            return prices, bench, None, supply, False
        except Exception as e:  # noqa: BLE001
            print(f"[data_layer] Real data path unavailable ({e}). "
                  f"Falling back to SYNTHETIC data for this demo run.")

    print("=" * 78)
    print("DATA SOURCE: SYNTHETIC (simulated). This is NOT real market history.")
    print("See data_layer.py docstring / README for what this demonstrates")
    print("and how to switch to real data on a machine with data-vendor access.")
    print("=" * 78)
    mp = SyntheticMarketDataProvider(seed=config.RANDOM_SEED)
    prices = mp.build(tickers, start, end)
    bench = mp.build_benchmark(start, end)

    sent_provider = SimulatedSentimentProvider(mp)
    sentiment = sent_provider.get_sentiment(tickers, start, end)

    sc_provider = SimulatedSupplyChainProvider(mp)
    supply = sc_provider.get_supply_chain_metrics(tickers, start, end)

    return prices, bench, sentiment, supply, True
