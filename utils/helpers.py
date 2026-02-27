"""
utils/helpers.py
Shared helper functions for the PEAD Earnings Prediction project.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. TICKER UNIVERSE
# ─────────────────────────────────────────────

def get_sp500_tickers():
    """Scrape current S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    tickers = table['Symbol'].str.replace('.', '-', regex=False).tolist()
    sectors = dict(zip(
        table['Symbol'].str.replace('.', '-', regex=False),
        table['GICS Sector']
    ))
    return tickers, sectors


# ─────────────────────────────────────────────
# 2. PRICE DATA
# ─────────────────────────────────────────────

def get_price_data(ticker, start="2019-01-01", end=None):
    """
    Download adjusted daily OHLCV from yfinance.
    Returns None if download fails or insufficient data.
    """
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start, end=end, auto_adjust=True)
        if df.empty or len(df) < 60:
            return None
        df.index = pd.to_datetime(df.index)
        if df.index.tzinfo is not None:
                df.index = df.index.tz_convert('America/New_York').tz_localize(None)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        return df.sort_index()
    except Exception:
        return None


def get_spx_returns(start="2019-01-01", end=None):
    """Download S&P 500 daily returns for alpha calculation."""
    spx = get_price_data("^GSPC", start=start, end=end)
    if spx is None:
        return None
    spx['spx_ret'] = spx['close'].pct_change()
    return spx[['close', 'spx_ret']].rename(columns={'close': 'spx_close'})


def get_macro_data(start="2019-01-01", end=None):
    """
    Download:
      ^TNX  — 10-Year Treasury Yield
      ^VIX  — CBOE Volatility Index
    Returns a merged daily DataFrame.
    """
    tnx = get_price_data("^TNX", start=start, end=end)
    vix = get_price_data("^VIX", start=start, end=end)

    if tnx is None or vix is None:
        return None

    macro = pd.DataFrame({
        'tnx': tnx['close'],
        'vix': vix['close']
    })
    macro = macro.ffill()

    # Regime flags
    macro['rate_regime'] = (macro['tnx'] >= 2.0).astype(int)   # 1 = High Rate
    macro['vix_regime']  = (macro['vix']  >= 20.0).astype(int) # 1 = Fear

    # Continuous features
    macro['tnx_1m_chg'] = macro['tnx'].pct_change(21)   # rate trend
    macro['vix_5d_chg'] = macro['vix'].pct_change(5)    # fear rising/falling

    return macro


# ─────────────────────────────────────────────
# 3. EARNINGS DATES & SURPRISE
# ─────────────────────────────────────────────

def get_earnings_history(ticker):
    """
    Pull earnings history from yfinance.
    Returns DataFrame with EPS actual, estimate, surprise.
    Note: yfinance does NOT provide analyst estimate StdDev for proper SUE.
    We compute a proxy SUE using rolling forecast error std.
    Capital IQ users should replace eps_std with true consensus dispersion.
    """
    try:
        t = yf.Ticker(ticker)
        cal = t.get_earnings_dates(limit=40)
        if cal is None or cal.empty:
            return None

        cal = cal.copy()
        cal.index = pd.to_datetime(cal.index).tz_localize(None)
        cal = cal.sort_index()

        # Keep only rows with actual results
        cal = cal.dropna(subset=['EPS Actual'])
        cal = cal.rename(columns={
            'EPS Actual':   'eps_actual',
            'EPS Estimate': 'eps_est',
            'Reported EPS': 'eps_reported'
        })

        # Raw surprise
        cal['eps_surprise'] = cal['eps_actual'] - cal['eps_est']

        # Proxy SUE: surprise / rolling std of past forecast errors
        # (Replace eps_std with Capital IQ dispersion if available)
        cal['eps_std'] = cal['eps_surprise'].rolling(4, min_periods=2).std().shift(1)
        cal['sue'] = cal['eps_surprise'] / (cal['eps_std'].abs() + 1e-9)
        cal['sue'] = cal['sue'].clip(-10, 10)  # winsorise outliers

        # Surprise as % of |estimate|
        cal['eps_surprise_pct'] = (
            cal['eps_surprise'] / cal['eps_est'].abs().replace(0, np.nan)
        )

        # Historical beat rate (trailing 4 quarters, computed at each row)
        cal['beat'] = (cal['eps_actual'] > cal['eps_est']).astype(int)
        cal['hist_beat_rate'] = cal['beat'].shift(1).rolling(4, min_periods=2).mean()

        # Consecutive beats streak
        def consec_streak(series):
            streaks = []
            count = 0
            for v in series:
                if v == 1:
                    count += 1
                else:
                    count = 0
                streaks.append(count)
            return pd.Series(streaks, index=series.index)
        cal['beat_streak'] = consec_streak(cal['beat'].shift(1).fillna(0))

        return cal

    except Exception:
        return None


# ─────────────────────────────────────────────
# 4. TECHNICAL FEATURE COMPUTATION
# ─────────────────────────────────────────────

def compute_technical_features(pre_prices):
    """
    Compute all pre-earnings technical signals.
    pre_prices: DataFrame with OHLCV columns, sorted ascending,
                covering ~30 trading days before earnings.
    """
    feats = {}
    close  = pre_prices['close']
    volume = pre_prices['volume']
    high   = pre_prices['high']
    low    = pre_prices['low']

    if len(close) < 5:
        return feats

    # ── Momentum ──
    for n in [5, 10, 14, 20]:
        if len(close) >= n + 1:
            feats[f'ret_{n}d'] = close.iloc[-1] / close.iloc[-(n+1)] - 1

    # ── RSI (14-day) ──
    delta = close.diff().dropna()
    gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rs    = gain / (loss + 1e-9)
    feats['rsi_14'] = float(100 - (100 / (1 + rs.iloc[-1])))

    # ── MA deviation ──
    if len(close) >= 20:
        feats['pct_above_ma20'] = close.iloc[-1] / close.tail(20).mean() - 1
    if len(close) >= 50:
        feats['pct_above_ma50'] = close.iloc[-1] / close.tail(50).mean() - 1

    # ── Realized volatility (annualised) ──
    log_ret = np.log(close / close.shift(1)).dropna()
    if len(log_ret) >= 10:
        feats['rvol_10d'] = float(log_ret.tail(10).std() * np.sqrt(252))
    if len(log_ret) >= 20:
        feats['rvol_20d'] = float(log_ret.tail(20).std() * np.sqrt(252))

    # ── Vol percentile (IV proxy) ──
    if len(log_ret) >= 60:
        roll_vol = log_ret.rolling(20).std() * np.sqrt(252)
        cur_vol  = roll_vol.iloc[-1]
        feats['rvol_pctile'] = float((roll_vol < cur_vol).mean())

    # ── Term structure proxy: short-term vol / long-term vol ──
    if len(log_ret) >= 60:
        st = log_ret.tail(10).std()
        lt = log_ret.tail(60).std()
        feats['vol_term_ratio'] = float(st / (lt + 1e-9))

    # ── Volume anomaly ──
    if len(volume) >= 10:
        vol_base = volume.iloc[:-5].mean()
        feats['vol_ratio_5d'] = float(volume.tail(5).mean() / (vol_base + 1))
        feats['vol_ratio_1d'] = float(volume.iloc[-1]        / (vol_base + 1))

    # ── ATR (normalised) ──
    if len(pre_prices) >= 14:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs()
        ], axis=1).max(axis=1)
        feats['atr_pct'] = float(tr.tail(14).mean() / (close.iloc[-1] + 1e-9))

    return feats


def compute_ticker_history_features(price_df, past_earn_dates):
    """
    Compute how this specific ticker has historically behaved on earnings.
    Captures idiosyncratic earnings vol (e.g. NFLX always moves big).
    """
    feats = {}
    past_moves = []

    for ed in past_earn_dates[-8:]:
        future = price_df[price_df.index > ed]
        prev   = price_df[price_df.index <= ed]
        if len(future) >= 1 and len(prev) >= 1:
            m = future.iloc[0]['close'] / prev.iloc[-1]['close'] - 1
            past_moves.append(m)

    if len(past_moves) >= 2:
        feats['ticker_avg_abs_move'] = float(np.mean(np.abs(past_moves)))
        feats['ticker_avg_move']     = float(np.mean(past_moves))
        feats['ticker_move_std']     = float(np.std(past_moves))
        feats['ticker_up_rate']      = float(np.mean([m > 0 for m in past_moves]))

    return feats


# ─────────────────────────────────────────────
# 5. TARGET VARIABLE
# ─────────────────────────────────────────────

def compute_targets(post_stock_ret, post_spx_ret):
    """
    PRIMARY: Excess return vs S&P 500 over 5 days post-earnings.
    Returns binary (1 = outperform) and the raw excess return.
    """
    excess = post_stock_ret - post_spx_ret
    binary = int(excess > 0)
    return binary, excess


def compute_magnitude_class(excess_ret, threshold=0.05):
    """
    3-class target:
      2 = Significant Beat  (excess > +5%)
      1 = Neutral           (-5% to +5%)
      0 = Significant Miss  (excess < -5%)
    """
    if excess_ret > threshold:
        return 2
    elif excess_ret < -threshold:
        return 0
    else:
        return 1
