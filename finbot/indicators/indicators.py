"""
FINBOT Technical Indicators Library
-----------------------------------
Comprehensive vectorized implementations of technical indicators
for quantitative trading and research.

This module is optimized for clarity, speed, and compatibility with pandas Series/DataFrames.
It supports SMA, EMA, RSI, MACD, Bollinger Bands, ATR, VWAP, and many others.

Dependencies:
    - numpy
    - pandas
    - scipy (optional, for linear regression-based indicators)

Author: Mohith
Version: 2.0.0
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union

# Optional SciPy support for linear regression
try:
    from scipy import stats
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# =====================================================================
# 🧮 Moving Averages
# =====================================================================

def sma(series: pd.Series, window: int = 14) -> pd.Series:
    """Simple Moving Average (SMA)."""
    return series.rolling(window, min_periods=1).mean()


def ema(series: pd.Series, window: int = 14) -> pd.Series:
    """Exponential Moving Average (EMA)."""
    return series.ewm(span=window, adjust=False).mean()


def wma(series: pd.Series, window: int = 14) -> pd.Series:
    """Weighted Moving Average (WMA)."""
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def dema(series: pd.Series, window: int = 20) -> pd.Series:
    """Double Exponential Moving Average (DEMA)."""
    e = ema(series, window)
    return 2 * e - ema(e, window)


def tema(series: pd.Series, window: int = 20) -> pd.Series:
    """Triple Exponential Moving Average (TEMA)."""
    e1 = ema(series, window)
    e2 = ema(e1, window)
    e3 = ema(e2, window)
    return 3 * (e1 - e2) + e3


def hull_ma(series: pd.Series, window: int = 21) -> pd.Series:
    """Hull Moving Average (HMA)."""
    half = int(window / 2)
    sqrt_w = int(np.sqrt(window))
    raw = 2 * wma(series, half) - wma(series, window)
    return wma(raw.dropna(), sqrt_w).reindex(series.index)


# =====================================================================
# 📈 Momentum Indicators
# =====================================================================

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def roc(series: pd.Series, window: int = 12) -> pd.Series:
    """Rate of Change (ROC)."""
    return series.pct_change(window) * 100


def trix(series: pd.Series, window: int = 15) -> pd.Series:
    """Triple Exponential Average (TRIX)."""
    e = ema(ema(ema(series, window), window), window)
    return e.pct_change() * 100


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Moving Average Convergence Divergence (MACD)."""
    close = df['close']
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({'macd': macd_line, 'signal': signal_line, 'histogram': hist}, index=df.index)


# =====================================================================
# 🧭 Volatility Indicators
# =====================================================================

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range (ATR)."""
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def bollinger(series: pd.Series, window: int = 20, n_std: int = 2) -> pd.DataFrame:
    """Bollinger Bands."""
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    bandwidth = (upper - lower) / ma
    percent_b = (series - lower) / (upper - lower)
    return pd.DataFrame({
        'ma': ma,
        'upper': upper,
        'lower': lower,
        'bandwidth': bandwidth,
        'percent_b': percent_b
    })


def keltner(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, multiplier: float = 1.5) -> pd.DataFrame:
    """Keltner Channels."""
    mid = ema(df['close'], ema_period)
    atrv = atr(df, atr_period)
    return pd.DataFrame({
        'mid': mid,
        'upper': mid + multiplier * atrv,
        'lower': mid - multiplier * atrv
    })


# =====================================================================
# 💰 Volume-Based Indicators
# =====================================================================

def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume (OBV)."""
    direction = np.sign(df['close'].diff()).fillna(0)
    return (direction * df['volume']).cumsum()


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume Weighted Average Price (VWAP)."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    pv = tp * df['volume']
    return pv.cumsum() / df['volume'].cumsum()


def money_flow_index(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Money Flow Index (MFI)."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    pos = mf.where(tp > tp.shift(1), 0).rolling(window).sum()
    neg = mf.where(tp < tp.shift(1), 0).rolling(window).sum()
    mfr = pos / neg.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


# =====================================================================
# 📊 Trend & Directional Indicators
# =====================================================================

def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average Directional Index (ADX)."""
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    up_move = df['high'].diff()
    down_move = -df['low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atrv = tr.rolling(window).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window).sum() / atrv)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window).sum() / atrv)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], 0) * 100
    return dx.rolling(window).mean()


# =====================================================================
# 📉 Statistical Indicators
# =====================================================================

def rolling_corr(s1: pd.Series, s2: pd.Series, window: int = 30) -> pd.Series:
    """Rolling Pearson correlation."""
    return s1.rolling(window).corr(s2)


def least_squares_ma(series: pd.Series, window: int = 14) -> pd.Series:
    """Least Squares Moving Average (regression-based)."""
    out = pd.Series(index=series.index, dtype=float)
    for i in range(window - 1, len(series)):
        y = series.iloc[i - window + 1:i + 1].values
        x = np.arange(len(y))
        if _HAS_SCIPY:
            slope, intercept, *_ = stats.linregress(x, y)
        else:
            slope, intercept = np.polyfit(x, y, 1)
        out.iloc[i] = intercept + slope * (len(y) - 1)
    return out


# =====================================================================
# 🧾 Utility
# =====================================================================

def normalize(series: pd.Series) -> pd.Series:
    """Normalize series between 0 and 1."""
    return (series - series.min()) / (series.max() - series.min())


def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Rolling z-score."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


__all__ = [
    # Moving Averages
    "sma", "ema", "wma", "dema", "tema", "hull_ma",
    # Momentum
    "rsi", "roc", "trix", "macd",
    # Volatility
    "atr", "bollinger", "keltner",
    # Volume
    "obv", "vwap", "money_flow_index",
    # Trend
    "adx",
    # Statistical
    "rolling_corr", "least_squares_ma",
    # Utility
    "normalize", "zscore",
]
