"""
FINBOT Indicators Package Initialization
---------------------------------------

This module exports unified, stateless technical indicator functions
implemented in :mod:`finbot.indicators.indicators`.

Preferred usage:
    from finbot.indicators import sma, ema, rsi, macd

Legacy class-based modules are retained for backward compatibility,
but using unified functions from `indicators.indicators` is recommended.

Author: Mohith
Version: 1.0.0
"""

import importlib

# -----------------------------------------------------------
# Safe dynamic import of indicator functions
# -----------------------------------------------------------
try:
    indicators_module = importlib.import_module("finbot.indicators.indicators")

    sma = getattr(indicators_module, "sma", None)
    ema = getattr(indicators_module, "ema", None)
    rsi = getattr(indicators_module, "rsi", None)
    stochastic = getattr(indicators_module, "stochastic", None)
    stochastic_rsi = getattr(indicators_module, "stochastic_rsi", None)
    macd = getattr(indicators_module, "macd", None)
    atr = getattr(indicators_module, "atr", None)
    vwap = getattr(indicators_module, "vwap", None)
    obv = getattr(indicators_module, "obv", None)
    bollinger = getattr(indicators_module, "bollinger", None)

except ImportError as e:
    print(f"[FINBOT] ⚠️ Warning: Could not import indicators ({e})")
    sma = ema = rsi = stochastic = stochastic_rsi = macd = atr = vwap = obv = bollinger = None

# -----------------------------------------------------------
# Public API
# -----------------------------------------------------------
__all__ = [
    "sma",
    "ema",
    "rsi",
    "stochastic",
    "stochastic_rsi",
    "macd",
    "atr",
    "vwap",
    "obv",
    "bollinger",
]
