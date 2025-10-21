"""
Trading strategies module for TradeBot.
"""

from finbot.strategies.base import BaseStrategy
from finbot.strategies.vwap_rsi_atr import VWAPRSIATRStrategy
from finbot.strategies.bollinger_breakout import BollingerBreakoutStrategy
from finbot.strategies.macd_crossover import MACDCrossoverStrategy

__all__ = [
    'BaseStrategy',
    'VWAPRSIATRStrategy',
    'BollingerBreakoutStrategy',
    'MACDCrossoverStrategy'
]
