"""
MACD (Moving Average Convergence Divergence) indicator implementation.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional
import warnings
from .indicators import macd as _macd


class MACD:
    """
    MACD (Moving Average Convergence Divergence) indicator.
    
    MACD consists of:
    - MACD line: 12-period EMA - 26-period EMA
    - Signal line: 9-period EMA of MACD line
    - Histogram: MACD line - Signal line
    
    Used for trend following and momentum analysis.
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Initialize MACD indicator.
        
        Args:
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.name = f"MACD({fast_period}, {slow_period}, {signal_period})"
    
    def calculate(self, data: pd.DataFrame, price_column: str = 'close') -> pd.DataFrame:
        warnings.warn("indicators.macd.MACD is deprecated — import macd from indicators.indicators instead", DeprecationWarning)
        """
        Calculate MACD values.
        
        Args:
            data: DataFrame with price data
            price_column: Price column to use (default: 'close')
            
        Returns:
            pd.DataFrame: MACD components
        """
        if data.empty:
            return pd.DataFrame()
        return _macd(data, fast=self.fast_period, slow=self.slow_period, signal=self.signal_period)
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate MACD trading signals.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            pd.DataFrame: MACD values and signals
        """
        macd_data = self.calculate(data)
        
        signals = macd_data.copy()
        signals['price'] = data['close']
        signals['signal'] = 0
        
        # Buy signal: MACD crosses above signal line
        macd_cross_up = (macd_data['macd'] > macd_data['signal']) & \
                       (macd_data['macd'].shift(1) <= macd_data['signal'].shift(1))
        signals.loc[macd_cross_up, 'signal'] = 1
        
        # Sell signal: MACD crosses below signal line
        macd_cross_down = (macd_data['macd'] < macd_data['signal']) & \
                         (macd_data['macd'].shift(1) >= macd_data['signal'].shift(1))
        signals.loc[macd_cross_down, 'signal'] = -1
        
        # Zero line crossover signals
        signals['zero_cross_up'] = (macd_data['macd'] > 0) & (macd_data['macd'].shift(1) <= 0)
        signals['zero_cross_down'] = (macd_data['macd'] < 0) & (macd_data['macd'].shift(1) >= 0)
        
        return signals
    
    def get_divergence(self, data: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
        """
        Detect MACD divergences with price.
        
        Args:
            data: DataFrame with OHLC data
            lookback: Number of periods for divergence detection
            
        Returns:
            pd.DataFrame: Divergence signals
        """
        macd_data = self.calculate(data)
        price = data['close']
        macd_line = macd_data['macd']
        
        # Calculate trends
        price_trend = price.rolling(window=lookback).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        )
        macd_trend = macd_line.rolling(window=lookback).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        )
        
        # Detect divergences
        bullish_div = (price_trend < 0) & (macd_trend > 0)
        bearish_div = (price_trend > 0) & (macd_trend < 0)
        
        return pd.DataFrame({
            'price': price,
            'macd': macd_line,
            'price_trend': price_trend,
            'macd_trend': macd_trend,
            'bullish_divergence': bullish_div,
            'bearish_divergence': bearish_div
        })
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for MACD calculation."""
        return not data.empty and len(data) >= max(self.fast_period, self.slow_period, self.signal_period)

