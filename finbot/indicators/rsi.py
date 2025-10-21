"""
Relative Strength Index (RSI) indicator implementation.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional
import warnings
from .indicators import rsi as _rsi


class RSI:
    """
    Relative Strength Index (RSI) indicator.
    
    RSI measures the speed and magnitude of price changes.
    It oscillates between 0 and 100 and is used to identify:
    - Overbought conditions (RSI > 70)
    - Oversold conditions (RSI < 30)
    - Divergences between price and momentum
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize RSI indicator.
        
        Args:
            period: Number of periods for RSI calculation (default: 14)
        """
        self.period = period
        self.name = f"RSI({period})"
    
    def calculate(self, data: pd.DataFrame, price_column: str = 'close') -> pd.Series:
        warnings.warn("indicators.rsi.RSI is deprecated — import rsi from indicators.indicators instead", DeprecationWarning)
        """
        Calculate RSI values.
        
        Args:
            data: DataFrame with price data
            price_column: Price column to use (default: 'close')
            
        Returns:
            pd.Series: RSI values
        """
        if data.empty or price_column not in data.columns:
            return pd.Series(dtype=float)

        return _rsi(data[price_column], window=self.period)
    
    def get_signals(self, data: pd.DataFrame, overbought: float = 70, 
                   oversold: float = 30) -> pd.DataFrame:
        """
        Generate RSI-based trading signals.
        
        Args:
            data: DataFrame with price data
            overbought: Overbought threshold (default: 70)
            oversold: Oversold threshold (default: 30)
            
        Returns:
            pd.DataFrame: RSI values and signals
        """
        rsi = self.calculate(data)
        
        signals = pd.DataFrame({
            'rsi': rsi,
            'signal': 0,
            'overbought': rsi > overbought,
            'oversold': rsi < oversold
        })
        
        # Buy signal: RSI crosses above oversold level
        signals.loc[(rsi > oversold) & (rsi.shift(1) <= oversold), 'signal'] = 1
        
        # Sell signal: RSI crosses below overbought level
        signals.loc[(rsi < overbought) & (rsi.shift(1) >= overbought), 'signal'] = -1
        
        return signals
    
    def get_divergence(self, data: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
        """
        Detect RSI divergences with price.
        
        Args:
            data: DataFrame with OHLC data
            lookback: Number of periods for divergence detection
            
        Returns:
            pd.DataFrame: Divergence signals
        """
        rsi = self.calculate(data)
        price = data['close']
        
        # Calculate price and RSI trends
        price_trend = price.rolling(window=lookback).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        )
        rsi_trend = rsi.rolling(window=lookback).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        )
        
        # Detect divergences
        bullish_div = (price_trend < 0) & (rsi_trend > 0)  # Price down, RSI up
        bearish_div = (price_trend > 0) & (rsi_trend < 0)  # Price up, RSI down
        
        return pd.DataFrame({
            'rsi': rsi,
            'price': price,
            'price_trend': price_trend,
            'rsi_trend': rsi_trend,
            'bullish_divergence': bullish_div,
            'bearish_divergence': bearish_div
        })
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for RSI calculation."""
        return not data.empty and len(data) >= self.period

