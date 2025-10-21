"""
Bollinger Bands indicator implementation.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional
import warnings
from .indicators import bollinger as _bollinger


class BollingerBands:
    """
    Bollinger Bands indicator.
    
    Bollinger Bands consist of a moving average and two standard deviation bands.
    Used for:
    - Identifying overbought/oversold conditions
    - Volatility analysis
    - Mean reversion strategies
    - Breakout detection
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialize Bollinger Bands.
        
        Args:
            period: Number of periods for moving average (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)
        """
        self.period = period
        self.std_dev = std_dev
        self.name = f"BB({period}, {std_dev})"
    
    def calculate(self, data: pd.DataFrame, price_column: str = 'close') -> pd.DataFrame:
        warnings.warn("indicators.bollinger.BollingerBands is deprecated — import bollinger from indicators.indicators instead", DeprecationWarning)
        """
        Calculate Bollinger Bands.
        
        Args:
            data: DataFrame with price data
            price_column: Price column to use (default: 'close')
            
        Returns:
            pd.DataFrame: Bollinger Bands (upper, middle, lower)
        """
        if data.empty or price_column not in data.columns:
            return pd.DataFrame()

        prices = data[price_column]
        return _bollinger(prices, window=self.period, n_std=self.std_dev).rename(columns={'ma': 'middle'})
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Bollinger Bands trading signals.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            pd.DataFrame: Bollinger Bands and signals
        """
        bands = self.calculate(data)
        price = data['close']
        
        signals = bands.copy()
        signals['price'] = price
        signals['signal'] = 0
        
        # Buy signal: price touches lower band
        signals.loc[price <= signals['lower'], 'signal'] = 1
        
        # Sell signal: price touches upper band
        signals.loc[price >= signals['upper'], 'signal'] = -1
        
        # Squeeze signal: low volatility
        signals['squeeze'] = signals['bandwidth'] < signals['bandwidth'].rolling(20).quantile(0.2)
        
        return signals
    
    def get_breakout_signals(self, data: pd.DataFrame, 
                           lookback: int = 5) -> pd.DataFrame:
        """
        Detect Bollinger Band breakouts.
        
        Args:
            data: DataFrame with price data
            lookback: Periods to confirm breakout
            
        Returns:
            pd.DataFrame: Breakout signals
        """
        bands = self.calculate(data)
        price = data['close']
        
        breakout_signals = pd.DataFrame({
            'price': price,
            'upper': bands['upper'],
            'lower': bands['lower'],
            'breakout_up': False,
            'breakout_down': False
        })
        
        # Upper breakout: price breaks above upper band
        breakout_up = (price > bands['upper']) & (price.shift(1) <= bands['upper'].shift(1))
        breakout_signals['breakout_up'] = breakout_up
        
        # Lower breakout: price breaks below lower band
        breakout_down = (price < bands['lower']) & (price.shift(1) >= bands['lower'].shift(1))
        breakout_signals['breakout_down'] = breakout_down
        
        return breakout_signals
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for Bollinger Bands calculation."""
        return not data.empty and len(data) >= self.period

