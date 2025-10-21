"""
Average True Range (ATR) indicator implementation.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional


class ATR:
    """
    Average True Range (ATR) indicator.
    
    ATR measures market volatility by calculating the average of true ranges
    over a specified period. It's commonly used for:
    - Setting stop-loss levels
    - Position sizing
    - Volatility-based trading strategies
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize ATR indicator.
        
        Args:
            period: Number of periods for ATR calculation (default: 14)
        """
        self.period = period
        self.name = f"ATR({period})"
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate ATR values.
        
        Args:
            data: DataFrame with OHLC data (columns: open, high, low, close)
            
        Returns:
            pd.Series: ATR values
        """
        if data.empty:
            return pd.Series(dtype=float)
        
        # Ensure required columns exist
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        # Calculate True Range (TR)
        tr = self._calculate_true_range(data)
        
        # Calculate ATR using exponential moving average
        atr = tr.ewm(span=self.period, adjust=False).mean()
        
        return atr
    
    def _calculate_true_range(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate True Range for each period.
        
        True Range is the maximum of:
        1. High - Low
        2. |High - Previous Close|
        3. |Low - Previous Close|
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate the three components
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        # True Range is the maximum of the three
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr
    
    def calculate_sma_atr(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate ATR using Simple Moving Average instead of EMA.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            pd.Series: ATR values using SMA
        """
        tr = self._calculate_true_range(data)
        atr_sma = tr.rolling(window=self.period).mean()
        
        return atr_sma
    
    def get_volatility_percentile(self, data: pd.DataFrame, 
                                 lookback: int = 252) -> pd.Series:
        """
        Get ATR volatility percentile over lookback period.
        
        Args:
            data: DataFrame with OHLC data
            lookback: Number of periods for percentile calculation
            
        Returns:
            pd.Series: ATR percentile values (0-100)
        """
        atr = self.calculate(data)
        atr_percentile = atr.rolling(window=lookback).rank(pct=True) * 100
        
        return atr_percentile
    
    def get_normalized_atr(self, data: pd.DataFrame) -> pd.Series:
        """
        Get normalized ATR (ATR / Close price).
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            pd.Series: Normalized ATR values
        """
        atr = self.calculate(data)
        close = data['close']
        
        normalized_atr = (atr / close) * 100  # As percentage
        
        return normalized_atr
    
    def get_atr_based_stops(self, data: pd.DataFrame, multiplier: float = 2.0,
                           side: str = 'long') -> pd.Series:
        """
        Calculate ATR-based stop loss levels.
        
        Args:
            data: DataFrame with OHLC data
            multiplier: ATR multiplier for stop distance
            side: 'long' or 'short' position
            
        Returns:
            pd.Series: Stop loss levels
        """
        atr = self.calculate(data)
        close = data['close']
        
        if side.lower() == 'long':
            # Long position: stop below current price
            stops = close - (atr * multiplier)
        else:
            # Short position: stop above current price
            stops = close + (atr * multiplier)
        
        return stops
    
    def get_atr_based_targets(self, data: pd.DataFrame, multiplier: float = 3.0,
                             side: str = 'long') -> pd.Series:
        """
        Calculate ATR-based profit targets.
        
        Args:
            data: DataFrame with OHLC data
            multiplier: ATR multiplier for target distance
            side: 'long' or 'short' position
            
        Returns:
            pd.Series: Profit target levels
        """
        atr = self.calculate(data)
        close = data['close']
        
        if side.lower() == 'long':
            # Long position: target above current price
            targets = close + (atr * multiplier)
        else:
            # Short position: target below current price
            targets = close - (atr * multiplier)
        
        return targets
    
    def get_position_size(self, data: pd.DataFrame, risk_amount: float,
                         risk_percent: float = 1.0) -> pd.Series:
        """
        Calculate position size based on ATR and risk amount.
        
        Args:
            data: DataFrame with OHLC data
            risk_amount: Dollar amount willing to risk
            risk_percent: Risk as percentage of account (alternative to risk_amount)
            
        Returns:
            pd.Series: Position sizes
        """
        atr = self.calculate(data)
        close = data['close']
        
        # Calculate stop distance (2 ATR)
        stop_distance = atr * 2
        
        # Position size = Risk Amount / Stop Distance
        position_size = risk_amount / stop_distance
        
        return position_size
    
    def is_high_volatility(self, data: pd.DataFrame, threshold: float = 2.0) -> pd.Series:
        """
        Identify periods of high volatility.
        
        Args:
            data: DataFrame with OHLC data
            threshold: ATR multiplier threshold for high volatility
            
        Returns:
            pd.Series: Boolean series indicating high volatility periods
        """
        atr = self.calculate(data)
        atr_mean = atr.rolling(window=20).mean()
        
        high_vol = atr > (atr_mean * threshold)
        
        return high_vol
    
    def get_atr_breakout_levels(self, data: pd.DataFrame, 
                               multiplier: float = 1.5) -> pd.DataFrame:
        """
        Calculate breakout levels based on ATR.
        
        Args:
            data: DataFrame with OHLC data
            multiplier: ATR multiplier for breakout levels
            
        Returns:
            pd.DataFrame: Breakout levels (upper and lower)
        """
        atr = self.calculate(data)
        close = data['close']
        
        upper_breakout = close + (atr * multiplier)
        lower_breakout = close - (atr * multiplier)
        
        return pd.DataFrame({
            'upper_breakout': upper_breakout,
            'lower_breakout': lower_breakout,
            'atr': atr,
            'close': close
        })
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data for ATR calculation.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            bool: True if data is valid
        """
        if data.empty:
            return False
        
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            return False
        
        # Check for sufficient data
        if len(data) < self.period:
            return False
        
        # Check for valid OHLC relationships
        invalid_data = (
            (data['high'] < data['low']) |
            (data['high'] < data['close']) |
            (data['low'] > data['close']) |
            (data['high'] < data['open']) |
            (data['low'] > data['open'])
        )
        
        if invalid_data.any():
            return False
        
        return True
