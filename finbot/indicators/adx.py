"""
Average Directional Index (ADX) indicator implementation.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional
import warnings
from .indicators import adx as _adx


class ADX:
    """
    Average Directional Index (ADX) indicator.
    
    ADX measures trend strength without indicating direction.
    Components:
    - +DI: Positive Directional Indicator
    - -DI: Negative Directional Indicator
    - ADX: Average Directional Index
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize ADX indicator.
        
        Args:
            period: Number of periods for ADX calculation (default: 14)
        """
        self.period = period
        self.name = f"ADX({period})"
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        warnings.warn("indicators.adx.ADX is deprecated — import adx from indicators.indicators instead", DeprecationWarning)
        """
        Calculate ADX values.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            pd.DataFrame: ADX components
        """
        if data.empty:
            return pd.DataFrame()
        return pd.DataFrame({'adx': _adx(data, window=self.period)})
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ADX-based trading signals.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            pd.DataFrame: ADX values and signals
        """
        adx_data = self.calculate(data)
        
        signals = adx_data.copy()
        signals['trend_strength'] = 'weak'
        signals['trend_direction'] = 'neutral'
        signals['signal'] = 0
        
        # Trend strength classification
        signals.loc[adx_data['adx'] > 25, 'trend_strength'] = 'strong'
        signals.loc[(adx_data['adx'] > 20) & (adx_data['adx'] <= 25), 'trend_strength'] = 'moderate'
        
        # Trend direction
        signals.loc[adx_data['di_plus'] > adx_data['di_minus'], 'trend_direction'] = 'bullish'
        signals.loc[adx_data['di_minus'] > adx_data['di_plus'], 'trend_direction'] = 'bearish'
        
        # Generate signals
        strong_bullish = (adx_data['adx'] > 25) & (adx_data['di_plus'] > adx_data['di_minus'])
        strong_bearish = (adx_data['adx'] > 25) & (adx_data['di_minus'] > adx_data['di_plus'])
        
        signals.loc[strong_bullish, 'signal'] = 1
        signals.loc[strong_bearish, 'signal'] = -1
        
        return signals
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for ADX calculation."""
        required_cols = ['high', 'low', 'close']
        return not data.empty and all(col in data.columns for col in required_cols) and len(data) >= self.period

