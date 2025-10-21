"""
Volume Weighted Average Price (VWAP) indicator implementation.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, List
import warnings
from .indicators import vwap as _vwap


class VWAP:
    """
    Volume Weighted Average Price (VWAP) indicator.
    
    VWAP is the average price weighted by volume over a specified period.
    It's commonly used for:
    - Intraday trading strategies
    - Institutional trading benchmarks
    - Mean reversion strategies
    - Support/resistance identification
    """
    
    def __init__(self, period: int = 20):
        """
        Initialize VWAP indicator.
        
        Args:
            period: Number of periods for VWAP calculation (default: 20)
        """
        self.period = period
        self.name = f"VWAP({period})"
    
    def calculate(self, data: pd.DataFrame, 
                  price_column: str = 'typical_price') -> pd.Series:
        warnings.warn("indicators.vwap.VWAP is deprecated — import vwap from indicators.indicators instead", DeprecationWarning)
        """
        Calculate VWAP values.
        
        Args:
            data: DataFrame with OHLCV data
            price_column: Price column to use ('typical_price', 'close', 'hl2', 'hlc3')
            
        Returns:
            pd.Series: VWAP values
        """
        # Delegate to unified vwap implementation (which uses tp=(h+l+c)/3)
        return _vwap(data)
    
    def _get_price(self, data: pd.DataFrame, price_column: str) -> pd.Series:
        """
        Get price series based on specified column.
        
        Args:
            data: DataFrame with OHLC data
            price_column: Price column specification
            
        Returns:
            pd.Series: Price series
        """
        if price_column == 'typical_price':
            if 'typical_price' in data.columns:
                return data['typical_price']
            else:
                # Calculate typical price: (High + Low + Close) / 3
                return (data['high'] + data['low'] + data['close']) / 3
        
        elif price_column == 'hl2':
            # High-Low average: (High + Low) / 2
            return (data['high'] + data['low']) / 2
        
        elif price_column == 'hlc3':
            # High-Low-Close average: (High + Low + Close) / 3
            return (data['high'] + data['low'] + data['close']) / 3
        
        elif price_column == 'close':
            return data['close']
        
        elif price_column in data.columns:
            return data[price_column]
        
        else:
            raise ValueError(f"Invalid price column: {price_column}")
    
    def calculate_session_vwap(self, data: pd.DataFrame, 
                              session_start: str = "09:30") -> pd.Series:
        """
        Calculate session-based VWAP (resets each trading session).
        
        Args:
            data: DataFrame with datetime index and OHLCV data
            session_start: Time to start new session (format: "HH:MM")
            
        Returns:
            pd.Series: Session-based VWAP values
        """
        if data.empty or not isinstance(data.index, pd.DatetimeIndex):
            return pd.Series(dtype=float)
        
        # Create session groups
        data_copy = data.copy()
        data_copy['session'] = self._create_session_groups(data_copy.index, session_start)
        
        # Calculate VWAP for each session
        price = self._get_price(data_copy, 'typical_price')
        volume_price = price * data_copy['volume']
        
        # Group by session and calculate cumulative VWAP
        session_vwap = []
        
        for session_id in data_copy['session'].unique():
            session_data = data_copy[data_copy['session'] == session_id]
            session_volume_price = volume_price[data_copy['session'] == session_id]
            session_volume = data_copy['volume'][data_copy['session'] == session_id]
            
            # Calculate cumulative VWAP within session
            cum_volume_price = session_volume_price.cumsum()
            cum_volume = session_volume.cumsum()
            session_vwap_series = cum_volume_price / cum_volume
            
            session_vwap.append(session_vwap_series)
        
        # Combine all sessions
        if session_vwap:
            return pd.concat(session_vwap).sort_index()
        else:
            return pd.Series(dtype=float)
    
    def _create_session_groups(self, index: pd.DatetimeIndex, 
                              session_start: str) -> pd.Series:
        """Create session groups based on session start time."""
        session_groups = []
        current_session = 0
        
        for timestamp in index:
            if timestamp.time().strftime("%H:%M") == session_start:
                current_session += 1
            session_groups.append(current_session)
        
        return pd.Series(session_groups, index=index)
    
    def get_vwap_deviations(self, data: pd.DataFrame, 
                           deviations: List[float] = [1, 2, 3]) -> pd.DataFrame:
        """
        Calculate VWAP standard deviation bands.
        
        Args:
            data: DataFrame with OHLCV data
            deviations: List of standard deviation multipliers
            
        Returns:
            pd.DataFrame: VWAP and deviation bands
        """
        vwap = self.calculate(data)
        price = self._get_price(data, 'typical_price')
        
        # Calculate price deviations from VWAP
        price_deviation = (price - vwap) ** 2
        
        # Calculate standard deviation
        std_dev = np.sqrt(price_deviation.rolling(window=self.period).mean())
        
        # Create deviation bands
        result = pd.DataFrame({'vwap': vwap})
        
        for dev in deviations:
            result[f'upper_{dev}std'] = vwap + (std_dev * dev)
            result[f'lower_{dev}std'] = vwap - (std_dev * dev)
        
        return result
    
    def get_vwap_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate VWAP-based trading signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: VWAP signals and levels
        """
        vwap = self.calculate(data)
        price = self._get_price(data, 'typical_price')
        
        # Calculate deviations
        deviations = self.get_vwap_deviations(data, [1, 2])
        
        # Generate signals
        signals = pd.DataFrame({
            'vwap': vwap,
            'price': price,
            'deviation': ((price - vwap) / vwap) * 100,  # Percentage deviation
            'signal': 0
        })
        
        # Buy signal: price below VWAP (oversold)
        signals.loc[price < deviations['lower_1std'], 'signal'] = 1
        
        # Sell signal: price above VWAP (overbought)
        signals.loc[price > deviations['upper_1std'], 'signal'] = -1
        
        # Strong signals at 2 standard deviations
        signals.loc[price < deviations['lower_2std'], 'signal'] = 2  # Strong buy
        signals.loc[price > deviations['upper_2std'], 'signal'] = -2  # Strong sell
        
        return signals
    
    def get_vwap_mean_reversion_zones(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identify mean reversion zones around VWAP.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: Mean reversion zones and signals
        """
        vwap = self.calculate(data)
        price = self._get_price(data, 'typical_price')
        
        # Calculate percentage deviation from VWAP
        deviation_pct = ((price - vwap) / vwap) * 100
        
        # Define mean reversion zones
        zones = pd.DataFrame({
            'vwap': vwap,
            'price': price,
            'deviation_pct': deviation_pct,
            'zone': 'neutral'
        })
        
        # Oversold zone (buy opportunity)
        zones.loc[deviation_pct < -2, 'zone'] = 'oversold'
        
        # Overbought zone (sell opportunity)
        zones.loc[deviation_pct > 2, 'zone'] = 'overbought'
        
        # Extreme zones
        zones.loc[deviation_pct < -4, 'zone'] = 'extreme_oversold'
        zones.loc[deviation_pct > 4, 'zone'] = 'extreme_overbought'
        
        return zones
    
    def get_vwap_trend(self, data: pd.DataFrame, lookback: int = 5) -> pd.Series:
        """
        Determine VWAP trend direction.
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Number of periods for trend calculation
            
        Returns:
            pd.Series: Trend values (1: uptrend, -1: downtrend, 0: sideways)
        """
        vwap = self.calculate(data)
        
        # Calculate VWAP slope
        vwap_slope = vwap.rolling(window=lookback).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        )
        
        # Determine trend
        trend = pd.Series(0, index=vwap.index)
        trend.loc[vwap_slope > 0.01, 'trend'] = 1    # Uptrend
        trend.loc[vwap_slope < -0.01, 'trend'] = -1  # Downtrend
        
        return trend
    
    def get_vwap_volume_profile(self, data: pd.DataFrame, 
                               price_bins: int = 20) -> pd.DataFrame:
        """
        Create volume profile around VWAP.
        
        Args:
            data: DataFrame with OHLCV data
            price_bins: Number of price bins for profile
            
        Returns:
            pd.DataFrame: Volume profile data
        """
        vwap = self.calculate(data)
        price = self._get_price(data, 'typical_price')
        volume = data['volume']
        
        # Create price bins around VWAP
        vwap_std = price.rolling(window=self.period).std()
        price_min = (vwap - 2 * vwap_std).min()
        price_max = (vwap + 2 * vwap_std).max()
        
        # Create bins
        bins = np.linspace(price_min, price_max, price_bins + 1)
        
        # Assign prices to bins
        price_bins_assigned = pd.cut(price, bins=bins, labels=False)
        
        # Calculate volume at each price level
        volume_profile = []
        for bin_idx in range(price_bins):
            bin_volume = volume[price_bins_assigned == bin_idx].sum()
            bin_center = (bins[bin_idx] + bins[bin_idx + 1]) / 2
            
            volume_profile.append({
                'price_level': bin_center,
                'volume': bin_volume,
                'vwap_distance': bin_center - vwap.iloc[-1] if len(vwap) > 0 else 0
            })
        
        return pd.DataFrame(volume_profile)
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data for VWAP calculation.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            bool: True if data is valid
        """
        if data.empty:
            return False
        
        if 'volume' not in data.columns:
            return False
        
        # Check for sufficient data
        if len(data) < self.period:
            return False
        
        # Check for valid volume data
        if data['volume'].isna().all() or (data['volume'] <= 0).all():
            return False
        
        # Check for valid OHLC data
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            return False
        
        return True
