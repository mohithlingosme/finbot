"""
Utility functions for technical indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import warnings


class IndicatorUtils:
    """Utility class for technical indicator calculations."""
    
    @staticmethod
    def calculate_returns(data: pd.DataFrame, price_column: str = 'close', 
                         method: str = 'simple') -> pd.Series:
        """
        Calculate price returns.
        
        Args:
            data: DataFrame with price data
            price_column: Price column to use
            method: 'simple' or 'log'
            
        Returns:
            pd.Series: Returns
        """
        if price_column not in data.columns:
            raise ValueError(f"Column '{price_column}' not found in data")
        
        prices = data[price_column]
        
        if method == 'simple':
            returns = prices.pct_change()
        elif method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            raise ValueError("Method must be 'simple' or 'log'")
        
        return returns
    
    @staticmethod
    def calculate_volatility(data: pd.DataFrame, price_column: str = 'close',
                           window: int = 20, annualize: bool = True) -> pd.Series:
        """
        Calculate rolling volatility.
        
        Args:
            data: DataFrame with price data
            price_column: Price column to use
            window: Rolling window size
            annualize: Whether to annualize volatility
            
        Returns:
            pd.Series: Volatility values
        """
        returns = IndicatorUtils.calculate_returns(data, price_column, method='log')
        volatility = returns.rolling(window=window).std()
        
        if annualize:
            volatility = volatility * np.sqrt(252)  # Assuming daily data
        
        return volatility
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02,
                             window: int = 252) -> pd.Series:
        """
        Calculate rolling Sharpe ratio.
        
        Args:
            returns: Returns series
            risk_free_rate: Risk-free rate (annual)
            window: Rolling window size
            
        Returns:
            pd.Series: Sharpe ratios
        """
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        sharpe = excess_returns.rolling(window=window).mean() / returns.rolling(window=window).std()
        
        return sharpe * np.sqrt(252)  # Annualize
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, pd.Series]:
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve: Cumulative returns or portfolio value
            
        Returns:
            Tuple: (max_drawdown, drawdown_series)
        """
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return max_drawdown, drawdown
    
    @staticmethod
    def calculate_bollinger_squeeze(data: pd.DataFrame, bb_period: int = 20,
                                  kc_period: int = 20, bb_std: float = 2.0,
                                  kc_std: float = 1.5) -> pd.Series:
        """
        Calculate Bollinger Band squeeze indicator.
        
        Args:
            data: DataFrame with OHLC data
            bb_period: Bollinger Bands period
            kc_period: Keltner Channels period
            bb_std: Bollinger Bands standard deviation
            kc_std: Keltner Channels ATR multiplier
            
        Returns:
            pd.Series: Squeeze indicator
        """
        # Calculate Bollinger Bands
        bb_middle = data['close'].rolling(window=bb_period).mean()
        bb_std_val = data['close'].rolling(window=bb_period).std()
        bb_upper = bb_middle + (bb_std_val * bb_std)
        bb_lower = bb_middle - (bb_std_val * bb_std)
        
        # Calculate Keltner Channels
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        kc_middle = typical_price.rolling(window=kc_period).mean()
        
        # True Range for Keltner Channels
        tr = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        atr = tr.rolling(window=kc_period).mean()
        
        kc_upper = kc_middle + (atr * kc_std)
        kc_lower = kc_middle - (atr * kc_std)
        
        # Squeeze occurs when Bollinger Bands are inside Keltner Channels
        squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
        
        return squeeze
    
    @staticmethod
    def calculate_pivot_points(data: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Calculate pivot points.
        
        Args:
            data: DataFrame with OHLC data
            method: 'standard', 'fibonacci', or 'woodie'
            
        Returns:
            pd.DataFrame: Pivot points and levels
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        if method == 'standard':
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)
            
        elif method == 'fibonacci':
            pivot = (high + low + close) / 3
            r1 = pivot + 0.382 * (high - low)
            r2 = pivot + 0.618 * (high - low)
            r3 = pivot + 1.0 * (high - low)
            s1 = pivot - 0.382 * (high - low)
            s2 = pivot - 0.618 * (high - low)
            s3 = pivot - 1.0 * (high - low)
            
        elif method == 'woodie':
            pivot = (high + low + 2 * close) / 4
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)
        
        else:
            raise ValueError("Method must be 'standard', 'fibonacci', or 'woodie'")
        
        return pd.DataFrame({
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        })
    
    @staticmethod
    def calculate_support_resistance(data: pd.DataFrame, window: int = 20,
                                   threshold: float = 0.02) -> pd.DataFrame:
        """
        Calculate dynamic support and resistance levels.
        
        Args:
            data: DataFrame with OHLC data
            window: Rolling window for levels
            threshold: Minimum distance threshold
            
        Returns:
            pd.DataFrame: Support and resistance levels
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate rolling highs and lows
        rolling_high = high.rolling(window=window).max()
        rolling_low = low.rolling(window=window).min()
        
        # Calculate dynamic support and resistance
        resistance = rolling_high * (1 + threshold)
        support = rolling_low * (1 - threshold)
        
        # Calculate mid-point
        mid_point = (resistance + support) / 2
        
        return pd.DataFrame({
            'support': support,
            'resistance': resistance,
            'mid_point': mid_point,
            'range': resistance - support
        })
    
    @staticmethod
    def calculate_volume_profile(data: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
        """
        Calculate volume profile.
        
        Args:
            data: DataFrame with OHLCV data
            bins: Number of price bins
            
        Returns:
            pd.DataFrame: Volume profile data
        """
        if 'volume' not in data.columns:
            raise ValueError("Data must contain 'volume' column")
        
        # Use typical price for volume profile
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Create price bins
        price_min = data['low'].min()
        price_max = data['high'].max()
        bin_edges = np.linspace(price_min, price_max, bins + 1)
        
        # Assign prices to bins
        price_bins = pd.cut(typical_price, bins=bin_edges, labels=False)
        
        # Calculate volume at each price level
        volume_profile = []
        for i in range(bins):
            bin_volume = data.loc[price_bins == i, 'volume'].sum()
            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            volume_profile.append({
                'price_level': bin_center,
                'volume': bin_volume,
                'volume_pct': bin_volume / data['volume'].sum() * 100
            })
        
        return pd.DataFrame(volume_profile)
    
    @staticmethod
    def calculate_momentum_oscillator(data: pd.DataFrame, period: int = 10) -> pd.Series:
        """
        Calculate momentum oscillator.
        
        Args:
            data: DataFrame with price data
            period: Momentum period
            
        Returns:
            pd.Series: Momentum values
        """
        close = data['close']
        momentum = close - close.shift(period)
        
        return momentum
    
    @staticmethod
    def calculate_rate_of_change(data: pd.DataFrame, period: int = 10) -> pd.Series:
        """
        Calculate rate of change.
        
        Args:
            data: DataFrame with price data
            period: ROC period
            
        Returns:
            pd.Series: Rate of change values
        """
        close = data['close']
        roc = ((close - close.shift(period)) / close.shift(period)) * 100
        
        return roc
    
    @staticmethod
    def normalize_indicator(values: pd.Series, method: str = 'minmax') -> pd.Series:
        """
        Normalize indicator values.
        
        Args:
            values: Series to normalize
            method: 'minmax' or 'zscore'
            
        Returns:
            pd.Series: Normalized values
        """
        if method == 'minmax':
            return (values - values.min()) / (values.max() - values.min())
        elif method == 'zscore':
            return (values - values.mean()) / values.std()
        else:
            raise ValueError("Method must be 'minmax' or 'zscore'")
    
    @staticmethod
    def smooth_indicator(values: pd.Series, window: int = 3, method: str = 'sma') -> pd.Series:
        """
        Smooth indicator values.
        
        Args:
            values: Series to smooth
            window: Smoothing window
            method: 'sma', 'ema', or 'median'
            
        Returns:
            pd.Series: Smoothed values
        """
        if method == 'sma':
            return values.rolling(window=window).mean()
        elif method == 'ema':
            return values.ewm(span=window).mean()
        elif method == 'median':
            return values.rolling(window=window).median()
        else:
            raise ValueError("Method must be 'sma', 'ema', or 'median'")
    
    @staticmethod
    def detect_outliers(values: pd.Series, method: str = 'iqr', 
                       threshold: float = 1.5) -> pd.Series:
        """
        Detect outliers in indicator values.
        
        Args:
            values: Series to analyze
            method: 'iqr' or 'zscore'
            threshold: Outlier threshold
            
        Returns:
            pd.Series: Boolean mask of outliers
        """
        if method == 'iqr':
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = (values < lower_bound) | (values > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((values - values.mean()) / values.std())
            outliers = z_scores > threshold
        
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")
        
        return outliers
    
    @staticmethod
    def validate_indicator_data(data: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate data for indicator calculation.
        
        Args:
            data: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            bool: True if data is valid
        """
        if data.empty:
            return False
        
        if not all(col in data.columns for col in required_columns):
            return False
        
        # Check for sufficient data
        if len(data) < 2:
            return False
        
        # Check for valid numeric data
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                return False
        
        return True
