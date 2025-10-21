"""
Helper utilities for TradeBot.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import re


class Helpers:
    """
    Utility helper functions for TradeBot.
    """
    
    @staticmethod
    def format_currency(amount: float, currency: str = "₹") -> str:
        """
        Format currency amount.
        
        Args:
            amount: Amount to format
            currency: Currency symbol
            
        Returns:
            Formatted currency string
        """
        return f"{currency}{amount:,.2f}"
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 2) -> str:
        """
        Format percentage value.
        
        Args:
            value: Value to format
            decimals: Number of decimal places
            
        Returns:
            Formatted percentage string
        """
        return f"{value:.{decimals}f}%"
    
    @staticmethod
    def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
        """
        Calculate returns from price series.
        
        Args:
            prices: Price series
            method: 'simple' or 'log'
            
        Returns:
            Returns series
        """
        if method == 'simple':
            return prices.pct_change()
        elif method == 'log':
            return np.log(prices / prices.shift(1))
        else:
            raise ValueError("Method must be 'simple' or 'log'")
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Returns series
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Sharpe ratio
        """
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        return (excess_returns.mean() / returns.std()) * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve: Portfolio value series
            
        Returns:
            Maximum drawdown percentage
        """
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min() * 100
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
        """
        Calculate volatility.
        
        Args:
            returns: Returns series
            annualize: Whether to annualize
            
        Returns:
            Volatility
        """
        vol = returns.std()
        if annualize:
            vol *= np.sqrt(252)
        return vol * 100
    
    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """
        Validate trading symbol format.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if valid
        """
        # Basic validation - alphanumeric and some special characters
        pattern = r'^[A-Z0-9.-]+$'
        return bool(re.match(pattern, symbol.upper()))
    
    @staticmethod
    def normalize_symbol(symbol: str, exchange: str = "NSE") -> str:
        """
        Normalize symbol format.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            
        Returns:
            Normalized symbol
        """
        symbol = symbol.upper().strip()
        
        # Add exchange suffix if not present
        if exchange == "NSE" and not symbol.endswith('.NS'):
            symbol += '.NS'
        elif exchange == "BSE" and not symbol.endswith('.BO'):
            symbol += '.BO'
        
        return symbol
    
    @staticmethod
    def get_market_hours(exchange: str = "NSE") -> Dict[str, str]:
        """
        Get market trading hours.
        
        Args:
            exchange: Exchange name
            
        Returns:
            Dictionary with market hours
        """
        market_hours = {
            "NSE": {"start": "09:15", "end": "15:30"},
            "BSE": {"start": "09:15", "end": "15:30"},
            "NASDAQ": {"start": "09:30", "end": "16:00"},
            "NYSE": {"start": "09:30", "end": "16:00"}
        }
        
        return market_hours.get(exchange, {"start": "09:00", "end": "17:00"})
    
    @staticmethod
    def is_market_open(exchange: str = "NSE") -> bool:
        """
        Check if market is currently open.
        
        Args:
            exchange: Exchange name
            
        Returns:
            True if market is open
        """
        print("Entering is_market_open")  # Debugging statement
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        
        market_hours = Helpers.get_market_hours(exchange)

        start_time = market_hours["start"]
        end_time = market_hours["end"]
        
        
        # Check if current time is within market hours
        result = start_time <= current_time <= end_time
        print("Exiting is_market_open")  # Debugging statement
        return result
    
    @staticmethod
    def calculate_position_size(portfolio_value: float, risk_amount: float,

                               stop_loss_price: float, entry_price: float) -> float:
        """
        Calculate position size based on risk.
        
        Args:
            portfolio_value: Total portfolio value
            risk_amount: Amount willing to risk
            stop_loss_price: Stop loss price
            entry_price: Entry price
            
        Returns:
            Position size
        """
        if stop_loss_price <= 0 or entry_price <= 0:
            return 0.0
        
        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share == 0:
            return 0.0
        
        position_size = risk_amount / risk_per_share
        
        # Ensure we don't exceed available capital
        max_position_value = portfolio_value * 0.95  # Use 95% of portfolio
        max_position_size = max_position_value / entry_price
        
        return min(position_size, max_position_size)
    
    @staticmethod
    def calculate_stop_loss(entry_price: float, atr: float, 
                           multiplier: float = 2.0, side: str = "long") -> float:
        """
        Calculate stop loss price using ATR.
        
        Args:
            entry_price: Entry price
            atr: Average True Range
            multiplier: ATR multiplier
            side: 'long' or 'short'
            
        Returns:
            Stop loss price
        """
        stop_distance = atr * multiplier
        
        if side.lower() == "long":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    @staticmethod
    def calculate_take_profit(entry_price: float, atr: float,
                             multiplier: float = 3.0, side: str = "long") -> float:
        """
        Calculate take profit price using ATR.
        
        Args:
            entry_price: Entry price
            atr: Average True Range
            multiplier: ATR multiplier
            side: 'long' or 'short'
            
        Returns:
            Take profit price
        """
        profit_distance = atr * multiplier
        
        if side.lower() == "long":
            return entry_price + profit_distance
        else:
            return entry_price - profit_distance
    
    @staticmethod
    def resample_data(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample OHLCV data to different timeframe.
        
        Args:
            data: OHLCV data
            timeframe: Target timeframe (e.g., '5T', '1H', '1D')
            
        Returns:
            Resampled data
        """
        if data.empty:
            return data
        
        # Set datetime index if not already set
        if 'datetime' in data.columns:
            data = data.set_index('datetime')
        
        # Define aggregation rules
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Resample
        resampled = data.resample(timeframe).agg(agg_rules)
        
        # Remove rows with NaN values
        resampled = resampled.dropna()
        
        return resampled.reset_index()
    
    @staticmethod
    def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """
        Add common technical indicators to data.
        
        Args:
            data: OHLCV data
            
        Returns:
            Data with technical indicators
        """
        df = data.copy()
        
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        return df
    
    @staticmethod
    def detect_patterns(data: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Detect basic chart patterns.
        
        Args:
            data: OHLCV data
            
        Returns:
            Dictionary of detected patterns
        """
        patterns = {
            'doji': [],
            'hammer': [],
            'shooting_star': [],
            'engulfing_bullish': [],
            'engulfing_bearish': []
        }
        
        if len(data) < 2:
            return patterns
        
        for i in range(len(data)):
            # Doji pattern
            if abs(data.iloc[i]['open'] - data.iloc[i]['close']) < (data.iloc[i]['high'] - data.iloc[i]['low']) * 0.1:
                patterns['doji'].append(i)
            
            # Hammer pattern
            if i > 0:
                body = abs(data.iloc[i]['open'] - data.iloc[i]['close'])
                lower_shadow = min(data.iloc[i]['open'], data.iloc[i]['close']) - data.iloc[i]['low']
                upper_shadow = data.iloc[i]['high'] - max(data.iloc[i]['open'], data.iloc[i]['close'])
                
                if lower_shadow > body * 2 and upper_shadow < body * 0.5:
                    patterns['hammer'].append(i)
                
                # Shooting star
                if upper_shadow > body * 2 and lower_shadow < body * 0.5:
                    patterns['shooting_star'].append(i)
                
                # Engulfing patterns
                prev_body = abs(data.iloc[i-1]['open'] - data.iloc[i-1]['close'])
                curr_body = abs(data.iloc[i]['open'] - data.iloc[i]['close'])
                
                if curr_body > prev_body:
                    # Bullish engulfing
                    if (data.iloc[i-1]['close'] < data.iloc[i-1]['open'] and 
                        data.iloc[i]['close'] > data.iloc[i]['open'] and
                        data.iloc[i]['open'] < data.iloc[i-1]['close'] and
                        data.iloc[i]['close'] > data.iloc[i-1]['open']):
                        patterns['engulfing_bullish'].append(i)
                    
                    # Bearish engulfing
                    elif (data.iloc[i-1]['close'] > data.iloc[i-1]['open'] and 
                          data.iloc[i]['close'] < data.iloc[i]['open'] and
                          data.iloc[i]['open'] > data.iloc[i-1]['close'] and
                          data.iloc[i]['close'] < data.iloc[i-1]['open']):
                        patterns['engulfing_bearish'].append(i)
        
        return patterns
