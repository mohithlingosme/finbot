"""
Position sizing algorithms for TradeBot.
"""

import numpy as np
from typing import Dict, Optional, Any
from enum import Enum


class PositionSizingMethod(Enum):
    """Position sizing methods."""
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    KELLY = "kelly"
    VOLATILITY = "volatility"
    ATR = "atr"


class PositionSizing:
    """
    Position sizing calculator using various methods.
    """
    
    def __init__(self, method: PositionSizingMethod = PositionSizingMethod.PERCENTAGE):
        """
        Initialize position sizing calculator.
        
        Args:
            method: Position sizing method
        """
        self.method = method
    
    def calculate_position_size(self, portfolio_value: float, price: float,
                               method: Optional[PositionSizingMethod] = None,
                               **kwargs) -> float:
        """
        Calculate position size.
        
        Args:
            portfolio_value: Total portfolio value
            price: Entry price
            method: Position sizing method (overrides instance method)
            **kwargs: Method-specific parameters
            
        Returns:
            float: Position size
        """
        method = method or self.method
        
        if method == PositionSizingMethod.FIXED:
            return self._fixed_sizing(portfolio_value, price, **kwargs)
        elif method == PositionSizingMethod.PERCENTAGE:
            return self._percentage_sizing(portfolio_value, price, **kwargs)
        elif method == PositionSizingMethod.KELLY:
            return self._kelly_sizing(portfolio_value, price, **kwargs)
        elif method == PositionSizingMethod.VOLATILITY:
            return self._volatility_sizing(portfolio_value, price, **kwargs)
        elif method == PositionSizingMethod.ATR:
            return self._atr_sizing(portfolio_value, price, **kwargs)
        else:
            raise ValueError(f"Unknown position sizing method: {method}")
    
    def _fixed_sizing(self, portfolio_value: float, price: float, 
                     fixed_amount: float = 10000) -> float:
        """Fixed dollar amount position sizing."""
        return fixed_amount / price
    
    def _percentage_sizing(self, portfolio_value: float, price: float,
                          percentage: float = 0.1) -> float:
        """Percentage of portfolio position sizing."""
        position_value = portfolio_value * percentage
        return position_value / price
    
    def _kelly_sizing(self, portfolio_value: float, price: float,
                     win_rate: float, avg_win: float, avg_loss: float,
                     kelly_fraction: float = 0.25) -> float:
        """
        Kelly Criterion position sizing.
        
        Args:
            portfolio_value: Total portfolio value
            price: Entry price
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade
            avg_loss: Average losing trade
            kelly_fraction: Kelly fraction to use (0-1)
        """
        if avg_loss == 0:
            return 0.0
        
        # Kelly percentage = (bp - q) / b
        # where b = avg_win / avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / abs(avg_loss)
        kelly_percentage = (b * win_rate - (1 - win_rate)) / b
        
        # Apply Kelly fraction for safety
        kelly_percentage *= kelly_fraction
        
        # Ensure non-negative
        kelly_percentage = max(0, kelly_percentage)
        
        # Cap at reasonable maximum
        kelly_percentage = min(kelly_percentage, 0.25)  # Max 25%
        
        position_value = portfolio_value * kelly_percentage
        return position_value / price
    
    def _volatility_sizing(self, portfolio_value: float, price: float,
                          volatility: float, target_volatility: float = 0.02) -> float:
        """
        Volatility-based position sizing.
        
        Args:
            portfolio_value: Total portfolio value
            price: Entry price
            volatility: Asset volatility (daily)
            target_volatility: Target portfolio volatility
        """
        if volatility == 0:
            return 0.0
        
        # Position size = Target Volatility / Asset Volatility
        position_ratio = target_volatility / volatility
        
        # Cap at reasonable maximum
        position_ratio = min(position_ratio, 0.25)  # Max 25%
        
        position_value = portfolio_value * position_ratio
        return position_value / price
    
    def _atr_sizing(self, portfolio_value: float, price: float,
                   atr: float, risk_per_trade: float = 0.02) -> float:
        """
        ATR-based position sizing.
        
        Args:
            portfolio_value: Total portfolio value
            price: Entry price
            atr: Average True Range
            risk_per_trade: Risk per trade as fraction of portfolio
        """
        if atr == 0:
            return 0.0
        
        # Risk amount
        risk_amount = portfolio_value * risk_per_trade
        
        # Stop distance (2 ATR)
        stop_distance = atr * 2
        
        # Position size = Risk Amount / Stop Distance
        position_size = risk_amount / stop_distance
        
        # Cap at reasonable maximum
        max_position_value = portfolio_value * 0.25  # Max 25%
        max_position_size = max_position_value / price
        
        return min(position_size, max_position_size)
