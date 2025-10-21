"""
Risk management system for TradeBot.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np

from finbot.core.portfolio import Portfolio
from finbot.utils.config_loader import ConfigLoader


class RiskManager:
    """
    Risk management system for controlling trading risk.
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize risk manager.
        
        Args:
            config: Configuration loader
        """
        self.config = config
        
        # Risk parameters
        self.max_position_size = config.get("risk.max_position_size", 0.1)  # 10%
        self.max_daily_loss = config.get("risk.max_daily_loss", 0.02)  # 2%
        self.max_open_positions = config.get("risk.max_open_positions", 5)
        self.stop_loss_atr_multiplier = config.get("risk.stop_loss_atr_multiplier", 2.0)
        self.take_profit_atr_multiplier = config.get("risk.take_profit_atr_multiplier", 3.0)
        
        # Daily tracking
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        
    def validate_order(self, order: Order, portfolio: Portfolio) -> bool:
        """
        Validate order against risk parameters.
        
        Args:
            order: Order to validate
            portfolio: Current portfolio
            
        Returns:
            bool: True if order passes risk checks
        """
        # Reset daily P&L if new day
        self._reset_daily_pnl_if_needed()
        
        # Check daily loss limit
        if self.daily_pnl <= -abs(portfolio.get_total_value() * self.max_daily_loss):
            return False
        
        # Check maximum open positions
        if len(portfolio.positions) >= self.max_open_positions:
            return False
        
        # Get risk parameters
        risk_params = {
            'max_position_size': self.max_position_size,
            'max_open_positions': self.max_open_positions
        }

        # Validate order using portfolio method
        if not portfolio.validate_order(order, risk_params, portfolio.get_total_value()):
            return False
        
        return True
    
    def check_risk_limits(self, portfolio: Portfolio) -> bool:
        """
        Check if portfolio is within risk limits.
        
        Args:
            portfolio: Portfolio to check
            
        Returns:
            bool: True if within limits
        """
        # Reset daily P&L if new day
        self._reset_daily_pnl_if_needed()
        
        # Check daily loss limit
        if self.daily_pnl <= -abs(portfolio.get_total_value() * self.max_daily_loss):
            return False
        
        # Check maximum open positions
        if len(portfolio.positions) > self.max_open_positions:
            return False
        
        return True
    
    def update_daily_pnl(self, pnl: float) -> None:
        """Update daily P&L tracking."""
        self.daily_pnl += pnl
    
    def _reset_daily_pnl_if_needed(self) -> None:
        """Reset daily P&L if new day."""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
    
    def calculate_position_size(self, portfolio: Portfolio, symbol: str, 
                               risk_amount: float, stop_loss_price: float,
                               entry_price: float) -> float:
        """
        Calculate position size based on risk management.
        
        Args:
            portfolio: Current portfolio
            symbol: Trading symbol
            risk_amount: Amount willing to risk
            stop_loss_price: Stop loss price
            entry_price: Entry price
            
        Returns:
            float: Position size
        """
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            return 0.0
        
        # Position size = Risk Amount / Risk per Share
        position_size = risk_amount / risk_per_share
        
        # Apply maximum position size limit
        portfolio_value = portfolio.get_total_value()
        max_position_value = portfolio_value * self.max_position_size
        max_position_size = max_position_value / entry_price
        
        return min(position_size, max_position_size)
    
    def get_risk_metrics(self, portfolio: Portfolio) -> Dict[str, Any]:
        """
        Get current risk metrics.
        
        Args:
            portfolio: Current portfolio
            
        Returns:
            Dict: Risk metrics
        """
        portfolio_value = portfolio.get_total_value()
        
        return {
            'portfolio_value': portfolio_value,
            'daily_pnl': self.daily_pnl,
            'daily_loss_limit': portfolio_value * self.max_daily_loss,
            'max_position_size': portfolio_value * self.max_position_size,
            'open_positions': len(portfolio.positions),
            'max_open_positions': self.max_open_positions,
            'risk_per_position': self.max_position_size * 100,  # Percentage
            'daily_loss_percentage': self.max_daily_loss * 100,
            'within_daily_limit': self.daily_pnl > -abs(portfolio_value * self.max_daily_loss),
            'within_position_limit': len(portfolio.positions) <= self.max_open_positions
        }
