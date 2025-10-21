"""
Tests for trading strategies.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from finbot.strategies.base import BaseStrategy
from finbot.strategies.vwap_rsi_atr import VWAPRSIATRStrategy



class TestStrategy(BaseStrategy):
    """Test strategy implementation."""
    
    def on_data(self, symbol: str, data: dict) -> None:
        """Test implementation."""
        pass


def test_base_strategy():
    """Test base strategy functionality."""
    config = {
        'enabled': True,
        'timeframe': '1d',
        'test_param': 10
    }
    
    strategy = TestStrategy("test_strategy", config)
    
    assert strategy.name == "test_strategy"
    assert strategy.enabled == True
    assert strategy.timeframe == "1d"
    assert strategy.is_enabled() == True
    
    # Test order generation
    order = strategy.generate_order("TEST", "buy", 100, 50.0)
    assert order.symbol == "TEST"
    assert order.order_type.value == "BUY"
    assert order.quantity == 100
    assert order.price == 50.0


def test_vwap_rsi_atr_strategy():
    """Test VWAP-RSI-ATR strategy."""
    config = {
        'enabled': True,
        'timeframe': '5m',
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'vwap_period': 20,
        'atr_period': 14,
        'atr_multiplier': 2.0,
        'max_position_size': 0.1,
        'risk_per_trade': 0.02
    }
    
    strategy = VWAPRSIATRStrategy("vwap_rsi_atr", config)
    
    assert strategy.name == "vwap_rsi_atr"
    assert strategy.rsi_period == 14
    assert strategy.vwap_period == 20
    assert strategy.atr_period == 14


def test_strategy_performance_tracking():
    """Test strategy performance tracking."""
    config = {'enabled': True}
    strategy = TestStrategy("test_strategy", config)
    
    # Initial performance should be zero
    performance = strategy.get_performance_summary()
    assert performance['total_trades'] == 0
    assert performance['winning_trades'] == 0
    assert performance['total_pnl'] == 0.0
    
    # Simulate some trades
    strategy.total_trades = 10
    strategy.winning_trades = 6
    strategy.total_pnl = 1000.0
    
    performance = strategy.get_performance_summary()
    assert performance['total_trades'] == 10
    assert performance['winning_trades'] == 6
    assert performance['win_rate'] == 60.0


def test_strategy_data_validation():
    """Test strategy data validation."""
    config = {'enabled': True}
    strategy = TestStrategy("test_strategy", config)
    
    # Valid data
    valid_data = {
        'open': 100.0,
        'high': 105.0,
        'low': 95.0,
        'close': 102.0,
        'volume': 1000
    }
    assert strategy.validate_data(valid_data) == True
    
    # Invalid data - missing fields
    invalid_data = {
        'open': 100.0,
        'high': 105.0,
        'low': 95.0
        # Missing close and volume
    }
    assert strategy.validate_data(invalid_data) == False
    
    # Invalid data - invalid OHLC relationships
    invalid_ohlc = {
        'open': 100.0,
        'high': 95.0,  # High < Low
        'low': 105.0,
        'close': 102.0,
        'volume': 1000
    }
    assert strategy.validate_data(invalid_ohlc) == False


if __name__ == "__main__":
    pytest.main([__file__])
