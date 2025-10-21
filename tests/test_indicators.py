"""
Tests for technical indicators.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from finbot.indicators.indicators import atr, rsi, vwap, bollinger, macd
from finbot.indicators.atr import ATR



@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate realistic OHLCV data
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    data = []
    for i, (date, close) in enumerate(zip(dates, close_prices)):
        high = close + abs(np.random.randn()) * 2
        low = close - abs(np.random.randn()) * 2
        open_price = close + np.random.randn() * 0.5
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'datetime': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)


def test_atr_calculation(sample_data):
    """Test ATR calculation."""
    result = atr(sample_data, window=14)
    
    assert len(result) == len(sample_data)
    assert not result.empty
    assert all(result.dropna() >= 0)  # ATR should be non-negative for valid values
    assert result.iloc[-1] > 0  # Latest ATR should be positive


def test_rsi_calculation(sample_data):
    """Test RSI calculation."""
    result = rsi(sample_data['close'], window=14)
    
    assert len(result) == len(sample_data)
    assert not result.empty
    assert all(0 <= val <= 100 for val in result.dropna())  # RSI should be between 0 and 100


def test_vwap_calculation(sample_data):
    """Test VWAP calculation."""
    result = vwap(sample_data)
    
    assert len(result) == len(sample_data)
    assert not result.empty
    assert all(result > 0)  # VWAP should be positive


def test_bollinger_bands(sample_data):
    """Test Bollinger Bands calculation."""
    result = bollinger(sample_data['close'], window=20, n_std=2.0)
    # Map 'ma' to 'middle' for backwards compatibility check
    if 'ma' in result.columns and 'middle' not in result.columns:
        result = result.rename(columns={'ma': 'middle'})
    
    assert len(result) == len(sample_data)
    assert not result.empty
    assert 'upper' in result.columns
    assert 'middle' in result.columns
    assert 'lower' in result.columns
    
    # Check that upper > middle > lower
    valid_data = result.dropna()
    assert all(valid_data['upper'] >= valid_data['middle'])
    assert all(valid_data['middle'] >= valid_data['lower'])


def test_macd_calculation(sample_data):
    result = macd(sample_data, fast=12, slow=26, signal=9)
    
    assert len(result) == len(sample_data)
    assert not result.empty
    assert 'macd' in result.columns
    assert 'signal' in result.columns
    assert 'histogram' in result.columns


def test_indicator_validation(sample_data):
    """Test indicator data validation."""
    # Using stateless atr() function for basic validation
    empty_data = pd.DataFrame()
    # atr() expects OHLC columns; calling with empty frame may raise or return empty — accept either
    import pytest as _pytest
    with _pytest.raises(Exception):
        _ = atr(empty_data, window=14)

    small_data = sample_data.head(5)
    # ATR on insufficient data should produce mostly NaNs (or be empty after dropna)
    small_atr = atr(small_data, window=14)
    assert small_atr.dropna().empty

    # ATR on valid data should produce non-empty non-NaN values
    full_atr = atr(sample_data, window=14)
    assert not full_atr.dropna().empty


if __name__ == "__main__":
    pytest.main([__file__])
