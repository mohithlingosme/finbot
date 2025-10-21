#!/usr/bin/env python3
"""
Comprehensive test script for newly created trading strategies.
Tests syntax, imports, initialization, signal generation, and edge cases.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

def create_test_data(num_bars: int = 100) -> pd.DataFrame:
    """Create synthetic OHLCV data for testing."""
    np.random.seed(42)  # For reproducible results

    # Generate base price series
    base_price = 100.0
    prices = [base_price]

    for i in range(num_bars - 1):
        # Random walk with slight upward bias
        change = np.random.normal(0.001, 0.02)  # Mean 0.1%, std 2%
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1.0))  # Floor at $1

    # Create OHLCV data
    data = []
    for i, close in enumerate(prices):
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = data[-1]['close'] if data else close * (1 + np.random.normal(0, 0.005))
        volume = np.random.randint(1000, 100000)

        data.append({
            'open': open_price,
            'high': max(high, open_price, close),
            'low': min(low, open_price, close),
            'close': close,
            'volume': volume,
            'timestamp': datetime.now() - timedelta(minutes=num_bars - i)
        })

    return pd.DataFrame(data)

def test_strategy_import(strategy_name: str) -> bool:
    """Test if a strategy can be imported successfully."""
    try:
        module_path = f"src.ai_engine.models.strategies.{strategy_name.lower()}"
        __import__(module_path)
        print(f"✓ Successfully imported {strategy_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to import {strategy_name}: {e}")
        return False

def test_strategy_initialization(strategy_class, strategy_name: str) -> bool:
    """Test strategy initialization and parameter setup."""
    try:
        # Test with default config
        config = {}
        strategy = strategy_class(strategy_name, config)

        # Check required attributes
        assert hasattr(strategy, 'name'), f"{strategy_name} missing 'name' attribute"
        assert hasattr(strategy, 'parameters'), f"{strategy_name} missing 'parameters' attribute"
        assert hasattr(strategy, 'enabled'), f"{strategy_name} missing 'enabled' attribute"

        # Test parameter setup
        params = strategy._setup_parameters()
        assert isinstance(params, dict), f"{strategy_name} _setup_parameters() should return dict"

        print(f"✓ Successfully initialized {strategy_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize {strategy_name}: {e}")
        return False

def test_strategy_data_processing(strategy_class, strategy_name: str) -> bool:
    """Test strategy data processing and signal generation."""
    try:
        config = {}
        strategy = strategy_class(strategy_name, config)

        # Create test data
        test_data = create_test_data(50)
        symbol = "TEST"

        # Test on_data method
        for idx, row in test_data.iterrows():
            market_data = {
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'timestamp': row['timestamp']
            }
            strategy.on_data(symbol, market_data)

        print(f"✓ Successfully processed data for {strategy_name}")
        return True
    except Exception as e:
        print(f"✗ Failed data processing for {strategy_name}: {e}")
        return False

def test_strategy_signal_calculation(strategy_class, strategy_name: str) -> bool:
    """Test strategy signal calculation from historical data."""
    try:
        config = {}
        strategy = strategy_class(strategy_name, config)

        # Create test data
        test_data = create_test_data(100)
        symbol = "TEST"

        # Test calculate_signals method
        signals = strategy.calculate_signals(test_data, symbol)

        # Verify signals structure
        if signals:
            required_keys = ['timestamp', 'symbol', 'signal_type', 'strength']
            for signal in signals:
                for key in required_keys:
                    assert key in signal, f"Signal missing required key: {key}"

        print(f"✓ Successfully calculated signals for {strategy_name} ({len(signals)} signals generated)")
        return True
    except Exception as e:
        print(f"✗ Failed signal calculation for {strategy_name}: {e}")
        return False

def test_strategy_info(strategy_class, strategy_name: str) -> bool:
    """Test strategy info method."""
    try:
        config = {}
        strategy = strategy_class(strategy_name, config)

        info = strategy.get_strategy_info()

        required_keys = ['name', 'type', 'description', 'parameters', 'indicators', 'timeframe', 'enabled']
        for key in required_keys:
            assert key in info, f"Strategy info missing required key: {key}"

        print(f"✓ Successfully retrieved info for {strategy_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to get strategy info for {strategy_name}: {e}")
        return False

def test_all_strategies():
    """Test all newly created strategies."""
    print("🧪 Starting comprehensive testing of new trading strategies...\n")

    # List of new strategies to test
    new_strategies = [
        ('ConservativeUpDownStrategy', 'conservative_up_down'),
        ('GreedyStrategy', 'greedy'),
        ('MeanReversionStrategy', 'mean_reversion'),
        ('MomentumStrategy', 'momentum'),
        ('RangeTradingStrategy', 'range_trading'),
        ('TrendFollowingStrategy', 'trend_following')
    ]

    results = {
        'imports': [],
        'initialization': [],
        'data_processing': [],
        'signal_calculation': [],
        'info': []
    }

    for strategy_class_name, module_name in new_strategies:
        print(f"\n--- Testing {strategy_class_name} ---")

        # Test import
        import_success = test_strategy_import(module_name)
        results['imports'].append((strategy_class_name, import_success))

        if not import_success:
            continue

        # Import the strategy class
        try:
            module = __import__(f"src.ai_engine.models.strategies.{module_name}", fromlist=[strategy_class_name])
            strategy_class = getattr(module, strategy_class_name)
        except Exception as e:
            print(f"✗ Failed to get strategy class {strategy_class_name}: {e}")
            continue

        # Test initialization
        init_success = test_strategy_initialization(strategy_class, strategy_class_name)
        results['initialization'].append((strategy_class_name, init_success))

        if not init_success:
            continue

        # Test data processing
        data_success = test_strategy_data_processing(strategy_class, strategy_class_name)
        results['data_processing'].append((strategy_class_name, data_success))

        # Test signal calculation
        signal_success = test_strategy_signal_calculation(strategy_class, strategy_class_name)
        results['signal_calculation'].append((strategy_class_name, signal_success))

        # Test strategy info
        info_success = test_strategy_info(strategy_class, strategy_class_name)
        results['info'].append((strategy_class_name, info_success))

    # Print summary
    print("\n" + "="*60)
    print("🧪 TESTING SUMMARY")
    print("="*60)

    all_passed = True
    for test_type, test_results in results.items():
        print(f"\n{test_type.upper()} TESTS:")
        for strategy_name, passed in test_results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {strategy_name}: {status}")
            if not passed:
                all_passed = False

    print(f"\n{'🎉 ALL TESTS PASSED!' if all_passed else '❌ SOME TESTS FAILED!'}")

    return all_passed

if __name__ == "__main__":
    success = test_all_strategies()
    sys.exit(0 if success else 1)
