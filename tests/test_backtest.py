"""
Unit tests for the FINBOT Backtesting Engine.
------------------------------------------------
Tests the end-to-end backtesting functionality including:
- Configuration setup
- Data preparation
- Strategy execution
- Portfolio updates
- Performance metrics calculation
- Multi-strategy comparison
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# -----------------------------------------------------------
# ✅ Add root path dynamically for pytest & VS Code execution
# -----------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print("[PYTEST] Added path:", sys.path[-1])

# -----------------------------------------------------------
# Imports from project modules
# -----------------------------------------------------------
from finbot.backtest.engine import BacktestEngine, BacktestConfig
from finbot.strategies.base import BaseStrategy





# -----------------------------------------------------------
# Mock Strategy for Testing
# -----------------------------------------------------------
class TestBacktestStrategy(BaseStrategy):
    """A simple buy-and-hold strategy used for testing."""

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.buy_signals = 0
        self.sell_signals = 0
        self.price_data = {}

    def on_data(self, symbol: str, data: dict) -> None:
        """Simulated signal logic: buy once, sell after 10 periods."""
        current_position = self.get_position(symbol)
        self.price_data.setdefault(symbol, []).append(data['close'])

        if current_position == 0 and self.buy_signals == 0:
            # Buy once at the start
            order = self.generate_order(symbol, "buy", 100, data['close'])
            self.submit_order(order)
            self.buy_signals += 1

        elif current_position > 0 and self.sell_signals == 0 and len(self.price_data[symbol]) > 10:
            # Sell after 10 periods
            order = self.generate_order(symbol, "sell", current_position, data['close'])
            self.submit_order(order)
            self.sell_signals += 1


# -----------------------------------------------------------
# Fixtures
# -----------------------------------------------------------
@pytest.fixture
def sample_backtest_data() -> dict[str, pd.DataFrame]:
    """Generate realistic mock OHLCV data for backtesting."""
    dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
    np.random.seed(42)

    base_price = 100
    returns = np.random.randn(50) * 0.02  # 2% volatility
    prices = base_price * np.exp(np.cumsum(returns))

    data = []
    for date, price in zip(dates, prices):
        high = price * (1 + abs(np.random.randn()) * 0.01)
        low = price * (1 - abs(np.random.randn()) * 0.01)
        open_price = price * (1 + np.random.randn() * 0.005)
        volume = np.random.randint(1000, 10000)

        data.append({
            "datetime": date,
            "open": open_price,
            "high": high,
            "low": low,
            "close": price,
            "volume": volume
        })

    df = pd.DataFrame(data)
    return {"TEST": df}


# -----------------------------------------------------------
# Unit Tests
# -----------------------------------------------------------
def test_backtest_config():
    """Test configuration setup."""
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 31),
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005
    )

    assert config.initial_capital == 100000
    assert config.commission == 0.001
    assert config.slippage == 0.0005
    assert config.start_date < config.end_date


def test_backtest_engine(sample_backtest_data: dict[str, pd.DataFrame]):
    """Test basic backtest execution and result structure."""
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 31),
        initial_capital=100000
    )

    engine = BacktestEngine(config)
    strategy = TestBacktestStrategy("test_strategy", {"enabled": True})

    results = engine.run_backtest(strategy, sample_backtest_data, ["TEST"])

    assert isinstance(results, dict)
    for key in ["total_return", "final_value", "total_trades", "equity_curve"]:
        assert key in results, f"Missing key in results: {key}"

    assert results["initial_capital"] == 100000
    assert results["final_value"] > 0


def test_backtest_performance_metrics(sample_backtest_data: dict[str, pd.DataFrame]):
    """Test that performance metrics are computed correctly."""
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 2, 1),
        initial_capital=100000
    )

    engine = BacktestEngine(config)
    strategy = TestBacktestStrategy("test_strategy", {"enabled": True})

    results = engine.run_backtest(strategy, sample_backtest_data, ["TEST"])

    for metric in ["total_return", "annualized_return", "sharpe_ratio", "max_drawdown", "win_rate"]:
        assert metric in results, f"Metric missing: {metric}"

    # Reasonable numeric sanity checks
    assert -100 <= results["total_return"] <= 2000
    assert results["final_value"] > 0


def test_equity_curve_structure(sample_backtest_data: dict[str, pd.DataFrame]):
    """Test the structure of the equity curve output."""
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 31),
        initial_capital=100000
    )

    engine = BacktestEngine(config)
    strategy = TestBacktestStrategy("test_strategy", {"enabled": True})

    results = engine.run_backtest(strategy, sample_backtest_data, ["TEST"])
    equity_curve = results["equity_curve"]

    assert isinstance(equity_curve, pd.DataFrame)
    assert not equity_curve.empty
    for col in ["timestamp", "total_value", "cash", "positions_value"]:
        assert col in equity_curve.columns

    assert equity_curve["total_value"].iloc[0] > 0


def test_strategy_comparison(sample_backtest_data: dict[str, pd.DataFrame]):
    """Test multi-strategy backtest comparison results."""
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 31),
        initial_capital=100000
    )

    engine = BacktestEngine(config)
    strategy1 = TestBacktestStrategy("strategy1", {"enabled": True})
    strategy2 = TestBacktestStrategy("strategy2", {"enabled": True})

    comparison_df = engine.compare_strategies([strategy1, strategy2], sample_backtest_data, ["TEST"])

    assert isinstance(comparison_df, pd.DataFrame)
    assert len(comparison_df) == 2
    for col in ["Strategy", "Total Return (%)", "Annualized Return (%)", "Final Value"]:
        assert col in comparison_df.columns


# -----------------------------------------------------------
# Entry point for direct execution
# -----------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--disable-warnings"])
# -----------------------------------------------------------
# End of file
