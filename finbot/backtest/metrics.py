"""
Performance metrics module for backtesting.
-------------------------------------------
Includes key financial performance calculations and a helper class
to analyze the results of backtests (equity curves and returns).
"""

import numpy as np
import pandas as pd


# ==========================================================
# ⚙️ Core Metric Calculation Functions
# ==========================================================

def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """
    Calculates simple returns from a price series.
    Args:
        prices (np.ndarray): Array of price values
    Returns:
        np.ndarray: Array of simple returns
    """
    if len(prices) < 2:
        return np.array([])
    return np.diff(prices) / prices[:-1]


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculates the annualized Sharpe ratio.
    Args:
        returns (np.ndarray): Array of returns
        risk_free_rate (float): Annual risk-free rate (default 0)
        periods_per_year (int): Number of trading periods per year
    Returns:
        float: Sharpe ratio
    """
    if returns.size == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    annualized_excess_return = np.mean(excess_returns) * periods_per_year
    annualized_volatility = np.std(returns) * np.sqrt(periods_per_year)
    return 0.0 if annualized_volatility == 0 else annualized_excess_return / annualized_volatility


def calculate_max_drawdown(values: np.ndarray) -> float:
    """
    Calculates the maximum drawdown from a series of equity values.
    Args:
        values (np.ndarray): Array of equity or portfolio values
    Returns:
        float: Maximum drawdown (as a negative percentage)
    """
    if len(values) == 0:
        return 0.0

    running_max = np.maximum.accumulate(values)
    drawdowns = (values - running_max) / running_max
    return np.min(drawdowns)  # Negative value representing drawdown


def calculate_annualized_return(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Calculates annualized return from period returns.
    Args:
        returns (np.ndarray): Array of returns
        periods_per_year (int): Number of periods per year
    Returns:
        float: Annualized return
    """
    if len(returns) == 0:
        return 0.0
    total_return = np.prod(1 + returns) - 1
    annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
    return annualized_return


def calculate_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Calculates annualized volatility.
    Args:
        returns (np.ndarray): Array of returns
        periods_per_year (int): Number of trading periods per year
    Returns:
        float: Annualized volatility
    """
    if len(returns) == 0:
        return 0.0
    return np.std(returns) * np.sqrt(periods_per_year)


# ==========================================================
# 📊 PerformanceMetrics Class Wrapper
# ==========================================================

class PerformanceMetrics:
    """
    Performance metrics helper for analyzing backtest results.
    Accepts equity curve DataFrame from BacktestEngine.
    """

    def __init__(self, equity_curve: pd.DataFrame, risk_free_rate: float = 0.0):
        if not isinstance(equity_curve, pd.DataFrame):
            raise TypeError("equity_curve must be a pandas DataFrame")

        if "total_value" not in equity_curve.columns:
            raise ValueError("equity_curve must contain a 'total_value' column")

        self.equity_curve = equity_curve.copy()
        self.returns = self._compute_returns()
        self.risk_free_rate = risk_free_rate

    def _compute_returns(self) -> np.ndarray:
        """Extract returns from the equity curve."""
        values = self.equity_curve["total_value"].values
        return calculate_returns(values)

    def total_return(self) -> float:
        """Calculate total portfolio return in %."""
        values = self.equity_curve["total_value"].values
        if len(values) < 2:
            return 0.0
        return (values[-1] / values[0] - 1) * 100

    def annualized_return(self) -> float:
        """Annualized return based on daily data."""
        return calculate_annualized_return(self.returns) * 100

    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio."""
        return calculate_sharpe_ratio(self.returns, self.risk_free_rate)

    def max_drawdown(self) -> float:
        """Maximum drawdown in %."""
        values = self.equity_curve["total_value"].values
        return calculate_max_drawdown(values) * 100

    def volatility(self) -> float:
        """Annualized volatility in %."""
        return calculate_volatility(self.returns) * 100

    def summary(self) -> dict:
        """Return a summary of all metrics."""
        return {
            "Total Return (%)": round(self.total_return(), 2),
            "Annualized Return (%)": round(self.annualized_return(), 2),
            "Sharpe Ratio": round(self.sharpe_ratio(), 2),
            "Max Drawdown (%)": round(self.max_drawdown(), 2),
            "Volatility (%)": round(self.volatility(), 2),
        }


# ==========================================================
# 🧪 Example Usage (for testing)
# ==========================================================
if __name__ == "__main__":
    # Simulated equity curve example
    prices = np.linspace(100, 120, 100) + np.random.normal(0, 2, 100)
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=100), "total_value": prices})

    metrics = PerformanceMetrics(df)
    print(metrics.summary())

    export = {
        "total_return": metrics.total_return(),
        "annualized_return": metrics.annualized_return(),
        "sharpe_ratio": metrics.sharpe_ratio(),
        "max_drawdown": metrics.max_drawdown(),
        "volatility": metrics.volatility()
    }
