"""
Visualization utilities for backtesting performance.
-----------------------------------------------------

Provides tools to visualize equity curves, drawdowns, and
return distributions from backtest results.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Optional


class Plotter:
    """
    Handles visualization of backtest results using matplotlib.
    Works directly with BacktestEngine equity_curve output.
    """

    def __init__(self, equity_curve: pd.DataFrame):
        """
        Args:
            equity_curve (pd.DataFrame): Must include 'timestamp' and 'total_value' columns.
        """
        if not isinstance(equity_curve, pd.DataFrame):
            raise TypeError("equity_curve must be a pandas DataFrame")
        if "timestamp" not in equity_curve.columns or "total_value" not in equity_curve.columns:
            raise ValueError("equity_curve must include 'timestamp' and 'total_value' columns")

        self.equity_curve = equity_curve.sort_values("timestamp").reset_index(drop=True)
        self.returns = self.equity_curve["total_value"].pct_change().dropna()
        plt.style.use("seaborn-v0_8-darkgrid")

    # ============================================================
    # 🔹 Core Plot: Portfolio Equity Curve
    # ============================================================
    def plot_equity_curve(self, title: str = "Equity Curve", save_path: Optional[str] = None):
        """
        Plots the total portfolio value over time.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.equity_curve["timestamp"], self.equity_curve["total_value"], label="Total Value", linewidth=2)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    # ============================================================
    # 🔹 Drawdown Plot
    # ============================================================
    def plot_drawdown(self, title: str = "Drawdown", save_path: Optional[str] = None):
        """
        Plots portfolio drawdowns over time.
        """
        equity = self.equity_curve["total_value"]
        drawdown = equity / equity.cummax() - 1

        plt.figure(figsize=(10, 4))
        plt.fill_between(self.equity_curve["timestamp"], drawdown, 0, color="red", alpha=0.3)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%)")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    # ============================================================
    # 🔹 Histogram of Returns
    # ============================================================
    def plot_return_distribution(self, bins: int = 30, title: str = "Daily Returns Distribution", save_path: Optional[str] = None):
        """
        Plots a histogram of daily returns to visualize return distribution.
        """
        plt.figure(figsize=(8, 5))
        plt.hist(self.returns * 100, bins=bins, color="steelblue", alpha=0.7, edgecolor="black")
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Daily Return (%)")
        plt.ylabel("Frequency")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    # ============================================================
    # 🔹 Cumulative Returns vs Benchmark
    # ============================================================
    def plot_vs_benchmark(
        self,
        benchmark_data: Optional[pd.Series] = None,
        benchmark_label: str = "Benchmark",
        title: str = "Equity vs Benchmark",
        save_path: Optional[str] = None
    ):
        """
        Compares portfolio equity curve with a benchmark (if provided).

        Args:
            benchmark_data (pd.Series): Benchmark price or index values with datetime index
            benchmark_label (str): Label for the benchmark
        """
        plt.figure(figsize=(10, 5))
        plt.plot(
            self.equity_curve["timestamp"],
            self.equity_curve["total_value"] / self.equity_curve["total_value"].iloc[0],
            label="Strategy",
            linewidth=2,
        )

        if benchmark_data is not None and isinstance(benchmark_data, pd.Series):
            benchmark_norm = benchmark_data / benchmark_data.iloc[0]
            plt.plot(benchmark_data.index, benchmark_norm, label=benchmark_label, linestyle="--", color="gray")

        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Normalized Value (Base 1.0)")
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    # ============================================================
    # 🔹 Strategy Comparison Plot
    # ============================================================
    @staticmethod
    def compare_strategies(results: List[Dict[str, any]], metric: str = "Total Return (%)", title: str = "Strategy Comparison"):
        """
        Plots a bar chart comparing multiple strategy performances.

        Args:
            results (List[Dict]): List of result dictionaries from BacktestEngine.compare_strategies()
            metric (str): Metric key to compare (e.g., 'Sharpe Ratio', 'Win Rate (%)')
        """
        if not results:
            raise ValueError("No results provided for comparison plot")

        df = pd.DataFrame(results)
        df = df.sort_values(by=metric, ascending=False)

        plt.figure(figsize=(10, 5))
        plt.bar(df["Strategy"], df[metric], color="teal", alpha=0.8)
        plt.title(f"{title} ({metric})", fontsize=14, fontweight="bold")
        plt.xlabel("Strategy")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
class BacktestPlotter(Plotter):
    """
    BacktestPlotter is an alias for Plotter to maintain backward compatibility.
    """
    pass
# ============================================================
# Example Usage (for testing purposes)
if __name__ == "__main__":
    # Generate sample equity curve data
    dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
    values = np.cumprod(1 + np.random.randn(100) * 0.01) * 1000  # Simulated portfolio values
    df = pd.DataFrame({"timestamp": dates, "total_value": values})

    plotter = BacktestPlotter(df)
    plotter.plot_equity_curve()
    plotter.plot_drawdown()
    plotter.plot_return_distribution()
    
    # Example benchmark data
    benchmark_values = np.cumprod(1 + np.random.randn(100) * 0.008) * 1000
    benchmark_series = pd.Series(benchmark_values, index=dates)
    plotter.plot_vs_benchmark(benchmark_data=benchmark_series, benchmark_label="S&P 500")
    
    # Example strategy comparison
    strategies = [
        {"Strategy": "Strategy A", "Total Return (%)": 15.2, "Sharpe Ratio": 1.5},
        {"Strategy": "Strategy B", "Total Return (%)": 10.5, "Sharpe Ratio": 1.2},
        {"Strategy": "Strategy C", "Total Return (%)": 20.1, "Sharpe Ratio": 1.8},
    ]
    Plotter.compare_strategies(strategies, metric="Total Return (%)")
    