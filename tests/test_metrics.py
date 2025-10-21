import unittest
import numpy as np
from finbot.backtest import metrics as bt_metrics



class TestMetrics(unittest.TestCase):

    def test_calculate_returns(self):
        prices = np.array([100, 110, 121, 133.1])
        expected_returns = np.array([0.1, 0.1, 0.1])
        np.testing.assert_allclose(bt_metrics.calculate_returns(prices), expected_returns)

    def test_calculate_sharpe_ratio(self):
        returns = np.array([0.1, 0.2, 0.15, 0.25, 0.05])
        risk_free_rate = 0.0
        periods_per_year = 252
        expected_sharpe_ratio = 4.464572047675872  # Manually calculated
        actual_sharpe_ratio = bt_metrics.calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
        self.assertAlmostEqual(actual_sharpe_ratio, expected_sharpe_ratio)

    def test_calculate_max_drawdown(self):
        returns = np.array([0.1, 0.2, -0.1, 0.15, -0.2])
        expected_max_drawdown = -0.22  # Manually calculated
        actual_max_drawdown = bt_metrics.calculate_max_drawdown(returns)
        self.assertAlmostEqual(actual_max_drawdown, expected_max_drawdown)

    def test_calculate_annualized_returns(self):
        returns = np.array([0.1, 0.2, 0.15, 0.25, 0.05])
        periods_per_year = 252
        expected_annualized_returns = 7.687777177737874 # Manually calculated
        actual_annualized_returns = bt_metrics.calculate_annualized_returns(returns, periods_per_year)
        self.assertAlmostEqual(actual_annualized_returns, expected_annualized_returns)

    def test_calculate_volatility(self):
        returns = np.array([0.1, 0.2, 0.15, 0.25, 0.05])
        periods_per_year = 252
        expected_volatility = 3.033150177884916 # Manually calculated
        actual_volatility = bt_metrics.calculate_volatility(returns, periods_per_year)
        self.assertAlmostEqual(actual_volatility, expected_volatility)


if __name__ == '__main__':
    unittest.main()
