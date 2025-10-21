"""
Backtesting engine for strategy testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from finbot.core.engine import TradingEngine
from finbot.core.portfolio import Portfolio
from finbot.core.orders import Order, OrderType, OrderStatus
from finbot.strategies.base import BaseStrategy
from data.loader import DataLoader
from finbot.config.config_loader import ConfigLoader


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005   # 0.05%
    benchmark: str = "NIFTY_50"


class BacktestEngine:
    """
    Backtesting engine for testing trading strategies.
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtesting engine.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.trades: List[Dict] = []
        self.equity_curve: pd.DataFrame = None
        self.daily_returns: pd.Series = None
        
    def run_backtest(self, strategy: BaseStrategy, data: Dict[str, pd.DataFrame],
                    symbols: List[str]) -> Dict[str, Any]:
        """
        Run backtest for a strategy.
        
        Args:
            strategy: Trading strategy to test
            data: Historical data for symbols
            symbols: List of symbols to test
            
        Returns:
            Dict: Backtest results
        """
        # Initialize portfolio
        portfolio = Portfolio(initial_capital=self.config.initial_capital)
        
        # Initialize trading engine
        engine = TradingEngine(self.config)
        engine.portfolio = portfolio
        engine.add_strategy(strategy, strategy.name)






        
        # Set up paper broker for backtesting
        from finbot.execution.paper import PaperBroker
        broker = PaperBroker(
            initial_capital=self.config.initial_capital,
            commission=self.config.commission,
            slippage=self.config.slippage
        )
        engine.set_broker(broker)
        
        # Prepare data
        all_data = self._prepare_data(data, symbols)
        
        # Run simulation
        self._simulate_trading(engine, all_data)
        
        # Calculate results
        results = self._calculate_results(portfolio, strategy)
        
        return results
    
    def _prepare_data(self, data: Dict[str, pd.DataFrame], symbols: List[str]) -> pd.DataFrame:
        """Prepare and align data for backtesting."""
        # Combine all symbols into single dataframe with multi-index
        combined_data = []
        
        for symbol in symbols:
            if symbol in data:
                df = data[symbol].copy()
                df['symbol'] = symbol
                combined_data.append(df)
        
        if not combined_data:
            raise ValueError("No data available for backtesting")
        
        # Combine all data
        all_data = pd.concat(combined_data, ignore_index=True)
        
        # Sort by timestamp
        all_data = all_data.sort_values('datetime')
        
        # Filter by date range
        all_data = all_data[
            (all_data['datetime'] >= self.config.start_date) &
            (all_data['datetime'] <= self.config.end_date)
        ]
        
        return all_data
    
    def _simulate_trading(self, engine: TradingEngine, data: pd.DataFrame) -> None:
        """Simulate trading over historical data."""
        engine.start()
        
        # Process data chronologically
        for _, row in data.iterrows():
            symbol = row['symbol']
            market_data = {
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'timestamp': row['datetime']
            }
            
            # Process market data
            engine.process_market_data(symbol, market_data)
            
            # Record portfolio value
            self._record_portfolio_value(engine.portfolio, row['datetime'])
        
        engine.stop()
    
    def _record_portfolio_value(self, portfolio: Portfolio, timestamp: datetime) -> None:
        """Record portfolio value at timestamp."""
        if self.equity_curve is None:
            self.equity_curve = pd.DataFrame(columns=['timestamp', 'total_value', 'cash', 'positions_value'])
        
        total_value = portfolio.get_total_value()
        cash = portfolio.cash
        positions_value = total_value - cash
        
        new_row = pd.DataFrame({
            'timestamp': [timestamp],
            'total_value': [total_value],
            'cash': [cash],
            'positions_value': [positions_value]
        })
        
        if self.equity_curve is None or self.equity_curve.empty:
            self.equity_curve = new_row
        else:
            self.equity_curve = pd.concat([self.equity_curve, new_row], ignore_index=True)

    
    def _calculate_results(self, portfolio: Portfolio, strategy: BaseStrategy) -> Dict[str, Any]:
        """Calculate backtest results."""
        if self.equity_curve is None or self.equity_curve.empty:
            return {}
        
        # Calculate returns
        self.equity_curve['returns'] = self.equity_curve['total_value'].pct_change()
        self.daily_returns = self.equity_curve.set_index('timestamp')['returns']
        
        # Calculate metrics
        total_return = (self.equity_curve['total_value'].iloc[-1] / self.config.initial_capital - 1) * 100
        
        
        # Calculate Sharpe ratio
        if len(self.daily_returns) > 1:
            sharpe_ratio = (self.daily_returns.mean() / (self.daily_returns.std() + 1e-8)) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Calculate maximum drawdown

        max_drawdown, _ = self._calculate_max_drawdown(self.equity_curve['total_value'])
        
        # Calculate win rate
        trades = portfolio.trade_history
        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate average trade metrics
        if trades:
            avg_win = np.mean([trade['pnl'] for trade in trades if trade['pnl'] > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([trade['pnl'] for trade in trades if trade['pnl'] < 0]) if (total_trades - winning_trades) > 0 else 0
            profit_factor = abs(avg_win * winning_trades / (avg_loss * (total_trades - winning_trades))) if avg_loss != 0 else 0
        else:
            avg_win = avg_loss = profit_factor = 0
        
        results = {
            'strategy_name': strategy.name,
            'start_date': self.config.start_date,
            'end_date': self.config.end_date,
            'initial_capital': self.config.initial_capital,
            'final_value': self.equity_curve['total_value'].iloc[-1],
            'total_return': total_return,
            'annualized_return': self._calculate_annualized_return(total_return),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'equity_curve': self.equity_curve,
            'trades': trades,
            'portfolio': {
                'cash': portfolio.cash,
                'positions': portfolio.get_positions_summary(),
                'unrealized_pnl': portfolio.get_unrealized_pnl(),
                'realized_pnl': portfolio.get_realized_pnl()
            }
        }
        
        return results
    
    def _calculate_annualized_return(self, total_return: float) -> float:
        """Calculate annualized return."""
        days = (self.config.end_date - self.config.start_date).days
        years = days / 365.25
        
        if years > 0:
            annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
        else:
            annualized_return = total_return
        
        return annualized_return
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> Tuple[float, pd.Series]:
        """Calculate maximum drawdown."""
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        return max_drawdown, drawdown
    
    def compare_strategies(self, strategies: List[BaseStrategy], data: Dict[str, pd.DataFrame],
                          symbols: List[str]) -> pd.DataFrame:
        """
        Compare multiple strategies.
        
        Args:
            strategies: List of strategies to compare
            data: Historical data
            symbols: List of symbols
            
        Returns:
            pd.DataFrame: Comparison results
        """
        results = []
        
        for strategy in strategies:
            try:
                result = self.run_backtest(strategy, data, symbols)
                
                results.append({
                    'Strategy': strategy.name,
                    'Total Return (%)': result.get('total_return', 0),
                    'Annualized Return (%)': result.get('annualized_return', 0),
                    'Sharpe Ratio': result.get('sharpe_ratio', 0),
                    'Max Drawdown (%)': result.get('max_drawdown', 0),
                    'Total Trades': result.get('total_trades', 0),
                    'Win Rate (%)': result.get('win_rate', 0),
                    'Profit Factor': result.get('profit_factor', 0),
                    'Final Value': result.get('final_value', 0)
                })
                
            except Exception as e:
                print(f"Error backtesting strategy {strategy.name}: {e}")
                results.append({
                    'Strategy': strategy.name,
                    'Total Return (%)': 0,
                    'Annualized Return (%)': 0,
                    'Sharpe Ratio': 0,
                    'Max Drawdown (%)': 0,
                    'Total Trades': 0,
                    'Win Rate (%)': 0,
                    'Profit Factor': 0,
                    'Final Value': self.config.initial_capital
                })
        
        return pd.DataFrame(results)
#======================TESTING=========================#
