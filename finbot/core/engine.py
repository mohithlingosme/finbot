"""
Main trading engine that orchestrates all components.
"""

from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from typing import List, Dict, Any
import os

from finbot.execution.broker import Broker
from finbot.strategies.base import BaseStrategy

from finbot.core.portfolio import Portfolio
from finbot.config.config_loader import ConfigLoader
from finbot.config import config_loader
from finbot.core.orders import Order, OrderStatus, OrderType



@dataclass
class EngineState:
    """Current state of the trading engine."""
    is_running: bool = False
    current_time: Optional[datetime] = None
    active_strategies: List[str] = field(default_factory=list)
    total_trades: int = 0
    daily_pnl: float = 0.0


class TradingEngine:
    """
    Main trading engine that coordinates all components.
    
    This class orchestrates data flow, strategy execution, order management,
    risk management, and monitoring.
    """
    
    def __init__(self, config):
        """Initialize the trading engine with configuration."""
        # Load configuration
        #config_path = os.path.join(os.getcwd(), "config.yaml")
        #self.config = config_loader.ConfigLoader(config_path)
        self.state = EngineState()
        self.config = config

        # Initialize core components
        self.portfolio = Portfolio(
            initial_capital=self.config.initial_capital
        )
        self.broker = None  # Will be set by broker factory
        #self.risk_manager = RiskManager(self.config)
        #self.logger = Logger(self.config)

        # Strategy management
        self.strategies: Dict[str, BaseStrategy] = {}
        self.active_strategies: List[BaseStrategy] = []

        #self.logger.info("Trading engine initialized")
    
    def add_strategy(self, strategy: BaseStrategy, name: str) -> None:
        """Add a strategy to the engine."""
        strategy.set_engine(self)
        self.strategies[name] = strategy
        
        if strategy.is_enabled():
            self.active_strategies.append(strategy)
            self.state.active_strategies.append(name)
        
        #self.logger.info(f"Strategy '{name}' added to engine")
    
    def set_broker(self, broker: Broker) -> None:
        """Set the broker for order execution."""
        self.broker = broker
        #self.logger.info(f"Broker set to: {broker.__class__.__name__}")
    
    def start(self) -> None:
        """Start the trading engine."""
        if self.state.is_running:
            #self.logger.warning("Engine is already running")
            return
        
        if not self.broker:
            raise ValueError("No broker set. Call set_broker() first.")
        
        self.state.is_running = True
        #self.logger.info("Trading engine started")
        
        # Initialize strategies
        for strategy in self.active_strategies:
            strategy.initialize()
    
    def stop(self) -> None:
        """Stop the trading engine."""
        if not self.state.is_running:
            #self.logger.warning("Engine is not running")
            return
        
        self.state.is_running = False
        
        # Cleanup strategies
        for strategy in self.active_strategies:
            strategy.cleanup()
        
        #self.logger.info("Trading engine stopped")
    
    def process_market_data(self, symbol: str, data: Dict) -> None:
        """
        Process incoming market data and trigger strategy updates.
        
        Args:
            symbol: Trading symbol
            data: Market data dictionary with OHLCV
        """
        if not self.state.is_running:
            return
        
        self.state.current_time = data.get('timestamp', datetime.now())
        
        # Update portfolio with latest prices
        self.portfolio.update_position_prices(symbol, data.get('close', 0))
        
        # Check risk limits
        #if not self.risk_manager.check_risk_limits(self.portfolio):
            #self.logger.warning("Risk limits exceeded, skipping strategy updates")
            #return
        
        # Update all active strategies
        for strategy in self.active_strategies:
            try:
                strategy.on_data(symbol, data)
            except Exception as e:
                #self.logger.error(f"Error in strategy {strategy.__class__.__name__}: {e}")
                pass
    
    def submit_order(self, order: Order) -> bool:
        """
        Submit an order through the broker.
        
        Args:
            order: Order to submit
            
        Returns:
            bool: True if order submitted successfully
        """
        if not self.broker:
            #self.logger.error("No broker available for order submission")
            return False
        
        # Risk check
        #if not self.risk_manager.validate_order(order, self.portfolio):
            #self.logger.warning("Risk limits exceeded, skipping strategy updates")
            return False
        
        try:
            # Submit to broker
            result = self.broker.submit_order(order)
            if result:
                self.portfolio.add_order(order)
                #self.logger.info(f"Order submitted: {order}")
                return True
            else:
                #self.logger.error(f"Error submitting order: {e}")
                return False
                
        except Exception as e:
            #self.logger.error(f"Error submitting order: {e}")
            return False
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        if not self.broker:
            #self.logger.error("No broker available for order submission")
            return False
        
        try:
            result = self.broker.cancel_order(order_id)
            if result:
                self.portfolio.cancel_order(order_id)
                #self.logger.info(f"Order cancelled: {order_id}")
            return result
        except Exception as e:
            #self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary."""
        return {
            'total_value': self.portfolio.get_total_value(),
            'cash': self.portfolio.cash,
            'positions': self.portfolio.get_positions_summary(),
            'unrealized_pnl': self.portfolio.get_unrealized_pnl(),
            'realized_pnl': self.portfolio.get_realized_pnl(),
            'total_trades': self.state.total_trades,
            'daily_pnl': self.state.daily_pnl
        }
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics."""
        # This would typically calculate Sharpe ratio, max drawdown, etc.
        # For now, return basic metrics
        return {
            'total_return': self.portfolio.get_total_return(),
            'sharpe_ratio': 0.0,  # TODO: Implement
            'max_drawdown': 0.0,   # TODO: Implement
            'win_rate': 0.0        # TODO: Implement
        }
