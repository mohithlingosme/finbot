"""
Base strategy class for all trading strategies.

Provides an abstract base class that strategy implementations should extend.
This file keeps the same public API you had, while ensuring both
`BaseStrategy` and the alias `Strategy` are available for imports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from datetime import datetime
import pandas as pd

# Local imports (will raise if core.orders is missing)
from core.orders import Order, OrderType, OrderStatus  # keep as-is

if TYPE_CHECKING:
    from core.engine import TradingEngine


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    All strategies must inherit from this class and implement the `on_data` method.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize base strategy.

        Args:
            name: Strategy name
            config: Strategy configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.enabled: bool = self.config.get("enabled", True)
        self.timeframe: str = self.config.get("timeframe", "1d")

        # Strategy state
        self.positions: Dict[str, float] = {}
        self.orders: List[Order] = []
        self.signals: List[Dict[str, Any]] = []

        # Performance tracking
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.total_pnl: float = 0.0

        # Engine reference (set by TradingEngine)
        self.engine: Optional["TradingEngine"] = None

        # Strategy parameters
        self.parameters: Dict[str, Any] = self._setup_parameters()

    # -------------------------
    # Overridable helpers
    # -------------------------
    def _setup_parameters(self) -> Dict[str, Any]:
        """
        Setup strategy parameters from config.
        Override in subclasses to define specific parameters.
        """
        return {}

    def set_engine(self, engine: "TradingEngine") -> None:
        """Set reference to trading engine."""
        self.engine = engine

    def is_enabled(self) -> bool:
        """Check if strategy is enabled."""
        return bool(self.enabled)

    def initialize(self) -> None:
        """
        Initialize strategy.
        Called once when strategy is added to the engine.
        Override in subclasses for strategy-specific initialization.
        """
        pass

    def cleanup(self) -> None:
        """
        Cleanup strategy.
        Called when strategy is removed or engine stops.
        Override in subclasses for strategy-specific cleanup.
        """
        pass

    # -------------------------
    # Core abstract API
    # -------------------------
    @abstractmethod
    def on_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """
        Process new market data. Must be implemented by subclasses.

        Args:
            symbol: Trading symbol
            data: Market data dictionary with OHLCV (keys: open, high, low, close, volume, datetime/timestamp)
        """
        ...

    # Optional helper for bulk signal generation
    def calculate_signals(self, data: pd.DataFrame, symbol: str) -> List[Dict[str, Any]]:
        """
        Calculate trading signals from market data.
        Override in subclasses to implement signal generation.

        Args:
            data: Historical market data (DataFrame)
            symbol: Trading symbol

        Returns:
            List[Dict]: List of signal dictionaries
        """
        return []

    # -------------------------
    # Order helpers
    # -------------------------
    def generate_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        order_type: str = "market",
    ) -> Order:
        """
        Generate an Order object used by the engine.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            order_type: 'market', 'limit', or 'stop'

        Returns:
            Order: Generated order
        """
        # Defensive: support different OrderType names (BUY/SELL vs LONG/SHORT, etc.)
        try:
            order_type_enum = OrderType.BUY if side.lower() == "buy" else OrderType.SELL
        except Exception:
            # fallback: if OrderType doesn't have BUY/SELL names, pass a string/type that Order expects
            order_type_enum = side.lower()

        order = Order(
            symbol=symbol,
            order_type=order_type_enum,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            strategy_id=self.name,
        )

        return order

    def submit_order(self, order: Order) -> bool:
        """
        Submit order through the trading engine.

        Returns True if submission succeeded, False otherwise.
        """
        if not self.engine:
            return False
        return self.engine.submit_order(order)

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order by id.
        """
        if not self.engine:
            return False
        return self.engine.cancel_order(order_id)

    # -------------------------
    # Portfolio helpers
    # -------------------------
    def get_position(self, symbol: str) -> float:
        """
        Get current position size for a symbol.
        """
        if not self.engine or not hasattr(self.engine, "portfolio"):
            return 0.0
        return self.engine.portfolio.get_position_size(symbol)

    def get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        if not self.engine or not hasattr(self.engine, "portfolio"):
            return 0.0
        return self.engine.portfolio.get_total_value()

    def get_available_cash(self) -> float:
        """Get available cash in portfolio."""
        if not self.engine or not hasattr(self.engine, "portfolio"):
            return 0.0
        return self.engine.portfolio.cash

    # -------------------------
    # Position sizing / risk
    # -------------------------
    def calculate_position_size(
        self, symbol: str, risk_amount: float, stop_loss_price: float, entry_price: float
    ) -> float:
        """
        Calculate position size based on risk management.

        Returns: position size (number of shares/contracts)
        """
        if stop_loss_price <= 0 or entry_price <= 0:
            return 0.0

        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share == 0:
            return 0.0

        position_size = risk_amount / risk_per_share

        available_cash = self.get_available_cash()
        max_position_value = available_cash * 0.95  # use up to 95% of cash
        max_position_size = max_position_value / entry_price

        return min(position_size, max_position_size)

    # -------------------------
    # Logging & utility
    # -------------------------
    def log_signal(self, symbol: str, signal_type: str, strength: float, data: Dict[str, Any], notes: str = "") -> None:
        """
        Log a trading signal. Keeps an internal ring buffer of signals.
        """
        signal = {
            "timestamp": data.get("timestamp", data.get("datetime", datetime.now())),
            "symbol": symbol,
            "signal_type": signal_type,
            "strength": strength,
            "price": data.get("close", 0),
            "volume": data.get("volume", 0),
            "notes": notes,
        }

        self.signals.append(signal)
        # keep last 1000
        if len(self.signals) > 1000:
            self.signals = self.signals[-1000:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get strategy performance summary.
        """
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0

        return {
            "strategy_name": self.name,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": max(0, self.total_trades - self.winning_trades),
            "win_rate": win_rate,
            "total_pnl": self.total_pnl,
            "current_positions": len(self.positions),
            "total_signals": len(self.signals),
        }

    def validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate incoming market data (basic checks).
        """
        required_fields = ["open", "high", "low", "close", "volume"]

        for field in required_fields:
            if field not in data:
                return False
            # allow non-numeric checks to pass safely
            try:
                if data[field] is None or float(data[field]) <= 0:
                    return False
            except Exception:
                return False

        # OHLC relationships
        if data["high"] < max(data["open"], data["close"]):
            return False
        if data["low"] > min(data["open"], data["close"]):
            return False

        return True

    def get_strategy_config(self) -> Dict[str, Any]:
        """Return the strategy configuration and parameters."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "timeframe": self.timeframe,
            "parameters": self.parameters,
            "config": self.config,
        }

    def update_parameter(self, parameter_name: str, value: Any) -> bool:
        """Update a parameter if it exists."""
        if parameter_name in self.parameters:
            self.parameters[parameter_name] = value
            return True
        return False

    def reset_strategy(self) -> None:
        """Reset strategy runtime state."""
        self.positions.clear()
        self.orders.clear()
        self.signals.clear()
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', enabled={self.enabled})"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', enabled={self.enabled}, "
            f"trades={self.total_trades}, pnl={self.total_pnl:.2f})"
        )


# Backwards-compatible alias: some modules import `Strategy`, others import `BaseStrategy`
Strategy = BaseStrategy
