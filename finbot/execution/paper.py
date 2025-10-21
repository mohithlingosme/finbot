"""
Paper trading broker implementation.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from datetime import datetime
import random

# ✅ FIXED: Use absolute imports to avoid relative import errors
from finbot.execution.broker import Broker
from finbot.core.orders import Order, OrderStatus, OrderType
from finbot.core.portfolio import Portfolio


class PaperBroker(Broker):
    """
    Paper trading broker for backtesting and live simulation.

    This class simulates a broker by executing trades instantly
    with configurable commission and slippage.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.0005,   # 0.05%
    ):
        """
        Initialize paper broker.

        Args:
            initial_capital: Starting capital
            commission: Commission rate per trade (fractional)
            slippage: Slippage rate (fractional)
        """
        config = {
            "initial_capital": initial_capital,
            "commission": commission,
            "slippage": slippage,
        }

        super().__init__("Paper Broker", config)

        self.portfolio = Portfolio(initial_capital)
        self.commission = commission
        self.slippage = slippage
        self.is_connected = False
        self.orders: Dict[str, Order] = {}

    # -------------------------------------------------------------------------
    # Broker Interface
    # -------------------------------------------------------------------------
    def connect(self) -> bool:
        """Connect to paper broker (always succeeds)."""
        self.is_connected = True
        return True

    def disconnect(self) -> bool:
        """Disconnect from paper broker."""
        self.is_connected = False
        return True

    # -------------------------------------------------------------------------
    # Order Handling
    # -------------------------------------------------------------------------
    def submit_order(self, order: Order) -> bool:
        """Submit order for simulated paper trading."""
        if not self.is_connected:
            print("[PaperBroker] Broker not connected.")
            return False

        # Save order reference
        self.orders[order.order_id] = order

        # Market orders execute immediately with slippage
        if order.is_market_order():
            fill_price = self._simulate_slippage(order)
            if fill_price:
                # Commission = price × qty × rate
                commission = order.quantity * fill_price * self.commission
                order.commission = commission

                # Execute the trade in the portfolio
                self.portfolio.execute_trade(order, fill_price)

                # Update order details
                order.status = OrderStatus.FILLED
                order.fill_price = fill_price
                order.fill_time = datetime.now()

                return True

        # TODO: Add support for limit/stop later if needed
        return False

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                order.status = OrderStatus.CANCELLED
                return True
        return False

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Retrieve order status by ID."""
        order = self.orders.get(order_id)
        return order.status if order else OrderStatus.REJECTED

    # -------------------------------------------------------------------------
    # Simulation Helpers
    # -------------------------------------------------------------------------
    def _simulate_slippage(self, order: Order) -> Optional[float]:
        """Simulate slippage for market order execution."""
        base_price = order.price or 100.0  # fallback
        slippage_amount = base_price * self.slippage

        if order.is_buy_order():
            # Buy orders get slightly worse price (higher)
            return base_price + slippage_amount
        elif order.is_sell_order():
            # Sell orders get slightly worse price (lower)
            return base_price - slippage_amount

        return base_price

    # -------------------------------------------------------------------------
    # Portfolio Access
    # -------------------------------------------------------------------------
    def get_positions(self) -> Dict[str, float]:
        """Return current symbol positions."""
        return {symbol: pos.quantity for symbol, pos in self.portfolio.positions.items()}

    def get_portfolio(self) -> Dict[str, Any]:
        """Return current portfolio snapshot."""
        return {
            "total_value": self.portfolio.get_total_value(),
            "cash": self.portfolio.cash,
            "positions": self.portfolio.get_positions_summary(),
            "unrealized_pnl": self.portfolio.get_unrealized_pnl(),
            "realized_pnl": self.portfolio.get_realized_pnl(),
        }

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"<PaperBroker connected={self.is_connected}, "
            f"cash={self.portfolio.cash:.2f}, "
            f"positions={len(self.portfolio.positions)}>"
        )
