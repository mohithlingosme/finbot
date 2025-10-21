"""
Order management system for tradebot.
"""

from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import uuid

@dataclass
class Position:
    """
    Represents a trading position.
    
    Attributes:
        symbol: Trading symbol (e.g., 'NIFTY', 'RELIANCE')
        quantity: Number of shares/units
        avg_price: Average purchase price
        current_price: Current market price
        unrealized_pnl: Unrealized profit/loss
    """
    symbol: str
    quantity: float
    avg_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    
    def update_price(self, price: float) -> None:
        """Update current price and P&L."""
        self.current_price = price
        self.unrealized_pnl = (self.current_price - self.avg_price) * self.quantity
    
    def get_value(self) -> float:
        """Get current value of the position."""
        return self.quantity * self.current_price


class OrderType(Enum):
    """Order type enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class OrderSide(Enum):
    """Order side enumeration."""
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Order:
    """
    Represents a trading order.
    
    Attributes:
        symbol: Trading symbol (e.g., 'NIFTY', 'RELIANCE')
        order_type: BUY or SELL
        quantity: Number of shares/units
        price: Limit price (None for market orders)
        stop_price: Stop loss price (for stop orders)
        order_id: Unique order identifier
        strategy_id: ID of strategy that generated this order
        status: Current order status
        created_at: Order creation timestamp
        fill_price: Actual fill price
        fill_time: Fill timestamp
        commission: Trading commission
        notes: Additional order notes
    """
    
    symbol: str
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None
    commission: float = 0.0
    notes: str = ""
    
    def is_market_order(self) -> bool:
        """Check if this is a market order."""
        return self.price is None
    
    def is_limit_order(self) -> bool:
        """Check if this is a limit order."""
        return self.price is not None
    
    def is_stop_order(self) -> bool:
        """Check if this is a stop order."""
        return self.stop_price is not None
    
    def is_buy_order(self) -> bool:
        """Check if this is a buy order."""
        return self.order_type == OrderType.BUY
    
    def is_sell_order(self) -> bool:
        """Check if this is a sell order."""
        return self.order_type == OrderType.SELL
    
    def get_notional_value(self) -> float:
        """Get notional value of the order."""
        if self.fill_price:
            return self.fill_price * self.quantity
        elif self.price:
            return self.price * self.quantity
        else:
            return 0.0  # Market order without fill price
    
    def get_total_cost(self) -> float:
        """Get total cost including commission."""
        return self.get_notional_value() + self.commission
    
    def __str__(self) -> str:
        """String representation of the order."""
        order_type_str = f"{self.order_type.value}"
        price_str = f"@{self.price}" if self.price else "MARKET"
        return f"{order_type_str} {self.quantity} {self.symbol} {price_str} [{self.status.value}]"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Order(symbol='{self.symbol}', type={self.order_type.value}, "
                f"quantity={self.quantity}, price={self.price}, status={self.status.value}, "
                f"id='{self.order_id}')")


class OrderManager:
    """
    Manages order lifecycle and execution.
    """
    
    def __init__(self):
        """Initialize order manager."""
        self.orders: dict = {}
        self.order_counter = 0
    
    def create_order(self, symbol: str, order_type: OrderType, quantity: float,
                    price: Optional[float] = None, stop_price: Optional[float] = None,
                    strategy_id: str = "", notes: str = "") -> Order:
        """
        Create a new order.
        
        Args:
            symbol: Trading symbol
            order_type: BUY or SELL
            quantity: Number of shares/units
            price: Limit price (None for market orders)
            stop_price: Stop loss price
            strategy_id: Strategy identifier
            notes: Additional notes
            
        Returns:
            Order: Created order object
        """
        order = Order(
            symbol=symbol,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            strategy_id=strategy_id,
            notes=notes
        )
        
        self.orders[order.order_id] = order
        self.order_counter += 1
        
        return order
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def update_order_status(self, order_id: str, status: OrderStatus,
                          fill_price: Optional[float] = None) -> bool:
        """
        Update order status.
        
        Args:
            order_id: Order identifier
            status: New status
            fill_price: Fill price if order is filled
            
        Returns:
            bool: True if update successful
        """
        order = self.get_order(order_id)
        if not order:
            return False
        
        order.status = status
        if fill_price is not None:
            order.fill_price = fill_price
            order.fill_time = datetime.now()
        
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        return self.update_order_status(order_id, OrderStatus.CANCELLED)
    
    def get_orders_by_symbol(self, symbol: str) -> list:
        """Get all orders for a symbol."""
        return [order for order in self.orders.values() if order.symbol == symbol]
    
    def get_orders_by_status(self, status: OrderStatus) -> list:
        """Get all orders with specific status."""
        return [order for order in self.orders.values() if order.status == status]
    
    def get_pending_orders(self) -> list:
        """Get all pending orders."""
        return self.get_orders_by_status(OrderStatus.PENDING)
    
    def get_filled_orders(self) -> list:
        """Get all filled orders."""
        return self.get_orders_by_status(OrderStatus.FILLED)
    
    def get_order_count(self) -> int:
        """Get total number of orders."""
        return len(self.orders)
    
    def clear_old_orders(self, days: int = 30) -> int:
        """
        Clear orders older than specified days.
        
        Args:
            days: Number of days
            
        Returns:
            int: Number of orders removed
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        old_orders = [
            order_id for order_id, order in self.orders.items()
            if order.created_at < cutoff_date
        ]
        
        for order_id in old_orders:
            del self.orders[order_id]
        
        return len(old_orders)
