"""
Portfolio management module for tracking positions, orders, and P&L.
"""

from datetime import datetime
from typing import Dict, List, Optional

from finbot.core.orders import Order, OrderType, Position
from finbot.core.orders import Order, OrderType



class Portfolio:
    """

    Portfolio manager for tracking positions, orders, and performance.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize portfolio with starting capital.
        
        Args:
            initial_capital: Starting cash amount
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trade_history: List[Dict] = []
        
        # Performance tracking
        self.realized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
    def add_order(self, order: Order) -> None:
        """Add an order to the portfolio."""
        self.orders.append(order)
    
    def cancel_order(self, order_id: str) -> None:
        """Cancel an order by ID."""
        for order in self.orders:
            if order.order_id == order_id and order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                break
    
    def update_position_prices(self, symbol: str, price: float) -> None:
        """Update position price for a symbol."""
        if symbol in self.positions:
            self.positions[symbol].update_price(price)
    
    def execute_trade(self, order: Order, fill_price: float, fill_time: datetime = None) -> None:
        """
        Execute a trade and update portfolio.
        
        Args:
            order: Executed order
            fill_price: Actual fill price
            fill_time: Fill timestamp
        """
        if fill_time is None:
            fill_time = datetime.now()
        
        symbol = order.symbol
        quantity = order.quantity
        side = 1 if order.order_type == OrderType.BUY else -1
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.fill_price = fill_price
        order.fill_time = fill_time
        
        # Calculate trade cost
        trade_cost = quantity * fill_price
        
        if symbol not in self.positions:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=side * quantity,
                avg_price=fill_price
            )
            self.cash -= side * trade_cost
        else:
            # Existing position
            position = self.positions[symbol]
            
            if (position.quantity > 0 and side > 0) or (position.quantity < 0 and side < 0):
                # Adding to position
                total_quantity = abs(position.quantity) + quantity
                total_cost = abs(position.quantity) * position.avg_price + trade_cost
                new_avg_price = total_cost / total_quantity
                
                position.quantity = side * total_quantity
                position.avg_price = new_avg_price
                self.cash -= side * trade_cost
            else:
                # Reducing position
                remaining_quantity = abs(position.quantity) - quantity
                if remaining_quantity <= 0:
                    # Position closed
                    realized_pnl = side * quantity * (fill_price - position.avg_price)
                    self.realized_pnl += realized_pnl
                    self.cash += side * trade_cost + realized_pnl
                    
                    # Record trade
                    self._record_trade(symbol, position.avg_price, fill_price, 
                                     abs(position.quantity), realized_pnl)
                    
                    del self.positions[symbol]
                else:
                    # Partial close
                    realized_pnl = side * quantity * (fill_price - position.avg_price)
                    self.realized_pnl += realized_pnl
                    self.cash += side * trade_cost + realized_pnl
                    
                    # Update position
                    position.quantity = side * remaining_quantity
                    
                    # Record trade
                    self._record_trade(symbol, position.avg_price, fill_price, 
                                     quantity, realized_pnl)
    
    def _record_trade(self, symbol: str, entry_price: float, exit_price: float, 
                     quantity: float, pnl: float) -> None:
        """Record a completed trade."""
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        
        self.trade_history.append({
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl': pnl,
            'timestamp': datetime.now()
        })
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def get_positions_summary(self) -> Dict[str, Dict]:
        """Get summary of all positions."""
        summary = {}
        for symbol, position in self.positions.items():
            summary[symbol] = {
                'quantity': position.quantity,
                'avg_price': position.avg_price,
                'current_price': position.current_price,
                'unrealized_pnl': position.unrealized_pnl,
                'value': position.get_value()
            }
        return summary
    
    def get_total_value(self) -> float:
        """Get total portfolio value."""
        positions_value = sum(pos.get_value() for pos in self.positions.values())
        return self.cash + positions_value
    
    def get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def get_realized_pnl(self) -> float:
        """Get total realized P&L."""
        return self.realized_pnl
    
    def get_total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)."""
        return self.realized_pnl + self.get_unrealized_pnl()
    
    def get_total_return(self) -> float:
        """Get total return percentage."""
        total_value = self.get_total_value()
        return (total_value - self.initial_capital) / self.initial_capital
    
    def get_win_rate(self) -> float:
        """Get win rate percentage."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders."""
        return [order for order in self.orders if order.status == OrderStatus.PENDING]
    
    def get_position_size(self, symbol: str) -> float:
        """Get position size for a symbol."""
        position = self.get_position(symbol)
        return position.quantity if position else 0.0
    
    def get_position_value(self, symbol: str) -> float:
        """Get position value for a symbol."""
        position = self.get_position(symbol)
        return position.get_value() if position else 0.0

    def validate_order(self, order: Order, risk_params: Dict[str, float], portfolio_value: float) -> bool:
        """
        Validate order against risk parameters.

        Args:
            order: Order to validate
            risk_params: Risk parameters
            portfolio_value: Current portfolio value

        Returns:
            bool: True if order passes risk checks
        """
        max_position_size = risk_params['max_position_size']
        max_open_positions = int(risk_params['max_open_positions'])

        # Check maximum open positions
        if len(self.positions) >= max_open_positions:
            return False

        # Check position size limit
        order_value = order.quantity * (order.price or 100.0)  # Use current price if available

        if order_value > portfolio_value * max_position_size:
            return False

        # Check if order would exceed maximum position size
        current_position = self.get_position_size(order.symbol)
        new_position_size = current_position + (order.quantity if order.is_buy_order() else -order.quantity)

        if abs(new_position_size) * (order.price or 100.0) > portfolio_value * max_position_size:
            return False

        return True
