"""
Zerodha broker implementation.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from execution.broker import Broker
from core.orders import Order, OrderStatus, OrderType

try:
    from kiteconnect import KiteConnect
except ImportError:
    KiteConnect = None


class ZerodhaBroker(Broker):
    """
    Zerodha broker implementation using KiteConnect API.
    """
    
    def __init__(self, api_key: str, api_secret: str, access_token: str = None):
        """
        Initialize Zerodha broker.
        
        Args:
            api_key: Zerodha API key
            api_secret: Zerodha API secret
            access_token: Access token (if available)
        """
        config = {
            'api_key': api_key,
            'api_secret': api_secret,
            'access_token': access_token
        }
        super().__init__("Zerodha", config)
        
        if KiteConnect is None:
            raise ImportError("kiteconnect package is required for Zerodha broker")
        
        self.kite = KiteConnect(api_key=api_key)
        if access_token:
            self.kite.set_access_token(access_token)
        
    def connect(self) -> bool:
        """Connect to Zerodha."""
        try:
            # Test connection
            profile = self.kite.profile()
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to Zerodha: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from Zerodha."""
        self.is_connected = False
        return True
    
    def submit_order(self, order: Order) -> bool:
        """Submit order to Zerodha."""
        if not self.is_connected:
            return False
        
        try:
            # Map order type to Zerodha format
            variety = "regular"
            product = "MIS"  # Intraday
            exchange = "NSE"
            
            # Map order type
            if order.order_type == OrderType.BUY:
                transaction_type = "BUY"
            else:
                transaction_type = "SELL"
            
            # Submit order
            order_id = self.kite.place_order(
                variety=variety,
                exchange=exchange,
                tradingsymbol=order.symbol,
                transaction_type=transaction_type,
                quantity=int(order.quantity),
                product=product,
                order_type="MARKET" if order.is_market_order() else "LIMIT",
                price=order.price if order.price else None
            )
            
            # Update order with Zerodha order ID
            order.order_id = str(order_id)
            order.status = OrderStatus.PENDING
            
            return True
            
        except Exception as e:
            print(f"Failed to submit order to Zerodha: {e}")
            return False
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        try:
            self.kite.cancel_order(variety="regular", order_id=order_id)
            return True
        except Exception as e:
            print(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        try:
            orders = self.kite.orders()
            for order in orders:
                if str(order['order_id']) == order_id:
                    status = order['status']
                    if status == "COMPLETE":
                        return OrderStatus.FILLED
                    elif status == "CANCELLED":
                        return OrderStatus.CANCELLED
                    elif status == "REJECTED":
                        return OrderStatus.REJECTED
                    else:
                        return OrderStatus.PENDING
            return OrderStatus.REJECTED
        except Exception as e:
            print(f"Failed to get order status: {e}")
            return OrderStatus.REJECTED
    
    def get_positions(self) -> Dict[str, float]:
        """Get current positions."""
        try:
            positions = self.kite.positions()
            position_dict = {}
            
            for pos in positions['day']:
                symbol = pos['tradingsymbol']
                quantity = pos['quantity']
                position_dict[symbol] = quantity
            
            return position_dict
        except Exception as e:
            print(f"Failed to get positions: {e}")
            return {}
    
    def get_portfolio(self) -> Dict[str, Any]:
        """Get portfolio information."""
        try:
            holdings = self.kite.holdings()
            positions = self.kite.positions()
            
            total_value = 0.0
            positions_summary = {}
            
            # Process holdings
            for holding in holdings:
                symbol = holding['tradingsymbol']
                quantity = holding['quantity']
                avg_price = holding['average_price']
                current_price = holding['last_price']
                
                total_value += quantity * current_price
                positions_summary[symbol] = {
                    'quantity': quantity,
                    'avg_price': avg_price,
                    'current_price': current_price,
                    'value': quantity * current_price
                }
            
            return {
                'total_value': total_value,
                'positions': positions_summary
            }
        except Exception as e:
            print(f"Failed to get portfolio: {e}")
            return {}
    
    def get_latest_price(self, symbol: str) -> float:
        """Get latest price for symbol."""
        try:
            quote = self.kite.quote(f"NSE:{symbol}")
            return quote[f"NSE:{symbol}"]['last_price']
        except Exception as e:
            print(f"Failed to get price for {symbol}: {e}")
            return 0.0
