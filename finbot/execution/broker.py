"""
Base broker class for order execution.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
from core.orders import Order, OrderStatus


class Broker(ABC):
    """
    Abstract base class for all brokers.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize broker.
        
        Args:
            name: Broker name
            config: Broker configuration
        """
        self.name = name
        self.config = config
        self.is_connected = False
        self.orders: Dict[str, Order] = {}
        
    @abstractmethod
    def connect(self) -> bool:
        """Connect to broker."""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from broker."""
        pass
    
    @abstractmethod
    def submit_order(self, order: Order) -> bool:
        """Submit order to broker."""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, float]:
        """Get current positions."""
        pass
    
    @abstractmethod
    def get_portfolio(self) -> Dict[str, Any]:
        """Get portfolio information."""
        pass
    
    def get_latest_price(self, symbol: str) -> float:
        """Get latest price for symbol."""
        pass
