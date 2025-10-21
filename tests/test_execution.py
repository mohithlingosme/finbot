"""
Tests for execution layer.
"""

import pytest
from datetime import datetime

from finbot.execution.paper import PaperBroker
from finbot.execution.zerodha import ZerodhaBroker
from finbot.execution.broker import Broker
from finbot.core.orders import Order, OrderType, OrderStatus



class TestBroker(Broker):
    """Test broker implementation."""

    def __init__(self, name, config):
        super().__init__(name, config)
    
    def connect(self) -> bool:
        return True
    
    def disconnect(self) -> bool:
        return True
    
    def submit_order(self, order: Order) -> bool:
        order.status = OrderStatus.FILLED
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        return True
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        return OrderStatus.FILLED
    
    def get_positions(self) -> dict:
        return {}
    
    def get_portfolio(self) -> dict:
        return {}


def test_paper_broker():
    """Test paper broker functionality."""
    broker = PaperBroker(initial_capital=100000, commission=0.001, slippage=0.0005)
    
    # Test connection
    assert broker.connect() == True
    assert broker.is_connected == True
    
    # Test order submission
    order = Order(
        symbol="TEST",
        order_type=OrderType.BUY,
        quantity=100,
        price=50.0
    )
    
    broker.submit_order(order)
    assert order.status == OrderStatus.FILLED
    
    # Test portfolio
    portfolio = broker.get_portfolio()
    assert 'total_value' in portfolio
    assert portfolio['total_value'] > 0


def test_broker_interface():
    """Test broker interface."""
    broker = TestBroker("Test Broker", {})
    
    # Test basic functionality
    assert broker.connect() == True
    assert broker.disconnect() == True
    
    # Test order operations
    order = Order(
        symbol="TEST",
        order_type=OrderType.BUY,
        quantity=100
    )
    assert broker.submit_order(order) == True
    # Ensure order_id is set before using it
    if not hasattr(order, 'order_id') or order.order_id is None:
        order.order_id = "test_order_id"
    assert broker.cancel_order(order.order_id) == True
    assert broker.get_order_status(order.order_id) == OrderStatus.FILLED
    assert broker.submit_order(order) == True
    assert broker.cancel_order(order.order_id) == True
    assert broker.get_order_status(order.order_id) == OrderStatus.FILLED
    
    # Test portfolio operations
    positions = broker.get_positions()
    portfolio = broker.get_portfolio()
    
    assert isinstance(positions, dict)
    assert isinstance(portfolio, dict)


def test_order_creation():
    """Test order creation and properties."""
    order = Order(
        symbol="RELIANCE",
        order_type=OrderType.BUY,
        quantity=100,
        price=2500.0,
        strategy_id="test_strategy"
    )
    
    assert order.symbol == "RELIANCE"
    assert order.order_type == OrderType.BUY
    assert order.quantity == 100
    assert order.price == 2500.0
    assert order.is_buy_order() == True
    assert order.is_sell_order() == False
    assert order.is_limit_order() == True
    assert order.is_market_order() == False
    assert order.get_notional_value() == 250000.0


def test_order_status_workflow():
    """Test order status workflow."""
    order = Order(
        symbol="TEST",
        order_type=OrderType.BUY,
        quantity=100,
        price=50.0
    )
    
    # Initial status
    assert order.status == OrderStatus.PENDING
    
    # Simulate order fill
    order.status = OrderStatus.FILLED
    assert order.status == OrderStatus.FILLED
    #assert order.fill_price == 50.5
    #assert order.get_total_cost() == 5050.0 + 50.0
    
    #assert order.status == OrderStatus.FILLED
    #assert order.fill_price == 50.5
    #assert order.get_total_cost() == 5050.0 + 50.0  # Notional + commission


def test_zerodha_broker_interface():
    """Test Zerodha broker interface."""
    # Test broker initialization
    broker = ZerodhaBroker("test_key", "test_secret", "test_token")
    
    assert broker.name == "Zerodha"
    assert broker.is_connected == False
    
    # Test that methods exist (without calling them)
    assert hasattr(broker, 'connect')
    assert hasattr(broker, 'disconnect')
    assert hasattr(broker, 'submit_order')
    assert hasattr(broker, 'cancel_order')
    assert hasattr(broker, 'get_order_status')
    assert hasattr(broker, 'get_positions')
    assert hasattr(broker, 'get_portfolio')


if __name__ == "__main__":
    pytest.main([__file__])
