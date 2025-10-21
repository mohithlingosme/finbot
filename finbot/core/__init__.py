"""
Core module initialization.
"""

# Lazy import pattern to avoid circular import issues
def get_trading_engine():
    from finbot.core.engine import TradingEngine
    return TradingEngine

def get_portfolio():
    from finbot.core.portfolio import Portfolio
    return Portfolio

def get_orders():
    from finbot.core.orders import Order, OrderType, OrderStatus
    return Order, OrderType, OrderStatus

__all__ = ["get_trading_engine", "get_portfolio", "get_orders"]
