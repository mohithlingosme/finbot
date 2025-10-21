"""
FINBOT Package Initialization

This module initializes the FINBOT trading framework.
It exposes core functionality, loads configuration files,
and ensures all subpackages are properly registered.

Author: Mohith
Version: 1.0.0
"""

import sys
from pathlib import Path

# -----------------------------------------------------------
# Metadata
# -----------------------------------------------------------
__version__ = "1.0.0"
__author__ = "Mohith"
__license__ = "MIT"

# -----------------------------------------------------------
# Directory setup
# -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Ensure BASE_DIR is in sys.path
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# -----------------------------------------------------------
# Configuration loader (using class-based ConfigLoader)
# -----------------------------------------------------------
try:
    from finbot.utils.config_loader import ConfigLoader
    config_loader = ConfigLoader()
    config = config_loader.config
except Exception as e:
    print(f"[FINBOT] ⚠️ Warning: Config not loaded ({e})")
    config = {}

# -----------------------------------------------------------
# Export key classes for top-level import convenience
# -----------------------------------------------------------
try:
    from finbot.execution.broker import Broker
    from finbot.core.portfolio import Portfolio
    from finbot.core.orders import Order, OrderStatus, OrderType
except ImportError as e:
    print(f"[FINBOT] ⚠️ Optional import failed: {e}")
    Broker = None
    Portfolio = None
    Order = None
    OrderStatus = None
    OrderType = None

# -----------------------------------------------------------
# Public API

# -----------------------------------------------------------
__all__ = [
    "Broker",
    "Portfolio",
    "Order",
    "OrderStatus",
    "OrderType",
    "config",
    "__version__",
]
