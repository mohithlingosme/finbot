"""
Logging system for TradeBot.
"""

import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger
from finbot.utils.config_loader import ConfigLoader


class Logger:
    """
    Centralized logging system for TradeBot.
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize logger.
        
        Args:
            config: Configuration loader
        """
        self.config = config
        self.log_level = config.get("general.log_level", "INFO")
        
        # Configure loguru
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Setup loguru logger configuration."""
        # Remove default handler
        logger.remove()
        
        # Console handler
        logger.add(
            sys.stdout,
            level=self.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True
        )
        
        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logger.add(
            log_dir / "tradebot_{time:YYYY-MM-DD}.log",
            level=self.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="1 day",
            retention="30 days",
            compression="zip"
        )
        
        # Trade log handler
        logger.add(
            log_dir / "trades_{time:YYYY-MM-DD}.log",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
            filter=lambda record: "TRADE" in record["extra"],
            rotation="1 day",
            retention="90 days"
        )
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        logger.bind(**kwargs).info(message)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        logger.bind(**kwargs).warning(message)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        logger.bind(**kwargs).error(message)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        logger.bind(**kwargs).debug(message)
    
    def log_trade(self, trade_data: Dict[str, Any]) -> None:
        """Log trade information."""
        trade_info = f"TRADE | {trade_data.get('symbol', 'Unknown')} | " \
                    f"{trade_data.get('side', 'Unknown')} | " \
                    f"Qty: {trade_data.get('quantity', 0)} | " \
                    f"Price: {trade_data.get('price', 0)} | " \
                    f"P&L: {trade_data.get('pnl', 0)}"
        
        logger.bind(TRADE=True).info(trade_info)
    
    def log_signal(self, signal_data: Dict[str, Any]) -> None:
        """Log trading signal."""
        signal_info = f"SIGNAL | {signal_data.get('symbol', 'Unknown')} | " \
                     f"{signal_data.get('signal_type', 'Unknown')} | " \
                     f"Strength: {signal_data.get('strength', 0)} | " \
                     f"Price: {signal_data.get('price', 0)}"
        
        logger.info(signal_info)
    
    def log_performance(self, performance_data: Dict[str, Any]) -> None:
        """Log performance metrics."""
        perf_info = f"PERFORMANCE | Total Return: {performance_data.get('total_return', 0)}% | " \
                   f"Sharpe: {performance_data.get('sharpe_ratio', 0)} | " \
                   f"Max DD: {performance_data.get('max_drawdown', 0)}%"
        
        logger.info(perf_info)
