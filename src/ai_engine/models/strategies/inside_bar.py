"""
Inside bar strategy implementation.

Detects inside bar patterns: a bar whose high/low is inside the previous (mother) bar.
Entry on breakout beyond mother bar; stop inside mother bar; target 1–3× ATR.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.ai_engine.models.strategies.base import BaseStrategy
from src.ai_engine.scripts.indicators.indicators import atr


class InsideBarStrategy(BaseStrategy):
    """
    Inside bar strategy.

    Identifies inside bars (compression) and enters on breakout beyond mother bar.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize inside bar strategy.

        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)

        # Strategy parameters
        self.atr_period = config.get('atr_period', 14)
        self.atr_multiplier = config.get('atr_multiplier', 2.0)
        self.target_multiplier = config.get('target_multiplier', 2.0)  # 1-3x ATR for target

        # Risk management
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% of portfolio
        self.risk_per_trade = config.get('risk_per_trade', 0.02)  # 2% risk per trade

        # Data storage
        self.price_data: Dict[str, pd.DataFrame] = {}

    def _setup_parameters(self) -> Dict[str, Any]:
        """Setup strategy parameters."""
        return {
            'atr_period': self.atr_period,
            'atr_multiplier': self.atr_multiplier,
            'target_multiplier': self.target_multiplier,
            'max_position_size': self.max_position_size,
            'risk_per_trade': self.risk_per_trade
        }

    def on_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """
        Process new market data.

        Args:
            symbol: Trading symbol
            data: Market data dictionary
        """
        if not self.validate_data(data):
            return

        # Update price data
        self._update_price_data(symbol, data)

        # Check if we have enough data
        if symbol not in self.price_data or len(self.price_data[symbol]) < 20:
            return

        # Calculate indicators
        current_data = self.price_data[symbol]

        try:
            # Calculate ATR
            atr_series = atr(current_data, window=self.atr_period)
            current_atr = atr_series.iloc[-1] if not atr_series.empty else None

            if current_atr is None:
                return

            # Get current values
            current_price = data['close']
            current_position = self.get_position(symbol)

            # Check for inside bar pattern and breakout
            self._check_inside_bar_and_breakout(symbol, current_data, current_atr, current_position, data)

        except Exception as e:
            print(f"Error processing data for {symbol}: {e}")

    def _update_price_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """Update price data for symbol."""
        if symbol not in self.price_data:
            self.price_data[symbol] = pd.DataFrame()

        # Create new row
        new_row = pd.DataFrame({
            'open': [data['open']],
            'high': [data['high']],
            'low': [data['low']],
            'close': [data['close']],
            'volume': [data['volume']],
            'timestamp': [data.get('timestamp', datetime.now())]
        })

        # Append to existing data
        self.price_data[symbol] = pd.concat([self.price_data[symbol], new_row], ignore_index=True)

        # Keep only recent data
        if len(self.price_data[symbol]) > 1000:
            self.price_data[symbol] = self.price_data[symbol].tail(1000).reset_index(drop=True)

    def _check_inside_bar_and_breakout(self, symbol: str, data: pd.DataFrame,
                                       atr: float, position: float, market_data: Dict[str, Any]) -> None:
        """
        Check for inside bar pattern and breakout signals.

        Args:
            symbol: Trading symbol
            data: Price data
            atr: Current ATR value
            position: Current position size
            market_data: Current market data
        """
        current_price = market_data['close']

        # Detect inside bar
        inside_bar_info = self._detect_inside_bar(data)

        if not inside_bar_info:
            return

        mother_high = inside_bar_info['mother_high']
        mother_low = inside_bar_info['mother_low']

        # Check for breakout above mother bar
        if current_price > mother_high and position == 0:
            # Calculate target and stop
            target_price = current_price + (atr * self.target_multiplier)
            stop_price = mother_low

            # Execute buy order
            self._execute_buy_order(symbol, current_price, stop_price, target_price, atr, market_data)

        # Check for breakout below mother bar
        elif current_price < mother_low and position == 0:
            # Calculate target and stop
            target_price = current_price - (atr * self.target_multiplier)
            stop_price = mother_high

            # Execute sell order
            self._execute_sell_order(symbol, current_price, stop_price, target_price, atr, market_data)

    def _detect_inside_bar(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Detect inside bar pattern in recent data.

        Returns dict with mother bar info or None if no pattern found.
        """
        if len(data) < 2:
            return None

        # Get last two bars
        current_bar = data.iloc[-1]
        mother_bar = data.iloc[-2]

        # Check if current bar is inside mother bar
        if (current_bar['high'] <= mother_bar['high'] and
            current_bar['low'] >= mother_bar['low']):
            return {
                'mother_high': mother_bar['high'],
                'mother_low': mother_bar['low'],
                'inside_high': current_bar['high'],
                'inside_low': current_bar['low']
            }

        return None

    def _execute_buy_order(self, symbol: str, price: float, stop_price: float,
                          target_price: float, atr: float, market_data: Dict[str, Any]) -> None:
        """Execute buy order."""
        # Calculate position size based on risk
        portfolio_value = self.get_portfolio_value()
        risk_amount = portfolio_value * self.risk_per_trade

        # Risk per share
        risk_per_share = price - stop_price
        if risk_per_share <= 0:
            return

        position_size = risk_amount / risk_per_share

        # Cap position size
        max_position_value = portfolio_value * self.max_position_size
        max_position_size = max_position_value / price
        position_size = min(position_size, max_position_size)

        if position_size > 0:
            # Create buy order
            order = self.generate_order(
                symbol=symbol,
                side='buy',
                quantity=position_size,
                price=price,
                order_type='market'
            )

            # Submit order
            if self.submit_order(order):
                print(f"INSIDE BAR BUY order submitted: {symbol} qty={position_size:.2f} price={price:.2f}")
                notes = f"Inside bar breakout up: Entry {price:.2f}, Stop {stop_price:.2f}, Target {target_price:.2f}"
                self.log_signal(symbol, "buy", 0.7, market_data, notes)
            else:
                print(f"Failed to submit BUY order for {symbol}")

    def _execute_sell_order(self, symbol: str, price: float, stop_price: float,
                           target_price: float, atr: float, market_data: Dict[str, Any]) -> None:
        """Execute sell order."""
        # Calculate position size based on risk
        portfolio_value = self.get_portfolio_value()
        risk_amount = portfolio_value * self.risk_per_trade

        # Risk per share
        risk_per_share = stop_price - price
        if risk_per_share <= 0:
            return

        position_size = risk_amount / risk_per_share

        # Cap position size
        max_position_value = portfolio_value * self.max_position_size
        max_position_size = max_position_value / price
        position_size = min(position_size, max_position_size)

        if position_size > 0:
            # Create sell order
            order = self.generate_order(
                symbol=symbol,
                side='sell',
                quantity=position_size,
                price=price,
                order_type='market'
            )

            # Submit order
            if self.submit_order(order):
                print(f"INSIDE BAR SELL order submitted: {symbol} qty={position_size:.2f} price={price:.2f}")
                notes = f"Inside bar breakout down: Entry {price:.2f}, Stop {stop_price:.2f}, Target {target_price:.2f}"
                self.log_signal(symbol, "sell", 0.7, market_data, notes)
            else:
                print(f"Failed to submit SELL order for {symbol}")

    def calculate_signals(self, data: pd.DataFrame, symbol: str) -> List[Dict[str, Any]]:
        """
        Calculate trading signals from historical data.

        Args:
            data: Historical market data
            symbol: Trading symbol

        Returns:
            List[Dict]: List of signal dictionaries
        """
        signals = []

        if len(data) < 20:
            return signals

        try:
            # Calculate ATR
            atr_series = atr(data, window=self.atr_period)

            for i in range(1, len(data)):
                window_data = data.iloc[:i+1]

                # Detect inside bar
                inside_bar_info = self._detect_inside_bar(window_data)

                if inside_bar_info:
                    current_price = data.iloc[i]['close']

                    # Check breakout up
                    if current_price > inside_bar_info['mother_high']:
                        current_atr = atr_series.iloc[i] if i < len(atr_series) else atr_series.iloc[-1]
                        signals.append({
                            'timestamp': data.index[i],
                            'symbol': symbol,
                            'signal_type': 'buy',
                            'strength': 0.7,
                            'price': current_price,
                            'stop_price': inside_bar_info['mother_low'],
                            'target_price': current_price + (current_atr * self.target_multiplier)
                        })

                    # Check breakout down
                    elif current_price < inside_bar_info['mother_low']:
                        current_atr = atr_series.iloc[i] if i < len(atr_series) else atr_series.iloc[-1]
                        signals.append({
                            'timestamp': data.index[i],
                            'symbol': symbol,
                            'signal_type': 'sell',
                            'strength': 0.7,
                            'price': current_price,
                            'stop_price': inside_bar_info['mother_high'],
                            'target_price': current_price - (current_atr * self.target_multiplier)
                        })

        except Exception as e:
            print(f"Error calculating signals for {symbol}: {e}")

        return signals

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': self.name,
            'type': 'Inside Bar',
            'description': 'Detects inside bar patterns and enters on breakouts beyond mother bar',
            'parameters': self.parameters,
            'indicators': ['ATR'],
            'timeframe': self.timeframe,
            'enabled': self.enabled
        }
