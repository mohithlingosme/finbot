"""
Outside bar strategy implementation.

Detects outside bar patterns: a bar that engulfs the previous bar(s).
Entry on breakout of the outside bar in the momentum direction; stop inside bar; target multiple ATRs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.ai_engine.models.strategies.base import BaseStrategy
from src.ai_engine.scripts.indicators.indicators import atr


class OutsideBarStrategy(BaseStrategy):
    """
    Outside bar strategy.

    Identifies outside bars (expansion) and enters on breakout in momentum direction.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize outside bar strategy.

        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)

        # Strategy parameters
        self.atr_period = config.get('atr_period', 14)
        self.atr_multiplier = config.get('atr_multiplier', 2.0)
        self.target_multiplier = config.get('target_multiplier', 2.0)  # Multiple ATRs for target

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

            # Check for outside bar pattern and breakout
            self._check_outside_bar_and_breakout(symbol, current_data, current_atr, current_position, data)

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

    def _check_outside_bar_and_breakout(self, symbol: str, data: pd.DataFrame,
                                       atr: float, position: float, market_data: Dict[str, Any]) -> None:
        """
        Check for outside bar pattern and breakout signals.

        Args:
            symbol: Trading symbol
            data: Price data
            atr: Current ATR value
            position: Current position size
            market_data: Current market data
        """
        current_price = market_data['close']

        # Detect outside bar
        outside_bar_info = self._detect_outside_bar(data)

        if not outside_bar_info:
            return

        outside_high = outside_bar_info['outside_high']
        outside_low = outside_bar_info['outside_low']
        momentum_direction = outside_bar_info['momentum_direction']

        # Check for breakout in momentum direction
        if momentum_direction == 'up' and current_price > outside_high and position == 0:
            # Calculate target and stop
            target_price = current_price + (atr * self.target_multiplier)
            stop_price = outside_low

            # Execute buy order
            self._execute_buy_order(symbol, current_price, stop_price, target_price, atr, market_data)

        elif momentum_direction == 'down' and current_price < outside_low and position == 0:
            # Calculate target and stop
            target_price = current_price - (atr * self.target_multiplier)
            stop_price = outside_high

            # Execute sell order
            self._execute_sell_order(symbol, current_price, stop_price, target_price, atr, market_data)

    def _detect_outside_bar(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Detect outside bar pattern in recent data.

        Returns dict with outside bar info or None if no pattern found.
        """
        if len(data) < 2:
            return None

        # Get last two bars
        current_bar = data.iloc[-1]
        previous_bar = data.iloc[-2]

        # Check if current bar engulfs previous bar
        if (current_bar['high'] >= previous_bar['high'] and
            current_bar['low'] <= previous_bar['low']):
            # Determine momentum direction
            if current_bar['close'] > previous_bar['close']:
                momentum_direction = 'up'
            else:
                momentum_direction = 'down'

            return {
                'outside_high': current_bar['high'],
                'outside_low': current_bar['low'],
                'previous_high': previous_bar['high'],
                'previous_low': previous_bar['low'],
                'momentum_direction': momentum_direction
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
                print(f"OUTSIDE BAR BUY order submitted: {symbol} qty={position_size:.2f} price={price:.2f}")
                notes = f"Outside bar breakout up: Entry {price:.2f}, Stop {stop_price:.2f}, Target {target_price:.2f}"
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
                print(f"OUTSIDE BAR SELL order submitted: {symbol} qty={position_size:.2f} price={price:.2f}")
                notes = f"Outside bar breakout down: Entry {price:.2f}, Stop {stop_price:.2f}, Target {target_price:.2f}"
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

                # Detect outside bar
                outside_bar_info = self._detect_outside_bar(window_data)

                if outside_bar_info:
                    current_price = data.iloc[i]['close']
                    momentum_direction = outside_bar_info['momentum_direction']

                    # Check breakout up
                    if momentum_direction == 'up' and current_price > outside_bar_info['outside_high']:
                        current_atr = atr_series.iloc[i] if i < len(atr_series) else atr_series.iloc[-1]
                        signals.append({
                            'timestamp': data.index[i],
                            'symbol': symbol,
                            'signal_type': 'buy',
                            'strength': 0.7,
                            'price': current_price,
                            'stop_price': outside_bar_info['outside_low'],
                            'target_price': current_price + (current_atr * self.target_multiplier)
                        })

                    # Check breakout down
                    elif momentum_direction == 'down' and current_price < outside_bar_info['outside_low']:
                        current_atr = atr_series.iloc[i] if i < len(atr_series) else atr_series.iloc[-1]
                        signals.append({
                            'timestamp': data.index[i],
                            'symbol': symbol,
                            'signal_type': 'sell',
                            'strength': 0.7,
                            'price': current_price,
                            'stop_price': outside_bar_info['outside_high'],
                            'target_price': current_price - (current_atr * self.target_multiplier)
                        })

        except Exception as e:
            print(f"Error calculating signals for {symbol}: {e}")

        return signals

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': self.name,
            'type': 'Outside Bar',
            'description': 'Detects outside bar patterns and enters on breakouts in momentum direction',
            'parameters': self.parameters,
            'indicators': ['ATR'],
            'timeframe': self.timeframe,
            'enabled': self.enabled
        }
