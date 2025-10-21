"""
Rectangle / Range strategy implementation.

Detects rectangle/range patterns: horizontal consolidation between support & resistance.
Entry at support/resistance or trade breakout; stop beyond box edge.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.ai_engine.models.strategies.base import BaseStrategy
from src.ai_engine.scripts.indicators.indicators import atr


class RectangleStrategy(BaseStrategy):
    """
    Rectangle / Range strategy.

    Identifies horizontal consolidation patterns and trades bounces or breakouts.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize rectangle strategy.

        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)

        # Strategy parameters
        self.lookback_period = config.get('lookback_period', 20)  # Bars to look for range
        self.range_threshold = config.get('range_threshold', 0.05)  # Max range % for consolidation
        self.breakout_mode = config.get('breakout_mode', True)  # True for breakout, False for mean-reversion
        self.atr_period = config.get('atr_period', 14)
        self.atr_multiplier = config.get('atr_multiplier', 2.0)

        # Risk management
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% of portfolio
        self.risk_per_trade = config.get('risk_per_trade', 0.02)  # 2% risk per trade

        # Data storage
        self.price_data: Dict[str, pd.DataFrame] = {}

    def _setup_parameters(self) -> Dict[str, Any]:
        """Setup strategy parameters."""
        return {
            'lookback_period': self.lookback_period,
            'range_threshold': self.range_threshold,
            'breakout_mode': self.breakout_mode,
            'atr_period': self.atr_period,
            'atr_multiplier': self.atr_multiplier,
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
        if symbol not in self.price_data or len(self.price_data[symbol]) < self.lookback_period + 10:
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

            # Check for rectangle pattern and signals
            self._check_rectangle_and_signal(symbol, current_data, current_atr, current_position, data)

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

    def _check_rectangle_and_signal(self, symbol: str, data: pd.DataFrame,
                                   atr: float, position: float, market_data: Dict[str, Any]) -> None:
        """
        Check for rectangle pattern and generate signals.

        Args:
            symbol: Trading symbol
            data: Price data
            atr: Current ATR value
            position: Current position size
            market_data: Current market data
        """
        current_price = market_data['close']

        # Detect rectangle
        rectangle_info = self._detect_rectangle(data)

        if not rectangle_info:
            return

        support = rectangle_info['support']
        resistance = rectangle_info['resistance']

        if self.breakout_mode:
            # Breakout mode
            if current_price > resistance and position == 0:
                # Breakout above resistance
                target_price = resistance + (resistance - support)  # Measured move
                stop_price = support

                self._execute_buy_order(symbol, current_price, stop_price, target_price, atr, market_data)

            elif current_price < support and position == 0:
                # Breakout below support
                target_price = support - (resistance - support)  # Measured move
                stop_price = resistance

                self._execute_sell_order(symbol, current_price, stop_price, target_price, atr, market_data)

        else:
            # Mean-reversion mode
            if current_price <= support * 1.001 and position == 0:  # Near support
                target_price = resistance
                stop_price = support - (atr * self.atr_multiplier)

                self._execute_buy_order(symbol, current_price, stop_price, target_price, atr, market_data)

            elif current_price >= resistance * 0.999 and position == 0:  # Near resistance
                target_price = support
                stop_price = resistance + (atr * self.atr_multiplier)

                self._execute_sell_order(symbol, current_price, stop_price, target_price, atr, market_data)

    def _detect_rectangle(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Detect rectangle/range pattern in recent data.

        Returns dict with rectangle info or None if no pattern found.
        """
        if len(data) < self.lookback_period:
            return None

        # Look at recent data
        recent_data = data.tail(self.lookback_period)

        # Find support and resistance levels
        support = recent_data['low'].min()
        resistance = recent_data['high'].max()

        # Check if price is consolidating (range is within threshold)
        range_pct = (resistance - support) / support
        if range_pct > self.range_threshold:
            return None

        # Check if price has tested both levels multiple times
        support_tests = 0
        resistance_tests = 0

        for i in range(len(recent_data)):
            if recent_data.iloc[i]['low'] <= support * 1.005:  # Within 0.5% of support
                support_tests += 1
            if recent_data.iloc[i]['high'] >= resistance * 0.995:  # Within 0.5% of resistance
                resistance_tests += 1

        # Need at least 2 tests of each level
        if support_tests < 2 or resistance_tests < 2:
            return None

        return {
            'support': support,
            'resistance': resistance,
            'range_pct': range_pct,
            'support_tests': support_tests,
            'resistance_tests': resistance_tests
        }

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
                mode = "breakout" if self.breakout_mode else "mean-reversion"
                print(f"RECTANGLE BUY order submitted: {symbol} qty={position_size:.2f} price={price:.2f} mode={mode}")
                notes = f"Rectangle {mode}: Entry {price:.2f}, Stop {stop_price:.2f}, Target {target_price:.2f}"
                self.log_signal(symbol, "buy", 0.6, market_data, notes)
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
                mode = "breakout" if self.breakout_mode else "mean-reversion"
                print(f"RECTANGLE SELL order submitted: {symbol} qty={position_size:.2f} price={price:.2f} mode={mode}")
                notes = f"Rectangle {mode}: Entry {price:.2f}, Stop {stop_price:.2f}, Target {target_price:.2f}"
                self.log_signal(symbol, "sell", 0.6, market_data, notes)
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

        if len(data) < self.lookback_period + 10:
            return signals

        try:
            # Calculate ATR
            atr_series = atr(data, window=self.atr_period)

            for i in range(self.lookback_period, len(data)):
                window_data = data.iloc[:i+1]

                # Detect rectangle
                rectangle_info = self._detect_rectangle(window_data)

                if rectangle_info:
                    current_price = data.iloc[i]['close']
                    support = rectangle_info['support']
                    resistance = rectangle_info['resistance']

                    if self.breakout_mode:
                        # Breakout signals
                        if current_price > resistance:
                            current_atr = atr_series.iloc[i] if i < len(atr_series) else atr_series.iloc[-1]
                            signals.append({
                                'timestamp': data.index[i],
                                'symbol': symbol,
                                'signal_type': 'buy',
                                'strength': 0.6,
                                'price': current_price,
                                'stop_price': support,
                                'target_price': resistance + (resistance - support)
                            })

                        elif current_price < support:
                            current_atr = atr_series.iloc[i] if i < len(atr_series) else atr_series.iloc[-1]
                            signals.append({
                                'timestamp': data.index[i],
                                'symbol': symbol,
                                'signal_type': 'sell',
                                'strength': 0.6,
                                'price': current_price,
                                'stop_price': resistance,
                                'target_price': support - (resistance - support)
                            })

                    else:
                        # Mean-reversion signals
                        if current_price <= support * 1.001:
                            current_atr = atr_series.iloc[i] if i < len(atr_series) else atr_series.iloc[-1]
                            signals.append({
                                'timestamp': data.index[i],
                                'symbol': symbol,
                                'signal_type': 'buy',
                                'strength': 0.6,
                                'price': current_price,
                                'stop_price': support - (current_atr * self.atr_multiplier),
                                'target_price': resistance
                            })

                        elif current_price >= resistance * 0.999:
                            current_atr = atr_series.iloc[i] if i < len(atr_series) else atr_series.iloc[-1]
                            signals.append({
                                'timestamp': data.index[i],
                                'symbol': symbol,
                                'signal_type': 'sell',
                                'strength': 0.6,
                                'price': current_price,
                                'stop_price': resistance + (current_atr * self.atr_multiplier),
                                'target_price': support
                            })

        except Exception as e:
            print(f"Error calculating signals for {symbol}: {e}")

        return signals

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        mode = "breakout" if self.breakout_mode else "mean-reversion"
        return {
            'name': self.name,
            'type': f'Rectangle ({mode})',
            'description': f'Detects rectangle patterns and trades {mode} signals',
            'parameters': self.parameters,
            'indicators': ['ATR'],
            'timeframe': self.timeframe,
            'enabled': self.enabled
        }
