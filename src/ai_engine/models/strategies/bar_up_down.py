"""
Bar Up/Down strategy implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from .base import BaseStrategy
from src.ai_engine.indicators.indicators import atr
from src.backend.core.orders import OrderType


class BarUpDownStrategy(BaseStrategy):
    """
    Bar Up/Down strategy.

    This strategy trades based on directional single-bar or multi-bar patterns:
    - Strong bullish bars (close > open and close > previous high) with volume spike
    - Strong bearish bars (close < open and close < previous low) with volume spike
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize Bar Up/Down strategy.

        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)

        # Strategy parameters
        self.volume_multiplier = config.get('volume_multiplier', 1.5)  # Volume spike threshold
        self.atr_period = config.get('atr_period', 14)
        self.atr_multiplier = config.get('atr_multiplier', 2.0)  # Risk multiplier
        self.target_multiplier = config.get('target_multiplier', 2.0)  # Profit target multiplier

        # Risk management
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% of portfolio
        self.risk_per_trade = config.get('risk_per_trade', 0.02)  # 2% risk per trade

        # Data storage
        self.price_data: Dict[str, pd.DataFrame] = {}

    def _setup_parameters(self) -> Dict[str, Any]:
        """Setup strategy parameters."""
        return {
            'volume_multiplier': self.volume_multiplier,
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
        if symbol not in self.price_data or len(self.price_data[symbol]) < max(self.atr_period, 2):
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

            # Check for bar patterns
            self._check_bar_patterns(symbol, current_data, current_atr, current_position, data)

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

    def _check_bar_patterns(self, symbol: str, data: pd.DataFrame, atr: float,
                           position: float, market_data: Dict[str, Any]) -> None:
        """
        Check for bullish/bearish bar patterns.

        Args:
            symbol: Trading symbol
            data: Price data
            atr: Current ATR value
            position: Current position size
            market_data: Current market data
        """
        current_price = market_data['close']
        current_volume = market_data['volume']

        # Get current and previous bars
        current_bar = data.iloc[-1]
        prev_bar = data.iloc[-2] if len(data) > 1 else None

        if prev_bar is None:
            return

        # Check volume spike (above average)
        avg_volume = data['volume'].rolling(window=20).mean().iloc[-1] if len(data) >= 20 else current_volume
        volume_spike = current_volume > (avg_volume * self.volume_multiplier)

        signal_type = "hold"
        signal_strength = 0.0
        notes = ""

        # Bullish bar pattern
        bullish_bar = (current_bar['close'] > current_bar['open'] and
                      current_bar['close'] > prev_bar['high'])

        # Bearish bar pattern
        bearish_bar = (current_bar['close'] < current_bar['open'] and
                       current_bar['close'] < prev_bar['low'])

        # Entry signals (no current position)
        if position == 0 and volume_spike:
            if bullish_bar:
                signal_type = "buy"
                signal_strength = 0.8
                notes = f"Bullish bar: Close {current_bar['close']:.2f} > Open {current_bar['open']:.2f} and Prev High {prev_bar['high']:.2f}"

            elif bearish_bar:
                signal_type = "sell_short"
                signal_strength = 0.8
                notes = f"Bearish bar: Close {current_bar['close']:.2f} < Open {current_bar['open']:.2f} and Prev Low {prev_bar['low']:.2f}"

        # Exit signals for existing positions
        elif position > 0:  # Long position
            # Stop loss: below current bar low
            if current_price < current_bar['low']:
                signal_type = "sell"
                signal_strength = 0.9
                notes = f"Stop loss: Price {current_price:.2f} < Bar Low {current_bar['low']:.2f}"

        elif position < 0:  # Short position
            # Stop loss: above current bar high
            if current_price > current_bar['high']:
                signal_type = "buy_to_cover"
                signal_strength = 0.9
                notes = f"Stop loss: Price {current_price:.2f} > Bar High {current_bar['high']:.2f}"

        # Execute trades
        if signal_type == "buy" and position == 0:
            self._execute_buy_order(symbol, current_price, atr, signal_strength, notes)

        elif signal_type == "sell" and position > 0:
            self._execute_sell_order(symbol, current_price, signal_strength, notes)

        elif signal_type == "sell_short" and position == 0:
            self._execute_sell_short_order(symbol, current_price, atr, signal_strength, notes)

        elif signal_type == "buy_to_cover" and position < 0:
            self._execute_buy_to_cover_order(symbol, current_price, signal_strength, notes)

        # Log signal
        if signal_type != "hold":
            self.log_signal(symbol, signal_type, signal_strength, market_data, notes)

    def _execute_buy_order(self, symbol: str, price: float, atr: float,
                          strength: float, notes: str) -> None:
        """Execute buy order."""
        # Calculate position size
        portfolio_value = self.get_portfolio_value()
        risk_amount = portfolio_value * self.risk_per_trade

        # Stop loss below current bar low
        current_data = self.price_data[symbol]
        stop_loss_price = current_data['low'].iloc[-1]

        # Ensure stop loss is valid
        if stop_loss_price >= price:
            return  # Cannot calculate position size if stop loss is at or above entry

        # Calculate position size based on risk
        position_size = risk_amount / (price - stop_loss_price)

        # Cap position size by max_position_size
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
                print(f"BUY order submitted: {symbol} qty={position_size:.2f} price={price:.2f}")
            else:
                print(f"Failed to submit BUY order for {symbol}")

    def _execute_sell_order(self, symbol: str, price: float, strength: float, notes: str) -> None:
        """Execute sell order."""
        current_position = self.get_position(symbol)

        if current_position > 0:
            # Create sell order
            order = self.generate_order(
                symbol=symbol,
                side='sell',
                quantity=current_position,
                price=price,
                order_type='market'
            )

            # Submit order
            if self.submit_order(order):
                print(f"SELL order submitted: {symbol} qty={current_position:.2f} price={price:.2f}")
            else:
                print(f"Failed to submit SELL order for {symbol}")

    def _execute_sell_short_order(self, symbol: str, price: float, atr: float,
                                 strength: float, notes: str) -> None:
        """Execute sell short order."""
        # Calculate position size
        portfolio_value = self.get_portfolio_value()
        risk_amount = portfolio_value * self.risk_per_trade

        # Stop loss above current bar high
        current_data = self.price_data[symbol]
        stop_loss_price = current_data['high'].iloc[-1]

        # Ensure stop loss is valid
        if stop_loss_price <= price:
            return  # Cannot calculate position size if stop loss is at or below entry

        # Calculate position size based on risk
        position_size = risk_amount / (stop_loss_price - price)

        # Cap position size by max_position_size
        max_position_value = portfolio_value * self.max_position_size
        max_position_size = max_position_value / price
        position_size = min(position_size, max_position_size)

        if position_size > 0:
            # Create sell short order (assuming OrderType supports it)
            order = self.generate_order(
                symbol=symbol,
                side='sell',
                quantity=position_size,
                price=price,
                order_type='market'
            )

            # Submit order
            if self.submit_order(order):
                print(f"SELL SHORT order submitted: {symbol} qty={position_size:.2f} price={price:.2f}")
            else:
                print(f"Failed to submit SELL SHORT order for {symbol}")

    def _execute_buy_to_cover_order(self, symbol: str, price: float, strength: float, notes: str) -> None:
        """Execute buy to cover order."""
        current_position = abs(self.get_position(symbol))  # Short position is negative

        if current_position > 0:
            # Create buy to cover order
            order = self.generate_order(
                symbol=symbol,
                side='buy',
                quantity=current_position,
                price=price,
                order_type='market'
            )

            # Submit order
            if self.submit_order(order):
                print(f"BUY TO COVER order submitted: {symbol} qty={current_position:.2f} price={price:.2f}")
            else:
                print(f"Failed to submit BUY TO COVER order for {symbol}")

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

        if len(data) < max(self.atr_period, 2):
            return signals

        try:
            # Calculate ATR
            atr_series = atr(data, window=self.atr_period)

            # Calculate average volume
            data['avg_volume'] = data['volume'].rolling(window=20).mean()

            for i in range(1, len(data)):
                if pd.isna(atr_series.iloc[i]):
                    continue

                current_bar = data.iloc[i]
                prev_bar = data.iloc[i-1]
                current_atr = atr_series.iloc[i]
                avg_volume = data['avg_volume'].iloc[i]

                # Check volume spike
                volume_spike = current_bar['volume'] > (avg_volume * self.volume_multiplier)

                # Bullish bar
                bullish_bar = (current_bar['close'] > current_bar['open'] and
                              current_bar['close'] > prev_bar['high'])

                # Bearish bar
                bearish_bar = (current_bar['close'] < current_bar['open'] and
                               current_bar['close'] < prev_bar['low'])

                signal_type = "hold"
                signal_strength = 0.0

                if volume_spike:
                    if bullish_bar:
                        signal_type = "buy"
                        signal_strength = 0.8
                    elif bearish_bar:
                        signal_type = "sell_short"
                        signal_strength = 0.8

                if signal_type != "hold":
                    signals.append({
                        'timestamp': data.index[i],
                        'symbol': symbol,
                        'signal_type': signal_type,
                        'strength': signal_strength,
                        'price': current_bar['close'],
                        'volume': current_bar['volume'],
                        'avg_volume': avg_volume,
                        'bullish_bar': bullish_bar,
                        'bearish_bar': bearish_bar,
                        'volume_spike': volume_spike
                    })

        except Exception as e:
            print(f"Error calculating signals for {symbol}: {e}")

        return signals

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': self.name,
            'type': 'Bar Up/Down',
            'description': 'Trades directional single/multi-bar patterns with volume confirmation',
            'parameters': self.parameters,
            'indicators': ['ATR'],
            'timeframe': self.timeframe,
            'enabled': self.enabled
        }
