"""
Bearish cup and handle strategy implementation.

Detects bearish inverted cup and handle patterns: large inverted rounded top with small consolidation before breakdown.
Entry on breakdown below handle low; stop above handle high; target measured from cup height.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.ai_engine.models.strategies.base import BaseStrategy
from src.ai_engine.scripts.indicators.indicators import atr


class BearishCupAndHandleStrategy(BaseStrategy):
    """
    Bearish cup and handle strategy.

    Identifies inverted cup and handle consolidation and enters on breakdown below handle.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize bearish cup and handle strategy.

        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)

        # Strategy parameters
        self.cup_height_min_pct = config.get('cup_height_min_pct', 10.0)  # Minimum cup height %
        self.cup_height_max_pct = config.get('cup_height_max_pct', 50.0)  # Maximum cup height %
        self.cup_duration_min = config.get('cup_duration_min', 20)  # Minimum bars for cup
        self.handle_duration_max = config.get('handle_duration_max', 15)  # Maximum bars for handle
        self.handle_rally_max_pct = config.get('handle_rally_max_pct', 15.0)  # Max handle rally %
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
            'cup_height_min_pct': self.cup_height_min_pct,
            'cup_height_max_pct': self.cup_height_max_pct,
            'cup_duration_min': self.cup_duration_min,
            'handle_duration_max': self.handle_duration_max,
            'handle_rally_max_pct': self.handle_rally_max_pct,
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
        if symbol not in self.price_data or len(self.price_data[symbol]) < self.cup_duration_min + self.handle_duration_max + 20:
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

            # Check for cup and handle pattern and breakout
            self._check_cup_handle_and_breakout(symbol, current_data, current_atr, current_position, data)

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

    def _check_cup_handle_and_breakout(self, symbol: str, data: pd.DataFrame,
                                      atr: float, position: float, market_data: Dict[str, Any]) -> None:
        """
        Check for bearish cup and handle pattern and breakout signals.

        Args:
            symbol: Trading symbol
            data: Price data
            atr: Current ATR value
            position: Current position size
            market_data: Current market data
        """
        current_price = market_data['close']

        # Detect cup and handle pattern
        pattern_info = self._detect_cup_and_handle(data)

        if not pattern_info:
            return

        handle_high = pattern_info['handle_high']
        handle_low = pattern_info['handle_low']
        cup_height = pattern_info['cup_height']

        # Check for breakdown below handle
        if current_price < handle_low and position == 0:
            # Calculate target and stop (measured move from cup height)
            target_price = current_price - cup_height
            stop_price = handle_high

            # Execute sell order
            self._execute_sell_order(symbol, current_price, stop_price, target_price, atr, market_data)

    def _detect_cup_and_handle(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Detect bearish inverted cup and handle pattern in recent data.

        Returns dict with pattern info or None if no pattern found.
        """
        if len(data) < self.cup_duration_min + self.handle_duration_max + 10:
            return None

        # Look for inverted cup formation (rounded top)
        # Find potential cup start and end
        recent_data = data.tail(self.cup_duration_min + self.handle_duration_max + 20)

        # Find the lowest low in the lookback period
        trough_idx = recent_data['low'].idxmin()
        trough_price = recent_data.loc[trough_idx, 'low']

        # Find the highest high after the trough (cup top)
        cup_data = recent_data.loc[trough_idx:]
        if len(cup_data) < self.cup_duration_min:
            return None

        top_idx = cup_data['high'].idxmax()
        top_price = cup_data.loc[top_idx, 'high']

        # Calculate cup height
        cup_height_pct = (top_price - trough_price) / trough_price * 100
        if not (self.cup_height_min_pct <= cup_height_pct <= self.cup_height_max_pct):
            return None

        # Check for rounded top (inverted U-shape)
        # Price should decline most of the way back to trough
        decline_data = cup_data.loc[top_idx:]
        if len(decline_data) < 5:
            return None

        decline_low = decline_data['low'].min()
        decline_pct = (top_price - decline_low) / (top_price - trough_price)

        if decline_pct < 0.7:  # Should decline at least 70% of the rally
            return None

        # Look for handle formation (small consolidation near the bottom)
        handle_start_idx = decline_data['low'].idxmin()
        handle_data = decline_data.loc[handle_start_idx:]

        if len(handle_data) > self.handle_duration_max or len(handle_data) < 3:
            return None

        # Handle should be a small rally
        handle_high = handle_data['high'].max()
        handle_low = handle_data['low'].min()
        handle_rally_pct = (handle_high - handle_low) / handle_low * 100

        if handle_rally_pct > self.handle_rally_max_pct:
            return None

        # Handle should be near the cup bottom
        bottom_price = trough_price * 1.05  # Within 5% of trough
        if handle_low > bottom_price:
            return None

        return {
            'cup_top': top_price,
            'cup_bottom': trough_price,
            'cup_height': top_price - trough_price,
            'cup_height_pct': cup_height_pct,
            'handle_high': handle_high,
            'handle_low': handle_low,
            'handle_rally_pct': handle_rally_pct
        }

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
                print(f"BEARISH CUP & HANDLE SELL order submitted: {symbol} qty={position_size:.2f} price={price:.2f}")
                notes = f"Inverted cup & handle breakdown: Entry {price:.2f}, Stop {stop_price:.2f}, Target {target_price:.2f}"
                self.log_signal(symbol, "sell", 0.8, market_data, notes)
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

        min_length = self.cup_duration_min + self.handle_duration_max + 20
        if len(data) < min_length:
            return signals

        try:
            # Calculate ATR
            atr_series = atr(data, window=self.atr_period)

            for i in range(min_length, len(data)):
                window_data = data.iloc[:i+1]

                # Detect cup and handle pattern
                pattern_info = self._detect_cup_and_handle(window_data)

                if pattern_info:
                    current_price = data.iloc[i]['close']

                    # Check breakdown
                    if current_price < pattern_info['handle_low']:
                        signals.append({
                            'timestamp': data.index[i],
                            'symbol': symbol,
                            'signal_type': 'sell',
                            'strength': 0.8,
                            'price': current_price,
                            'handle_high': pattern_info['handle_high'],
                            'handle_low': pattern_info['handle_low'],
                            'cup_height': pattern_info['cup_height'],
                            'stop_price': pattern_info['handle_high'],
                            'target_price': current_price - pattern_info['cup_height']
                        })

        except Exception as e:
            print(f"Error calculating signals for {symbol}: {e}")

        return signals

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': self.name,
            'type': 'Bearish Cup and Handle',
            'description': 'Detects bearish inverted cup and handle patterns and enters on breakdowns below handle',
            'parameters': self.parameters,
            'indicators': ['ATR'],
            'timeframe': self.timeframe,
            'enabled': self.enabled
        }
