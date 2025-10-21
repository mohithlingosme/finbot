"""
Bullish cup and handle strategy implementation.

Detects bullish cup and handle patterns: large rounded base (cup) with small consolidation (handle) before breakout.
Entry on breakout above handle high; stop below handle low; target measured from cup depth.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.ai_engine.models.strategies.base import BaseStrategy
from src.ai_engine.scripts.indicators.indicators import atr


class BullishCupAndHandleStrategy(BaseStrategy):
    """
    Bullish cup and handle strategy.

    Identifies cup and handle consolidation and enters on breakout above handle.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize bullish cup and handle strategy.

        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)

        # Strategy parameters
        self.cup_depth_min_pct = config.get('cup_depth_min_pct', 10.0)  # Minimum cup depth %
        self.cup_depth_max_pct = config.get('cup_depth_max_pct', 50.0)  # Maximum cup depth %
        self.cup_duration_min = config.get('cup_duration_min', 20)  # Minimum bars for cup
        self.handle_duration_max = config.get('handle_duration_max', 15)  # Maximum bars for handle
        self.handle_pullback_max_pct = config.get('handle_pullback_max_pct', 15.0)  # Max handle pullback %
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
            'cup_depth_min_pct': self.cup_depth_min_pct,
            'cup_depth_max_pct': self.cup_depth_max_pct,
            'cup_duration_min': self.cup_duration_min,
            'handle_duration_max': self.handle_duration_max,
            'handle_pullback_max_pct': self.handle_pullback_max_pct,
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
        Check for bullish cup and handle pattern and breakout signals.

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
        cup_depth = pattern_info['cup_depth']

        # Check for breakout above handle
        if current_price > handle_high and position == 0:
            # Calculate target and stop (measured move from cup depth)
            target_price = current_price + cup_depth
            stop_price = handle_low

            # Execute buy order
            self._execute_buy_order(symbol, current_price, stop_price, target_price, atr, market_data)

    def _detect_cup_and_handle(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Detect bullish cup and handle pattern in recent data.

        Returns dict with pattern info or None if no pattern found.
        """
        if len(data) < self.cup_duration_min + self.handle_duration_max + 10:
            return None

        # Look for cup formation (rounded bottom)
        # Find potential cup start and end
        recent_data = data.tail(self.cup_duration_min + self.handle_duration_max + 20)

        # Find the highest high in the lookback period
        peak_idx = recent_data['high'].idxmax()
        peak_price = recent_data.loc[peak_idx, 'high']

        # Find the lowest low after the peak (cup bottom)
        cup_data = recent_data.loc[peak_idx:]
        if len(cup_data) < self.cup_duration_min:
            return None

        bottom_idx = cup_data['low'].idxmin()
        bottom_price = cup_data.loc[bottom_idx, 'low']

        # Calculate cup depth
        cup_depth_pct = (peak_price - bottom_price) / peak_price * 100
        if not (self.cup_depth_min_pct <= cup_depth_pct <= self.cup_depth_max_pct):
            return None

        # Check for rounded bottom (U-shape)
        # Price should recover most of the way back to peak
        recovery_data = cup_data.loc[bottom_idx:]
        if len(recovery_data) < 5:
            return None

        recovery_high = recovery_data['high'].max()
        recovery_pct = (recovery_high - bottom_price) / (peak_price - bottom_price)

        if recovery_pct < 0.7:  # Should recover at least 70% of the drop
            return None

        # Look for handle formation (small consolidation near the peak)
        handle_start_idx = recovery_data['high'].idxmax()
        handle_data = recovery_data.loc[handle_start_idx:]

        if len(handle_data) > self.handle_duration_max or len(handle_data) < 3:
            return None

        # Handle should be a small pullback
        handle_high = handle_data['high'].max()
        handle_low = handle_data['low'].min()
        handle_pullback_pct = (handle_high - handle_low) / handle_high * 100

        if handle_pullback_pct > self.handle_pullback_max_pct:
            return None

        # Handle should be near the cup rim
        rim_price = peak_price * 0.95  # Within 5% of peak
        if handle_high < rim_price:
            return None

        return {
            'cup_peak': peak_price,
            'cup_bottom': bottom_price,
            'cup_depth': peak_price - bottom_price,
            'cup_depth_pct': cup_depth_pct,
            'handle_high': handle_high,
            'handle_low': handle_low,
            'handle_pullback_pct': handle_pullback_pct
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
                print(f"BULLISH CUP & HANDLE BUY order submitted: {symbol} qty={position_size:.2f} price={price:.2f}")
                notes = f"Cup & handle breakout: Entry {price:.2f}, Stop {stop_price:.2f}, Target {target_price:.2f}"
                self.log_signal(symbol, "buy", 0.8, market_data, notes)
            else:
                print(f"Failed to submit BUY order for {symbol}")

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

                    # Check breakout
                    if current_price > pattern_info['handle_high']:
                        signals.append({
                            'timestamp': data.index[i],
                            'symbol': symbol,
                            'signal_type': 'buy',
                            'strength': 0.8,
                            'price': current_price,
                            'handle_high': pattern_info['handle_high'],
                            'handle_low': pattern_info['handle_low'],
                            'cup_depth': pattern_info['cup_depth'],
                            'stop_price': pattern_info['handle_low'],
                            'target_price': current_price + pattern_info['cup_depth']
                        })

        except Exception as e:
            print(f"Error calculating signals for {symbol}: {e}")

        return signals

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': self.name,
            'type': 'Bullish Cup and Handle',
            'description': 'Detects bullish cup and handle patterns and enters on breakouts above handle',
            'parameters': self.parameters,
            'indicators': ['ATR'],
            'timeframe': self.timeframe,
            'enabled': self.enabled
        }
