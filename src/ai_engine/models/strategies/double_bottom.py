"""
Double bottom strategy implementation.

Detects double bottom patterns: two troughs at similar levels indicating potential reversal.
Entry on break above the neckline (between troughs); stop below recent low; target measured from trough-to-neckline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.ai_engine.models.strategies.base import BaseStrategy
from src.ai_engine.scripts.indicators.indicators import atr


class DoubleBottomStrategy(BaseStrategy):
    """
    Double bottom strategy.

    Identifies double bottom reversal patterns and enters on breakout above neckline.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize double bottom strategy.

        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)

        # Strategy parameters
        self.lookback_period = config.get('lookback_period', 50)  # Bars to look for pattern
        self.trough_tolerance_pct = config.get('trough_tolerance_pct', 3.0)  # Max % difference between troughs
        self.neckline_confirm_bars = config.get('neckline_confirm_bars', 3)  # Bars for neckline confirmation
        self.volume_confirmation = config.get('volume_confirmation', True)  # Require volume on breakout
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
            'trough_tolerance_pct': self.trough_tolerance_pct,
            'neckline_confirm_bars': self.neckline_confirm_bars,
            'volume_confirmation': self.volume_confirmation,
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

            # Check for double bottom pattern and breakout
            self._check_double_bottom_and_breakout(symbol, current_data, current_atr, current_position, data)

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

    def _check_double_bottom_and_breakout(self, symbol: str, data: pd.DataFrame,
                                         atr: float, position: float, market_data: Dict[str, Any]) -> None:
        """
        Check for double bottom pattern and breakout signals.

        Args:
            symbol: Trading symbol
            data: Price data
            atr: Current ATR value
            position: Current position size
            market_data: Current market data
        """
        current_price = market_data['close']
        current_volume = market_data['volume']

        # Detect double bottom pattern
        pattern_info = self._detect_double_bottom(data)

        if not pattern_info:
            return

        neckline = pattern_info['neckline']
        trough_height = pattern_info['trough_height']

        # Check for breakout above neckline
        if current_price > neckline and position == 0:
            # Volume confirmation if required
            if self.volume_confirmation:
                avg_volume = data['volume'].rolling(window=20).mean().iloc[-1] if len(data) >= 20 else current_volume
                volume_confirmed = current_volume > avg_volume
                if not volume_confirmed:
                    return

            # Calculate target and stop
            target_price = neckline + trough_height  # Measured move
            stop_price = pattern_info['second_trough']  # Below second trough

            # Execute buy order
            self._execute_buy_order(symbol, current_price, stop_price, target_price, atr, market_data)

    def _detect_double_bottom(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Detect double bottom pattern in recent data.

        Returns dict with pattern info or None if no pattern found.
        """
        if len(data) < self.lookback_period:
            return None

        # Look at recent data
        recent_data = data.tail(self.lookback_period)

        # Find troughs (local minima)
        lows = recent_data['low']
        troughs = []

        for i in range(2, len(lows) - 2):
            if (lows.iloc[i] < lows.iloc[i-1] and
                lows.iloc[i] < lows.iloc[i-2] and
                lows.iloc[i] < lows.iloc[i+1] and
                lows.iloc[i] < lows.iloc[i+2]):
                troughs.append((i, lows.iloc[i]))

        if len(troughs) < 2:
            return None

        # Get the two lowest troughs
        troughs.sort(key=lambda x: x[1])
        first_trough_idx, first_trough = troughs[0]
        second_trough_idx, second_trough = troughs[1]

        # Ensure troughs are in chronological order
        if first_trough_idx > second_trough_idx:
            first_trough_idx, second_trough_idx = second_trough_idx, first_trough_idx
            first_trough, second_trough = second_trough, first_trough

        # Check if troughs are at similar levels
        trough_diff_pct = abs(first_trough - second_trough) / first_trough * 100
        if trough_diff_pct > self.trough_tolerance_pct:
            return None

        # Find neckline (peak between troughs)
        between_data = recent_data.iloc[first_trough_idx:second_trough_idx+1]
        neckline_idx = between_data['high'].idxmax()
        neckline = between_data.loc[neckline_idx, 'high']

        # Ensure neckline is above both troughs
        if neckline <= first_trough or neckline <= second_trough:
            return None

        # Check for confirmation (price tested neckline)
        after_second_trough = recent_data.iloc[second_trough_idx:]
        if len(after_second_trough) < self.neckline_confirm_bars:
            return None

        # Price should have declined back to neckline area after second trough
        min_after_trough = after_second_trough['low'].min()
        neckline_tolerance = neckline * 0.98  # Within 2% of neckline

        if min_after_trough > neckline_tolerance:
            return None

        # Calculate pattern measurements
        trough_height = neckline - (first_trough + second_trough) / 2

        return {
            'first_trough': first_trough,
            'second_trough': second_trough,
            'neckline': neckline,
            'trough_height': trough_height,
            'first_trough_idx': first_trough_idx,
            'second_trough_idx': second_trough_idx,
            'neckline_idx': neckline_idx
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
                print(f"DOUBLE BOTTOM BUY order submitted: {symbol} qty={position_size:.2f} price={price:.2f}")
                notes = f"Double bottom breakout: Entry {price:.2f}, Stop {stop_price:.2f}, Target {target_price:.2f}"
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

        if len(data) < self.lookback_period + 10:
            return signals

        try:
            # Calculate ATR
            atr_series = atr(data, window=self.atr_period)

            for i in range(self.lookback_period, len(data)):
                window_data = data.iloc[:i+1]

                # Detect double bottom pattern
                pattern_info = self._detect_double_bottom(window_data)

                if pattern_info:
                    current_price = data.iloc[i]['close']
                    current_volume = data.iloc[i]['volume']

                    # Check breakout
                    if current_price > pattern_info['neckline']:
                        # Volume confirmation if required
                        if self.volume_confirmation:
                            avg_volume = window_data['volume'].rolling(window=20).mean().iloc[-1]
                            volume_confirmed = current_volume > avg_volume
                            if not volume_confirmed:
                                continue

                        signals.append({
                            'timestamp': data.index[i],
                            'symbol': symbol,
                            'signal_type': 'buy',
                            'strength': 0.8,
                            'price': current_price,
                            'first_trough': pattern_info['first_trough'],
                            'second_trough': pattern_info['second_trough'],
                            'neckline': pattern_info['neckline'],
                            'trough_height': pattern_info['trough_height'],
                            'stop_price': pattern_info['second_trough'],
                            'target_price': pattern_info['neckline'] + pattern_info['trough_height']
                        })

        except Exception as e:
            print(f"Error calculating signals for {symbol}: {e}")

        return signals

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': self.name,
            'type': 'Double Bottom',
            'description': 'Detects double bottom reversal patterns and enters on breakouts above neckline',
            'parameters': self.parameters,
            'indicators': ['ATR'],
            'timeframe': self.timeframe,
            'enabled': self.enabled
        }
