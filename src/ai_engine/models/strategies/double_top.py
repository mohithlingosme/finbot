"""
Double top strategy implementation.

Detects double top patterns: two peaks at similar levels indicating potential reversal.
Entry on break below the neckline (between peaks); stop above recent high; target measured from peak-to-neckline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.ai_engine.models.strategies.base import BaseStrategy
from src.ai_engine.scripts.indicators.indicators import atr


class DoubleTopStrategy(BaseStrategy):
    """
    Double top strategy.

    Identifies double top reversal patterns and enters on breakdown below neckline.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize double top strategy.

        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)

        # Strategy parameters
        self.lookback_period = config.get('lookback_period', 50)  # Bars to look for pattern
        self.peak_tolerance_pct = config.get('peak_tolerance_pct', 3.0)  # Max % difference between peaks
        self.neckline_confirm_bars = config.get('neckline_confirm_bars', 3)  # Bars for neckline confirmation
        self.volume_confirmation = config.get('volume_confirmation', True)  # Require volume on breakdown
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
            'peak_tolerance_pct': self.peak_tolerance_pct,
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

            # Check for double top pattern and breakdown
            self._check_double_top_and_breakdown(symbol, current_data, current_atr, current_position, data)

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

    def _check_double_top_and_breakdown(self, symbol: str, data: pd.DataFrame,
                                       atr: float, position: float, market_data: Dict[str, Any]) -> None:
        """
        Check for double top pattern and breakdown signals.

        Args:
            symbol: Trading symbol
            data: Price data
            atr: Current ATR value
            position: Current position size
            market_data: Current market data
        """
        current_price = market_data['close']
        current_volume = market_data['volume']

        # Detect double top pattern
        pattern_info = self._detect_double_top(data)

        if not pattern_info:
            return

        neckline = pattern_info['neckline']
        peak_height = pattern_info['peak_height']

        # Check for breakdown below neckline
        if current_price < neckline and position == 0:
            # Volume confirmation if required
            if self.volume_confirmation:
                avg_volume = data['volume'].rolling(window=20).mean().iloc[-1] if len(data) >= 20 else current_volume
                volume_confirmed = current_volume > avg_volume
                if not volume_confirmed:
                    return

            # Calculate target and stop
            target_price = neckline - peak_height  # Measured move
            stop_price = pattern_info['second_peak']  # Above second peak

            # Execute sell order
            self._execute_sell_order(symbol, current_price, stop_price, target_price, atr, market_data)

    def _detect_double_top(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Detect double top pattern in recent data.

        Returns dict with pattern info or None if no pattern found.
        """
        if len(data) < self.lookback_period:
            return None

        # Look at recent data
        recent_data = data.tail(self.lookback_period)

        # Find peaks (local maxima)
        highs = recent_data['high']
        peaks = []

        for i in range(2, len(highs) - 2):
            if (highs.iloc[i] > highs.iloc[i-1] and
                highs.iloc[i] > highs.iloc[i-2] and
                highs.iloc[i] > highs.iloc[i+1] and
                highs.iloc[i] > highs.iloc[i+2]):
                peaks.append((i, highs.iloc[i]))

        if len(peaks) < 2:
            return None

        # Get the two highest peaks
        peaks.sort(key=lambda x: x[1], reverse=True)
        first_peak_idx, first_peak = peaks[0]
        second_peak_idx, second_peak = peaks[1]

        # Ensure peaks are in chronological order
        if first_peak_idx > second_peak_idx:
            first_peak_idx, second_peak_idx = second_peak_idx, first_peak_idx
            first_peak, second_peak = second_peak, first_peak

        # Check if peaks are at similar levels
        peak_diff_pct = abs(first_peak - second_peak) / first_peak * 100
        if peak_diff_pct > self.peak_tolerance_pct:
            return None

        # Find neckline (trough between peaks)
        between_data = recent_data.iloc[first_peak_idx:second_peak_idx+1]
        neckline_idx = between_data['low'].idxmin()
        neckline = between_data.loc[neckline_idx, 'low']

        # Ensure neckline is below both peaks
        if neckline >= first_peak or neckline >= second_peak:
            return None

        # Check for confirmation (price tested neckline)
        after_second_peak = recent_data.iloc[second_peak_idx:]
        if len(after_second_peak) < self.neckline_confirm_bars:
            return None

        # Price should have rallied back to neckline area after second peak
        max_after_peak = after_second_peak['high'].max()
        neckline_tolerance = neckline * 1.02  # Within 2% of neckline

        if max_after_peak < neckline_tolerance:
            return None

        # Calculate pattern measurements
        peak_height = (first_peak + second_peak) / 2 - neckline

        return {
            'first_peak': first_peak,
            'second_peak': second_peak,
            'neckline': neckline,
            'peak_height': peak_height,
            'first_peak_idx': first_peak_idx,
            'second_peak_idx': second_peak_idx,
            'neckline_idx': neckline_idx
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
                print(f"DOUBLE TOP SELL order submitted: {symbol} qty={position_size:.2f} price={price:.2f}")
                notes = f"Double top breakdown: Entry {price:.2f}, Stop {stop_price:.2f}, Target {target_price:.2f}"
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

        if len(data) < self.lookback_period + 10:
            return signals

        try:
            # Calculate ATR
            atr_series = atr(data, window=self.atr_period)

            for i in range(self.lookback_period, len(data)):
                window_data = data.iloc[:i+1]

                # Detect double top pattern
                pattern_info = self._detect_double_top(window_data)

                if pattern_info:
                    current_price = data.iloc[i]['close']
                    current_volume = data.iloc[i]['volume']

                    # Check breakdown
                    if current_price < pattern_info['neckline']:
                        # Volume confirmation if required
                        if self.volume_confirmation:
                            avg_volume = window_data['volume'].rolling(window=20).mean().iloc[-1]
                            volume_confirmed = current_volume > avg_volume
                            if not volume_confirmed:
                                continue

                        signals.append({
                            'timestamp': data.index[i],
                            'symbol': symbol,
                            'signal_type': 'sell',
                            'strength': 0.8,
                            'price': current_price,
                            'first_peak': pattern_info['first_peak'],
                            'second_peak': pattern_info['second_peak'],
                            'neckline': pattern_info['neckline'],
                            'peak_height': pattern_info['peak_height'],
                            'stop_price': pattern_info['second_peak'],
                            'target_price': pattern_info['neckline'] - pattern_info['peak_height']
                        })

        except Exception as e:
            print(f"Error calculating signals for {symbol}: {e}")

        return signals

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': self.name,
            'type': 'Double Top',
            'description': 'Detects double top reversal patterns and enters on breakdowns below neckline',
            'parameters': self.parameters,
            'indicators': ['ATR'],
            'timeframe': self.timeframe,
            'enabled': self.enabled
        }
