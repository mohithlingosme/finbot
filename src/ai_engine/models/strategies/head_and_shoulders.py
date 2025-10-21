"""
Head and Shoulders pattern strategy implementation.

Detects head and shoulders (and inverse) patterns: left shoulder, head (higher peak), right shoulder; neckline break signals reversal.
Entry on neckline break; stop above right shoulder; target ≈ head-to-neck distance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.ai_engine.models.strategies.base import BaseStrategy
from src.ai_engine.scripts.indicators.indicators import atr


class HeadAndShouldersStrategy(BaseStrategy):
    """
    Head and Shoulders pattern strategy.

    Detects classic head and shoulders reversal patterns.
    Enters on neckline breakout with volume confirmation.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize head and shoulders strategy.

        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)

        # Strategy parameters
        self.lookback = config.get('lookback', 50)  # Bars to look for pattern
        self.min_pattern_height = config.get('min_pattern_height', 3.0)  # Minimum % height
        self.neckline_tolerance = config.get('neckline_tolerance', 0.5)  # % tolerance for neckline
        self.volume_multiplier = config.get('volume_multiplier', 1.2)  # Volume confirmation
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
            'lookback': self.lookback,
            'min_pattern_height': self.min_pattern_height,
            'neckline_tolerance': self.neckline_tolerance,
            'volume_multiplier': self.volume_multiplier,
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
        if symbol not in self.price_data or len(self.price_data[symbol]) < self.lookback:
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
            current_position = self.get_position(symbol)

            # Check for head and shoulders patterns
            self._check_head_and_shoulders_patterns(symbol, current_data, current_atr, current_position, data)

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

    def _check_head_and_shoulders_patterns(self, symbol: str, data: pd.DataFrame,
                                         atr: float, position: float, market_data: Dict[str, Any]) -> None:
        """
        Check for head and shoulders patterns and generate signals.

        Args:
            symbol: Trading symbol
            data: Price data
            atr: Current ATR value
            position: Current position size
            market_data: Current market data
        """
        current_price = market_data['close']
        current_volume = market_data['volume']

        # Detect head and shoulders pattern
        pattern_info = self._detect_head_and_shoulders(data)

        if not pattern_info:
            return

        pattern_type = pattern_info['type']
        neckline = pattern_info['neckline']
        head_height = pattern_info['head_height']

        # Check for neckline breakout
        if pattern_type == 'bearish' and current_price < neckline:
            # Volume confirmation
            avg_volume = data['volume'].rolling(window=20).mean().iloc[-1] if len(data) >= 20 else current_volume
            volume_confirmation = current_volume > (avg_volume * self.volume_multiplier)

            if volume_confirmation and position >= 0:  # Allow if flat or long
                # Calculate target and stop
                target_price = neckline - head_height
                stop_price = pattern_info['right_shoulder'] + (atr * 0.5)

                # Execute short order
                self._execute_sell_short_order(symbol, current_price, stop_price, target_price, atr, market_data)

        elif pattern_type == 'bullish' and current_price > neckline:
            # Volume confirmation
            avg_volume = data['volume'].rolling(window=20).mean().iloc[-1] if len(data) >= 20 else current_volume
            volume_confirmation = current_volume > (avg_volume * self.volume_multiplier)

            if volume_confirmation and position == 0:
                # Calculate target and stop
                target_price = neckline + head_height
                stop_price = pattern_info['right_shoulder'] - (atr * 0.5)

                # Execute buy order
                self._execute_buy_order(symbol, current_price, stop_price, target_price, atr, market_data)

    def _detect_head_and_shoulders(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Detect head and shoulders pattern in recent data.

        Returns dict with pattern info or None if no pattern found.
        """
        if len(data) < 20:
            return None

        # Look for peaks (local maxima) in recent data
        highs = data['high'].values
        peaks = []

        # Find local peaks
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                peaks.append((i, highs[i]))

        if len(peaks) < 3:
            return None

        # Look for three peaks with middle one highest
        for i in range(len(peaks) - 2):
            left_shoulder_idx, left_shoulder = peaks[i]
            head_idx, head = peaks[i+1]
            right_shoulder_idx, right_shoulder = peaks[i+2]

            # Check pattern structure
            if not (head > left_shoulder and head > right_shoulder):
                continue

            # Check shoulders are roughly equal height
            shoulder_diff_pct = abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder) * 100
            if shoulder_diff_pct > 10:  # Shoulders within 10% of each other
                continue

            # Find neckline (troughs between peaks)
            left_trough = data['low'].iloc[left_shoulder_idx:head_idx].min()
            right_trough = data['low'].iloc[head_idx:right_shoulder_idx].min()
            neckline = (left_trough + right_trough) / 2

            # Check neckline is below head
            if neckline >= head:
                continue

            # Check pattern height
            pattern_height = head - neckline
            pattern_height_pct = pattern_height / neckline * 100
            if pattern_height_pct < self.min_pattern_height:
                continue

            # Determine pattern type based on trend
            recent_trend = data['close'].iloc[-10:].pct_change().sum()
            if recent_trend > 0.02:  # Uptrend -> bearish H&S
                pattern_type = 'bearish'
            elif recent_trend < -0.02:  # Downtrend -> bullish inverse H&S
                pattern_type = 'bullish'
            else:
                continue

            return {
                'type': pattern_type,
                'neckline': neckline,
                'head_height': pattern_height,
                'left_shoulder': left_shoulder,
                'head': head,
                'right_shoulder': right_shoulder,
                'left_shoulder_idx': left_shoulder_idx,
                'head_idx': head_idx,
                'right_shoulder_idx': right_shoulder_idx
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
                print(f"H&S BUY order submitted: {symbol} qty={position_size:.2f} price={price:.2f}")
                notes = f"Bullish H&S breakout: Entry {price:.2f}, Stop {stop_price:.2f}, Target {target_price:.2f}"
                self.log_signal(symbol, "buy", 0.9, market_data, notes)
            else:
                print(f"Failed to submit BUY order for {symbol}")

    def _execute_sell_short_order(self, symbol: str, price: float, stop_price: float,
                                target_price: float, atr: float, market_data: Dict[str, Any]) -> None:
        """Execute sell short order."""
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
            # Create sell short order
            order = self.generate_order(
                symbol=symbol,
                side='sell',
                quantity=position_size,
                price=price,
                order_type='market'
            )

            # Submit order
            if self.submit_order(order):
                print(f"H&S SHORT order submitted: {symbol} qty={position_size:.2f} price={price:.2f}")
                notes = f"Bearish H&S breakout: Short {price:.2f}, Stop {stop_price:.2f}, Target {target_price:.2f}"
                self.log_signal(symbol, "sell_short", 0.9, market_data, notes)
            else:
                print(f"Failed to submit SHORT order for {symbol}")

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

        if len(data) < self.lookback:
            return signals

        try:
            # Calculate ATR
            atr_series = atr(data, window=self.atr_period)

            for i in range(self.lookback, len(data)):
                window_data = data.iloc[:i+1]

                # Detect pattern
                pattern_info = self._detect_head_and_shoulders(window_data)

                if pattern_info:
                    current_price = data.iloc[i]['close']
                    current_volume = data.iloc[i]['volume']

                    # Check breakout conditions
                    neckline = pattern_info['neckline']
                    avg_volume = window_data['volume'].rolling(window=20).mean().iloc[-1]
                    volume_confirmation = current_volume > (avg_volume * self.volume_multiplier)
                    current_atr = atr_series.iloc[i] if i < len(atr_series) else atr_series.iloc[-1]

                    if pattern_info['type'] == 'bearish' and current_price < neckline and volume_confirmation:
                        signals.append({
                            'timestamp': data.index[i],
                            'symbol': symbol,
                            'signal_type': 'sell_short',
                            'strength': 0.9,
                            'price': current_price,
                            'pattern': 'head_and_shoulders_bearish',
                            'neckline': neckline,
                            'stop_price': pattern_info['right_shoulder'] + (current_atr * 0.5),
                            'target_price': neckline - pattern_info['head_height']
                        })

                    elif pattern_info['type'] == 'bullish' and current_price > neckline and volume_confirmation:
                        signals.append({
                            'timestamp': data.index[i],
                            'symbol': symbol,
                            'signal_type': 'buy',
                            'strength': 0.9,
                            'price': current_price,
                            'pattern': 'head_and_shoulders_bullish',
                            'neckline': neckline,
                            'stop_price': pattern_info['right_shoulder'] - (current_atr * 0.5),
                            'target_price': neckline + pattern_info['head_height']
                        })

        except Exception as e:
            print(f"Error calculating signals for {symbol}: {e}")

        return signals

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': self.name,
            'type': 'Head and Shoulders',
            'description': 'Detects head and shoulders reversal patterns with neckline breakout confirmation',
            'parameters': self.parameters,
            'indicators': ['ATR'],
            'timeframe': self.timeframe,
            'enabled': self.enabled
        }
