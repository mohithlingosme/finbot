"""
Bearish flag strategy implementation.

Detects bearish flag patterns: sharp drop (flagpole) followed by a small upward-sloping channel (flag).
Short entry on breakout below flag; stop above flag high; target ≈ flagpole height.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.ai_engine.models.strategies.base import BaseStrategy
from src.ai_engine.scripts.indicators.indicators import atr


class BearishFlagStrategy(BaseStrategy):
    """
    Bearish flag strategy.

    Identifies flagpole (sharp drop) followed by consolidation channel.
    Enters short on breakout below the channel with volume confirmation.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize bearish flag strategy.

        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)

        # Strategy parameters
        self.flagpole_min_pct = config.get('flagpole_min_pct', 5.0)  # Minimum flagpole % drop
        self.flagpole_max_pct = config.get('flagpole_max_pct', 20.0)  # Maximum flagpole % drop
        self.channel_lookback = config.get('channel_lookback', 10)  # Bars to look for channel
        self.channel_slope_threshold = config.get('channel_slope_threshold', 0.1)  # Max upward slope
        self.breakout_volume_multiplier = config.get('breakout_volume_multiplier', 1.2)  # Volume confirmation
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
            'flagpole_min_pct': self.flagpole_min_pct,
            'flagpole_max_pct': self.flagpole_max_pct,
            'channel_lookback': self.channel_lookback,
            'channel_slope_threshold': self.channel_slope_threshold,
            'breakout_volume_multiplier': self.breakout_volume_multiplier,
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
        if symbol not in self.price_data or len(self.price_data[symbol]) < self.channel_lookback + 20:
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

            # Check for flag pattern and breakout
            self._check_flag_pattern_and_breakout(symbol, current_data, current_atr, current_position, data)

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

    def _check_flag_pattern_and_breakout(self, symbol: str, data: pd.DataFrame,
                                       atr: float, position: float, market_data: Dict[str, Any]) -> None:
        """
        Check for bearish flag pattern and breakout signals.

        Args:
            symbol: Trading symbol
            data: Price data
            atr: Current ATR value
            position: Current position size
            market_data: Current market data
        """
        current_price = market_data['close']
        current_volume = market_data['volume']

        # Detect flag pattern
        flag_info = self._detect_bearish_flag(data)

        if not flag_info:
            return

        flag_high = flag_info['flag_high']
        flag_low = flag_info['flag_low']
        flagpole_height = flag_info['flagpole_height']

        # Check for breakout below flag
        if current_price < flag_low:
            # Volume confirmation
            avg_volume = data['volume'].rolling(window=20).mean().iloc[-1] if len(data) >= 20 else current_volume
            volume_confirmation = current_volume > (avg_volume * self.breakout_volume_multiplier)

            if volume_confirmation and position >= 0:  # Allow if flat or long (close long first if needed)
                # Calculate target and stop
                target_price = current_price - flagpole_height
                stop_price = flag_high

                # Execute short order
                self._execute_sell_short_order(symbol, current_price, stop_price, target_price, atr, market_data)

    def _detect_bearish_flag(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Detect bearish flag pattern in recent data.

        Returns dict with flag info or None if no pattern found.
        """
        if len(data) < self.channel_lookback + 10:
            return None

        # Look for flagpole: recent sharp drop
        recent_lows = data['low'].tail(20)
        flagpole_start_idx = recent_lows.idxmax()  # Start of drop
        flagpole_end_idx = len(data) - 1

        if flagpole_end_idx - flagpole_start_idx < 5:
            return None

        flagpole_start_price = data.loc[flagpole_start_idx, 'high']
        flagpole_end_price = data.loc[flagpole_end_idx, 'low']
        flagpole_pct = (flagpole_start_price - flagpole_end_price) / flagpole_start_price * 100

        # Check flagpole size
        if not (self.flagpole_min_pct <= flagpole_pct <= self.flagpole_max_pct):
            return None

        # Look for consolidation channel after flagpole
        channel_data = data.tail(self.channel_lookback)

        # Calculate channel slope (using highs and lows)
        highs = channel_data['high']
        lows = channel_data['low']

        # Simple linear regression for slope
        x = np.arange(len(channel_data))
        high_slope = np.polyfit(x, highs, 1)[0]
        low_slope = np.polyfit(x, lows, 1)[0]

        # Channel should be slightly upward sloping
        avg_slope = (high_slope + low_slope) / 2
        if avg_slope < self.channel_slope_threshold:
            return None

        # Channel boundaries
        channel_high = highs.max()
        channel_low = lows.min()

        # Flagpole height
        flagpole_height = flagpole_start_price - flagpole_end_price

        return {
            'flag_high': channel_high,
            'flag_low': channel_low,
            'flagpole_height': flagpole_height,
            'flagpole_start': flagpole_start_price,
            'flagpole_end': flagpole_end_price
        }

    def _execute_sell_short_order(self, symbol: str, price: float, stop_price: float,
                                target_price: float, atr: float, market_data: Dict[str, Any]) -> None:
        """Execute sell short order."""
        # Calculate position size based on risk
        portfolio_value = self.get_portfolio_value()
        risk_amount = portfolio_value * self.risk_per_trade

        # Risk per share (stop above entry)
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
                print(f"BEARISH FLAG SHORT order submitted: {symbol} qty={position_size:.2f} price={price:.2f}")
                notes = f"Bearish flag breakout: Short {price:.2f}, Stop {stop_price:.2f}, Target {target_price:.2f}"
                self.log_signal(symbol, "sell_short", 0.8, market_data, notes)
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

        if len(data) < self.channel_lookback + 20:
            return signals

        try:
            # Calculate ATR
            atr_series = atr(data, window=self.atr_period)

            for i in range(self.channel_lookback + 10, len(data)):
                window_data = data.iloc[:i+1]

                # Detect flag pattern
                flag_info = self._detect_bearish_flag(window_data)

                if flag_info:
                    current_price = data.iloc[i]['close']
                    current_volume = data.iloc[i]['volume']

                    # Check breakout
                    if current_price < flag_info['flag_low']:
                        # Volume confirmation
                        avg_volume = window_data['volume'].rolling(window=20).mean().iloc[-1]
                        volume_confirmation = current_volume > (avg_volume * self.breakout_volume_multiplier)

                        if volume_confirmation:
                            signals.append({
                                'timestamp': data.index[i],
                                'symbol': symbol,
                                'signal_type': 'sell_short',
                                'strength': 0.8,
                                'price': current_price,
                                'flag_high': flag_info['flag_high'],
                                'flag_low': flag_info['flag_low'],
                                'flagpole_height': flag_info['flagpole_height'],
                                'stop_price': flag_info['flag_high'],
                                'target_price': current_price - flag_info['flagpole_height']
                            })

        except Exception as e:
            print(f"Error calculating signals for {symbol}: {e}")

        return signals

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': self.name,
            'type': 'Bearish Flag',
            'description': 'Detects bearish flag patterns and enters short on breakouts with volume confirmation',
            'parameters': self.parameters,
            'indicators': ['ATR'],
            'timeframe': self.timeframe,
            'enabled': self.enabled
        }
