"""
Doji candlestick strategy implementation.

Detects doji patterns: open ≈ close — indecision.
Entry context-dependent — in trend can signal pause/reversal if supported by volume; stop beyond next swing.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.ai_engine.models.strategies.base import BaseStrategy
from src.ai_engine.scripts.indicators.indicators import atr, sma


class DojiStrategy(BaseStrategy):
    """
    Doji candlestick strategy.

    Detects various doji patterns (regular, dragonfly, gravestone).
    Enters on confirmation with trend and volume filters.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize doji strategy.

        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)

        # Strategy parameters
        self.doji_threshold = config.get('doji_threshold', 0.05)  # Max body % of range for doji
        self.shadow_ratio = config.get('shadow_ratio', 2.0)  # Min shadow ratio for dragonfly/gravestone
        self.volume_multiplier = config.get('volume_multiplier', 1.1)  # Volume confirmation
        self.atr_period = config.get('atr_period', 14)
        self.atr_multiplier = config.get('atr_multiplier', 2.0)
        self.lookback = config.get('lookback', 10)  # Bars to check for trend/confirmation

        # Risk management
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% of portfolio
        self.risk_per_trade = config.get('risk_per_trade', 0.02)  # 2% risk per trade

        # Data storage
        self.price_data: Dict[str, pd.DataFrame] = {}

    def _setup_parameters(self) -> Dict[str, Any]:
        """Setup strategy parameters."""
        return {
            'doji_threshold': self.doji_threshold,
            'shadow_ratio': self.shadow_ratio,
            'volume_multiplier': self.volume_multiplier,
            'atr_period': self.atr_period,
            'atr_multiplier': self.atr_multiplier,
            'lookback': self.lookback,
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
        if symbol not in self.price_data or len(self.price_data[symbol]) < self.lookback + 5:
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

            # Check for doji patterns
            self._check_doji_patterns(symbol, current_data, current_atr, current_position, data)

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

    def _check_doji_patterns(self, symbol: str, data: pd.DataFrame,
                           atr: float, position: float, market_data: Dict[str, Any]) -> None:
        """
        Check for doji patterns and generate signals.

        Args:
            symbol: Trading symbol
            data: Price data
            atr: Current ATR value
            position: Current position size
            market_data: Current market data
        """
        # Get current candle
        curr_candle = data.iloc[-1]

        # Detect doji pattern
        doji_info = self._detect_doji(curr_candle, market_data)

        if not doji_info:
            return

        doji_type = doji_info['type']
        strength = doji_info['strength']

        # Check trend context and confirmation
        trend_context = self._get_trend_context(data, self.lookback)

        # Regular doji - context dependent
        if doji_type == 'regular':
            if trend_context == 'uptrend' and position >= 0:  # Potential reversal in uptrend
                # Wait for bearish confirmation
                if self._check_confirmation(data, 'bearish'):
                    self._execute_sell_short_order(symbol, curr_candle['close'], atr, strength, market_data)

            elif trend_context == 'downtrend' and position == 0:  # Potential reversal in downtrend
                # Wait for bullish confirmation
                if self._check_confirmation(data, 'bullish'):
                    self._execute_buy_order(symbol, curr_candle['close'], atr, strength, market_data)

        # Dragonfly doji - bullish at support
        elif doji_type == 'dragonfly':
            if trend_context in ['downtrend', 'sideways'] and position == 0:
                if self._check_confirmation(data, 'bullish'):
                    self._execute_buy_order(symbol, curr_candle['close'], atr, strength, market_data)

        # Gravestone doji - bearish at resistance
        elif doji_type == 'gravestone':
            if trend_context in ['uptrend', 'sideways'] and position >= 0:
                if self._check_confirmation(data, 'bearish'):
                    self._execute_sell_short_order(symbol, curr_candle['close'], atr, strength, market_data)

    def _detect_doji(self, candle: pd.Series, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Detect doji pattern in candle.

        Returns dict with doji info or None.
        """
        open_price = candle['open']
        close_price = candle['close']
        high_price = candle['high']
        low_price = candle['low']

        # Calculate body and shadows
        body_size = abs(close_price - open_price)
        total_range = high_price - low_price

        if total_range == 0:
            return None

        body_ratio = body_size / total_range

        # Check if it's a doji (small body)
        if body_ratio > self.doji_threshold:
            return None

        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price

        # Volume confirmation
        avg_volume = self.price_data[list(self.price_data.keys())[0]]['volume'].rolling(window=10).mean().iloc[-1]
        volume_confirmation = market_data['volume'] > (avg_volume * self.volume_multiplier)

        if not volume_confirmation:
            return None

        # Classify doji type
        if lower_shadow > upper_shadow * self.shadow_ratio:
            doji_type = 'dragonfly'  # Long lower shadow
            strength = 0.7
        elif upper_shadow > lower_shadow * self.shadow_ratio:
            doji_type = 'gravestone'  # Long upper shadow
            strength = 0.7
        else:
            doji_type = 'regular'  # Balanced
            strength = 0.6

        return {
            'type': doji_type,
            'strength': strength,
            'body_ratio': body_ratio,
            'upper_shadow': upper_shadow,
            'lower_shadow': lower_shadow
        }

    def _get_trend_context(self, data: pd.DataFrame, lookback: int) -> str:
        """
        Determine trend context from recent bars.

        Returns: 'uptrend', 'downtrend', or 'sideways'
        """
        recent_data = data.tail(lookback)

        # Simple trend detection using SMA slope
        sma_period = min(10, len(recent_data))
        if len(recent_data) < sma_period:
            return 'sideways'

        sma_values = sma(recent_data['close'], window=sma_period)
        if len(sma_values) < 2:
            return 'sideways'

        sma_slope = sma_values.iloc[-1] - sma_values.iloc[-2]

        if sma_slope > 0.001:  # Positive slope
            return 'uptrend'
        elif sma_slope < -0.001:  # Negative slope
            return 'downtrend'
        else:
            return 'sideways'

    def _check_confirmation(self, data: pd.DataFrame, direction: str) -> bool:
        """
        Check for confirmation candle after doji.

        Args:
            data: Price data
            direction: 'bullish' or 'bearish'

        Returns: True if confirmed
        """
        if len(data) < 2:
            return False

        prev_candle = data.iloc[-2]  # Doji candle
        curr_candle = data.iloc[-1]  # Confirmation candle

        if direction == 'bullish':
            # Bullish confirmation: close above doji high
            return curr_candle['close'] > prev_candle['high']
        elif direction == 'bearish':
            # Bearish confirmation: close below doji low
            return curr_candle['close'] < prev_candle['low']

        return False

    def _execute_buy_order(self, symbol: str, price: float, atr: float,
                          strength: float, market_data: Dict[str, Any]) -> None:
        """Execute buy order."""
        # Calculate position size based on risk
        portfolio_value = self.get_portfolio_value()
        risk_amount = portfolio_value * self.risk_per_trade

        # Stop loss below recent swing low
        recent_low = self.price_data[symbol]['low'].tail(10).min()
        stop_price = recent_low - (atr * 0.5)
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
                print(f"DOJI BUY order submitted: {symbol} qty={position_size:.2f} price={price:.2f}")
                notes = f"Doji bullish signal: Entry {price:.2f}, Stop {stop_price:.2f}"
                self.log_signal(symbol, "buy", strength, market_data, notes)
            else:
                print(f"Failed to submit BUY order for {symbol}")

    def _execute_sell_short_order(self, symbol: str, price: float, atr: float,
                                strength: float, market_data: Dict[str, Any]) -> None:
        """Execute sell short order."""
        # Calculate position size based on risk
        portfolio_value = self.get_portfolio_value()
        risk_amount = portfolio_value * self.risk_per_trade

        # Stop loss above recent swing high
        recent_high = self.price_data[symbol]['high'].tail(10).max()
        stop_price = recent_high + (atr * 0.5)
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
                print(f"DOJI SHORT order submitted: {symbol} qty={position_size:.2f} price={price:.2f}")
                notes = f"Doji bearish signal: Short {price:.2f}, Stop {stop_price:.2f}"
                self.log_signal(symbol, "sell_short", strength, market_data, notes)
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

        if len(data) < self.lookback + 5:
            return signals

        try:
            # Calculate ATR
            atr_series = atr(data, window=self.atr_period)

            for i in range(self.lookback, len(data)):
                candle = data.iloc[i]
                market_data_at_i = {
                    'volume': candle['volume'],
                    'high': candle['high'],
                    'low': candle['low']
                }

                # Detect doji
                doji_info = self._detect_doji(candle, market_data_at_i)

                if doji_info:
                    # Check trend context
                    window_data = data.iloc[max(0, i-self.lookback):i+1]
                    trend_context = self._get_trend_context(window_data, self.lookback)

                    # Check confirmation (next candle if available)
                    confirmation = False
                    if i + 1 < len(data):
                        next_candle = data.iloc[i+1]
                        if doji_info['type'] == 'dragonfly' or (doji_info['type'] == 'regular' and trend_context == 'downtrend'):
                            confirmation = next_candle['close'] > candle['high']
                        elif doji_info['type'] == 'gravestone' or (doji_info['type'] == 'regular' and trend_context == 'uptrend'):
                            confirmation = next_candle['close'] < candle['low']

                    current_atr = atr_series.iloc[i] if i < len(atr_series) else atr_series.iloc[-1]

                    if confirmation:
                        if doji_info['type'] == 'dragonfly' or (doji_info['type'] == 'regular' and trend_context == 'downtrend'):
                            recent_low = window_data['low'].min()
                            stop_price = recent_low - (current_atr * 0.5)
                            signals.append({
                                'timestamp': data.index[i],
                                'symbol': symbol,
                                'signal_type': 'buy',
                                'strength': doji_info['strength'],
                                'price': candle['close'],
                                'pattern': f"{doji_info['type']}_doji",
                                'stop_price': stop_price,
                                'trend_context': trend_context
                            })

                        elif doji_info['type'] == 'gravestone' or (doji_info['type'] == 'regular' and trend_context == 'uptrend'):
                            recent_high = window_data['high'].max()
                            stop_price = recent_high + (current_atr * 0.5)
                            signals.append({
                                'timestamp': data.index[i],
                                'symbol': symbol,
                                'signal_type': 'sell_short',
                                'strength': doji_info['strength'],
                                'price': candle['close'],
                                'pattern': f"{doji_info['type']}_doji",
                                'stop_price': stop_price,
                                'trend_context': trend_context
                            })

        except Exception as e:
            print(f"Error calculating signals for {symbol}: {e}")

        return signals

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': self.name,
            'type': 'Doji Candlestick',
            'description': 'Detects doji patterns (regular, dragonfly, gravestone) with trend and confirmation filters',
            'parameters': self.parameters,
            'indicators': ['ATR', 'SMA'],
            'timeframe': self.timeframe,
            'enabled': self.enabled
        }
