"""
Inverted Hammer candlestick strategy implementation.

Detects inverted hammer patterns: small body at bottom with long upper shadow — bullish reversal.
Entry buy after confirmation close above pattern; stop below low.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.ai_engine.models.strategies.base import BaseStrategy
from src.ai_engine.scripts.indicators.indicators import atr


class InvertedHammerStrategy(BaseStrategy):
    """
    Inverted Hammer candlestick strategy.

    Detects inverted hammer patterns: shooting star-like in downtrend, signals potential reversal.
    Small body at bottom with long upper shadow, bullish when in downtrend.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize inverted hammer strategy.

        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)

        # Strategy parameters
        self.body_ratio = config.get('body_ratio', 0.3)  # Max body % of total range
        self.shadow_ratio = config.get('shadow_ratio', 2.0)  # Min upper shadow ratio to body
        self.volume_multiplier = config.get('volume_multiplier', 1.1)  # Volume confirmation
        self.atr_period = config.get('atr_period', 14)
        self.atr_multiplier = config.get('atr_multiplier', 2.0)
        self.lookback = config.get('lookback', 5)  # Bars to check for trend

        # Risk management
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% of portfolio
        self.risk_per_trade = config.get('risk_per_trade', 0.02)  # 2% risk per trade

        # Data storage
        self.price_data: Dict[str, pd.DataFrame] = {}

    def _setup_parameters(self) -> Dict[str, Any]:
        """Setup strategy parameters."""
        return {
            'body_ratio': self.body_ratio,
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

            # Check for inverted hammer patterns
            self._check_inverted_hammer_patterns(symbol, current_data, current_atr, current_position, data)

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

    def _check_inverted_hammer_patterns(self, symbol: str, data: pd.DataFrame,
                                      atr: float, position: float, market_data: Dict[str, Any]) -> None:
        """
        Check for inverted hammer patterns and generate signals.

        Args:
            symbol: Trading symbol
            data: Price data
            atr: Current ATR value
            position: Current position size
            market_data: Current market data
        """
        # Get current candle
        curr_candle = data.iloc[-1]

        # Detect inverted hammer pattern
        inverted_hammer_info = self._detect_inverted_hammer(curr_candle, market_data)

        if not inverted_hammer_info:
            return

        # Check trend context (must be in downtrend for inverted hammer)
        trend_context = self._get_trend_context(data, self.lookback)

        # Inverted hammer only works in downtrends
        if trend_context == 'downtrend' and position == 0:
            # Wait for bullish confirmation (next candle closes above inverted hammer high)
            if self._check_confirmation(data, 'bullish'):
                self._execute_buy_order(symbol, curr_candle['close'], atr, inverted_hammer_info['strength'], market_data)

    def _detect_inverted_hammer(self, candle: pd.Series, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Detect inverted hammer pattern in candle.

        Inverted Hammer: same structure as shooting star but appears in downtrend.
        Small body at bottom, long upper shadow, little/no lower shadow.
        """
        open_price = candle['open']
        close_price = candle['close']
        high_price = candle['high']
        low_price = candle['low']

        # Calculate body and shadows
        body_high = max(open_price, close_price)
        body_low = min(open_price, close_price)
        body_size = body_high - body_low

        upper_shadow = high_price - body_high
        lower_shadow = body_low - low_price

        total_range = high_price - low_price

        if total_range == 0:
            return None

        # Inverted hammer criteria (same as shooting star):
        # 1. Small body (relative to total range)
        body_ratio = body_size / total_range
        if body_ratio > self.body_ratio:
            return None

        # 2. Long upper shadow (at least 2x body size)
        if upper_shadow < body_size * self.shadow_ratio:
            return None

        # 3. Little to no lower shadow (pattern at bottom)
        if lower_shadow > body_size * 0.5:  # Allow small lower shadow
            return None

        # 4. Body near bottom of range
        body_position = (body_high - low_price) / total_range
        if body_position > 0.3:  # Body should be in lower 30% of range
            return None

        # Volume confirmation
        avg_volume = self.price_data[list(self.price_data.keys())[0]]['volume'].rolling(window=10).mean().iloc[-1]
        volume_confirmation = market_data['volume'] > (avg_volume * self.volume_multiplier)

        if not volume_confirmation:
            return None

        return {
            'type': 'inverted_hammer',
            'strength': 0.8,
            'body_ratio': body_ratio,
            'upper_shadow_ratio': upper_shadow / body_size,
            'lower_shadow_ratio': lower_shadow / body_size
        }

    def _get_trend_context(self, data: pd.DataFrame, lookback: int) -> str:
        """
        Determine trend context from recent bars.
        """
        recent_data = data.tail(lookback)

        # Simple trend detection
        start_close = recent_data['close'].iloc[0]
        end_close = recent_data['close'].iloc[-1]
        change_pct = (end_close - start_close) / start_close * 100

        if change_pct > 2.0:
            return 'uptrend'
        elif change_pct < -2.0:
            return 'downtrend'
        else:
            return 'sideways'

    def _check_confirmation(self, data: pd.DataFrame, direction: str) -> bool:
        """
        Check for confirmation candle after inverted hammer.
        """
        if len(data) < 2:
            return False

        inverted_hammer_candle = data.iloc[-2]  # Inverted hammer candle
        conf_candle = data.iloc[-1]  # Confirmation candle

        if direction == 'bullish':
            # Bullish confirmation: close above inverted hammer high
            return conf_candle['close'] > inverted_hammer_candle['high']

        return False

    def _execute_buy_order(self, symbol: str, price: float, atr: float,
                          strength: float, market_data: Dict[str, Any]) -> None:
        """Execute buy order."""
        # Calculate position size based on risk
        portfolio_value = self.get_portfolio_value()
        risk_amount = portfolio_value * self.risk_per_trade

        # Stop loss below inverted hammer low
        stop_price = market_data['low'] - (atr * 0.5)
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
                print(f"INVERTED HAMMER BUY order submitted: {symbol} qty={position_size:.2f} price={price:.2f}")
                notes = f"Inverted hammer pattern: Entry {price:.2f}, Stop {stop_price:.2f}"
                self.log_signal(symbol, "buy", strength, market_data, notes)
            else:
                print(f"Failed to submit BUY order for {symbol}")

    def calculate_signals(self, data: pd.DataFrame, symbol: str) -> List[Dict[str, Any]]:
        """
        Calculate trading signals from historical data.
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

                # Detect inverted hammer
                inverted_hammer_info = self._detect_inverted_hammer(candle, market_data_at_i)

                if inverted_hammer_info:
                    # Check trend context
                    window_data = data.iloc[max(0, i-self.lookback):i+1]
                    trend_context = self._get_trend_context(window_data, self.lookback)

                    # Check confirmation
                    confirmation = False
                    if i + 1 < len(data):
                        next_candle = data.iloc[i+1]
                        confirmation = next_candle['close'] > candle['high']

                    current_atr = atr_series.iloc[i] if i < len(atr_series) else atr_series.iloc[-1]

                    if confirmation and trend_context == 'downtrend':
                        stop_price = candle['low'] - (current_atr * 0.5)
                        signals.append({
                            'timestamp': data.index[i],
                            'symbol': symbol,
                            'signal_type': 'buy',
                            'strength': inverted_hammer_info['strength'],
                            'price': candle['close'],
                            'pattern': 'inverted_hammer',
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
            'type': 'Inverted Hammer Candlestick',
            'description': 'Detects inverted hammer patterns (shooting star in downtrend) with trend and confirmation filters',
            'parameters': self.parameters,
            'indicators': ['ATR'],
            'timeframe': self.timeframe,
            'enabled': self.enabled
        }
