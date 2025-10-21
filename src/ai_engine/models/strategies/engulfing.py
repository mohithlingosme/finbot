"""
Engulfing candlestick strategy implementation.

Detects bullish/bearish engulfing patterns: second candle fully engulfs previous candle's body.
Bullish engulfing: up bar engulfs down bar; Bearish engulfing: down bar engulfs up bar.
Entry on confirmation close; stop beyond engulfing candle; target ATR multiples.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.ai_engine.models.strategies.base import BaseStrategy
from src.ai_engine.scripts.indicators.indicators import atr


class EngulfingStrategy(BaseStrategy):
    """
    Engulfing candlestick strategy.

    Detects bullish and bearish engulfing patterns.
    Enters on confirmation with volume and trend filters.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize engulfing strategy.

        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)

        # Strategy parameters
        self.min_body_ratio = config.get('min_body_ratio', 0.3)  # Minimum body size relative to total range
        self.engulf_ratio = config.get('engulf_ratio', 1.0)  # Engulfing ratio (1.0 = full engulf)
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
            'min_body_ratio': self.min_body_ratio,
            'engulf_ratio': self.engulf_ratio,
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

            # Check for engulfing patterns
            self._check_engulfing_patterns(symbol, current_data, current_atr, current_position, data)

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

    def _check_engulfing_patterns(self, symbol: str, data: pd.DataFrame,
                                atr: float, position: float, market_data: Dict[str, Any]) -> None:
        """
        Check for engulfing patterns and generate signals.

        Args:
            symbol: Trading symbol
            data: Price data
            atr: Current ATR value
            position: Current position size
            market_data: Current market data
        """
        # Get last two candles
        if len(data) < 2:
            return

        prev_candle = data.iloc[-2]
        curr_candle = data.iloc[-1]

        # Detect engulfing pattern
        engulfing_info = self._detect_engulfing(prev_candle, curr_candle, market_data)

        if not engulfing_info:
            return

        pattern_type = engulfing_info['type']
        strength = engulfing_info['strength']

        # Check trend context
        trend_context = self._get_trend_context(data, self.lookback)

        # Bullish engulfing
        if pattern_type == 'bullish' and position == 0:
            # Should occur in downtrend or at support
            if trend_context in ['downtrend', 'sideways']:
                self._execute_buy_order(symbol, curr_candle['close'], atr, strength, market_data)

        # Bearish engulfing
        elif pattern_type == 'bearish' and position >= 0:  # Allow if flat or long
            # Should occur in uptrend or at resistance
            if trend_context in ['uptrend', 'sideways']:
                self._execute_sell_short_order(symbol, curr_candle['close'], atr, strength, market_data)

    def _detect_engulfing(self, prev_candle: pd.Series, curr_candle: pd.Series,
                         market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Detect engulfing pattern between two candles.

        Returns dict with pattern info or None.
        """
        # Calculate body sizes
        prev_body_high = max(prev_candle['open'], prev_candle['close'])
        prev_body_low = min(prev_candle['open'], prev_candle['close'])
        prev_body_size = prev_body_high - prev_body_low

        curr_body_high = max(curr_candle['open'], curr_candle['close'])
        curr_body_low = min(curr_candle['open'], curr_candle['close'])
        curr_body_size = curr_body_high - curr_body_low

        # Minimum body size check
        prev_range = prev_candle['high'] - prev_candle['low']
        curr_range = curr_candle['high'] - curr_candle['low']

        if prev_body_size / prev_range < self.min_body_ratio or curr_body_size / curr_range < self.min_body_ratio:
            return None

        # Bullish engulfing: current up candle engulfs previous down candle
        if (prev_candle['close'] < prev_candle['open'] and  # Previous is bearish
            curr_candle['close'] > curr_candle['open'] and  # Current is bullish
            curr_body_low <= prev_body_low and  # Engulfs lower
            curr_body_high >= prev_body_high):  # Engulfs higher

            # Volume confirmation
            avg_volume = self.price_data[list(self.price_data.keys())[0]]['volume'].rolling(window=10).mean().iloc[-1]
            volume_confirmation = market_data['volume'] > (avg_volume * self.volume_multiplier)

            if volume_confirmation:
                return {
                    'type': 'bullish',
                    'strength': 0.8,
                    'prev_body_size': prev_body_size,
                    'curr_body_size': curr_body_size
                }

        # Bearish engulfing: current down candle engulfs previous up candle
        elif (prev_candle['close'] > prev_candle['open'] and  # Previous is bullish
              curr_candle['close'] < curr_candle['open'] and  # Current is bearish
              curr_body_low <= prev_body_low and  # Engulfs lower
              curr_body_high >= prev_body_high):  # Engulfs higher

            # Volume confirmation
            avg_volume = self.price_data[list(self.price_data.keys())[0]]['volume'].rolling(window=10).mean().iloc[-1]
            volume_confirmation = market_data['volume'] > (avg_volume * self.volume_multiplier)

            if volume_confirmation:
                return {
                    'type': 'bearish',
                    'strength': 0.8,
                    'prev_body_size': prev_body_size,
                    'curr_body_size': curr_body_size
                }

        return None

    def _get_trend_context(self, data: pd.DataFrame, lookback: int) -> str:
        """
        Determine trend context from recent bars.

        Returns: 'uptrend', 'downtrend', or 'sideways'
        """
        recent_data = data.tail(lookback)

        # Simple trend detection: compare closes
        start_close = recent_data['close'].iloc[0]
        end_close = recent_data['close'].iloc[-1]
        change_pct = (end_close - start_close) / start_close * 100

        if change_pct > 2.0:
            return 'uptrend'
        elif change_pct < -2.0:
            return 'downtrend'
        else:
            return 'sideways'

    def _execute_buy_order(self, symbol: str, price: float, atr: float,
                          strength: float, market_data: Dict[str, Any]) -> None:
        """Execute buy order."""
        # Calculate position size based on risk
        portfolio_value = self.get_portfolio_value()
        risk_amount = portfolio_value * self.risk_per_trade

        # Stop loss below the engulfing candle low
        stop_price = market_data['low'] - (atr * 0.5)  # Tighter stop
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
                print(f"ENGULFING BUY order submitted: {symbol} qty={position_size:.2f} price={price:.2f}")
                notes = f"Bullish engulfing: Entry {price:.2f}, Stop {stop_price:.2f}"
                self.log_signal(symbol, "buy", strength, market_data, notes)
            else:
                print(f"Failed to submit BUY order for {symbol}")

    def _execute_sell_short_order(self, symbol: str, price: float, atr: float,
                                strength: float, market_data: Dict[str, Any]) -> None:
        """Execute sell short order."""
        # Calculate position size based on risk
        portfolio_value = self.get_portfolio_value()
        risk_amount = portfolio_value * self.risk_per_trade

        # Stop loss above the engulfing candle high
        stop_price = market_data['high'] + (atr * 0.5)  # Tighter stop
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
                print(f"ENGULFING SHORT order submitted: {symbol} qty={position_size:.2f} price={price:.2f}")
                notes = f"Bearish engulfing: Short {price:.2f}, Stop {stop_price:.2f}"
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

            for i in range(1, len(data)):
                prev_candle = data.iloc[i-1]
                curr_candle = data.iloc[i]

                # Detect engulfing
                engulfing_info = self._detect_engulfing(prev_candle, curr_candle, {
                    'volume': curr_candle['volume'],
                    'high': curr_candle['high'],
                    'low': curr_candle['low']
                })

                if engulfing_info:
                    # Check trend context
                    window_data = data.iloc[max(0, i-self.lookback):i+1]
                    trend_context = self._get_trend_context(window_data, self.lookback)

                    current_atr = atr_series.iloc[i] if i < len(atr_series) else atr_series.iloc[-1]

                    if engulfing_info['type'] == 'bullish' and trend_context in ['downtrend', 'sideways']:
                        signals.append({
                            'timestamp': data.index[i],
                            'symbol': symbol,
                            'signal_type': 'buy',
                            'strength': engulfing_info['strength'],
                            'price': curr_candle['close'],
                            'pattern': 'bullish_engulfing',
                            'stop_price': curr_candle['low'] - (current_atr * 0.5),
                            'trend_context': trend_context
                        })

                    elif engulfing_info['type'] == 'bearish' and trend_context in ['uptrend', 'sideways']:
                        signals.append({
                            'timestamp': data.index[i],
                            'symbol': symbol,
                            'signal_type': 'sell_short',
                            'strength': engulfing_info['strength'],
                            'price': curr_candle['close'],
                            'pattern': 'bearish_engulfing',
                            'stop_price': curr_candle['high'] + (current_atr * 0.5),
                            'trend_context': trend_context
                        })

        except Exception as e:
            print(f"Error calculating signals for {symbol}: {e}")

        return signals

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': self.name,
            'type': 'Engulfing Candlestick',
            'description': 'Detects bullish/bearish engulfing patterns with trend and volume confirmation',
            'parameters': self.parameters,
            'indicators': ['ATR'],
            'timeframe': self.timeframe,
            'enabled': self.enabled
        }
