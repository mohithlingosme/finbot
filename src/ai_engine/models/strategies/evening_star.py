"""
Evening Star candlestick strategy implementation.

Detects evening star patterns: three-candle bearish reversal — bullish, small (doji/star), bearish.
Entry short after third candle confirmation; stop above pattern high.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.ai_engine.models.strategies.base import BaseStrategy
from src.ai_engine.scripts.indicators.indicators import atr


class EveningStarStrategy(BaseStrategy):
    """
    Evening Star candlestick strategy.

    Detects evening star patterns (three-candle bearish reversal).
    First candle bullish, second small body (star), third bearish engulfing.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize evening star strategy.

        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)

        # Strategy parameters
        self.star_body_ratio = config.get('star_body_ratio', 0.3)  # Max body size for star candle
        self.volume_multiplier = config.get('volume_multiplier', 1.1)  # Volume confirmation
        self.atr_period = config.get('atr_period', 14)
        self.atr_multiplier = config.get('atr_multiplier', 2.0)
        self.lookback = config.get('lookback', 10)  # Bars to check for trend

        # Risk management
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% of portfolio
        self.risk_per_trade = config.get('risk_per_trade', 0.02)  # 2% risk per trade

        # Data storage
        self.price_data: Dict[str, pd.DataFrame] = {}

    def _setup_parameters(self) -> Dict[str, Any]:
        """Setup strategy parameters."""
        return {
            'star_body_ratio': self.star_body_ratio,
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

            # Check for evening star patterns
            self._check_evening_star_patterns(symbol, current_data, current_atr, current_position, data)

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

    def _check_evening_star_patterns(self, symbol: str, data: pd.DataFrame,
                                   atr: float, position: float, market_data: Dict[str, Any]) -> None:
        """
        Check for evening star patterns and generate signals.

        Args:
            symbol: Trading symbol
            data: Price data
            atr: Current ATR value
            position: Current position size
            market_data: Current market data
        """
        # Need at least 3 candles
        if len(data) < 3:
            return

        # Get last three candles
        third_candle = data.iloc[-1]
        second_candle = data.iloc[-2]
        first_candle = data.iloc[-3]

        # Detect evening star pattern
        star_info = self._detect_evening_star(first_candle, second_candle, third_candle, market_data)

        if not star_info:
            return

        # Check trend context (should be in uptrend)
        trend_context = self._get_trend_context(data, self.lookback)

        # Evening star works best in uptrends
        if trend_context == 'uptrend' and position >= 0:  # Allow if flat or long
            # Entry on third candle close
            self._execute_sell_short_order(symbol, third_candle['close'], atr, star_info['strength'], market_data)

    def _detect_evening_star(self, first: pd.Series, second: pd.Series, third: pd.Series,
                           market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Detect evening star pattern in three candles.

        Evening Star: 1st bullish, 2nd small body (star), 3rd bearish engulfing.
        """
        # 1. First candle: bullish (close > open)
        if first['close'] <= first['open']:
            return None

        # 2. Second candle: small body (star) - can be doji-like
        second_body = abs(second['close'] - second['open'])
        second_range = second['high'] - second['low']

        if second_range == 0:
            return None

        second_body_ratio = second_body / second_range
        if second_body_ratio > self.star_body_ratio:
            return None

        # Star should gap up from first candle (ideal but not strict requirement)
        # star_gap = first['high'] < second['low']

        # 3. Third candle: bearish (close < open) and should engulf or close below star
        if third['close'] >= third['open']:
            return None

        # Third candle should close below midpoint of first candle
        first_midpoint = (first['open'] + first['close']) / 2
        if third['close'] >= first_midpoint:
            return None

        # Volume confirmation (higher on third candle)
        avg_volume = self.price_data[list(self.price_data.keys())[0]]['volume'].rolling(window=10).mean().iloc[-1]
        volume_confirmation = market_data['volume'] > (avg_volume * self.volume_multiplier)

        if not volume_confirmation:
            return None

        return {
            'type': 'evening_star',
            'strength': 0.9,
            'first_body_size': abs(first['close'] - first['open']),
            'star_body_ratio': second_body_ratio,
            'third_body_size': abs(third['close'] - third['open'])
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

        if change_pct > 3.0:
            return 'uptrend'
        elif change_pct < -3.0:
            return 'downtrend'
        else:
            return 'sideways'

    def _execute_sell_short_order(self, symbol: str, price: float, atr: float,
                                strength: float, market_data: Dict[str, Any]) -> None:
        """Execute sell short order."""
        # Calculate position size based on risk
        portfolio_value = self.get_portfolio_value()
        risk_amount = portfolio_value * self.risk_per_trade

        # Stop loss above the highest high of the pattern
        pattern_high = max(market_data['high'], self.price_data[symbol]['high'].tail(2).max())
        stop_price = pattern_high + (atr * 0.5)
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
                print(f"EVENING STAR SHORT order submitted: {symbol} qty={position_size:.2f} price={price:.2f}")
                notes = f"Evening star pattern: Short {price:.2f}, Stop {stop_price:.2f}"
                self.log_signal(symbol, "sell_short", strength, market_data, notes)
            else:
                print(f"Failed to submit SHORT order for {symbol}")

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

            for i in range(self.lookback + 2, len(data)):
                first = data.iloc[i-2]
                second = data.iloc[i-1]
                third = data.iloc[i]

                market_data_at_i = {
                    'volume': third['volume'],
                    'high': third['high'],
                    'low': third['low']
                }

                # Detect evening star
                star_info = self._detect_evening_star(first, second, third, market_data_at_i)

                if star_info:
                    # Check trend context
                    window_data = data.iloc[max(0, i-self.lookback):i+1]
                    trend_context = self._get_trend_context(window_data, self.lookback)

                    current_atr = atr_series.iloc[i] if i < len(atr_series) else atr_series.iloc[-1]

                    if trend_context == 'uptrend':
                        pattern_high = max(third['high'], second['high'], first['high'])
                        stop_price = pattern_high + (current_atr * 0.5)
                        signals.append({
                            'timestamp': data.index[i],
                            'symbol': symbol,
                            'signal_type': 'sell_short',
                            'strength': star_info['strength'],
                            'price': third['close'],
                            'pattern': 'evening_star',
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
            'type': 'Evening Star Candlestick',
            'description': 'Detects evening star three-candle bearish reversal patterns',
            'parameters': self.parameters,
            'indicators': ['ATR'],
            'timeframe': self.timeframe,
            'enabled': self.enabled
        }
