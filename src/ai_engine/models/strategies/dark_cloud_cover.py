"""
Dark Cloud Cover candlestick strategy implementation.

Detects dark cloud cover patterns: bearish reversal where second candle opens above first high but closes below midpoint.
Entry short after confirmation; stop above pattern high.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.ai_engine.models.strategies.base import BaseStrategy
from src.ai_engine.scripts.indicators.indicators import atr


class DarkCloudCoverStrategy(BaseStrategy):
    """
    Dark Cloud Cover candlestick strategy.

    Detects dark cloud cover patterns: two-candle bearish reversal.
    First candle bullish, second bearish opening above first high but closing below first midpoint.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize dark cloud cover strategy.

        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)

        # Strategy parameters
        self.penetration_ratio = config.get('penetration_ratio', 0.5)  # Min penetration below first midpoint
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
            'penetration_ratio': self.penetration_ratio,
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

            # Check for dark cloud cover patterns
            self._check_dark_cloud_cover_patterns(symbol, current_data, current_atr, current_position, data)

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

    def _check_dark_cloud_cover_patterns(self, symbol: str, data: pd.DataFrame,
                                       atr: float, position: float, market_data: Dict[str, Any]) -> None:
        """
        Check for dark cloud cover patterns and generate signals.

        Args:
            symbol: Trading symbol
            data: Price data
            atr: Current ATR value
            position: Current position size
            market_data: Current market data
        """
        # Need at least 2 candles
        if len(data) < 2:
            return

        # Get last two candles
        second_candle = data.iloc[-1]
        first_candle = data.iloc[-2]

        # Detect dark cloud cover pattern
        dark_cloud_info = self._detect_dark_cloud_cover(first_candle, second_candle, market_data)

        if not dark_cloud_info:
            return

        # Check trend context (should be in uptrend)
        trend_context = self._get_trend_context(data, self.lookback)

        # Dark cloud cover works best in uptrends
        if trend_context == 'uptrend' and position >= 0:  # Allow if flat or long
            # Entry on second candle close
            self._execute_sell_short_order(symbol, second_candle['close'], atr, dark_cloud_info['strength'], market_data)

    def _detect_dark_cloud_cover(self, first: pd.Series, second: pd.Series,
                               market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Detect dark cloud cover pattern in two candles.

        Dark Cloud Cover: 1st bullish, 2nd bearish opening above 1st high but closing below 1st midpoint.
        """
        # 1. First candle: bullish (close > open)
        if first['close'] <= first['open']:
            return None

        # 2. Second candle: bearish (close < open)
        if second['close'] >= second['open']:
            return None

        # 3. Second candle opens above first candle's high (gap up)
        if second['open'] <= first['high']:
            return None

        # 4. Second candle closes below first candle's midpoint
        first_midpoint = (first['open'] + first['close']) / 2
        penetration = (first_midpoint - second['close']) / (first_midpoint - first['open'])

        if penetration < self.penetration_ratio:
            return None

        # Volume confirmation (higher on second candle)
        avg_volume = self.price_data[list(self.price_data.keys())[0]]['volume'].rolling(window=10).mean().iloc[-1]
        volume_confirmation = market_data['volume'] > (avg_volume * self.volume_multiplier)

        if not volume_confirmation:
            return None

        return {
            'type': 'dark_cloud_cover',
            'strength': 0.8,
            'penetration': penetration,
            'first_body_size': abs(first['close'] - first['open']),
            'second_body_size': abs(second['close'] - second['open'])
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

    def _execute_sell_short_order(self, symbol: str, price: float, atr: float,
                                strength: float, market_data: Dict[str, Any]) -> None:
        """Execute sell short order."""
        # Calculate position size based on risk
        portfolio_value = self.get_portfolio_value()
        risk_amount = portfolio_value * self.risk_per_trade

        # Stop loss above the higher high of the pattern
        pattern_high = max(market_data['high'], self.price_data[symbol]['high'].tail(1).max())
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
                print(f"DARK CLOUD COVER SHORT order submitted: {symbol} qty={position_size:.2f} price={price:.2f}")
                notes = f"Dark cloud cover pattern: Short {price:.2f}, Stop {stop_price:.2f}"
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

            for i in range(1, len(data)):
                first = data.iloc[i-1]
                second = data.iloc[i]

                market_data_at_i = {
                    'volume': second['volume'],
                    'high': second['high'],
                    'low': second['low']
                }

                # Detect dark cloud cover
                dark_cloud_info = self._detect_dark_cloud_cover(first, second, market_data_at_i)

                if dark_cloud_info:
                    # Check trend context
                    window_data = data.iloc[max(0, i-self.lookback):i+1]
                    trend_context = self._get_trend_context(window_data, self.lookback)

                    current_atr = atr_series.iloc[i] if i < len(atr_series) else atr_series.iloc[-1]

                    if trend_context == 'uptrend':
                        pattern_high = max(second['high'], first['high'])
                        stop_price = pattern_high + (current_atr * 0.5)
                        signals.append({
                            'timestamp': data.index[i],
                            'symbol': symbol,
                            'signal_type': 'sell_short',
                            'strength': dark_cloud_info['strength'],
                            'price': second['close'],
                            'pattern': 'dark_cloud_cover',
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
            'type': 'Dark Cloud Cover Candlestick',
            'description': 'Detects dark cloud cover two-candle bearish reversal patterns',
            'parameters': self.parameters,
            'indicators': ['ATR'],
            'timeframe': self.timeframe,
            'enabled': self.enabled
        }
