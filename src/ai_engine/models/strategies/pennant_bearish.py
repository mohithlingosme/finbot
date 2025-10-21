"""
Bearish pennant strategy implementation.

Detects bearish pennant patterns: small symmetrical triangle after strong down move.
Entry on breakdown continuation in pole direction; stop inside pennant; target ≈ pole height.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.ai_engine.models.strategies.base import BaseStrategy
from src.ai_engine.scripts.indicators.indicators import atr


class BearishPennantStrategy(BaseStrategy):
    """
    Bearish pennant strategy.

    Identifies pennant consolidation after strong downtrend and enters on breakdown.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize bearish pennant strategy.

        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)

        # Strategy parameters
        self.pole_min_pct = config.get('pole_min_pct', 5.0)  # Minimum pole % decrease
        self.pole_max_pct = config.get('pole_max_pct', 20.0)  # Maximum pole % decrease
        self.pennant_lookback = config.get('pennant_lookback', 10)  # Bars for pennant formation
        self.convergence_threshold = config.get('convergence_threshold', 0.02)  # Triangle convergence
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
            'pole_min_pct': self.pole_min_pct,
            'pole_max_pct': self.pole_max_pct,
            'pennant_lookback': self.pennant_lookback,
            'convergence_threshold': self.convergence_threshold,
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
        if symbol not in self.price_data or len(self.price_data[symbol]) < self.pennant_lookback + 20:
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

            # Check for pennant pattern and breakout
            self._check_pennant_and_breakout(symbol, current_data, current_atr, current_position, data)

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

    def _check_pennant_and_breakout(self, symbol: str, data: pd.DataFrame,
                                   atr: float, position: float, market_data: Dict[str, Any]) -> None:
        """
        Check for bearish pennant pattern and breakout signals.

        Args:
            symbol: Trading symbol
            data: Price data
            atr: Current ATR value
            position: Current position size
            market_data: Current market data
        """
        current_price = market_data['close']

        # Detect pennant pattern
        pennant_info = self._detect_bearish_pennant(data)

        if not pennant_info:
            return

        pennant_high = pennant_info['pennant_high']
        pennant_low = pennant_info['pennant_low']
        pole_height = pennant_info['pole_height']

        # Check for breakdown below pennant
        if current_price < pennant_low and position == 0:
            # Calculate target and stop
            target_price = current_price - pole_height
            stop_price = pennant_high

            # Execute sell order
            self._execute_sell_order(symbol, current_price, stop_price, target_price, atr, market_data)

    def _detect_bearish_pennant(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Detect bearish pennant pattern in recent data.

        Returns dict with pennant info or None if no pattern found.
        """
        if len(data) < self.pennant_lookback + 10:
            return None

        # Look for pole: recent sharp drop
        recent_lows = data['low'].tail(20)
        pole_start_idx = recent_lows.idxmax()
        pole_end_idx = len(data) - 1

        if pole_end_idx - pole_start_idx < 5:
            return None

        pole_start_price = data.loc[pole_start_idx, 'high']
        pole_end_price = data.loc[pole_end_idx, 'low']
        pole_pct = (pole_start_price - pole_end_price) / pole_start_price * 100

        # Check pole size
        if not (self.pole_min_pct <= pole_pct <= self.pole_max_pct):
            return None

        # Look for pennant: small symmetrical triangle after pole
        pennant_data = data.tail(self.pennant_lookback)

        # Calculate trendlines
        highs = pennant_data['high']
        lows = pennant_data['low']

        # Upper trendline (downward sloping)
        x = np.arange(len(pennant_data))
        upper_slope = np.polyfit(x, highs, 1)[0]

        # Lower trendline (upward sloping)
        lower_slope = np.polyfit(x, lows, 1)[0]

        # Check for convergence (slopes should be converging)
        if upper_slope >= 0 or lower_slope <= 0:
            return None

        # Check convergence rate
        convergence = abs(upper_slope) + abs(lower_slope)
        if convergence < self.convergence_threshold:
            return None

        # Pennant boundaries
        pennant_high = highs.max()
        pennant_low = lows.min()

        # Pole height
        pole_height = pole_start_price - pole_end_price

        return {
            'pennant_high': pennant_high,
            'pennant_low': pennant_low,
            'pole_height': pole_height,
            'pole_start': pole_start_price,
            'pole_end': pole_end_price,
            'convergence': convergence
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
                print(f"BEARISH PENNANT SELL order submitted: {symbol} qty={position_size:.2f} price={price:.2f}")
                notes = f"Bearish pennant breakdown: Entry {price:.2f}, Stop {stop_price:.2f}, Target {target_price:.2f}"
                self.log_signal(symbol, "sell", 0.7, market_data, notes)
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

        if len(data) < self.pennant_lookback + 20:
            return signals

        try:
            # Calculate ATR
            atr_series = atr(data, window=self.atr_period)

            for i in range(self.pennant_lookback + 10, len(data)):
                window_data = data.iloc[:i+1]

                # Detect pennant pattern
                pennant_info = self._detect_bearish_pennant(window_data)

                if pennant_info:
                    current_price = data.iloc[i]['close']

                    # Check breakdown
                    if current_price < pennant_info['pennant_low']:
                        signals.append({
                            'timestamp': data.index[i],
                            'symbol': symbol,
                            'signal_type': 'sell',
                            'strength': 0.7,
                            'price': current_price,
                            'pennant_high': pennant_info['pennant_high'],
                            'pennant_low': pennant_info['pennant_low'],
                            'pole_height': pennant_info['pole_height'],
                            'stop_price': pennant_info['pennant_high'],
                            'target_price': current_price - pennant_info['pole_height']
                        })

        except Exception as e:
            print(f"Error calculating signals for {symbol}: {e}")

        return signals

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': self.name,
            'type': 'Bearish Pennant',
            'description': 'Detects bearish pennant patterns and enters on breakdowns',
            'parameters': self.parameters,
            'indicators': ['ATR'],
            'timeframe': self.timeframe,
            'enabled': self.enabled
        }
