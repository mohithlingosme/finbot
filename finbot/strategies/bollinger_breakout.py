"""
Bollinger Bands breakout strategy implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from finbot.strategies.base import BaseStrategy
from finbot.indicators.indicators import bollinger, atr
from finbot.core.orders import OrderType


class BollingerBreakoutStrategy(BaseStrategy):
    """
    Bollinger Bands breakout strategy.
    
    This strategy:
    - Identifies periods of low volatility (squeeze)
    - Waits for breakouts from the squeeze
    - Uses ATR for position sizing and stop losses
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize Bollinger Bands breakout strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)
        
        # Strategy parameters
        self.bb_period = config.get('period', 20)
        self.bb_std = config.get('std_dev', 2.0)
        self.lookback = config.get('lookback', 5)
        self.atr_period = config.get('atr_period', 14)
        self.atr_multiplier = config.get('atr_multiplier', 2.0)
        
        # Risk management
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% of portfolio
        self.risk_per_trade = config.get('risk_per_trade', 0.02)  # 2% risk per trade
        
        # Data storage
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.squeeze_periods: Dict[str, List[int]] = {}
        
    def _setup_parameters(self) -> Dict[str, Any]:
        """Setup strategy parameters."""
        return {
            'bb_period': self.bb_period,
            'bb_std': self.bb_std,
            'lookback': self.lookback,
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
        if symbol not in self.price_data or len(self.price_data[symbol]) < max(self.bb_period, self.atr_period):
            return
        
        # Calculate indicators
        current_data = self.price_data[symbol]
        
        try:
            # Calculate Bollinger Bands (returns DataFrame with 'ma' column)
            bb_data = bollinger(current_data['close'], window=self.bb_period, n_std=self.bb_std)
            if bb_data.empty:
                return
            # keep compatibility: 'ma' -> 'middle'
            if 'ma' in bb_data.columns and 'middle' not in bb_data.columns:
                bb_data = bb_data.rename(columns={'ma': 'middle'})

            # Calculate ATR
            atr_series = atr(current_data, window=self.atr_period)
            current_atr = atr_series.iloc[-1] if not atr_series.empty else None
            
            if current_atr is None:
                return
            
            # Get current values
            current_price = data['close']
            current_position = self.get_position(symbol)
            
            # Check for squeeze and breakout
            self._check_squeeze_and_breakout(symbol, current_data, bb_data, current_atr, current_position, data)
            
        except Exception as e:
            print(f"Error processing data for {symbol}: {e}")
    
    def _update_price_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """Update price data for symbol."""
        if symbol not in self.price_data:
            self.price_data[symbol] = pd.DataFrame()
            self.squeeze_periods[symbol] = []
        
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
    
    def _check_squeeze_and_breakout(self, symbol: str, data: pd.DataFrame, bb_data: pd.DataFrame,
                                   atr: float, position: float, market_data: Dict[str, Any]) -> None:
        """
        Check for squeeze conditions and breakout signals.
        
        Args:
            symbol: Trading symbol
            data: Price data
            bb_data: Bollinger Bands data
            atr: Current ATR value
            position: Current position size
            market_data: Current market data
        """
        current_price = market_data['close']
        
        # Check if we're in a squeeze (low volatility)
        current_bandwidth = bb_data['bandwidth'].iloc[-1] if not bb_data['bandwidth'].empty else 0
        avg_bandwidth = bb_data['bandwidth'].rolling(window=20).mean().iloc[-1] if len(bb_data) >= 20 else 0
        
        is_squeeze = current_bandwidth < (avg_bandwidth * 0.8)  # 20% below average
        
        # Track squeeze periods
        if is_squeeze:
            self.squeeze_periods[symbol].append(len(data) - 1)
        else:
            self.squeeze_periods[symbol] = []
        
        # Check for breakout after squeeze
        if len(self.squeeze_periods[symbol]) >= self.lookback:  # Minimum squeeze period
            self._check_breakout_signals(symbol, data, bb_data, atr, position, market_data)
    
    def _check_breakout_signals(self, symbol: str, data: pd.DataFrame, bb_data: pd.DataFrame,
                               atr: float, position: float, market_data: Dict[str, Any]) -> None:
        """Check for breakout signals."""
        current_price = market_data['close']
        current_volume = market_data['volume']
        
        # Get Bollinger Bands levels
        upper_band = bb_data['upper'].iloc[-1]
        lower_band = bb_data['lower'].iloc[-1]
        middle_band = bb_data['middle'].iloc[-1]
        
        # Check for volume confirmation (volume above average)
        avg_volume = data['volume'].rolling(window=20).mean().iloc[-1] if len(data) >= 20 else current_volume
        volume_confirmation = current_volume > (avg_volume * 1.2)
        
        signal_type = "hold"
        signal_strength = 0.0
        notes = ""
        
        # Breakout signals
        if position == 0:  # No current position
            # Upper breakout (bullish)
            if current_price > upper_band and volume_confirmation:
                signal_type = "buy"
                signal_strength = 0.8
                notes = f"Upper breakout: Price {current_price:.2f} > Upper Band {upper_band:.2f}"
            
            # Lower breakout (bearish) - for short selling if enabled
            elif current_price < lower_band and volume_confirmation:
                signal_type = "sell_short"
                signal_strength = 0.8
                notes = f"Lower breakout: Price {current_price:.2f} < Lower Band {lower_band:.2f}"
        
        # Exit signals for existing positions
        elif position > 0:  # Long position
            # Exit if price falls back to middle band or below
            if current_price <= middle_band:
                signal_type = "sell"
                signal_strength = 0.7
                notes = f"Exit long: Price {current_price:.2f} <= Middle Band {middle_band:.2f}"
            
            # Stop loss: Price below lower band
            elif current_price < lower_band:
                signal_type = "sell"
                signal_strength = 0.9
                notes = f"Stop loss: Price {current_price:.2f} < Lower Band {lower_band:.2f}"
        
        # Execute trades
        if signal_type == "buy" and position == 0:
            self._execute_buy_order(symbol, current_price, atr, signal_strength, notes)
        
        elif signal_type == "sell" and position > 0:
            self._execute_sell_order(symbol, current_price, signal_strength, notes)
        
        # Log signal
        if signal_type != "hold":
            self.log_signal(symbol, signal_type, signal_strength, market_data, notes)
    
    def _execute_buy_order(self, symbol: str, price: float, atr: float,
                           strength: float, notes: str) -> None:
        """Execute buy order."""
        # Calculate position size
        portfolio_value = self.get_portfolio_value()
        risk_amount = portfolio_value * self.risk_per_trade

        # Calculate stop loss price (below lower Bollinger Band)
        current_data = self.price_data[symbol]
        bb_data = bollinger(current_data['close'], window=self.bb_period, n_std=self.bb_std)
        lower_band = bb_data['lower'].iloc[-1] if not bb_data.empty else price - (atr * self.atr_multiplier)
        stop_loss_price = lower_band

        # Ensure stop loss is valid
        if stop_loss_price >= price:
            return  # Cannot calculate position size if stop loss is at or above entry

        # Calculate position size based on risk
        position_size = risk_amount / (price - stop_loss_price)

        # Cap position size by max_position_size
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
                print(f"BUY order submitted: {symbol} qty={position_size:.2f} price={price:.2f}")
                # Clear squeeze periods after successful trade
                self.squeeze_periods[symbol] = []
            else:
                print(f"Failed to submit BUY order for {symbol}")
    
    def _execute_sell_order(self, symbol: str, price: float, strength: float, notes: str) -> None:
        """Execute sell order."""
        current_position = self.get_position(symbol)
        
        if current_position > 0:
            # Create sell order
            order = self.generate_order(
                symbol=symbol,
                side='sell',
                quantity=current_position,
                price=price,
                order_type='market'
            )
            
            # Submit order
            if self.submit_order(order):
                print(f"SELL order submitted: {symbol} qty={current_position:.2f} price={price:.2f}")
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
        
        if len(data) < max(self.bb_period, self.atr_period):
            return signals
        
        try:
            # Calculate indicators
            bb_data = bollinger(data['close'], window=self.bb_period, n_std=self.bb_std)
            if 'ma' in bb_data.columns and 'middle' not in bb_data.columns:
                bb_data = bb_data.rename(columns={'ma': 'middle'})
            atr = atr(data, window=self.atr_period)
            
            if bb_data.empty:
                return signals
            
            # Calculate average bandwidth for squeeze detection
            bb_data['avg_bandwidth'] = bb_data['bandwidth'].rolling(window=20).mean()
            bb_data['is_squeeze'] = bb_data['bandwidth'] < (bb_data['avg_bandwidth'] * 0.8)
            
            # Track squeeze periods
            squeeze_periods = []
            
            for i in range(len(data)):
                if pd.isna(bb_data.iloc[i]['upper']) or pd.isna(atr.iloc[i]):
                    continue
                
                price = data.iloc[i]['close']
                upper_band = bb_data.iloc[i]['upper']
                lower_band = bb_data.iloc[i]['lower']
                middle_band = bb_data.iloc[i]['middle']
                is_squeeze = bb_data.iloc[i]['is_squeeze']
                
                # Track squeeze periods
                if is_squeeze:
                    squeeze_periods.append(i)
                else:
                    squeeze_periods = []
                
                # Check for breakout after minimum squeeze period
                if len(squeeze_periods) >= self.lookback:
                    signal_type = "hold"
                    signal_strength = 0.0
                    
                    # Upper breakout
                    if price > upper_band:
                        signal_type = "buy"
                        signal_strength = 0.8
                    
                    # Lower breakout
                    elif price < lower_band:
                        signal_type = "sell_short"
                        signal_strength = 0.8
                    
                    if signal_type != "hold":
                        signals.append({
                            'timestamp': data.index[i],
                            'symbol': symbol,
                            'signal_type': signal_type,
                            'strength': signal_strength,
                            'price': price,
                            'upper_band': upper_band,
                            'lower_band': lower_band,
                            'middle_band': middle_band,
                            'bandwidth': bb_data.iloc[i]['bandwidth'],
                            'squeeze_periods': len(squeeze_periods)
                        })
        
        except Exception as e:
            print(f"Error calculating signals for {symbol}: {e}")
        
        return signals
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': self.name,
            'type': 'Bollinger Bands Breakout',
            'description': 'Breakout strategy using Bollinger Bands squeeze and volume confirmation',
            'parameters': self.parameters,
            'indicators': ['Bollinger Bands', 'ATR'],
            'timeframe': self.timeframe,
            'enabled': self.enabled
        }
