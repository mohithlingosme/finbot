"""
MACD crossover strategy implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from .base import BaseStrategy
from finbot.indicators.indicators import macd, atr
from finbot.core.orders import OrderType


class MACDCrossoverStrategy(BaseStrategy):
    """
    MACD crossover strategy.
    
    This strategy:
    - Uses MACD line and signal line crossovers for entry signals
    - Uses zero line crossovers for trend confirmation
    - Uses ATR for position sizing and stop losses
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize MACD crossover strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)
        
        # Strategy parameters
        self.fast_period = config.get('fast_period', 12)
        self.slow_period = config.get('slow_period', 26)
        self.signal_period = config.get('signal_period', 9)
        self.atr_period = config.get('atr_period', 14)
        self.atr_multiplier = config.get('atr_multiplier', 2.0)
        
        # Risk management
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% of portfolio
        self.risk_per_trade = config.get('risk_per_trade', 0.02)  # 2% risk per trade
        
        # Signal confirmation
        self.require_zero_cross = config.get('require_zero_cross', True)
        self.min_macd_strength = config.get('min_macd_strength', 0.0)
        
        # Stateless functions will be used when calculating
        self.fast_period = self.fast_period
        self.slow_period = self.slow_period
        self.signal_period = self.signal_period
        self.atr_period = self.atr_period
        
        # Data storage
        self.price_data: Dict[str, pd.DataFrame] = {}
        
    def _setup_parameters(self) -> Dict[str, Any]:
        """Setup strategy parameters."""
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period,
            'atr_period': self.atr_period,
            'atr_multiplier': self.atr_multiplier,
            'max_position_size': self.max_position_size,
            'risk_per_trade': self.risk_per_trade,
            'require_zero_cross': self.require_zero_cross,
            'min_macd_strength': self.min_macd_strength
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
        min_periods = max(self.fast_period, self.slow_period, self.signal_period, self.atr_period)
        if symbol not in self.price_data or len(self.price_data[symbol]) < min_periods:
            return
        
        # Calculate indicators
        current_data = self.price_data[symbol]
        
        try:
            # Calculate MACD and ATR
            macd_data = macd(current_data, fast=self.fast_period, slow=self.slow_period, signal=self.signal_period)
            if macd_data.empty:
                return

            atr_series = atr(current_data, window=self.atr_period)
            current_atr = atr_series.iloc[-1] if not atr_series.empty else None
            
            if current_atr is None:
                return
            
            # Get current values
            current_price = data['close']
            current_position = self.get_position(symbol)
            
            # Process MACD signals
            self._process_macd_signals(symbol, macd_data, current_atr, current_position, data)
            
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
    
    def _process_macd_signals(self, symbol: str, macd_data: pd.DataFrame, atr: float,
                             position: float, market_data: Dict[str, Any]) -> None:
        """
        Process MACD crossover signals.
        
        Args:
            symbol: Trading symbol
            macd_data: MACD indicator data
            atr: Current ATR value
            position: Current position size
            market_data: Current market data
        """
        current_price = market_data['close']
        
        # Get current MACD values
        current_macd = macd_data['macd'].iloc[-1]
        current_signal = macd_data['signal'].iloc[-1]
        current_histogram = macd_data['histogram'].iloc[-1]
        previous_macd = macd_data['macd'].iloc[-2] if len(macd_data) > 1 else 0
        previous_signal = macd_data['signal'].iloc[-2] if len(macd_data) > 1 else 0
        
        signal_type = "hold"
        signal_strength = 0.0
        notes = ""
        
        # Check for MACD line crossovers
        macd_cross_up = (current_macd > current_signal) and (previous_macd <= previous_signal)
        macd_cross_down = (current_macd < current_signal) and (previous_macd >= previous_signal)
        
        # Check for zero line crossovers
        zero_cross_up = (current_macd > 0) and (previous_macd <= 0)
        zero_cross_down = (current_macd < 0) and (previous_macd >= 0)
        
        # Entry signals
        if position == 0:  # No current position
            # Bullish signal: MACD crosses above signal line
            if macd_cross_up and abs(current_macd) > self.min_macd_strength:
                # Optional: Require zero line cross for stronger signal
                if not self.require_zero_cross or zero_cross_up:
                    signal_type = "buy"
                    signal_strength = 0.8
                    notes = f"MACD bullish crossover: MACD={current_macd:.4f}, Signal={current_signal:.4f}"
                    
                    # Increase strength if zero line cross
                    if zero_cross_up:
                        signal_strength = 0.9
                        notes += " + Zero line cross"
        
        # Exit signals for existing positions
        elif position > 0:  # Long position
            # Bearish signal: MACD crosses below signal line
            if macd_cross_down:
                signal_type = "sell"
                signal_strength = 0.8
                notes = f"MACD bearish crossover: MACD={current_macd:.4f}, Signal={current_signal:.4f}"
            
            # Strong exit: MACD crosses below zero line
            elif zero_cross_down:
                signal_type = "sell"
                signal_strength = 0.9
                notes = f"MACD zero line cross down: MACD={current_macd:.4f}"
            
            # Stop loss based on histogram
            elif current_histogram < 0 and abs(current_histogram) > (atr * self.atr_multiplier):
                signal_type = "sell"
                signal_strength = 0.7
                notes = f"Stop loss: Negative histogram {current_histogram:.4f}"
        
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
        
        # Calculate stop loss price based on ATR
        stop_loss_price = price - (atr * self.atr_multiplier)
        
        # Calculate position size
        position_size = self.calculate_position_size(symbol, risk_amount, stop_loss_price, price)
        
        # Apply maximum position size limit
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
        
        min_periods = max(self.fast_period, self.slow_period, self.signal_period, self.atr_period)
        if len(data) < min_periods:
            return signals
        
        try:
            # Calculate indicators
            macd_data = macd(data, fast=self.fast_period, slow=self.slow_period, signal=self.signal_period)
            atr = atr(data, window=self.atr_period)
            
            if macd_data.empty:
                return signals
            
            # Generate signals for each data point
            for i in range(1, len(data)):  # Start from 1 to compare with previous values
                if pd.isna(macd_data.iloc[i]['macd']) or pd.isna(atr.iloc[i]):
                    continue
                
                current_macd = macd_data.iloc[i]['macd']
                current_signal = macd_data.iloc[i]['signal']
                current_histogram = macd_data.iloc[i]['histogram']
                previous_macd = macd_data.iloc[i-1]['macd']
                previous_signal = macd_data.iloc[i-1]['signal']
                
                price = data.iloc[i]['close']
                
                # Check for crossovers
                macd_cross_up = (current_macd > current_signal) and (previous_macd <= previous_signal)
                macd_cross_down = (current_macd < current_signal) and (previous_macd >= previous_signal)
                zero_cross_up = (current_macd > 0) and (previous_macd <= 0)
                zero_cross_down = (current_macd < 0) and (previous_macd >= 0)
                
                signal_type = "hold"
                signal_strength = 0.0
                
                # Buy signals
                if macd_cross_up and abs(current_macd) > self.min_macd_strength:
                    if not self.require_zero_cross or zero_cross_up:
                        signal_type = "buy"
                        signal_strength = 0.8
                        if zero_cross_up:
                            signal_strength = 0.9
                
                # Sell signals
                elif macd_cross_down or zero_cross_down:
                    signal_type = "sell"
                    signal_strength = 0.8 if macd_cross_down else 0.9
                
                if signal_type != "hold":
                    signals.append({
                        'timestamp': data.index[i],
                        'symbol': symbol,
                        'signal_type': signal_type,
                        'strength': signal_strength,
                        'price': price,
                        'macd': current_macd,
                        'signal': current_signal,
                        'histogram': current_histogram,
                        'zero_cross': zero_cross_up or zero_cross_down,
                        'macd_cross': macd_cross_up or macd_cross_down
                    })
        
        except Exception as e:
            print(f"Error calculating signals for {symbol}: {e}")
        
        return signals
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': self.name,
            'type': 'MACD Crossover',
            'description': 'MACD line and signal line crossover strategy with zero line confirmation',
            'parameters': self.parameters,
            'indicators': ['MACD', 'ATR'],
            'timeframe': self.timeframe,
            'enabled': self.enabled
        }
