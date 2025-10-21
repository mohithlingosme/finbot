"""
VWAP-RSI-ATR combined strategy implementation.
"""

import pandas as pd
from typing import Dict, List, Any
from datetime import datetime

from finbot.strategies.base import BaseStrategy
from finbot.indicators.indicators import vwap, rsi, atr


class VWAPRSIATRStrategy(BaseStrategy):
    """
    Combined VWAP, RSI, and ATR strategy.
    
    This strategy uses:
    - VWAP for mean reversion signals
    - RSI for momentum confirmation
    - ATR for position sizing and stop losses
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize VWAP-RSI-ATR strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)
        
        # Strategy parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.vwap_period = config.get('vwap_period', 20)
        self.atr_period = config.get('atr_period', 14)
        self.atr_multiplier = config.get('atr_multiplier', 2.0)
        
        # Risk management
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% of portfolio
        self.risk_per_trade = config.get('risk_per_trade', 0.02)  # 2% risk per trade
        
        # Data storage for indicators
        self.price_data: Dict[str, pd.DataFrame] = {}
        
    def _setup_parameters(self) -> Dict[str, Any]:
        """Setup strategy parameters."""
        return {
            'rsi_period': self.rsi_period,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'vwap_period': self.vwap_period,
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
        if symbol not in self.price_data or len(self.price_data[symbol]) < max(self.rsi_period, self.vwap_period, self.atr_period):
            return
        
        # Calculate indicators
        current_data = self.price_data[symbol]
        
        try:
            # Calculate VWAP, RSI and ATR using unified functions
            vwap_series = vwap(current_data)
            current_vwap = vwap_series.iloc[-1] if not vwap_series.empty else None

            rsi_series = rsi(current_data['close'], window=self.rsi_period)
            current_rsi = rsi_series.iloc[-1] if not rsi_series.empty else None

            atr_series = atr(current_data, window=self.atr_period)
            current_atr = atr_series.iloc[-1] if not atr_series.empty else None
            
            if current_vwap is None or current_rsi is None or current_atr is None:
                return
            
            # Get current price and position
            current_price = data['close']
            current_position = self.get_position(symbol)
            
            # Generate signals
            self._process_signals(symbol, current_price, current_vwap, current_rsi, current_atr, current_position, data)
            
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
        
        # Keep only recent data (last 1000 bars)
        if len(self.price_data[symbol]) > 1000:
            self.price_data[symbol] = self.price_data[symbol].tail(1000).reset_index(drop=True)
    
    def _process_signals(self, symbol: str, price: float, vwap: float, rsi: float, 
                        atr: float, position: float, data: Dict[str, Any]) -> None:
        """
        Process trading signals based on indicators.
        
        Args:
            symbol: Trading symbol
            price: Current price
            vwap: Current VWAP
            rsi: Current RSI
            atr: Current ATR
            position: Current position size
            data: Market data
        """
        # Calculate price deviation from VWAP
        vwap_deviation = ((price - vwap) / vwap) * 100
        
        # Signal generation logic
        signal_strength = 0.0
        signal_type = "hold"
        notes = ""
        
        # Buy signals (mean reversion)
        if position == 0:  # No current position
            # Strong buy: Price below VWAP, RSI oversold
            if vwap_deviation < -1.0 and rsi < self.rsi_oversold:
                signal_type = "buy"
                signal_strength = 0.8
                notes = f"Strong buy: Price {vwap_deviation:.2f}% below VWAP, RSI {rsi:.1f}"
            
            # Moderate buy: Price below VWAP, RSI not overbought
            elif vwap_deviation < -0.5 and rsi < 50:
                signal_type = "buy"
                signal_strength = 0.6
                notes = f"Buy: Price {vwap_deviation:.2f}% below VWAP, RSI {rsi:.1f}"
        
        # Sell signals (take profit or stop loss)
        elif position > 0:  # Long position
            # Take profit: Price above VWAP, RSI overbought
            if vwap_deviation > 1.0 and rsi > self.rsi_overbought:
                signal_type = "sell"
                signal_strength = 0.8
                notes = f"Take profit: Price {vwap_deviation:.2f}% above VWAP, RSI {rsi:.1f}"
            
            # Stop loss: Price below VWAP by more than ATR
            elif price < (vwap - atr * self.atr_multiplier):
                signal_type = "sell"
                signal_strength = 0.9
                notes = f"Stop loss: Price below VWAP by {atr * self.atr_multiplier:.2f}"
        
        # Execute trades based on signals
        if signal_type == "buy" and position == 0:
            self._execute_buy_order(symbol, price, atr, signal_strength, notes)
        
        elif signal_type == "sell" and position > 0:
            self._execute_sell_order(symbol, price, signal_strength, notes)
        
        # Log signal
        if signal_type != "hold":
            self.log_signal(symbol, signal_type, signal_strength, data, notes)
    
    def _execute_buy_order(self, symbol: str, price: float, atr: float, 
                          strength: float, notes: str) -> None:
        """Execute buy order."""
        # Calculate position size based on risk management
        portfolio_value = self.get_portfolio_value()
        risk_amount = portfolio_value * self.risk_per_trade
        
        # Calculate stop loss price
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
                price=price,  # Market order
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
                price=price,  # Market order
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
        
        if len(data) < max(self.rsi_period, self.vwap_period, self.atr_period):
            return signals
        
        try:
            # Calculate indicators
            vwap_series = vwap(data)
            rsi_series = rsi(data['close'], window=self.rsi_period)
            atr_series = atr(data, window=self.atr_period)
            
            # Generate signals for each data point
            for i in range(len(data)):
                if pd.isna(vwap_series.iloc[i]) or pd.isna(rsi_series.iloc[i]) or pd.isna(atr_series.iloc[i]):
                    continue
                
                price = data.iloc[i]['close']
                vwap_val = vwap_series.iloc[i]
                rsi_val = rsi_series.iloc[i]
                atr_val = atr_series.iloc[i]
                vwap_deviation = ((price - vwap_val) / vwap_val) * 100
                
                # Determine signal
                signal_type = "hold"
                signal_strength = 0.0
                
                # Buy conditions
                if vwap_deviation < -1.0 and rsi_val < self.rsi_oversold:
                    signal_type = "buy"
                    signal_strength = 0.8
                elif vwap_deviation < -0.5 and rsi_val < 50:
                    signal_type = "buy"
                    signal_strength = 0.6
                
                # Sell conditions
                elif vwap_deviation > 1.0 and rsi_val > self.rsi_overbought:
                    signal_type = "sell"
                    signal_strength = 0.8
                
                if signal_type != "hold":
                    signals.append({
                        'timestamp': data.index[i],
                        'symbol': symbol,
                        'signal_type': signal_type,
                        'strength': signal_strength,
                        'price': price,
                        'vwap': vwap_val,
                        'rsi': rsi_val,
                        'atr': atr_val,
                        'vwap_deviation': vwap_deviation
                    })
        
        except Exception as e:
            print(f"Error calculating signals for {symbol}: {e}")
        
        return signals
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': self.name,
            'type': 'VWAP-RSI-ATR',
            'description': 'Combined VWAP mean reversion with RSI momentum and ATR risk management',
            'parameters': self.parameters,
            'indicators': ['VWAP', 'RSI', 'ATR'],
            'timeframe': self.timeframe,
            'enabled': self.enabled
        }
