# TradeBot - Modular Trading Bot Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, modular trading bot framework built in Python for intraday trading in Indian markets (NSE/BSE) and global assets. TradeBot provides a complete ecosystem for strategy development, backtesting, risk management, and live trading.

## 🚀 Features

### Core Capabilities
- **Multi-Asset Support**: Trade Indian markets (NSE/BSE) and global assets
- **Modular Architecture**: Clean, extensible design with separate modules for different functionalities
- **Strategy Framework**: Easy-to-use base classes for developing custom trading strategies
- **Backtesting Engine**: Comprehensive backtesting with realistic commission and slippage modeling
- **Risk Management**: Built-in position sizing, stop-loss, and portfolio risk controls
- **Zerodha Integration**: Native support for Zerodha Kite API with paper trading for backtesting

### Technical Indicators
- **ATR (Average True Range)**: Volatility measurement and position sizing
- **VWAP (Volume Weighted Average Price)**: Institutional benchmark and mean reversion
- **RSI (Relative Strength Index)**: Momentum oscillator for overbought/oversold conditions
- **Bollinger Bands**: Volatility-based support/resistance and breakout detection
- **MACD**: Trend following and momentum analysis
- **ADX**: Trend strength measurement

### Sample Strategies
- **VWAP-RSI-ATR Strategy**: Mean reversion using VWAP with RSI confirmation and ATR risk management
- **Bollinger Bands Breakout**: Volatility breakout strategy with volume confirmation
- **MACD Crossover**: Classic trend-following strategy with zero-line confirmation

### Risk Management
- **Position Sizing**: Multiple algorithms (Fixed, Percentage, Kelly Criterion, Volatility-based, ATR-based)
- **Stop Loss**: ATR-based dynamic stop losses
- **Portfolio Limits**: Maximum position size, daily loss limits, and open position limits
- **Real-time Monitoring**: Continuous risk assessment and alerts

### Data Management
- **Multiple Sources**: Yahoo Finance, Alpha Vantage with automatic fallback
- **Caching**: Local storage with configurable formats (CSV, Parquet, HDF5, SQLite)
- **Data Validation**: OHLC consistency checks and quality assurance
- **Resampling**: Flexible time frame conversion

### Monitoring & Logging
- **Real-time Dashboard**: Streamlit-based web interface
- **Comprehensive Logging**: Trade logs, performance metrics, and error tracking
- **Alert System**: Email and Slack notifications
- **Performance Analytics**: Sharpe ratio, maximum drawdown, win rate, and more

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/tradebot.git
cd tradebot

# Install dependencies
pip install -r requirements.txt

# Run your first backtest
python main.py --mode backtest --symbols RELIANCE TCS HDFCBANK --start-date 2023-01-01 --end-date 2023-12-31
```

### Dependencies
The framework uses several key libraries:

**Core Data Science:**
- pandas, numpy, matplotlib, scipy

**Technical Analysis:**
- ta, TA-Lib

**Backtesting:**
- backtrader, vectorbt

**Zerodha API:**
- kiteconnect (Official Zerodha API)

**Utilities:**
- loguru, streamlit, fastapi, PyYAML, requests

## ⚙️ Configuration

TradeBot uses YAML configuration files for easy customization. Copy and modify `config.yaml`:

```yaml
# General Settings
general:
  project_name: "tradebot"
  log_level: "INFO"

# Market Configuration
markets:
  indian:
    exchanges: ["NSE", "BSE"]
    trading_hours:
      start: "09:15"
      end: "15:30"

# Strategy Configuration
strategies:
  vwap_rsi_atr:
    enabled: true
    timeframe: "5m"
    rsi_period: 14
    rsi_oversold: 30
    rsi_overbought: 70

# Risk Management
risk:
  max_position_size: 0.1  # 10% of portfolio
  max_daily_loss: 0.02    # 2% of portfolio
  max_open_positions: 5

# Zerodha Broker Configuration
brokers:
  zerodha:
    api_key: "your_api_key"  # Get from https://kite.trade/
    api_secret: "your_api_secret"  # Get from https://kite.trade/
    access_token: "your_access_token"  # Generate using login flow
```

## 🎯 Usage

### Backtesting
```bash
# Basic backtest
python main.py --mode backtest --symbols RELIANCE TCS --start-date 2023-01-01 --end-date 2023-12-31

# Advanced backtest with custom symbols
python main.py --mode backtest --symbols NIFTY_50 BANKNIFTY --start-date 2023-06-01 --end-date 2023-12-31
```

### Live Trading
```bash
# Start live trading with Zerodha (requires API credentials)
python main.py --mode live --symbols RELIANCE TCS HDFCBANK

# Note: Backtesting uses paper trading automatically
```

### Dashboard
```bash
# Launch monitoring dashboard
python main.py --mode dashboard

# Access dashboard at http://localhost:8501
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_indicators.py
pytest tests/test_strategies.py
pytest tests/test_backtest.py
```

## 🏗️ Architecture

```
tradebot/
├── core/                    # Core trading engine
│   ├── engine.py           # Main trading orchestrator
│   ├── portfolio.py        # Portfolio management
│   └── orders.py           # Order management system
├── data/                   # Data management
│   ├── fetcher.py          # Multi-source data fetching
│   ├── loader.py           # Data preprocessing
│   └── storage.py          # Caching and persistence
├── indicators/             # Technical indicators
│   ├── atr.py             # Average True Range
│   ├── vwap.py            # Volume Weighted Average Price
│   ├── rsi.py             # Relative Strength Index
│   ├── bollinger.py       # Bollinger Bands
│   ├── macd.py            # MACD
│   └── adx.py             # Average Directional Index
├── strategies/             # Trading strategies
│   ├── base.py            # Base strategy class
│   ├── vwap_rsi_atr.py    # VWAP-RSI-ATR strategy
│   ├── bollinger_breakout.py # Bollinger breakout strategy
│   └── macd_crossover.py  # MACD crossover strategy
├── backtest/              # Backtesting engine
│   ├── engine.py          # Backtesting orchestrator
│   ├── metrics.py         # Performance metrics
│   └── plot.py            # Visualization
├── execution/             # Broker integration
│   ├── broker.py          # Base broker interface
│   ├── zerodha.py         # Zerodha integration
│   └── paper.py           # Paper trading (for backtesting)
├── risk/                  # Risk management
│   ├── risk_manager.py    # Risk controls
│   └── position_sizing.py # Position sizing algorithms
├── monitoring/            # Monitoring and logging
│   ├── logger.py          # Centralized logging
│   ├── alerts.py          # Alert system
│   └── dashboard.py       # Web dashboard
├── utils/                 # Utilities
│   ├── config_loader.py   # Configuration management
│   └── helpers.py         # Helper functions
├── tests/                 # Test suite
└── main.py               # Entry point
```

## 📊 Sample Results

### VWAP-RSI-ATR Strategy Backtest Results
```
=== Backtest Results ===
Strategy: VWAP_RSI_ATR
Total Return: 15.23%
Annualized Return: 14.87%
Sharpe Ratio: 1.42
Max Drawdown: -8.45%
Total Trades: 127
Win Rate: 58.27%
Final Value: ₹115,230.45
```

### Strategy Comparison
```
Strategy Comparison:
    Strategy  Total Return (%)  Sharpe Ratio  Max Drawdown (%)  Win Rate (%)
VWAP_RSI_ATR          15.23          1.42             -8.45         58.27
Bollinger_Breakout    12.87          1.18            -12.34         55.12
MACD_Crossover         8.45          0.95            -15.67         52.89
```

## 🔧 Developing Custom Strategies

### Creating a New Strategy
```python
from strategies.base import BaseStrategy
from core.orders import OrderType

class MyCustomStrategy(BaseStrategy):
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        # Initialize your strategy parameters
        
    def on_data(self, symbol: str, data: dict) -> None:
        """Process new market data"""
        current_price = data['close']
        current_position = self.get_position(symbol)
        
        # Your strategy logic here
        if self.should_buy(symbol, data):
            order = self.generate_order(symbol, "buy", 100, current_price)
            self.submit_order(order)
        elif self.should_sell(symbol, data):
            order = self.generate_order(symbol, "sell", current_position, current_price)
            self.submit_order(order)
    
    def should_buy(self, symbol: str, data: dict) -> bool:
        """Your buy signal logic"""
        return False  # Implement your logic
    
    def should_sell(self, symbol: str, data: dict) -> bool:
        """Your sell signal logic"""
        return False  # Implement your logic
```

### Adding to Configuration
```yaml
strategies:
  my_custom_strategy:
    enabled: true
    timeframe: "15m"
    # Add your strategy parameters here
```

## 🔐 Zerodha Integration

### Setup Instructions
1. **Create Zerodha Account**: Sign up at [Zerodha](https://zerodha.com/)
2. **Get API Credentials**: 
   - Visit [Kite Developer Console](https://kite.trade/)
   - Create a new app to get API Key and Secret
3. **Generate Access Token**: Use the login flow to generate access token
4. **Update Configuration**:
```yaml
brokers:
  zerodha:
    api_key: "your_api_key"
    api_secret: "your_api_secret"
    access_token: "your_access_token"
```

### API Access
- **Paper Trading**: For testing and backtesting (no real money)
- **Live Trading**: Real money trading (requires proper KYC and funds)
- **Market Data**: Real-time and historical data access
- **Order Management**: Place, modify, and cancel orders

## 📈 Performance Monitoring

### Dashboard Features
- Real-time portfolio monitoring
- Strategy performance tracking
- Trade history and analytics
- Risk metrics visualization
- P&L charts and equity curves

### Alert System
- Trade execution notifications
- Performance milestone alerts
- Risk limit breaches
- Error notifications

## 🧪 Testing

The framework includes comprehensive tests for all components:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tradebot

# Run specific test categories
pytest tests/test_indicators.py    # Technical indicators
pytest tests/test_strategies.py    # Trading strategies
pytest tests/test_backtest.py      # Backtesting engine
pytest tests/test_execution.py     # Broker integration
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone repository
git clone https://github.com/yourusername/tradebot.git
cd tradebot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest

# Run linting
flake8 tradebot/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**IMPORTANT**: This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Always do your own research and consider consulting with a financial advisor before making investment decisions.

The authors and contributors are not responsible for any financial losses incurred through the use of this software. Use at your own risk.

## 📞 Support

- 📧 Email: support@tradebot.dev
- 💬 Discord: [TradeBot Community](https://discord.gg/tradebot)
- 📖 Documentation: [docs.tradebot.dev](https://docs.tradebot.dev)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/tradebot/issues)

## 🙏 Acknowledgments

- [TA-Lib](https://ta-lib.org/) for technical analysis functions
- [Backtrader](https://www.backtrader.com/) for backtesting inspiration
- [Streamlit](https://streamlit.io/) for dashboard framework
- [Loguru](https://github.com/Delgan/loguru) for excellent logging

---

**Happy Trading! 📈**