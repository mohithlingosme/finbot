"""
Main entry point for TradeBot.
"""

import argparse
import sys
from datetime import datetime, timedelta
from typing import Dict, List

from finbot.core.engine import TradingEngine
from finbot.strategies.vwap_rsi_atr import VWAPRSIATRStrategy
from finbot.strategies.bollinger_breakout import BollingerBreakoutStrategy
from finbot.strategies.macd_crossover import MACDCrossoverStrategy
from finbot.backtest.engine import BacktestEngine, BacktestConfig
from finbot.data.loader import DataLoader
from finbot.execution.paper import PaperBroker
from finbot.execution.zerodha import ZerodhaBroker
from finbot.monitoring.logger import Logger
from finbot.monitoring.dashboard import Dashboard
from finbot.config.config_loader import ConfigLoader


def setup_strategies(config: ConfigLoader) -> List:
    """Setup trading strategies based on configuration."""
    strategies = []
    
    # VWAP-RSI-ATR Strategy
    if config.get("strategies.vwap_rsi_atr.enabled", False):
        strategy_config = {
            'enabled': True,
            'timeframe': config.get("strategies.vwap_rsi_atr.timeframe", "5m"),
            'rsi_period': config.get("strategies.vwap_rsi_atr.rsi_period", 14),
            'rsi_oversold': config.get("strategies.vwap_rsi_atr.rsi_oversold", 30),
            'rsi_overbought': config.get("strategies.vwap_rsi_atr.rsi_overbought", 70),
            'vwap_period': config.get("strategies.vwap_rsi_atr.vwap_period", 20),
            'atr_period': config.get("strategies.vwap_rsi_atr.atr_period", 14),
            'atr_multiplier': config.get("strategies.vwap_rsi_atr.atr_multiplier", 2.0),
            'max_position_size': config.get("risk.max_position_size", 0.1),
            'risk_per_trade': config.get("risk.max_daily_loss", 0.02)
        }
        strategies.append(VWAPRSIATRStrategy("VWAP_RSI_ATR", strategy_config))
    
    # Bollinger Bands Breakout Strategy
    if config.get("strategies.bollinger_breakout.enabled", False):
        strategy_config = {
            'enabled': True,
            'timeframe': config.get("strategies.bollinger_breakout.timeframe", "15m"),
            'period': config.get("strategies.bollinger_breakout.period", 20),
            'std_dev': config.get("strategies.bollinger_breakout.std_dev", 2),
            'lookback': config.get("strategies.bollinger_breakout.lookback", 5),
            'atr_period': config.get("strategies.bollinger_breakout.atr_period", 14),
            'atr_multiplier': config.get("strategies.bollinger_breakout.atr_multiplier", 2.0),
            'max_position_size': config.get("risk.max_position_size", 0.1),
            'risk_per_trade': config.get("risk.max_daily_loss", 0.02)
        }
        strategies.append(BollingerBreakoutStrategy("Bollinger_Breakout", strategy_config))
    
    # MACD Crossover Strategy
    if config.get("strategies.macd_crossover.enabled", False):
        strategy_config = {
            'enabled': True,
            'timeframe': config.get("strategies.macd_crossover.timeframe", "1h"),
            'fast_period': config.get("strategies.macd_crossover.fast_period", 12),
            'slow_period': config.get("strategies.macd_crossover.slow_period", 26),
            'signal_period': config.get("strategies.macd_crossover.signal_period", 9),
            'atr_period': config.get("strategies.macd_crossover.atr_period", 14),
            'atr_multiplier': config.get("strategies.macd_crossover.atr_multiplier", 2.0),
            'max_position_size': config.get("risk.max_position_size", 0.1),
            'risk_per_trade': config.get("risk.max_daily_loss", 0.02)
        }
        strategies.append(MACDCrossoverStrategy("MACD_Crossover", strategy_config))
    
    return strategies


def setup_broker(config: ConfigLoader, live_trading: bool = True):
    """Setup broker based on configuration."""
    
    if live_trading:
        # Live trading with Zerodha
        api_key = config.get("brokers.zerodha.api_key", "")
        api_secret = config.get("brokers.zerodha.api_secret", "")
        access_token = config.get("brokers.zerodha.access_token", "")
        
        if not api_key or not api_secret:
            print("Error: Zerodha API credentials not configured in config.yaml")
            print("Please add your Zerodha API credentials:")
            print("brokers:")
            print("  zerodha:")
            print("    api_key: 'your_api_key'")
            print("    api_secret: 'your_api_secret'")
            print("    access_token: 'your_access_token'")
            return None
        
        return ZerodhaBroker(api_key, api_secret, access_token)
    
    else:
        # Paper trading for backtesting
        return PaperBroker(
            initial_capital=config.get("backtest.initial_capital", 100000),
            commission=config.get("backtest.commission", 0.001),
            slippage=config.get("backtest.slippage", 0.0005)
        )


def run_backtest(config: ConfigLoader, symbols: List[str], start_date: datetime, end_date: datetime):
    """Run backtesting."""
    print(f"Running backtest from {start_date} to {end_date}")
    print(f"Symbols: {', '.join(symbols)}")
    
    # Setup backtest configuration
    backtest_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=config.get("backtest.initial_capital", 100000),
        commission=config.get("backtest.commission", 0.001),
        slippage=config.get("backtest.slippage", 0.0005)
    )
    
    # Setup strategies
    strategies = setup_strategies(config)
    if not strategies:
        print("No strategies enabled for backtesting")
        return
    
    # Load historical data
    data_loader = DataLoader(config)
    data = {}
    
    for symbol in symbols:
        try:
            symbol_data = data_loader.load_data(symbol, start_date, end_date, "1d")
            data[symbol] = symbol_data
            print(f"Loaded {len(symbol_data)} records for {symbol}")
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
    
    if not data:
        print("No data loaded for backtesting")
        return
    
    # Run backtest
    engine = BacktestEngine(backtest_config)
    
    if len(strategies) == 1:
        # Single strategy backtest
        results = engine.run_backtest(strategies[0], data, symbols)
        
        print("\n=== Backtest Results ===")
        print(f"Strategy: {results['strategy_name']}")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Annualized Return: {results['annualized_return']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print(f"Final Value: ₹{results['final_value']:,.2f}")
    
    else:
        # Multiple strategy comparison
        comparison = engine.compare_strategies(strategies, data, symbols)
        print("\n=== Strategy Comparison ===")
        print(comparison.to_string(index=False))


def run_live_trading(config: ConfigLoader, symbols: List[str]):
    """Run live trading with Zerodha."""
    print(f"Starting live trading with Zerodha for symbols: {', '.join(symbols)}")
    
    # Setup trading engine
    engine = TradingEngine(config)
    
    # Setup Zerodha broker
    broker = setup_broker(config, live_trading=True)
    if not broker:
        print("Failed to setup Zerodha broker")
        return
    
    engine.set_broker(broker)
    
    # Setup strategies
    strategies = setup_strategies(config)
    if not strategies:
        print("No strategies enabled for live trading")
        return
    
    for strategy in strategies:
        engine.add_strategy(strategy, strategy.name)
    
    # Setup monitoring
    logger = Logger(config)
    
    # Start engine
    engine.start()
    
    print("Trading engine started. Press Ctrl+C to stop.")
    
    try:
        # Main trading loop
        while True:
            # In a real implementation, this would fetch live market data
            # For now, we'll just keep the engine running
            import time
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nStopping trading engine...")
        engine.stop()
        print("Trading engine stopped.")


def run_dashboard(config: ConfigLoader):
    """Run monitoring dashboard."""
    print("Starting TradeBot dashboard...")
    
    # Setup trading engine (for demonstration)
    engine = TradingEngine(config)
    
    # Setup dashboard
    dashboard = Dashboard(config)
    
    if dashboard.enabled:
        print(f"Dashboard will be available at http://{dashboard.host}:{dashboard.port}")
        dashboard.run(engine)
    else:
        print("Dashboard is disabled in configuration")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="TradeBot - Modular Trading Bot Framework")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--mode", choices=["backtest", "live", "dashboard"], 
                       default="backtest", help="Operation mode")
    parser.add_argument("--symbols", nargs="+", default=["RELIANCE", "TCS", "HDFCBANK"],
                       help="Trading symbols")
    parser.add_argument("--start-date", type=str, default="2023-01-01",
                       help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2023-12-31",
                       help="Backtest end date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = ConfigLoader(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        sys.exit(1)
    
    # Run based on mode
    if args.mode == "backtest":
        run_backtest(config, args.symbols, start_date, end_date)
    
    elif args.mode == "live":
        run_live_trading(config, args.symbols)
    
    elif args.mode == "dashboard":
        run_dashboard(config)


if __name__ == "__main__":
    main()
