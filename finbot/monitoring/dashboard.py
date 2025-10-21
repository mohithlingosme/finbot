"""
Dashboard for TradeBot monitoring.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from finbot.utils.config_loader import ConfigLoader


class Dashboard:
    """
    Streamlit dashboard for TradeBot monitoring.
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize dashboard.
        
        Args:
            config: Configuration loader
        """
        self.config = config
        self.enabled = config.get("monitoring.dashboard.enabled", True)
        self.port = config.get("monitoring.dashboard.port", 8501)
        self.host = config.get("monitoring.dashboard.host", "localhost")

    def run(self, engine) -> None:
        """
        Run the dashboard.

        Args:
            engine: Trading engine instance
        """
        if not self.enabled:
            return
        
        st.set_page_config(
            page_title="TradeBot Dashboard",
            page_icon="📈",
            layout="wide"
        )

        st.title("📈 TradeBot Dashboard")

        # Sidebar
        self._create_sidebar(engine)

        # Main content
        self._create_overview(engine)
        self._create_portfolio_view(engine)
        self._create_performance_view(engine)
        self._create_trades_view(engine)
        self._create_strategies_view(engine)

    def _create_sidebar(self, engine) -> None:
        """Create sidebar with controls."""
        st.sidebar.header("Controls")

        # Engine status
        status = "🟢 Running" if engine.state.is_running else "🔴 Stopped"
        st.sidebar.metric("Engine Status", status)

        # Portfolio value
        portfolio_value = engine.portfolio.get_total_value()
        st.sidebar.metric("Portfolio Value", f"₹{portfolio_value:,.2f}")
        
        # Active strategies
        active_strategies = len(engine.active_strategies)
        st.sidebar.metric("Active Strategies", active_strategies)
        
        # Controls
        if st.sidebar.button("Start Engine"):
            from finbot.core.engine import TradingEngine
            engine.start()
            st.success("Engine started!")
        
        if st.sidebar.button("Stop Engine"):
            from finbot.core.engine import TradingEngine
            engine.stop()
            st.success("Engine stopped!")
    
    def _create_overview(self, engine) -> None:
        """Create overview section."""
        st.header("📊 Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_value = engine.portfolio.get_total_value()
            st.metric("Total Value", f"₹{total_value:,.2f}")
        
        with col2:
            cash = engine.portfolio.cash
            st.metric("Cash", f"₹{cash:,.2f}")
        
        with col3:
            unrealized_pnl = engine.portfolio.get_unrealized_pnl()
            st.metric("Unrealized P&L", f"₹{unrealized_pnl:,.2f}")
        
        with col4:
            realized_pnl = engine.portfolio.get_realized_pnl()
            st.metric("Realized P&L", f"₹{realized_pnl:,.2f}")
    
    def _create_portfolio_view(self, engine) -> None:
        """Create portfolio view."""
        st.header("💼 Portfolio")
        
        positions = engine.portfolio.get_positions_summary()
        
        if positions:
            df = pd.DataFrame(positions).T
            df = df.reset_index()
            df.columns = ['Symbol', 'Quantity', 'Avg Price', 'Current Price', 'Unrealized P&L', 'Value']
            
            st.dataframe(df, use_container_width=True)
            
            # Portfolio pie chart
            if len(positions) > 1:
                fig = px.pie(
                    df, 
                    values='Value', 
                    names='Symbol', 
                    title="Portfolio Allocation"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No positions currently held.")
    
    def _create_performance_view(self, engine) -> None:
        """Create performance view."""
        st.header("📈 Performance")
        
        # Performance metrics
        metrics = engine.get_performance_metrics()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_return = engine.portfolio.get_total_return() * 100
            st.metric("Total Return", f"{total_return:.2f}%")
        
        with col2:
            win_rate = engine.portfolio.get_win_rate() * 100
            st.metric("Win Rate", f"{win_rate:.2f}%")
        
        with col3:
            total_trades = engine.state.total_trades
            st.metric("Total Trades", total_trades)
        
        # Equity curve (if available)
        if hasattr(engine, 'equity_curve') and engine.equity_curve is not None:
            st.subheader("Equity Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=engine.equity_curve['timestamp'],
                y=engine.equity_curve['total_value'],
                mode='lines',
                name='Portfolio Value'
            ))
            fig.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Date",
                yaxis_title="Value (₹)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _create_trades_view(self, engine) -> None:
        """Create trades view."""
        st.header("💱 Recent Trades")
        
        trades = engine.portfolio.trade_history
        
        if trades:
            # Show last 20 trades
            recent_trades = trades[-20:]
            df = pd.DataFrame(recent_trades)
            
            # Format the dataframe
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp', ascending=False)
            
            st.dataframe(df, use_container_width=True)
            
            # P&L distribution
            if len(trades) > 1:
                st.subheader("P&L Distribution")
                fig = px.histogram(
                    df, 
                    x='pnl', 
                    nbins=20,
                    title="Trade P&L Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trades executed yet.")
    
    def _create_strategies_view(self, engine) -> None:
        """Create strategies view."""
        st.header("🎯 Strategies")
        
        for strategy in engine.active_strategies:
            with st.expander(f"Strategy: {strategy.name}"):
                # Strategy info
                info = strategy.get_strategy_info()
                st.json(info)
                
                # Strategy performance
                performance = strategy.get_performance_summary()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Trades", performance['total_trades'])
                
                with col2:
                    st.metric("Win Rate", f"{performance['win_rate']:.2f}%")
                
                with col3:
                    st.metric("Total P&L", f"₹{performance['total_pnl']:.2f}")
                
                with col4:
                    st.metric("Signals", performance['total_signals'])
                
                # Recent signals
                if strategy.signals:
                    st.subheader("Recent Signals")
                    recent_signals = strategy.signals[-10:]
                    signals_df = pd.DataFrame(recent_signals)
                    st.dataframe(signals_df, use_container_width=True)
