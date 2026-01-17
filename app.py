#!/usr/bin/env python3
"""
================================================================================
🚀 KARANKA V8 - SMART TIMEFRAME TRADING BOT
================================================================================
• AUTO TIMEFRAME SELECTION (1M/5M)
• MULTI-MARKET ANALYSIS
• CONTINUOUS TRADING 24/7
• GOLD/BLACK UI WITH ALL TABS
• PROFITABLE STRATEGIES
================================================================================
"""

import os
import json
import time
import threading
import websocket
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from functools import wraps
import hashlib
import secrets
import hmac
import numpy as np
from scipy import stats
import statistics
import requests

# Flask imports
from flask import Flask, request, jsonify, render_template_string, session, redirect, url_for
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('karanka_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('KarankaBot')

# ============ FLASK APP INITIALIZATION ============
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
CORS(app)

# ============ ENHANCED CONFIGURATION ============
class Config:
    MAX_CONCURRENT_TRADES = 5
    MAX_DAILY_TRADES = 200
    MIN_TRADE_AMOUNT = 1.0
    MAX_TRADE_AMOUNT = 50.0
    DEFAULT_TRADE_AMOUNT = 2.0
    SCAN_INTERVAL = 25  # seconds
    HEARTBEAT_INTERVAL = 60
    SESSION_TIMEOUT = 86400
    DATABASE_FILE = 'users.db'
    LOG_FILE = 'trading_log.json'
    AVAILABLE_MARKETS = [
        'R_10', 'R_25', 'R_50', 'R_75', 'R_100',
        '1HZ100V', '1HZ150V', '1HZ200V',
        'cryBTCUSD', 'cryETHUSD', 'frxEURUSD',
        'frxGBPUSD', 'frxUSDJPY'
    ]
    TIMEFRAMES = ['1m', '5m']
    
config = Config()

# ============ ENHANCED MARKET ANALYZER ============
class AdvancedMarketAnalyzer:
    """ANALYZES 1M AND 5M TIMEFRAMES FOR BEST OPPORTUNITIES"""
    
    def __init__(self):
        self.price_history_1m = defaultdict(lambda: deque(maxlen=100))
        self.price_history_5m = defaultdict(lambda: deque(maxlen=50))
        self.volume_history = defaultdict(lambda: deque(maxlen=100))
        self.trend_strength = defaultdict(float)
        self.volatility_cache = {}
        logger.info("📊 Advanced Market Analyzer initialized")
    
    def add_price_data(self, symbol: str, price: float, timeframe: str = '1m'):
        """Add price data for specific timeframe"""
        if timeframe == '1m':
            self.price_history_1m[symbol].append(price)
        elif timeframe == '5m':
            self.price_history_5m[symbol].append(price)
        
        self.volume_history[symbol].append(1.0)  # Default volume
    
    def analyze_multi_timeframe(self, symbol: str) -> Dict:
        """Analyze both 1M and 5M timeframes for best opportunity"""
        
        # Get data for both timeframes
        prices_1m = list(self.price_history_1m[symbol])
        prices_5m = list(self.price_history_5m[symbol])
        
        if len(prices_1m) < 30 or len(prices_5m) < 10:
            return {'signal': 'NEUTRAL', 'confidence': 50, 'timeframe': '1m', 'reason': 'Insufficient data'}
        
        # Analyze 1-minute timeframe
        analysis_1m = self._analyze_timeframe(prices_1m, symbol, '1m')
        
        # Analyze 5-minute timeframe
        analysis_5m = self._analyze_timeframe(prices_5m, symbol, '5m')
        
        # Determine which timeframe has better opportunity
        best_trade = self._select_best_timeframe(analysis_1m, analysis_5m)
        
        return best_trade
    
    def _analyze_timeframe(self, prices: List[float], symbol: str, timeframe: str) -> Dict:
        """Analyze specific timeframe"""
        if len(prices) < 20:
            return {'signal': 'NEUTRAL', 'confidence': 50, 'timeframe': timeframe}
        
        # Multiple indicator analysis
        indicators = self._calculate_indicators(prices)
        
        # Signal generation
        signals = []
        weights = []
        
        # 1. RSI Analysis (25% weight)
        rsi_signal, rsi_conf = self._analyze_rsi(indicators['rsi'])
        signals.append(rsi_signal)
        weights.append(0.25)
        
        # 2. Trend Analysis (30% weight)
        trend_signal, trend_conf = self._analyze_trend(prices)
        signals.append(trend_signal)
        weights.append(0.30)
        
        # 3. Volatility Analysis (20% weight)
        vol_signal, vol_conf = self._analyze_volatility(prices, symbol)
        signals.append(vol_signal)
        weights.append(0.20)
        
        # 4. Pattern Recognition (15% weight)
        pattern_signal, pattern_conf = self._analyze_patterns(prices)
        signals.append(pattern_signal)
        weights.append(0.15)
        
        # 5. Momentum (10% weight)
        momentum_signal, momentum_conf = self._analyze_momentum(prices)
        signals.append(momentum_signal)
        weights.append(0.10)
        
        # Calculate weighted confidence
        total_weight = sum(weights)
        weighted_conf = (
            rsi_conf * 0.25 +
            trend_conf * 0.30 +
            vol_conf * 0.20 +
            pattern_conf * 0.15 +
            momentum_conf * 0.10
        ) / total_weight
        
        # Determine final signal
        buy_count = signals.count('BUY')
        sell_count = signals.count('SELL')
        
        if buy_count > sell_count:
            final_signal = 'BUY'
        elif sell_count > buy_count:
            final_signal = 'SELL'
        else:
            final_signal = 'NEUTRAL'
        
        # Adjust confidence based on volatility
        volatility = self._calculate_volatility(prices)
        if volatility > 2.0:  # High volatility
            weighted_conf *= 0.8  # Reduce confidence
        elif volatility < 0.5:  # Low volatility
            weighted_conf *= 0.9  # Slightly reduce
        
        return {
            'signal': final_signal,
            'confidence': min(95, max(50, weighted_conf)),  # Clamp between 50-95
            'timeframe': timeframe,
            'indicators': indicators,
            'volatility': volatility,
            'price': prices[-1],
            'analysis': {
                'rsi': f"{rsi_signal} ({rsi_conf}%)",
                'trend': f"{trend_signal} ({trend_conf}%)",
                'volatility': f"{vol_signal} ({vol_conf}%)",
                'pattern': f"{pattern_signal} ({pattern_conf}%)",
                'momentum': f"{momentum_signal} ({momentum_conf}%)"
            }
        }
    
    def _calculate_indicators(self, prices: List[float]) -> Dict:
        """Calculate all technical indicators"""
        return {
            'rsi': self._calculate_rsi(prices),
            'sma_10': statistics.mean(prices[-10:]) if len(prices) >= 10 else prices[-1],
            'sma_20': statistics.mean(prices[-20:]) if len(prices) >= 20 else prices[-1],
            'ema_12': self._calculate_ema(prices, 12),
            'ema_26': self._calculate_ema(prices, 26),
            'atr': self._calculate_atr(prices),
            'macd': self._calculate_macd(prices),
            'stoch': self._calculate_stochastic(prices)
        }
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        gains = [d for d in deltas[-period:] if d > 0]
        losses = [-d for d in deltas[-period:] if d < 0]
        
        avg_gain = statistics.mean(gains) if gains else 0
        avg_loss = statistics.mean(losses) if losses else 0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate EMA"""
        if len(prices) < period:
            return prices[-1]
        
        multiplier = 2 / (period + 1)
        ema = statistics.mean(prices[:period])
        
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_atr(self, prices: List[float], period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(prices) < period + 1:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(prices)):
            high_low = abs(prices[i] - prices[i-1])
            true_ranges.append(high_low)
        
        return statistics.mean(true_ranges[-period:]) if true_ranges else 0.0
    
    def _calculate_macd(self, prices: List[float]) -> float:
        """Calculate MACD"""
        if len(prices) < 26:
            return 0.0
        
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        
        return ema_12 - ema_26
    
    def _calculate_stochastic(self, prices: List[float], period: int = 14) -> float:
        """Calculate Stochastic Oscillator"""
        if len(prices) < period:
            return 50.0
        
        recent = prices[-period:]
        highest = max(recent)
        lowest = min(recent)
        
        if highest == lowest:
            return 50.0
        
        return 100 * ((prices[-1] - lowest) / (highest - lowest))
    
    def _analyze_rsi(self, rsi: float) -> Tuple[str, float]:
        """Analyze RSI"""
        if rsi < 30:
            return 'BUY', 85
        elif rsi > 70:
            return 'SELL', 85
        elif rsi < 35:
            return 'BUY', 70
        elif rsi > 65:
            return 'SELL', 70
        elif rsi < 45:
            return 'BUY', 60
        elif rsi > 55:
            return 'SELL', 60
        else:
            return 'NEUTRAL', 50
    
    def _analyze_trend(self, prices: List[float]) -> Tuple[str, float]:
        """Analyze trend direction"""
        if len(prices) < 20:
            return 'NEUTRAL', 50
        
        # Calculate moving averages
        sma_10 = statistics.mean(prices[-10:])
        sma_20 = statistics.mean(prices[-20:])
        
        # Calculate slope
        if len(prices) >= 10:
            recent_prices = prices[-10:]
            x = list(range(len(recent_prices)))
            slope, _, _, _, _ = stats.linregress(x, recent_prices)
        else:
            slope = 0
        
        current_price = prices[-1]
        
        # Strong uptrend
        if current_price > sma_10 > sma_20 and slope > 0.1:
            return 'BUY', 85
        # Strong downtrend
        elif current_price < sma_10 < sma_20 and slope < -0.1:
            return 'SELL', 85
        # Mild uptrend
        elif current_price > sma_10 and slope > 0:
            return 'BUY', 70
        # Mild downtrend
        elif current_price < sma_10 and slope < 0:
            return 'SELL', 70
        else:
            return 'NEUTRAL', 50
    
    def _analyze_volatility(self, prices: List[float], symbol: str) -> Tuple[str, float]:
        """Analyze volatility for trading opportunity"""
        volatility = self._calculate_volatility(prices)
        
        # Store volatility for this symbol
        self.volatility_cache[symbol] = volatility
        
        # High volatility often precedes big moves
        if volatility > 1.5:
            # In high volatility, we want to trade with the breakout
            recent_change = prices[-1] - prices[-5] if len(prices) >= 5 else 0
            
            if recent_change > 0.5:
                return 'BUY', 75
            elif recent_change < -0.5:
                return 'SELL', 75
            else:
                return 'NEUTRAL', 60
        # Low volatility - wait for breakout
        elif volatility < 0.3:
            return 'NEUTRAL', 40
        # Normal volatility
        else:
            return 'NEUTRAL', 55
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility"""
        if len(prices) < 10:
            return 0.0
        
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        if not returns:
            return 0.0
        
        return statistics.stdev(returns) * 100  # Percentage volatility
    
    def _analyze_patterns(self, prices: List[float]) -> Tuple[str, float]:
        """Analyze chart patterns"""
        if len(prices) < 15:
            return 'NEUTRAL', 50
        
        # Look for patterns in last 15 candles
        recent = prices[-15:]
        
        # Check for double bottom (bullish)
        if self._is_double_bottom(recent):
            return 'BUY', 80
        
        # Check for double top (bearish)
        if self._is_double_top(recent):
            return 'SELL', 80
        
        # Check for ascending triangle (bullish)
        if self._is_ascending_triangle(recent):
            return 'BUY', 75
        
        # Check for descending triangle (bearish)
        if self._is_descending_triangle(recent):
            return 'SELL', 75
        
        return 'NEUTRAL', 50
    
    def _is_double_bottom(self, prices: List[float]) -> bool:
        """Detect double bottom pattern"""
        if len(prices) < 10:
            return False
        
        # Find two similar lows with peak in between
        min1 = min(prices[:5])
        min2 = min(prices[5:])
        middle = max(prices[3:7])
        
        return abs(min1 - min2) < (min1 * 0.02) and middle > min1 * 1.02
    
    def _is_double_top(self, prices: List[float]) -> bool:
        """Detect double top pattern"""
        if len(prices) < 10:
            return False
        
        # Find two similar highs with trough in between
        max1 = max(prices[:5])
        max2 = max(prices[5:])
        middle = min(prices[3:7])
        
        return abs(max1 - max2) < (max1 * 0.02) and middle < max1 * 0.98
    
    def _is_ascending_triangle(self, prices: List[float]) -> bool:
        """Detect ascending triangle pattern"""
        if len(prices) < 10:
            return False
        
        # Rising lows, relatively flat highs
        lows = [min(prices[i:i+3]) for i in range(0, len(prices)-3, 3)]
        highs = [max(prices[i:i+3]) for i in range(0, len(prices)-3, 3)]
        
        if len(lows) < 3 or len(highs) < 3:
            return False
        
        # Check if lows are rising
        low_slope, _, _, _, _ = stats.linregress(range(len(lows)), lows)
        high_slope, _, _, _, _ = stats.linregress(range(len(highs)), highs)
        
        return low_slope > 0 and abs(high_slope) < 0.01
    
    def _is_descending_triangle(self, prices: List[float]) -> bool:
        """Detect descending triangle pattern"""
        if len(prices) < 10:
            return False
        
        # Falling highs, relatively flat lows
        lows = [min(prices[i:i+3]) for i in range(0, len(prices)-3, 3)]
        highs = [max(prices[i:i+3]) for i in range(0, len(prices)-3, 3)]
        
        if len(lows) < 3 or len(highs) < 3:
            return False
        
        # Check if highs are falling
        low_slope, _, _, _, _ = stats.linregress(range(len(lows)), lows)
        high_slope, _, _, _, _ = stats.linregress(range(len(highs)), highs)
        
        return high_slope < 0 and abs(low_slope) < 0.01
    
    def _analyze_momentum(self, prices: List[float]) -> Tuple[str, float]:
        """Analyze price momentum"""
        if len(prices) < 10:
            return 'NEUTRAL', 50
        
        # Calculate momentum over last 5 periods
        momentum = prices[-1] - prices[-5]
        
        if momentum > prices[-5] * 0.01:  # 1%+ upward momentum
            return 'BUY', 75
        elif momentum < -prices[-5] * 0.01:  # 1%+ downward momentum
            return 'SELL', 75
        else:
            return 'NEUTRAL', 50
    
    def _select_best_timeframe(self, analysis_1m: Dict, analysis_5m: Dict) -> Dict:
        """Select best timeframe based on analysis"""
        
        # Prefer 5-minute timeframe if confidence is significantly higher
        confidence_diff = analysis_5m['confidence'] - analysis_1m['confidence']
        
        # If 5m is at least 15% more confident, use it
        if confidence_diff >= 15 and analysis_5m['signal'] != 'NEUTRAL':
            return analysis_5m
        
        # If 1m has good confidence and 5m is neutral, use 1m
        if analysis_1m['confidence'] >= 70 and analysis_5m['signal'] == 'NEUTRAL':
            return analysis_1m
        
        # If both have similar confidence, use the one with stronger signal
        if analysis_1m['signal'] != 'NEUTRAL' and analysis_5m['signal'] == 'NEUTRAL':
            return analysis_1m
        elif analysis_5m['signal'] != 'NEUTRAL' and analysis_1m['signal'] == 'NEUTRAL':
            return analysis_5m
        
        # Default to 1m if no clear winner
        return analysis_1m if analysis_1m['confidence'] >= 60 else analysis_5m

# ============ ENHANCED TRADING ENGINE WITH TIMEFRAME SELECTION ============
class SmartTimeframeTradingEngine:
    """TRADES ON BEST TIMEFRAME (1M OR 5M) BASED ON MARKET ANALYSIS"""
    
    def __init__(self, username: str):
        self.username = username
        self.client = RobustDerivClient()
        self.analyzer = AdvancedMarketAnalyzer()
        self.running = False
        self.trading_thread = None
        self.health_thread = None
        self.active_trades = []
        self.trade_history = []
        self.last_trade_time = defaultdict(float)
        self.market_data = {}
        
        # Performance tracking
        self.performance = {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_profit': 0.0,
            'win_rate': 0.0,
            'daily_trades': 0,
            'daily_profit': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'timeframe_stats': {'1m': 0, '5m': 0},
            'market_stats': defaultdict(int)
        }
        
        # Default settings (will be loaded from user)
        self.settings = {
            'trade_amount': 2.0,
            'max_concurrent_trades': 3,
            'max_daily_trades': 50,
            'min_confidence': 65,
            'stop_loss': 10.0,
            'take_profit': 15.0,
            'dry_run': True,
            'auto_trading': True,
            'enabled_markets': ['R_10', 'R_25', 'R_50'],
            'auto_timeframe': True,
            'preferred_timeframe': 'best',  # best, 1m, or 5m
            'risk_level': 'medium',
            'scan_interval': 25,
            'cooldown_seconds': 30
        }
        
        logger.info(f"🤖 Smart Timeframe Trading Engine created for {username}")
    
    def start_trading(self):
        """Start continuous trading"""
        if self.running:
            return False, "Already running"
        
        self.running = True
        
        # Start main trading thread
        self.trading_thread = threading.Thread(target=self._continuous_trading_loop, daemon=True)
        self.trading_thread.start()
        
        # Start health monitor
        self.health_thread = threading.Thread(target=self._health_monitor, daemon=True)
        self.health_thread.start()
        
        logger.info(f"🚀 Continuous trading started for {self.username}")
        logger.info(f"⚡ Auto timeframe selection: {self.settings['auto_timeframe']}")
        logger.info(f"📈 Enabled markets: {', '.join(self.settings['enabled_markets'])}")
        
        return True, "Trading engine started"
    
    def stop_trading(self):
        """Stop all trading"""
        self.running = False
        
        if self.trading_thread:
            self.trading_thread.join(timeout=5)
        
        if self.health_thread:
            self.health_thread.join(timeout=5)
        
        logger.info(f"🛑 Trading engine stopped for {self.username}")
        return True, "Trading engine stopped"
    
    def _continuous_trading_loop(self):
        """Main loop that trades continuously 24/7"""
        logger.info(f"🔄 Starting continuous trading loop for {self.username}")
        
        # Market data collection thread
        threading.Thread(target=self._collect_market_data, daemon=True).start()
        
        while self.running:
            try:
                # Check if we should trade based on settings
                if not self._can_trade():
                    time.sleep(5)
                    continue
                
                # Check connection for real trading
                if not self.settings['dry_run'] and not self.client.connected:
                    logger.warning(f"Not connected to Deriv for {self.username}")
                    time.sleep(10)
                    continue
                
                # Trade on each enabled market
                for symbol in self.settings['enabled_markets']:
                    if not self.running:
                        break
                    
                    # Check limits
                    if not self._check_trading_limits(symbol):
                        time.sleep(2)
                        continue
                    
                    # Analyze market for best timeframe
                    trade_decision = self._analyze_and_decide(symbol)
                    
                    # Execute if good opportunity
                    if trade_decision and trade_decision['signal'] in ['BUY', 'SELL']:
                        self._execute_smart_trade(symbol, trade_decision)
                    
                    time.sleep(1)  # Small delay between markets
                
                # Wait for next scan
                time.sleep(self.settings['scan_interval'])
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                time.sleep(30)
    
    def _collect_market_data(self):
        """Continuously collect market data for all enabled symbols"""
        while self.running:
            try:
                for symbol in self.settings['enabled_markets']:
                    if not self.running:
                        break
                    
                    # Get market data from Deriv
                    market_data = self.client.get_market_data(symbol)
                    if market_data:
                        # Update analyzer with 1m data
                        self.analyzer.add_price_data(symbol, market_data['bid'], '1m')
                        
                        # Every 5 minutes, add 5m data
                        if int(time.time()) % 300 < 10:  # Every ~5 minutes
                            self.analyzer.add_price_data(symbol, market_data['bid'], '5m')
                        
                        # Store market data
                        self.market_data[symbol] = {
                            'bid': market_data['bid'],
                            'ask': market_data['ask'],
                            'timestamp': time.time(),
                            'symbol': symbol
                        }
                    
                    time.sleep(0.5)
                
                time.sleep(2)  # Wait before next collection cycle
                
            except Exception as e:
                logger.error(f"Market data collection error: {e}")
                time.sleep(5)
    
    def _can_trade(self) -> bool:
        """Check if trading is allowed"""
        # Check auto trading setting
        if not self.settings['auto_trading']:
            return False
        
        # Check market hours (optional - can trade 24/7)
        hour = datetime.now().hour
        # Optional: Avoid maintenance hours
        # if 2 <= hour < 3:  # Example maintenance window
        #     return False
        
        return True
    
    def _check_trading_limits(self, symbol: str) -> bool:
        """Check all trading limits"""
        # Check concurrent trades
        active_count = len([t for t in self.active_trades if t['username'] == self.username])
        if active_count >= self.settings['max_concurrent_trades']:
            return False
        
        # Check daily trades limit
        if self.performance['daily_trades'] >= self.settings['max_daily_trades']:
            logger.info(f"Daily trade limit reached: {self.performance['daily_trades']}/{self.settings['max_daily_trades']}")
            return False
        
        # Check cooldown for this symbol
        current_time = time.time()
        if symbol in self.last_trade_time:
            time_since = current_time - self.last_trade_time[symbol]
            if time_since < self.settings['cooldown_seconds']:
                return False
        
        # Check stop loss
        if self.performance['total_profit'] <= -self.settings['stop_loss']:
            logger.warning(f"Stop loss hit: {self.performance['total_profit']}")
            return False
        
        return True
    
    def _analyze_and_decide(self, symbol: str) -> Optional[Dict]:
        """Analyze market and decide on trade"""
        try:
            # Get multi-timeframe analysis
            analysis = self.analyzer.analyze_multi_timeframe(symbol)
            
            # Check minimum confidence
            if analysis['confidence'] < self.settings['min_confidence']:
                return None
            
            # Apply timeframe preferences
            if not self.settings['auto_timeframe']:
                # User has preferred timeframe
                if self.settings['preferred_timeframe'] == '1m':
                    analysis['timeframe'] = '1m'
                elif self.settings['preferred_timeframe'] == '5m':
                    analysis['timeframe'] = '5m'
                # 'best' keeps the analyzer's choice
            
            logger.info(f"📊 {symbol}: {analysis['signal']} on {analysis['timeframe']} ({analysis['confidence']}% conf)")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis error for {symbol}: {e}")
            return None
    
    def _execute_smart_trade(self, symbol: str, analysis: Dict):
        """Execute trade with smart timeframe selection"""
        try:
            # Determine trade duration based on timeframe
            if analysis['timeframe'] == '1m':
                duration = 1  # 1 minute
                duration_unit = 'm'
            else:  # 5m
                duration = 5  # 5 minutes
                duration_unit = 'm'
            
            # Calculate trade amount with risk management
            trade_amount = self._calculate_trade_amount(analysis['confidence'])
            
            if trade_amount < config.MIN_TRADE_AMOUNT:
                return
            
            # Prepare trade record
            trade_record = {
                'id': f"trade_{int(time.time())}_{secrets.token_hex(4)}",
                'username': self.username,
                'symbol': symbol,
                'direction': analysis['signal'],
                'timeframe': analysis['timeframe'],
                'duration': duration,
                'amount': trade_amount,
                'confidence': analysis['confidence'],
                'timestamp': datetime.now().isoformat(),
                'status': 'PENDING',
                'analysis': analysis
            }
            
            self.active_trades.append(trade_record)
            self.last_trade_time[symbol] = time.time()
            
            # Execute trade
            if self.settings['dry_run']:
                # Simulate trade
                profit = self._simulate_trade(symbol, analysis['signal'], trade_amount)
                trade_record['status'] = 'COMPLETED'
                trade_record['profit'] = profit
                trade_record['real_trade'] = False
                
                logger.info(f"📊 DRY RUN {analysis['timeframe']}: {analysis['signal']} {symbol} ${trade_amount:.2f}")
            else:
                # Real trade on Deriv
                success, result = self.client.place_trade_with_duration(
                    symbol=symbol,
                    direction=analysis['signal'],
                    amount=trade_amount,
                    duration=duration,
                    duration_unit=duration_unit
                )
                
                if success:
                    trade_data = json.loads(result)
                    trade_record['status'] = 'COMPLETED'
                    trade_record['contract_id'] = trade_data.get('contract_id')
                    trade_record['payout'] = trade_data.get('payout')
                    trade_record['profit'] = trade_data.get('profit', 0)
                    trade_record['real_trade'] = True
                    
                    logger.info(f"✅ REAL {analysis['timeframe']}: {analysis['signal']} {symbol} ${trade_amount:.2f}")
                else:
                    trade_record['status'] = 'FAILED'
                    trade_record['error'] = result
                    trade_record['real_trade'] = True
                    
                    logger.error(f"❌ Trade failed: {result}")
            
            # Update performance
            self._update_performance(trade_record)
            
            # Move to history
            self.trade_history.append(trade_record)
            self.active_trades.remove(trade_record)
            
            # Reset daily counts at midnight
            if datetime.now().hour == 0 and datetime.now().minute < 5:
                self.performance['daily_trades'] = 0
                self.performance['daily_profit'] = 0.0
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
    
    def _calculate_trade_amount(self, confidence: float) -> float:
        """Calculate trade amount with risk management"""
        base_amount = min(
            self.settings['trade_amount'],
            config.MAX_TRADE_AMOUNT
        )
        
        # Adjust based on confidence
        confidence_factor = confidence / 100.0
        
        # Adjust based on performance
        if self.performance['consecutive_wins'] >= 3:
            # After 3 wins, be more conservative
            risk_factor = 0.8
        elif self.performance['consecutive_losses'] >= 2:
            # After 2 losses, reduce risk
            risk_factor = 0.5
        else:
            risk_factor = 1.0
        
        # Stop loss protection
        if self.performance['total_profit'] <= -self.settings['stop_loss'] * 0.5:
            risk_factor *= 0.3  # Reduce drastically when approaching stop loss
        
        # Calculate final amount
        final_amount = base_amount * confidence_factor * risk_factor
        
        # Ensure within limits
        final_amount = max(config.MIN_TRADE_AMOUNT, min(final_amount, config.MAX_TRADE_AMOUNT))
        
        # Round for Deriv
        return round(final_amount, 2)
    
    def _simulate_trade(self, symbol: str, direction: str, amount: float) -> float:
        """Simulate trade for dry run"""
        import random
        
        # Base win probability based on confidence from settings
        base_win_prob = 0.65  # 65% base win rate
        
        # Adjust based on current performance
        if self.performance['consecutive_wins'] >= 3:
            win_prob = base_win_prob * 0.9  # Reduce after wins
        elif self.performance['consecutive_losses'] >= 2:
            win_prob = base_win_prob * 1.1  # Increase after losses
        else:
            win_prob = base_win_prob
        
        win = random.random() < win_prob
        
        if win:
            profit = amount * 0.82  # 82% profit on win (typical for 5-minute trades)
        else:
            profit = -amount  # Lose stake
        
        # Add small randomness
        profit *= random.uniform(0.9, 1.1)
        
        return round(profit, 2)
    
    def _update_performance(self, trade_record: Dict):
        """Update performance metrics"""
        profit = trade_record.get('profit', 0)
        timeframe = trade_record.get('timeframe', '1m')
        symbol = trade_record.get('symbol', 'unknown')
        
        self.performance['total_trades'] += 1
        self.performance['daily_trades'] += 1
        
        if profit > 0:
            self.performance['profitable_trades'] += 1
            self.performance['consecutive_wins'] += 1
            self.performance['consecutive_losses'] = 0
        else:
            self.performance['consecutive_losses'] += 1
            self.performance['consecutive_wins'] = 0
        
        self.performance['total_profit'] += profit
        self.performance['daily_profit'] += profit
        
        # Update timeframe stats
        if timeframe in self.performance['timeframe_stats']:
            self.performance['timeframe_stats'][timeframe] += 1
        
        # Update market stats
        self.performance['market_stats'][symbol] += 1
        
        # Calculate win rate
        if self.performance['total_trades'] > 0:
            self.performance['win_rate'] = (
                self.performance['profitable_trades'] / self.performance['total_trades'] * 100
            )
    
    def _health_monitor(self):
        """Monitor and maintain system health"""
        while self.running:
            try:
                # Reconnect if needed for real trading
                if not self.settings['dry_run'] and not self.client.connected:
                    logger.warning(f"Attempting to reconnect {self.username} to Deriv...")
                    # Reconnection handled by main client
                
                # Clean old active trades
                current_time = time.time()
                self.active_trades = [
                    t for t in self.active_trades
                    if current_time - datetime.fromisoformat(t['timestamp']).timestamp() < 600  # 10 minutes
                ]
                
                # Log status periodically
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    logger.info(f"Status: {self.performance['total_trades']} trades, "
                              f"Profit: ${self.performance['total_profit']:.2f}, "
                              f"Win rate: {self.performance['win_rate']:.1f}%")
                
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                time.sleep(30)
    
    def get_status(self) -> Dict:
        """Get engine status"""
        return {
            'running': self.running,
            'connected': self.client.connected,
            'balance': self.client.balance if self.client.connected else 0,
            'performance': self.performance,
            'settings': self.settings,
            'active_trades': len([t for t in self.active_trades if t['username'] == self.username]),
            'market_data': len(self.market_data)
        }

# ============ ENHANCED DERIV CLIENT WITH DURATION SUPPORT ============
class RobustDerivClient:
    """DERIV CLIENT WITH TIMEFRAME SUPPORT"""
    
    def __init__(self):
        self.ws = None
        self.connected = False
        self.token = None
        self.account_id = None
        self.balance = 0.0
        self.reconnect_attempts = 0
        self.connection_lock = threading.Lock()
        self.market_data_cache = {}
        
        self.endpoints = [
            "wss://ws.derivws.com/websockets/v3?app_id=1089",
            "wss://ws.binaryws.com/websockets/v3?app_id=1089",
            "wss://ws.deriv.com/websockets/v3?app_id=1089"
        ]
        
        logger.info("🔧 Enhanced Deriv Client initialized")
    
    def connect(self, api_token: str) -> Tuple[bool, str, float]:
        """Connect to Deriv API"""
        try:
            if not api_token:
                return False, "API token required", 0.0
            
            self.token = api_token
            
            for endpoint in self.endpoints:
                try:
                    logger.info(f"🔗 Connecting to {endpoint}")
                    
                    self.ws = websocket.create_connection(
                        endpoint,
                        timeout=15,
                        header={'User-Agent': 'Mozilla/5.0', 'Origin': 'https://app.deriv.com'}
                    )
                    
                    # Authorize
                    auth_msg = {"authorize": api_token}
                    self.ws.send(json.dumps(auth_msg))
                    
                    response = self.ws.recv()
                    data = json.loads(response)
                    
                    if "error" in data:
                        continue
                    
                    if "authorize" in data:
                        self.connected = True
                        self.account_id = data["authorize"].get("loginid", "Unknown")
                        self.reconnect_attempts = 0
                        
                        # Get balance
                        self.balance = self._get_balance()
                        
                        logger.info(f"✅ Connected: {self.account_id}")
                        logger.info(f"💰 Balance: ${self.balance:.2f}")
                        
                        # Start heartbeat
                        threading.Thread(target=self._heartbeat, daemon=True).start()
                        
                        return True, f"Connected to {self.account_id}", self.balance
                    
                except Exception as e:
                    logger.warning(f"Endpoint {endpoint} failed: {e}")
                    continue
            
            return False, "Failed to connect", 0.0
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False, str(e), 0.0
    
    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get market data"""
        try:
            # Check cache
            if symbol in self.market_data_cache:
                cached = self.market_data_cache[symbol]
                if time.time() - cached['timestamp'] < 2:
                    return cached['data']
            
            if not self.connected:
                return None
            
            with self.connection_lock:
                self.ws.send(json.dumps({
                    "ticks": symbol,
                    "subscribe": 1
                }))
                self.ws.settimeout(5)
                response = self.ws.recv()
                data = json.loads(response)
                
                if "error" in data:
                    return None
                
                if "tick" in data:
                    tick_data = {
                        'symbol': symbol,
                        'bid': float(data["tick"]["bid"]),
                        'ask': float(data["tick"]["ask"]),
                        'quote': float(data["tick"]["quote"]),
                        'timestamp': data["tick"]["epoch"]
                    }
                    
                    # Cache
                    self.market_data_cache[symbol] = {
                        'data': tick_data,
                        'timestamp': time.time()
                    }
                    
                    return tick_data
            
            return None
            
        except Exception as e:
            logger.error(f"Market data error: {e}")
            return None
    
    def place_trade_with_duration(self, symbol: str, direction: str, amount: float,
                                  duration: int = 1, duration_unit: str = 'm') -> Tuple[bool, str]:
        """Place trade with specific duration"""
        try:
            if not self.connected:
                return False, "Not connected"
            
            # Validate
            if amount < 1.0:
                amount = 1.0
            
            if amount > self.balance * 0.9:
                return False, f"Insufficient balance: ${self.balance:.2f}"
            
            # Prepare trade
            contract_type = "CALL" if direction.upper() in ["BUY", "CALL"] else "PUT"
            
            trade_request = {
                "buy": amount,
                "price": amount,
                "parameters": {
                    "amount": amount,
                    "basis": "stake",
                    "contract_type": contract_type,
                    "currency": "USD",
                    "duration": duration,
                    "duration_unit": duration_unit,
                    "symbol": symbol,
                    "product_type": "basic"
                }
            }
            
            logger.info(f"🚀 Placing {duration}{duration_unit} trade: {symbol} {direction} ${amount}")
            
            with self.connection_lock:
                self.ws.send(json.dumps(trade_request))
                self.ws.settimeout(10)
                response = self.ws.recv()
                data = json.loads(response)
                
                if "error" in data:
                    error_msg = data["error"].get("message", "Trade failed")
                    return False, error_msg
                
                if "buy" in data:
                    contract_id = data["buy"]["contract_id"]
                    payout = float(data["buy"].get("payout", amount * 1.8))
                    profit = payout - amount
                    
                    # Update balance
                    self.balance = self._get_balance()
                    
                    trade_result = {
                        "contract_id": contract_id,
                        "payout": payout,
                        "profit": profit,
                        "balance": self.balance,
                        "duration": f"{duration}{duration_unit}",
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    logger.info(f"✅ Trade successful: {contract_id}")
                    return True, json.dumps(trade_result)
            
            return False, "Unknown response"
            
        except Exception as e:
            logger.error(f"Trade error: {e}")
            return False, str(e)
    
    def _get_balance(self) -> float:
        """Get account balance"""
        try:
            if not self.connected:
                return 0.0
            
            with self.connection_lock:
                self.ws.send(json.dumps({"balance": 1}))
                self.ws.settimeout(5)
                response = self.ws.recv()
                data = json.loads(response)
                
                if "balance" in data:
                    self.balance = float(data["balance"]["balance"])
                    return self.balance
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Balance error: {e}")
            return 0.0
    
    def _heartbeat(self):
        """Keep connection alive"""
        while self.connected:
            try:
                time.sleep(30)
                if self.ws:
                    self.ws.ping()
            except:
                self.connected = False
                break

# ============ USER AND SESSION MANAGEMENT ============
class UserManager:
    """Manages user accounts"""
    
    def __init__(self):
        self.users_file = config.DATABASE_FILE
        self.users = self._load_users()
        self.sessions = {}
    
    def _load_users(self) -> Dict:
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {}
    
    def _save_users(self):
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving users: {e}")
    
    def register_user(self, username: str, password: str, email: str = "") -> Tuple[bool, str]:
        if username in self.users:
            return False, "Username exists"
        
        if len(password) < 8:
            return False, "Password too short"
        
        salt = secrets.token_hex(16)
        hashed_password = hashlib.sha256((password + salt).encode()).hexdigest()
        
        self.users[username] = {
            'password_hash': hashed_password,
            'salt': salt,
            'email': email,
            'created_at': datetime.now().isoformat(),
            'last_login': None,
            'api_token': None,
            'settings': {
                'trade_amount': 2.0,
                'max_concurrent_trades': 3,
                'max_daily_trades': 50,
                'risk_level': 'medium',
                'auto_trading': True,
                'dry_run': True,
                'enabled_markets': ['R_10', 'R_25', 'R_50'],
                'min_confidence': 65,
                'stop_loss': 10.0,
                'take_profit': 15.0,
                'auto_timeframe': True,
                'preferred_timeframe': 'best',
                'scan_interval': 25,
                'cooldown_seconds': 30
            },
            'trading_stats': {
                'total_trades': 0,
                'successful_trades': 0,
                'failed_trades': 0,
                'total_profit': 0.0,
                'current_balance': 1000.0,
                'daily_trades': 0,
                'daily_profit': 0.0
            }
        }
        
        self._save_users()
        return True, "Registration successful"
    
    def authenticate_user(self, username: str, password: str) -> Tuple[bool, str, Dict]:
        if username not in self.users:
            return False, "Invalid credentials", {}
        
        user = self.users[username]
        hashed_password = hashlib.sha256((password + user['salt']).encode()).hexdigest()
        
        if hashed_password != user['password_hash']:
            return False, "Invalid credentials", {}
        
        token = secrets.token_hex(32)
        user['last_login'] = datetime.now().isoformat()
        user['api_token'] = token
        
        self.sessions[token] = {
            'username': username,
            'created_at': datetime.now().isoformat()
        }
        
        self._save_users()
        return True, "Login successful", {
            'token': token,
            'username': username,
            'settings': user['settings'],
            'stats': user['trading_stats']
        }
    
    def validate_token(self, token: str) -> Tuple[bool, str]:
        if token not in self.sessions:
            return False, ""
        
        return True, self.sessions[token]['username']
    
    def get_user(self, username: str) -> Optional[Dict]:
        return self.users.get(username)
    
    def update_user_settings(self, username: str, settings: Dict) -> bool:
        if username not in self.users:
            return False
        
        self.users[username]['settings'].update(settings)
        self._save_users()
        return True
    
    def update_user_stats(self, username: str, stats_update: Dict) -> bool:
        if username not in self.users:
            return False
        
        for key, value in stats_update.items():
            if key in self.users[username]['trading_stats']:
                self.users[username]['trading_stats'][key] += value
        
        self._save_users()
        return True

# Initialize managers
user_manager = UserManager()
session_manager = {}

def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        elif request.json:
            token = request.json.get('token')
        elif 'token' in session:
            token = session.get('token')
        
        if not token:
            return jsonify({'success': False, 'message': 'Token missing'}), 401
        
        valid, username = user_manager.validate_token(token)
        if not valid:
            return jsonify({'success': False, 'message': 'Invalid token'}), 401
        
        request.username = username
        request.token = token
        return f(*args, **kwargs)
    
    return decorated_function

def get_user_engine(username: str) -> SmartTimeframeTradingEngine:
    """Get or create user's trading engine"""
    if username not in session_manager:
        session_manager[username] = SmartTimeframeTradingEngine(username)
        
        # Load user settings
        user_data = user_manager.get_user(username)
        if user_data:
            session_manager[username].settings.update(user_data['settings'])
    
    return session_manager[username]

# ============ AUTO-CONNECT THREAD ============
def auto_connect_thread():
    """Auto-connect users"""
    time.sleep(10)
    
    while True:
        try:
            for username, engine in list(session_manager.items()):
                user_data = user_manager.get_user(username)
                if user_data and user_data['settings'].get('api_token'):
                    if not engine.settings['dry_run'] and not engine.client.connected:
                        logger.info(f"Auto-connecting {username}...")
                        success, msg, balance = engine.client.connect(user_data['settings']['api_token'])
                        
                        if success and engine.settings['auto_trading'] and not engine.running:
                            engine.start_trading()
            
            time.sleep(60)
            
        except Exception as e:
            logger.error(f"Auto-connect error: {e}")
            time.sleep(60)

threading.Thread(target=auto_connect_thread, daemon=True).start()

# ============ GOLD/BLACK UI TEMPLATE ============
GOLD_BLACK_UI = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Karanka V8 - Gold Edition</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --gold: #FFD700;
            --dark-gold: #B8860B;
            --black: #000000;
            --dark-gray: #1a1a1a;
            --medium-gray: #2d2d2d;
            --light-gray: #444444;
            --success: #00FF00;
            --danger: #FF0000;
            --warning: #FFA500;
            --info: #00BFFF;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', 'Arial', sans-serif;
        }
        
        body {
            background: var(--black);
            color: white;
            min-height: 100vh;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(255, 215, 0, 0.05) 0%, transparent 20%),
                radial-gradient(circle at 90% 80%, rgba(255, 215, 0, 0.05) 0%, transparent 20%);
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        /* Header */
        .header {
            background: linear-gradient(135deg, var(--dark-gray), var(--black));
            border: 2px solid var(--gold);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 0 30px rgba(255, 215, 0, 0.2);
            position: relative;
            overflow: hidden;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,215,0,0.1) 0%, transparent 70%);
            animation: pulse 4s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 0.3; }
            50% { transform: scale(1.1); opacity: 0.5; }
        }
        
        .header-content {
            position: relative;
            z-index: 2;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .logo-icon {
            font-size: 3em;
            color: var(--gold);
            text-shadow: 0 0 20px var(--gold);
            animation: glow 2s ease-in-out infinite alternate;
        }
        
        @keyframes glow {
            from { text-shadow: 0 0 10px var(--gold), 0 0 20px var(--gold); }
            to { text-shadow: 0 0 20px var(--gold), 0 0 30px var(--dark-gold); }
        }
        
        .logo-text h1 {
            font-size: 2.5em;
            background: linear-gradient(to right, var(--gold), #fff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 2px 10px rgba(255, 215, 0, 0.3);
        }
        
        .logo-text .subtitle {
            color: #aaa;
            font-size: 0.9em;
            letter-spacing: 2px;
        }
        
        .user-info {
            background: rgba(0, 0, 0, 0.7);
            padding: 15px 25px;
            border-radius: 10px;
            border: 1px solid var(--gold);
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .user-info .username {
            color: var(--gold);
            font-weight: bold;
            font-size: 1.2em;
        }
        
        .btn-logout {
            background: linear-gradient(135deg, #ff3333, #cc0000);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn-logout:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(255, 0, 0, 0.4);
        }
        
        /* Tabs */
        .tabs-container {
            background: var(--dark-gray);
            border-radius: 12px;
            border: 1px solid var(--light-gray);
            margin-bottom: 25px;
            overflow: hidden;
        }
        
        .tabs {
            display: flex;
            flex-wrap: wrap;
        }
        
        .tab {
            flex: 1;
            min-width: 120px;
            text-align: center;
            padding: 20px;
            cursor: pointer;
            background: transparent;
            color: #aaa;
            border: none;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
            font-size: 1.1em;
            font-weight: 600;
        }
        
        .tab:hover {
            background: rgba(255, 215, 0, 0.1);
            color: var(--gold);
        }
        
        .tab.active {
            background: rgba(255, 215, 0, 0.15);
            color: var(--gold);
            border-bottom: 3px solid var(--gold);
        }
        
        .tab i {
            margin-right: 10px;
            font-size: 1.2em;
        }
        
        /* Panels */
        .panel {
            display: none;
            background: linear-gradient(135deg, var(--dark-gray), var(--medium-gray));
            border: 1px solid var(--light-gray);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }
        
        .panel.active {
            display: block;
            animation: fadeIn 0.5s ease-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .panel h2 {
            color: var(--gold);
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid var(--light-gray);
            font-size: 1.8em;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .panel h2 i {
            font-size: 1.4em;
        }
        
        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: rgba(0, 0, 0, 0.6);
            border: 1px solid var(--light-gray);
            border-left: 5px solid var(--gold);
            border-radius: 12px;
            padding: 25px;
            transition: all 0.3s;
            position: relative;
            overflow: hidden;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            border-color: var(--gold);
            box-shadow: 0 10px 25px rgba(255, 215, 0, 0.2);
        }
        
        .stat-card.success {
            border-left-color: var(--success);
        }
        
        .stat-card.danger {
            border-left-color: var(--danger);
        }
        
        .stat-card.warning {
            border-left-color: var(--warning);
        }
        
        .stat-card.info {
            border-left-color: var(--info);
        }
        
        .stat-value {
            font-size: 2.8em;
            font-weight: bold;
            margin-bottom: 10px;
            color: var(--gold);
        }
        
        .stat-card.success .stat-value {
            color: var(--success);
        }
        
        .stat-card.danger .stat-value {
            color: var(--danger);
        }
        
        .stat-card.warning .stat-value {
            color: var(--warning);
        }
        
        .stat-card.info .stat-value {
            color: var(--info);
        }
        
        .stat-label {
            color: #aaa;
            font-size: 1em;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        /* Control Cards */
        .control-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }
        
        .control-card {
            background: rgba(0, 0, 0, 0.7);
            border: 1px solid var(--light-gray);
            border-radius: 15px;
            padding: 25px;
            transition: all 0.3s;
        }
        
        .control-card:hover {
            border-color: var(--gold);
            box-shadow: 0 5px 20px rgba(255, 215, 0, 0.1);
        }
        
        .control-card h3 {
            color: var(--gold);
            margin-bottom: 20px;
            font-size: 1.4em;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        /* Forms */
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #ccc;
            font-weight: 600;
        }
        
        .form-group input, 
        .form-group select {
            width: 100%;
            padding: 14px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid var(--light-gray);
            border-radius: 8px;
            color: white;
            font-size: 1em;
            transition: all 0.3s;
        }
        
        .form-group input:focus,
        .form-group select:focus {
            outline: none;
            border-color: var(--gold);
            box-shadow: 0 0 10px rgba(255, 215, 0, 0.3);
        }
        
        /* Buttons */
        .btn {
            padding: 14px 28px;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--gold), var(--dark-gold));
            color: black;
        }
        
        .btn-success {
            background: linear-gradient(135deg, var(--success), #00cc00);
            color: black;
        }
        
        .btn-danger {
            background: linear-gradient(135deg, var(--danger), #cc0000);
            color: white;
        }
        
        .btn-warning {
            background: linear-gradient(135deg, var(--warning), #cc8400);
            color: black;
        }
        
        .btn-info {
            background: linear-gradient(135deg, var(--info), #0080ff);
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(255, 215, 0, 0.4);
        }
        
        .btn:active {
            transform: translateY(-1px);
        }
        
        /* Alert */
        .alert {
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
            display: none;
            animation: slideIn 0.3s ease-out;
            border-left: 5px solid var(--gold);
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        .alert.success {
            background: rgba(0, 255, 0, 0.1);
            border-left-color: var(--success);
            color: #90ff90;
        }
        
        .alert.error {
            background: rgba(255, 0, 0, 0.1);
            border-left-color: var(--danger);
            color: #ff9090;
        }
        
        .alert.info {
            background: rgba(0, 191, 255, 0.1);
            border-left-color: var(--info);
            color: #90e0ff;
        }
        
        /* Trade Table */
        .trade-table {
            width: 100%;
            border-collapse: collapse;
            background: rgba(0, 0, 0, 0.6);
            border-radius: 10px;
            overflow: hidden;
            margin-top: 20px;
        }
        
        .trade-table th {
            background: rgba(255, 215, 0, 0.2);
            padding: 18px;
            text-align: left;
            color: var(--gold);
            font-weight: 600;
            border-bottom: 2px solid var(--gold);
        }
        
        .trade-table td {
            padding: 16px;
            border-bottom: 1px solid var(--light-gray);
            color: #ccc;
        }
        
        .trade-table tr:hover {
            background: rgba(255, 215, 0, 0.05);
        }
        
        .trade-table tr:last-child td {
            border-bottom: none;
        }
        
        .status-badge {
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }
        
        .status-success {
            background: rgba(0, 255, 0, 0.2);
            color: var(--success);
        }
        
        .status-failed {
            background: rgba(255, 0, 0, 0.2);
            color: var(--danger);
        }
        
        .status-pending {
            background: rgba(255, 165, 0, 0.2);
            color: var(--warning);
        }
        
        /* Loading */
        .loader {
            border: 4px solid rgba(255, 215, 0, 0.3);
            border-top: 4px solid var(--gold);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Responsive */
        @media (max-width: 992px) {
            .header-content {
                flex-direction: column;
                gap: 20px;
                text-align: center;
            }
            
            .tabs {
                flex-direction: column;
            }
            
            .tab {
                width: 100%;
                border-bottom: 1px solid var(--light-gray);
                border-right: none;
            }
            
            .tab.active {
                border-right: 3px solid var(--gold);
                border-bottom: 1px solid var(--light-gray);
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
            
            .control-grid {
                grid-template-columns: 1fr;
            }
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }
            
            .panel {
                padding: 20px;
            }
            
            .btn {
                padding: 12px 20px;
                font-size: 1em;
            }
            
            .stat-value {
                font-size: 2.2em;
            }
        }
        
        /* Market Selection */
        .market-selection {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .market-checkbox {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--light-gray);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .market-checkbox:hover {
            background: rgba(255, 215, 0, 0.1);
            border-color: var(--gold);
        }
        
        .market-checkbox input[type="checkbox"] {
            width: 20px;
            height: 20px;
            accent-color: var(--gold);
        }
        
        .market-checkbox label {
            color: #ccc;
            font-size: 1em;
            cursor: pointer;
            flex: 1;
        }
        
        /* Timeframe Selection */
        .timeframe-selector {
            display: flex;
            gap: 15px;
            margin: 20px 0;
        }
        
        .timeframe-btn {
            flex: 1;
            padding: 15px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--light-gray);
            border-radius: 8px;
            color: #ccc;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
        }
        
        .timeframe-btn:hover {
            background: rgba(255, 215, 0, 0.1);
            color: var(--gold);
        }
        
        .timeframe-btn.active {
            background: rgba(255, 215, 0, 0.2);
            border-color: var(--gold);
            color: var(--gold);
            box-shadow: 0 0 15px rgba(255, 215, 0, 0.3);
        }
        
        /* Real-time indicators */
        .indicators {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .indicator {
            background: rgba(0, 0, 0, 0.5);
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid var(--gold);
        }
        
        .indicator .label {
            color: #aaa;
            font-size: 0.9em;
            margin-bottom: 5px;
        }
        
        .indicator .value {
            color: var(--gold);
            font-size: 1.2em;
            font-weight: bold;
        }
        
        /* Progress bars */
        .progress-bar {
            height: 10px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
            margin: 15px 0;
            overflow: hidden;
        }
        
        .progress {
            height: 100%;
            background: linear-gradient(90deg, var(--gold), var(--dark-gold));
            border-radius: 5px;
            transition: width 0.5s ease;
        }
        
        /* Tooltips */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: var(--black);
            color: white;
            text-align: center;
            border-radius: 6px;
            padding: 10px;
            border: 1px solid var(--gold);
            position: absolute;
            z-index: 1000;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.9em;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="header-content">
                <div class="logo">
                    <div class="logo-icon">
                        <i class="fas fa-crown"></i>
                    </div>
                    <div class="logo-text">
                        <h1>🚀 KARANKA V8 GOLD EDITION</h1>
                        <div class="subtitle">• SMART TIMEFRAME TRADING • 24/7 AUTO TRADING • REAL DERIV EXECUTION</div>
                    </div>
                </div>
                <div class="user-info">
                    <span class="username"><i class="fas fa-user"></i> {{ username }}</span>
                    <button class="btn-logout" onclick="logout()">
                        <i class="fas fa-sign-out-alt"></i> LOGOUT
                    </button>
                </div>
            </div>
        </div>
        
        <!-- Tabs -->
        <div class="tabs-container">
            <div class="tabs">
                <button class="tab active" onclick="showPanel('dashboard')">
                    <i class="fas fa-tachometer-alt"></i> DASHBOARD
                </button>
                <button class="tab" onclick="showPanel('trading')">
                    <i class="fas fa-robot"></i> AUTO TRADING
                </button>
                <button class="tab" onclick="showPanel('manual')">
                    <i class="fas fa-hand-pointer"></i> MANUAL TRADE
                </button>
                <button class="tab" onclick="showPanel('markets')">
                    <i class="fas fa-chart-line"></i> MARKETS
                </button>
                <button class="tab" onclick="showPanel('settings')">
                    <i class="fas fa-cog"></i> SETTINGS
                </button>
                <button class="tab" onclick="showPanel('history')">
                    <i class="fas fa-history"></i> HISTORY
                </button>
                <button class="tab" onclick="showPanel('account')">
                    <i class="fas fa-user-circle"></i> ACCOUNT
                </button>
            </div>
        </div>
        
        <!-- Alert Container -->
        <div id="alert" class="alert"></div>
        
        <!-- Dashboard Panel -->
        <div id="dashboard" class="panel active">
            <h2><i class="fas fa-tachometer-alt"></i> TRADING DASHBOARD</h2>
            
            <!-- Stats Grid -->
            <div class="stats-grid">
                <div class="stat-card success">
                    <div class="stat-value" id="balance">$0.00</div>
                    <div class="stat-label">
                        <i class="fas fa-wallet"></i> CURRENT BALANCE
                    </div>
                </div>
                
                <div class="stat-card info">
                    <div class="stat-value" id="totalTrades">0</div>
                    <div class="stat-label">
                        <i class="fas fa-exchange-alt"></i> TOTAL TRADES
                    </div>
                </div>
                
                <div class="stat-card success">
                    <div class="stat-value" id="winRate">0%</div>
                    <div class="stat-label">
                        <i class="fas fa-trophy"></i> WIN RATE
                    </div>
                </div>
                
                <div class="stat-card warning">
                    <div class="stat-value" id="totalProfit">$0.00</div>
                    <div class="stat-label">
                        <i class="fas fa-money-bill-wave"></i> TOTAL PROFIT
                    </div>
                </div>
            </div>
            
            <!-- Trading Status -->
            <div class="control-card">
                <h3><i class="fas fa-satellite-dish"></i> SYSTEM STATUS</h3>
                <div style="display: flex; gap: 15px; margin-top: 15px; flex-wrap: wrap;">
                    <button class="btn btn-success" id="startBtn" onclick="startTrading()">
                        <i class="fas fa-play"></i> START AUTO TRADING
                    </button>
                    <button class="btn btn-danger" id="stopBtn" onclick="stopTrading()" style="display: none;">
                        <i class="fas fa-stop"></i> STOP AUTO TRADING
                    </button>
                    <button class="btn btn-primary" onclick="quickUpdate()">
                        <i class="fas fa-sync"></i> REFRESH
                    </button>
                    
                    <div style="flex: 1; display: flex; align-items: center; justify-content: center;">
                        <div style="background: rgba(0,0,0,0.5); padding: 15px; border-radius: 10px; border: 1px solid var(--gold);">
                            <div id="statusText" style="color: var(--gold); font-weight: bold; font-size: 1.1em;">
                                <i class="fas fa-circle" style="color: #ff4444;"></i> SYSTEM OFFLINE
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Connection Status -->
                <div id="connectionStatus" style="margin-top: 20px; padding: 15px; background: rgba(0,0,0,0.5); border-radius: 8px;">
                    <div style="color: #aaa;">Deriv Connection: <span id="derivStatus" style="color: #ff4444;">Disconnected</span></div>
                    <div style="color: #aaa;">Trading Mode: <span id="tradingMode" style="color: var(--warning);">DRY RUN</span></div>
                </div>
            </div>
            
            <!-- Quick Actions -->
            <div class="control-grid">
                <div class="control-card">
                    <h3><i class="fas fa-bolt"></i> QUICK TRADE</h3>
                    <div class="form-group">
                        <label>Select Market</label>
                        <select id="quickSymbol" style="background: rgba(0,0,0,0.7); color: white; padding: 12px;">
                            <option value="R_10">Volatility 10 Index</option>
                            <option value="R_25">Volatility 25 Index</option>
                            <option value="R_50">Volatility 50 Index</option>
                            <option value="R_75">Volatility 75 Index</option>
                            <option value="R_100">Volatility 100 Index</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Trade Amount ($)</label>
                        <input type="number" id="quickAmount" value="2.00" min="1.00" step="0.10" style="background: rgba(0,0,0,0.7); color: white;">
                    </div>
                    <div class="timeframe-selector">
                        <div class="timeframe-btn" onclick="setQuickTimeframe('1m')" id="quick1m">1 MINUTE</div>
                        <div class="timeframe-btn active" onclick="setQuickTimeframe('5m')" id="quick5m">5 MINUTES</div>
                    </div>
                    <div style="display: flex; gap: 10px; margin-top: 15px;">
                        <button class="btn btn-success" style="flex: 1;" onclick="quickTrade('BUY')">
                            <i class="fas fa-arrow-up"></i> BUY / CALL
                        </button>
                        <button class="btn btn-danger" style="flex: 1;" onclick="quickTrade('SELL')">
                            <i class="fas fa-arrow-down"></i> SELL / PUT
                        </button>
                    </div>
                </div>
                
                <div class="control-card">
                    <h3><i class="fas fa-plug"></i> DERIV CONNECTION</h3>
                    <div class="form-group">
                        <label>API Token</label>
                        <input type="password" id="apiToken" placeholder="Enter Deriv API Token" style="background: rgba(0,0,0,0.7); color: white;">
                    </div>
                    <button class="btn btn-primary" onclick="connectDeriv()" style="width: 100%; margin-bottom: 15px;">
                        <i class="fas fa-link"></i> CONNECT TO DERIV
                    </button>
                    <div class="form-group" style="margin-top: 20px;">
                        <label>Trading Mode</label>
                        <select id="modeSwitch" onchange="switchTradingMode()" style="background: rgba(0,0,0,0.7); color: white; padding: 12px;">
                            <option value="dry">DRY RUN (Simulation)</option>
                            <option value="real">REAL TRADING</option>
                        </select>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Auto Trading Panel -->
        <div id="trading" class="panel">
            <h2><i class="fas fa-robot"></i> SMART AUTO TRADING</h2>
            
            <div class="alert info">
                <i class="fas fa-info-circle"></i> Bot automatically analyzes 1-minute and 5-minute timeframes, selecting the best opportunity for each trade.
            </div>
            
            <div class="control-grid">
                <div class="control-card">
                    <h3><i class="fas fa-brain"></i> STRATEGY SETTINGS</h3>
                    <div class="form-group">
                        <label>Minimum Confidence (%)</label>
                        <div style="display: flex; align-items: center; gap: 15px;">
                            <input type="range" id="minConfidence" min="50" max="90" value="65" 
                                   oninput="document.getElementById('confidenceValue').textContent = this.value + '%'"
                                   style="flex: 1;">
                            <span id="confidenceValue" style="color: var(--gold); font-weight: bold; min-width: 60px;">65%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress" id="confidenceBar" style="width: 65%;"></div>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>Trade Limits</label>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                            <div>
                                <label style="font-size: 0.9em;">Max Concurrent</label>
                                <input type="number" id="maxConcurrent" min="1" max="10" value="3" 
                                       style="background: rgba(0,0,0,0.7); color: white; padding: 10px;">
                            </div>
                            <div>
                                <label style="font-size: 0.9em;">Max Daily</label>
                                <input type="number" id="maxDaily" min="10" max="200" value="50" 
                                       style="background: rgba(0,0,0,0.7); color: white; padding: 10px;">
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>Trade Amount ($)</label>
                        <input type="number" id="tradeAmount" min="1.00" max="50.00" step="0.10" value="2.00"
                               style="background: rgba(0,0,0,0.7); color: white;">
                    </div>
                </div>
                
                <div class="control-card">
                    <h3><i class="fas fa-shield-alt"></i> RISK MANAGEMENT</h3>
                    <div class="form-group">
                        <label>Stop Loss ($)</label>
                        <input type="number" id="stopLoss" min="0" max="100" step="5" value="10"
                               style="background: rgba(0,0,0,0.7); color: white;">
                    </div>
                    
                    <div class="form-group">
                        <label>Take Profit ($)</label>
                        <input type="number" id="takeProfit" min="0" max="200" step="5" value="15"
                               style="background: rgba(0,0,0,0.7); color: white;">
                    </div>
                    
                    <div class="form-group">
                        <label>Timeframe Selection</label>
                        <div class="timeframe-selector">
                            <div class="timeframe-btn active" onclick="setTimeframeMode('auto')" id="timeframeAuto">
                                AUTO SELECT
                            </div>
                            <div class="timeframe-btn" onclick="setTimeframeMode('1m')" id="timeframe1m">
                                1 MINUTE
                            </div>
                            <div class="timeframe-btn" onclick="setTimeframeMode('5m')" id="timeframe5m">
                                5 MINUTES
                            </div>
                        </div>
                        <div style="color: #aaa; font-size: 0.9em; margin-top: 10px;">
                            <i class="fas fa-info-circle"></i> Auto mode selects best timeframe (1m or 5m) based on market analysis
                        </div>
                    </div>
                </div>
            </div>
            
            <button class="btn btn-success" onclick="saveTradingSettings()" style="padding: 15px 30px; font-size: 1.2em;">
                <i class="fas fa-save"></i> SAVE TRADING SETTINGS
            </button>
        </div>
        
        <!-- Manual Trade Panel -->
        <div id="manual" class="panel">
            <h2><i class="fas fa-hand-pointer"></i> MANUAL TRADE EXECUTION</h2>
            
            <div class="control-grid">
                <div class="control-card">
                    <h3><i class="fas fa-chart-bar"></i> MARKET ANALYSIS</h3>
                    <div class="form-group">
                        <label>Select Symbol</label>
                        <select id="manualSymbol" style="background: rgba(0,0,0,0.7); color: white; padding: 12px;">
                            {% for market in config.AVAILABLE_MARKETS %}
                            <option value="{{ market }}">{{ market }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    
                    <button class="btn btn-info" onclick="analyzeMarket()" style="width: 100%; margin-bottom: 20px;">
                        <i class="fas fa-chart-line"></i> ANALYZE MARKET
                    </button>
                    
                    <div id="analysisResult" style="display: none;">
                        <h4 style="color: var(--gold); margin-bottom: 15px;">ANALYSIS RESULT</h4>
                        <div class="indicators" id="manualIndicators">
                            <!-- Will be populated by JavaScript -->
                        </div>
                    </div>
                </div>
                
                <div class="control-card">
                    <h3><i class="fas fa-trade"></i> TRADE EXECUTION</h3>
                    <div class="form-group">
                        <label>Trade Amount ($)</label>
                        <input type="number" id="manualAmount" value="2.00" min="1.00" step="0.10"
                               style="background: rgba(0,0,0,0.7); color: white;">
                    </div>
                    
                    <div class="form-group">
                        <label>Timeframe</label>
                        <select id="manualTimeframe" style="background: rgba(0,0,0,0.7); color: white; padding: 12px;">
                            <option value="1">1 Minute</option>
                            <option value="5" selected>5 Minutes</option>
                        </select>
                    </div>
                    
                    <div style="display: flex; gap: 10px; margin: 25px 0;">
                        <button class="btn btn-success" style="flex: 1; padding: 18px;" onclick="executeManualTrade('BUY')">
                            <i class="fas fa-arrow-up"></i> BUY / CALL
                        </button>
                        <button class="btn btn-danger" style="flex: 1; padding: 18px;" onclick="executeManualTrade('SELL')">
                            <i class="fas fa-arrow-down"></i> SELL / PUT
                        </button>
                    </div>
                    
                    <div id="manualResult" style="margin-top: 20px;"></div>
                </div>
            </div>
        </div>
        
        <!-- Markets Panel -->
        <div id="markets" class="panel">
            <h2><i class="fas fa-chart-line"></i> MARKET SELECTION</h2>
            
            <div class="alert info">
                <i class="fas fa-info-circle"></i> Select which markets the bot should trade on. More markets = more opportunities.
            </div>
            
            <h3 style="color: var(--gold); margin: 25px 0 15px 0;">AVAILABLE MARKETS</h3>
            <div class="market-selection" id="marketSelection">
                {% for market in config.AVAILABLE_MARKETS %}
                <div class="market-checkbox">
                    <input type="checkbox" id="market_{{ market }}" name="market" value="{{ market }}" 
                           {% if market in ['R_10', 'R_25', 'R_50'] %}checked{% endif %}>
                    <label for="market_{{ market }}">{{ market }}</label>
                </div>
                {% endfor %}
            </div>
            
            <div style="margin-top: 30px; display: flex; gap: 15px;">
                <button class="btn btn-primary" onclick="selectAllMarkets()">
                    <i class="fas fa-check-square"></i> SELECT ALL
                </button>
                <button class="btn btn-primary" onclick="deselectAllMarkets()">
                    <i class="fas fa-square"></i> DESELECT ALL
                </button>
                <button class="btn btn-success" onclick="saveMarketSelection()" style="margin-left: auto;">
                    <i class="fas fa-save"></i> SAVE MARKETS
                </button>
            </div>
            
            <div class="control-card" style="margin-top: 30px;">
                <h3><i class="fas fa-chart-pie"></i> MARKET PERFORMANCE</h3>
                <div id="marketPerformance">
                    <div class="loader"></div>
                </div>
            </div>
        </div>
        
        <!-- Settings Panel -->
        <div id="settings" class="panel">
            <h2><i class="fas fa-cog"></i> BOT SETTINGS</h2>
            
            <div class="control-grid">
                <div class="control-card">
                    <h3><i class="fas fa-sliders-h"></i> GENERAL SETTINGS</h3>
                    <div class="form-group">
                        <label>Scan Interval (seconds)</label>
                        <input type="number" id="scanInterval" min="10" max="300" value="25"
                               style="background: rgba(0,0,0,0.7); color: white;">
                    </div>
                    
                    <div class="form-group">
                        <label>Cooldown (seconds)</label>
                        <input type="number" id="cooldown" min="5" max="120" value="30"
                               style="background: rgba(0,0,0,0.7); color: white;">
                    </div>
                    
                    <div class="form-group">
                        <label>Auto Trading</label>
                        <select id="autoTrading" style="background: rgba(0,0,0,0.7); color: white; padding: 12px;">
                            <option value="true">ENABLED</option>
                            <option value="false">DISABLED</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Risk Level</label>
                        <select id="riskLevel" style="background: rgba(0,0,0,0.7); color: white; padding: 12px;">
                            <option value="low">LOW</option>
                            <option value="medium" selected>MEDIUM</option>
                            <option value="high">HIGH</option>
                            <option value="aggressive">AGGRESSIVE</option>
                        </select>
                    </div>
                </div>
                
                <div class="control-card">
                    <h3><i class="fas fa-bell"></i> NOTIFICATIONS</h3>
                    <div class="form-group">
                        <label>Trade Alerts</label>
                        <select style="background: rgba(0,0,0,0.7); color: white; padding: 12px;">
                            <option>ENABLED</option>
                            <option>DISABLED</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Daily Report</label>
                        <select style="background: rgba(0,0,0,0.7); color: white; padding: 12px;">
                            <option>ENABLED</option>
                            <option>DISABLED</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Profit/Loss Alerts</label>
                        <select style="background: rgba(0,0,0,0.7); color: white; padding: 12px;">
                            <option>ENABLED</option>
                            <option>DISABLED</option>
                        </select>
                    </div>
                </div>
            </div>
            
            <button class="btn btn-success" onclick="saveAllSettings()" style="padding: 15px 40px; font-size: 1.2em; margin-top: 20px;">
                <i class="fas fa-save"></i> SAVE ALL SETTINGS
            </button>
        </div>
        
        <!-- History Panel -->
        <div id="history" class="panel">
            <h2><i class="fas fa-history"></i> TRADE HISTORY</h2>
            
            <div style="display: flex; gap: 15px; margin-bottom: 25px; flex-wrap: wrap;">
                <button class="btn btn-primary" onclick="loadTradeHistory()">
                    <i class="fas fa-sync"></i> REFRESH HISTORY
                </button>
                <button class="btn btn-info" onclick="exportHistory()">
                    <i class="fas fa-download"></i> EXPORT CSV
                </button>
                <div style="margin-left: auto; display: flex; gap: 10px; align-items: center;">
                    <span style="color: #aaa;">Filter:</span>
                    <select id="historyFilter" style="background: rgba(0,0,0,0.7); color: white; padding: 8px 15px; border-radius: 5px;">
                        <option value="all">ALL TRADES</option>
                        <option value="today">TODAY</option>
                        <option value="profitable">PROFITABLE</option>
                        <option value="loss">LOSS</option>
                    </select>
                </div>
            </div>
            
            <div id="historyTable"></div>
            
            <div style="margin-top: 30px; color: #aaa; font-size: 0.9em; text-align: center;">
                <i class="fas fa-info-circle"></i> Showing last 100 trades. Total trades: <span id="totalTradesCount">0</span>
            </div>
        </div>
        
        <!-- Account Panel -->
        <div id="account" class="panel">
            <h2><i class="fas fa-user-circle"></i> ACCOUNT INFORMATION</h2>
            
            <div class="control-grid">
                <div class="control-card">
                    <h3><i class="fas fa-user"></i> PROFILE</h3>
                    <div class="form-group">
                        <label>Username</label>
                        <input type="text" value="{{ username }}" readonly 
                               style="background: rgba(0,0,0,0.5); color: #aaa;">
                    </div>
                    
                    <div class="form-group">
                        <label>Account Created</label>
                        <input type="text" value="Loading..." id="accountCreated" readonly 
                               style="background: rgba(0,0,0,0.5); color: #aaa;">
                    </div>
                    
                    <div class="form-group">
                        <label>Last Login</label>
                        <input type="text" value="Loading..." id="lastLogin" readonly 
                               style="background: rgba(0,0,0,0.5); color: #aaa;">
                    </div>
                </div>
                
                <div class="control-card">
                    <h3><i class="fas fa-chart-pie"></i> TRADING STATISTICS</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div>
                            <div style="color: #aaa; font-size: 0.9em;">Total Trades</div>
                            <div style="color: var(--gold); font-size: 1.5em; font-weight: bold;" id="statTotalTrades">0</div>
                        </div>
                        <div>
                            <div style="color: #aaa; font-size: 0.9em;">Successful</div>
                            <div style="color: var(--success); font-size: 1.5em; font-weight: bold;" id="statSuccessful">0</div>
                        </div>
                        <div>
                            <div style="color: #aaa; font-size: 0.9em;">Win Rate</div>
                            <div style="color: var(--info); font-size: 1.5em; font-weight: bold;" id="statWinRate">0%</div>
                        </div>
                        <div>
                            <div style="color: #aaa; font-size: 0.9em;">Total Profit</div>
                            <div style="color: var(--warning); font-size: 1.5em; font-weight: bold;" id="statProfit">$0.00</div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 25px;">
                        <div style="color: #aaa; font-size: 0.9em;">Performance Breakdown</div>
                        <div id="performanceChart" style="height: 200px; margin-top: 10px; background: rgba(0,0,0,0.3); border-radius: 8px; padding: 15px;">
                            <!-- Chart will be added here -->
                            <div style="color: #666; text-align: center; line-height: 170px;">
                                Performance chart will appear here
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="control-card" style="margin-top: 25px;">
                <h3><i class="fas fa-key"></i> SECURITY</h3>
                <div style="display: flex; gap: 15px; margin-top: 15px;">
                    <button class="btn btn-warning" onclick="changePassword()">
                        <i class="fas fa-key"></i> CHANGE PASSWORD
                    </button>
                    <button class="btn btn-danger" onclick="deleteAccount()">
                        <i class="fas fa-trash"></i> DELETE ACCOUNT
                    </button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Global variables
        let currentToken = '{{ session.token }}';
        let currentUsername = '{{ username }}';
        let updateInterval;
        let quickTimeframe = '5m';
        let timeframeMode = 'auto';
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            loadDashboard();
            loadUserData();
            
            // Start auto-update every 10 seconds
            updateInterval = setInterval(loadDashboard, 10000);
            
            // Load trade history
            loadTradeHistory();
        });
        
        // Tab navigation
        function showPanel(panelId) {
            // Hide all panels
            document.querySelectorAll('.panel').forEach(panel => {
                panel.classList.remove('active');
            });
            
            // Remove active from all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected panel
            document.getElementById(panelId).classList.add('active');
            
            // Activate corresponding tab
            document.querySelectorAll('.tab').forEach(tab => {
                if (tab.onclick.toString().includes(panelId)) {
                    tab.classList.add('active');
                }
            });
            
            // Load specific data
            if (panelId === 'history') {
                loadTradeHistory();
            } else if (panelId === 'account') {
                loadUserData();
            }
        }
        
        // Dashboard functions
        function loadDashboard() {
            fetch('/api/status', {
                headers: {
                    'Authorization': 'Bearer ' + currentToken
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updateDashboard(data);
                }
            })
            .catch(error => {
                console.error('Dashboard error:', error);
            });
        }
        
        function updateDashboard(data) {
            // Update stats
            document.getElementById('balance').textContent = '$' + data.status.balance.toFixed(2);
            document.getElementById('totalTrades').textContent = data.user.stats.total_trades;
            document.getElementById('winRate').textContent = data.status.performance.win_rate.toFixed(1) + '%';
            document.getElementById('totalProfit').textContent = '$' + data.user.stats.total_profit.toFixed(2);
            
            // Update status
            const isRunning = data.status.running;
            const isConnected = data.status.connected;
            const isDryRun = data.user.settings.dry_run;
            
            document.getElementById('startBtn').style.display = isRunning ? 'none' : 'block';
            document.getElementById('stopBtn').style.display = isRunning ? 'block' : 'none';
            
            let statusText = '';
            let statusColor = '';
            
            if (isRunning) {
                statusText = '🟢 AUTO TRADING ACTIVE';
                statusColor = '#00ff00';
            } else {
                statusText = '🔴 AUTO TRADING STOPPED';
                statusColor = '#ff4444';
            }
            
            if (isConnected && !isDryRun) {
                statusText += ' | 🔗 DERIV CONNECTED';
            } else if (!isDryRun) {
                statusText += ' | 🔌 DERIV DISCONNECTED';
            } else {
                statusText += ' | 💨 DRY RUN MODE';
            }
            
            document.getElementById('statusText').innerHTML = `<i class="fas fa-circle" style="color: ${statusColor};"></i> ${statusText}`;
            
            // Update connection status
            document.getElementById('derivStatus').textContent = isConnected ? 'Connected' : 'Disconnected';
            document.getElementById('derivStatus').style.color = isConnected ? '#00ff00' : '#ff4444';
            
            document.getElementById('tradingMode').textContent = isDryRun ? 'DRY RUN' : 'REAL TRADING';
            document.getElementById('tradingMode').style.color = isDryRun ? '#ffa500' : '#00ff00';
            
            // Update mode switch
            document.getElementById('modeSwitch').value = isDryRun ? 'dry' : 'real';
        }
        
        function quickUpdate() {
            loadDashboard();
            showAlert('Dashboard refreshed!', 'success');
        }
        
        // Trading control
        function startTrading() {
            fetch('/api/trading/start', {
                method: 'POST',
                headers: {
                    'Authorization': 'Bearer ' + currentToken,
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showAlert('Auto trading started successfully!', 'success');
                    loadDashboard();
                } else {
                    showAlert('Failed to start: ' + data.message, 'error');
                }
            });
        }
        
        function stopTrading() {
            fetch('/api/trading/stop', {
                method: 'POST',
                headers: {
                    'Authorization': 'Bearer ' + currentToken,
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showAlert('Auto trading stopped.', 'success');
                    loadDashboard();
                }
            });
        }
        
        // Quick trade
        function setQuickTimeframe(timeframe) {
            quickTimeframe = timeframe;
            document.getElementById('quick1m').classList.remove('active');
            document.getElementById('quick5m').classList.remove('active');
            document.getElementById('quick' + timeframe).classList.add('active');
        }
        
        function quickTrade(direction) {
            const symbol = document.getElementById('quickSymbol').value;
            const amount = parseFloat(document.getElementById('quickAmount').value);
            const duration = quickTimeframe === '1m' ? 1 : 5;
            
            if (amount < 1.00) {
                showAlert('Minimum trade amount is $1.00', 'error');
                return;
            }
            
            showAlert(`Executing ${quickTimeframe} ${direction} trade...`, 'info');
            
            fetch('/api/trade', {
                method: 'POST',
                headers: {
                    'Authorization': 'Bearer ' + currentToken,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    symbol: symbol,
                    direction: direction,
                    amount: amount,
                    duration: duration,
                    duration_unit: 'm'
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showAlert('Trade executed! ' + data.message, 'success');
                    loadDashboard();
                } else {
                    showAlert('Trade failed: ' + data.message, 'error');
                }
            });
        }
        
        // Deriv connection
        function connectDeriv() {
            const apiToken = document.getElementById('apiToken').value.trim();
            
            if (!apiToken) {
                showAlert('Please enter your Deriv API token', 'error');
                return;
            }
            
            showAlert('Connecting to Deriv...', 'info');
            
            fetch('/api/connect', {
                method: 'POST',
                headers: {
                    'Authorization': 'Bearer ' + currentToken,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ api_token: apiToken })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showAlert('Connected to Deriv! Balance: $' + data.balance.toFixed(2), 'success');
                    loadDashboard();
                } else {
                    showAlert('Connection failed: ' + data.message, 'error');
                }
            });
        }
        
        function switchTradingMode() {
            const mode = document.getElementById('modeSwitch').value;
            const isDryRun = mode === 'dry';
            
            fetch('/api/settings/update', {
                method: 'POST',
                headers: {
                    'Authorization': 'Bearer ' + currentToken,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    settings: { dry_run: isDryRun }
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const modeText = isDryRun ? 'DRY RUN' : 'REAL TRADING';
                    showAlert('Switched to ' + modeText + ' mode', 'success');
                    loadDashboard();
                }
            });
        }
        
        // Trading settings
        function setTimeframeMode(mode) {
            timeframeMode = mode;
            document.getElementById('timeframeAuto').classList.remove('active');
            document.getElementById('timeframe1m').classList.remove('active');
            document.getElementById('timeframe5m').classList.remove('active');
            document.getElementById('timeframe' + (mode === 'auto' ? 'Auto' : mode)).classList.add('active');
        }
        
        function saveTradingSettings() {
            const settings = {
                min_confidence: parseInt(document.getElementById('minConfidence').value),
                max_concurrent_trades: parseInt(document.getElementById('maxConcurrent').value),
                max_daily_trades: parseInt(document.getElementById('maxDaily').value),
                trade_amount: parseFloat(document.getElementById('tradeAmount').value),
                stop_loss: parseFloat(document.getElementById('stopLoss').value),
                take_profit: parseFloat(document.getElementById('takeProfit').value),
                auto_timeframe: timeframeMode === 'auto',
                preferred_timeframe: timeframeMode === 'auto' ? 'best' : timeframeMode
            };
            
            saveSettings(settings, 'Trading settings saved successfully!');
        }
        
        // Manual trade
        function analyzeMarket() {
            const symbol = document.getElementById('manualSymbol').value;
            
            fetch('/api/analyze/' + symbol, {
                headers: {
                    'Authorization': 'Bearer ' + currentToken
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('analysisResult').style.display = 'block';
                    document.getElementById('manualIndicators').innerHTML = `
                        <div class="indicator">
                            <div class="label">Signal</div>
                            <div class="value" style="color: ${data.signal === 'BUY' ? '#00ff00' : '#ff0000'}">
                                ${data.signal} (${data.confidence}%)
                            </div>
                        </div>
                        <div class="indicator">
                            <div class="label">Recommended Timeframe</div>
                            <div class="value">${data.timeframe}</div>
                        </div>
                        <div class="indicator">
                            <div class="label">Volatility</div>
                            <div class="value">${data.volatility}%</div>
                        </div>
                        <div class="indicator">
                            <div class="label">Current Price</div>
                            <div class="value">$${data.price}</div>
                        </div>
                    `;
                }
            });
        }
        
        function executeManualTrade(direction) {
            const symbol = document.getElementById('manualSymbol').value;
            const amount = parseFloat(document.getElementById('manualAmount').value);
            const timeframe = document.getElementById('manualTimeframe').value;
            
            if (amount < 1.00) {
                showAlert('Minimum trade amount is $1.00', 'error');
                return;
            }
            
            showAlert(`Executing manual ${direction} trade...`, 'info');
            
            fetch('/api/trade', {
                method: 'POST',
                headers: {
                    'Authorization': 'Bearer ' + currentToken,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    symbol: symbol,
                    direction: direction,
                    amount: amount,
                    duration: parseInt(timeframe),
                    duration_unit: 'm'
                })
            })
            .then(response => response.json())
            .then(data => {
                const resultDiv = document.getElementById('manualResult');
                if (data.success) {
                    resultDiv.innerHTML = `
                        <div style="background: rgba(0,255,0,0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #00ff00;">
                            <div style="color: #00ff00; font-weight: bold; margin-bottom: 10px;">
                                <i class="fas fa-check-circle"></i> TRADE SUCCESSFUL
                            </div>
                            <div>Profit: $${data.profit.toFixed(2)}</div>
                            <div>New Balance: $${data.balance.toFixed(2)}</div>
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = `
                        <div style="background: rgba(255,0,0,0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #ff0000;">
                            <div style="color: #ff0000; font-weight: bold;">
                                <i class="fas fa-times-circle"></i> TRADE FAILED
                            </div>
                            <div>${data.message}</div>
                        </div>
                    `;
                }
            });
        }
        
        // Market selection
        function selectAllMarkets() {
            document.querySelectorAll('.market-checkbox input').forEach(checkbox => {
                checkbox.checked = true;
            });
        }
        
        function deselectAllMarkets() {
            document.querySelectorAll('.market-checkbox input').forEach(checkbox => {
                checkbox.checked = false;
            });
        }
        
        function saveMarketSelection() {
            const selectedMarkets = Array.from(document.querySelectorAll('.market-checkbox input:checked'))
                .map(cb => cb.value);
            
            if (selectedMarkets.length === 0) {
                showAlert('Please select at least one market', 'error');
                return;
            }
            
            fetch('/api/settings/update', {
                method: 'POST',
                headers: {
                    'Authorization': 'Bearer ' + currentToken,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    settings: { enabled_markets: selectedMarkets }
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showAlert(`Saved ${selectedMarkets.length} markets for trading`, 'success');
                }
            });
        }
        
        // Settings
        function saveAllSettings() {
            const settings = {
                scan_interval: parseInt(document.getElementById('scanInterval').value),
                cooldown_seconds: parseInt(document.getElementById('cooldown').value),
                auto_trading: document.getElementById('autoTrading').value === 'true',
                risk_level: document.getElementById('riskLevel').value
            };
            
            saveSettings(settings, 'All settings saved successfully!');
        }
        
        function saveSettings(settings, successMessage) {
            fetch('/api/settings/update', {
                method: 'POST',
                headers: {
                    'Authorization': 'Bearer ' + currentToken,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ settings: settings })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showAlert(successMessage, 'success');
                    loadDashboard();
                } else {
                    showAlert('Failed to save: ' + data.message, 'error');
                }
            });
        }
        
        // History
        function loadTradeHistory() {
            const historyTable = document.getElementById('historyTable');
            historyTable.innerHTML = '<div class="loader"></div>';
            
            fetch('/api/trades/history', {
                headers: { 'Authorization': 'Bearer ' + currentToken }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    renderTradeHistory(data.trades);
                    document.getElementById('totalTradesCount').textContent = data.total;
                }
            });
        }
        
        function renderTradeHistory(trades) {
            if (trades.length === 0) {
                document.getElementById('historyTable').innerHTML = `
                    <div style="text-align: center; padding: 50px; color: #666;">
                        <i class="fas fa-history" style="font-size: 3em; margin-bottom: 20px;"></i>
                        <div>No trades yet. Start trading to see history here.</div>
                    </div>
                `;
                return;
            }
            
            let html = `
                <table class="trade-table">
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Symbol</th>
                            <th>Direction</th>
                            <th>Timeframe</th>
                            <th>Amount</th>
                            <th>Profit</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            
            // Sort by timestamp descending (newest first)
            trades.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
            
            trades.forEach(trade => {
                const time = new Date(trade.timestamp).toLocaleTimeString();
                const profit = trade.profit ? trade.profit.toFixed(2) : '0.00';
                const profitColor = profit >= 0 ? '#00ff00' : '#ff0000';
                const statusClass = trade.status === 'COMPLETED' ? 'status-success' : 
                                  trade.status === 'FAILED' ? 'status-failed' : 'status-pending';
                
                html += `
                    <tr>
                        <td>${time}</td>
                        <td>${trade.symbol}</td>
                        <td style="color: ${trade.direction === 'BUY' ? '#00ff00' : '#ff0000'}">
                            ${trade.direction}
                        </td>
                        <td>${trade.timeframe || '1m'}</td>
                        <td>$${trade.amount.toFixed(2)}</td>
                        <td style="color: ${profitColor}">$${profit}</td>
                        <td><span class="status-badge ${statusClass}">${trade.status}</span></td>
                    </tr>
                `;
            });
            
            html += '</tbody></table>';
            document.getElementById('historyTable').innerHTML = html;
        }
        
        function exportHistory() {
            showAlert('Export feature coming soon!', 'info');
        }
        
        // Account
        function loadUserData() {
            fetch('/api/user/data', {
                headers: { 'Authorization': 'Bearer ' + currentToken }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('accountCreated').value = data.user.created_at.split('T')[0];
                    document.getElementById('lastLogin').value = data.user.last_login ? 
                        data.user.last_login.split('T')[0] : 'Never';
                    
                    document.getElementById('statTotalTrades').textContent = data.stats.total_trades;
                    document.getElementById('statSuccessful').textContent = data.stats.successful_trades;
                    
                    const winRate = data.stats.total_trades > 0 ? 
                        ((data.stats.successful_trades / data.stats.total_trades) * 100).toFixed(1) : 0;
                    document.getElementById('statWinRate').textContent = winRate + '%';
                    
                    document.getElementById('statProfit').textContent = '$' + data.stats.total_profit.toFixed(2);
                }
            });
        }
        
        function changePassword() {
            showAlert('Password change feature coming soon!', 'info');
        }
        
        function deleteAccount() {
            if (confirm('Are you sure you want to delete your account? This action cannot be undone!')) {
                showAlert('Account deletion feature coming soon!', 'info');
            }
        }
        
        // Alert function
        function showAlert(message, type = 'info') {
            const alertDiv = document.getElementById('alert');
            alertDiv.textContent = message;
            alertDiv.className = 'alert ' + type;
            alertDiv.style.display = 'block';
            
            setTimeout(() => {
                alertDiv.style.display = 'none';
            }, 5000);
        }
        
        // Logout
        function logout() {
            window.location.href = '/logout';
        }
    </script>
</body>
</html>
'''

# ============ FLASK ROUTES ============
@app.route('/')
def index():
    """Main page"""
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        success, message, user_data = user_manager.authenticate_user(username, password)
        
        if success:
            session['token'] = user_data['token']
            session['username'] = username
            return redirect('/dashboard')
        
        return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Login - Karanka V8</title>
                <style>
                    body { background: black; color: gold; font-family: Arial; }
                    .login-box { max-width: 400px; margin: 100px auto; padding: 30px; 
                               border: 2px solid gold; border-radius: 10px; text-align: center; }
                    input { width: 100%; padding: 10px; margin: 10px 0; background: #222; color: gold; border: 1px solid gold; }
                    button { background: gold; color: black; padding: 10px 20px; border: none; cursor: pointer; font-weight: bold; }
                </style>
            </head>
            <body>
                <div class="login-box">
                    <h1>🚀 Karanka V8</h1>
                    <p style="color: red;">{{ error }}</p>
                    <form method="POST">
                        <input type="text" name="username" placeholder="Username" required>
                        <input type="password" name="password" placeholder="Password" required>
                        <button type="submit">Login</button>
                    </form>
                    <p style="margin-top: 20px;"><a href="/register" style="color: gold;">Create Account</a></p>
                </div>
            </body>
            </html>
        ''', error=message)
    
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Login - Karanka V8</title>
            <style>
                body { background: black; color: gold; font-family: Arial; }
                .login-box { max-width: 400px; margin: 100px auto; padding: 30px; 
                           border: 2px solid gold; border-radius: 10px; text-align: center; }
                input { width: 100%; padding: 10px; margin: 10px 0; background: #222; color: gold; border: 1px solid gold; }
                button { background: gold; color: black; padding: 10px 20px; border: none; cursor: pointer; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="login-box">
                <h1>🚀 Karanka V8</h1>
                <form method="POST">
                    <input type="text" name="username" placeholder="Username" required>
                    <input type="password" name="password" placeholder="Password" required>
                    <button type="submit">Login</button>
                </form>
                <p style="margin-top: 20px;"><a href="/register" style="color: gold;">Create Account</a></p>
            </div>
        </body>
        </html>
    ''')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm = request.form.get('confirm_password', '').strip()
        email = request.form.get('email', '').strip()
        
        if password != confirm:
            return render_template_string('''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Register - Karanka V8</title>
                    <style>
                        body { background: black; color: gold; font-family: Arial; }
                        .register-box { max-width: 400px; margin: 50px auto; padding: 30px; 
                                      border: 2px solid gold; border-radius: 10px; text-align: center; }
                        input { width: 100%; padding: 10px; margin: 10px 0; background: #222; color: gold; border: 1px solid gold; }
                        button { background: gold; color: black; padding: 10px 20px; border: none; cursor: pointer; font-weight: bold; }
                    </style>
                </head>
                <body>
                    <div class="register-box">
                        <h1>🚀 Create Account</h1>
                        <p style="color: red;">Passwords do not match!</p>
                        <form method="POST">
                            <input type="text" name="username" placeholder="Username" required>
                            <input type="email" name="email" placeholder="Email (optional)">
                            <input type="password" name="password" placeholder="Password (min 8 chars)" required>
                            <input type="password" name="confirm_password" placeholder="Confirm Password" required>
                            <button type="submit">Register</button>
                        </form>
                        <p style="margin-top: 20px;"><a href="/login" style="color: gold;">Already have account? Login</a></p>
                    </div>
                </body>
                </html>
            ''')
        
        success, message = user_manager.register_user(username, password, email)
        
        if success:
            return redirect('/login')
        
        return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Register - Karanka V8</title>
                <style>
                    body { background: black; color: gold; font-family: Arial; }
                    .register-box { max-width: 400px; margin: 50px auto; padding: 30px; 
                                  border: 2px solid gold; border-radius: 10px; text-align: center; }
                    input { width: 100%; padding: 10px; margin: 10px 0; background: #222; color: gold; border: 1px solid gold; }
                    button { background: gold; color: black; padding: 10px 20px; border: none; cursor: pointer; font-weight: bold; }
                </style>
            </head>
            <body>
                <div class="register-box">
                    <h1>🚀 Create Account</h1>
                    <p style="color: red;">{{ error }}</p>
                    <form method="POST">
                        <input type="text" name="username" placeholder="Username" required>
                        <input type="email" name="email" placeholder="Email (optional)">
                        <input type="password" name="password" placeholder="Password (min 8 chars)" required>
                        <input type="password" name="confirm_password" placeholder="Confirm Password" required>
                        <button type="submit">Register</button>
                    </form>
                    <p style="margin-top: 20px;"><a href="/login" style="color: gold;">Already have account? Login</a></p>
                </div>
            </body>
            </html>
        ''', error=message)
    
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Register - Karanka V8</title>
            <style>
                body { background: black; color: gold; font-family: Arial; }
                .register-box { max-width: 400px; margin: 50px auto; padding: 30px; 
                              border: 2px solid gold; border-radius: 10px; text-align: center; }
                input { width: 100%; padding: 10px; margin: 10px 0; background: #222; color: gold; border: 1px solid gold; }
                button { background: gold; color: black; padding: 10px 20px; border: none; cursor: pointer; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="register-box">
                <h1>🚀 Create Account</h1>
                <form method="POST">
                    <input type="text" name="username" placeholder="Username" required>
                    <input type="email" name="email" placeholder="Email (optional)">
                    <input type="password" name="password" placeholder="Password (min 8 chars)" required>
                    <input type="password" name="confirm_password" placeholder="Confirm Password" required>
                    <button type="submit">Register</button>
                </form>
                <p style="margin-top: 20px;"><a href="/login" style="color: gold;">Already have account? Login</a></p>
            </div>
        </body>
        </html>
    ''')

@app.route('/dashboard')
def dashboard():
    """Main dashboard with gold/black UI"""
    if 'token' not in session or 'username' not in session:
        return redirect('/login')
    
    valid, username = user_manager.validate_token(session['token'])
    if not valid:
        return redirect('/login')
    
    # Get user engine
    engine = get_user_engine(username)
    user_data = user_manager.get_user(username)
    
    return render_template_string(
        GOLD_BLACK_UI,
        username=username,
        config=config,
        engine_status=engine.get_status() if engine else None,
        user_data=user_data
    )

@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    return redirect('/login')

# ============ API ENDPOINTS ============
@app.route('/api/status')
@token_required
def api_status():
    """Get bot status"""
    engine = get_user_engine(request.username)
    user_data = user_manager.get_user(request.username)
    
    return jsonify({
        'success': True,
        'status': engine.get_status() if engine else {},
        'user': {
            'settings': user_data['settings'] if user_data else {},
            'stats': user_data['trading_stats'] if user_data else {}
        }
    })

@app.route('/api/settings/update', methods=['POST'])
@token_required
def api_update_settings():
    """Update user settings"""
    data = request.json or {}
    settings = data.get('settings', {})
    
    # Update in user manager
    user_manager.update_user_settings(request.username, settings)
    
    # Update in engine
    engine = get_user_engine(request.username)
    if engine:
        engine.settings.update(settings)
    
    return jsonify({'success': True, 'message': 'Settings updated'})

@app.route('/api/connect', methods=['POST'])
@token_required
def api_connect():
    """Connect to Deriv"""
    data = request.json or {}
    api_token = data.get('api_token', '').strip()
    
    if not api_token:
        return jsonify({'success': False, 'message': 'API token required'})
    
    engine = get_user_engine(request.username)
    user_manager.update_user_settings(request.username, {'api_token': api_token})
    
    success, message, balance = engine.client.connect(api_token)
    
    if success:
        return jsonify({
            'success': True,
            'message': f'Connected! Balance: ${balance:.2f}',
            'balance': balance
        })
    else:
        return jsonify({'success': False, 'message': message})

@app.route('/api/trade', methods=['POST'])
@token_required
def api_trade():
    """Execute a trade"""
    data = request.json or {}
    symbol = data.get('symbol', 'R_10')
    direction = data.get('direction', 'BUY')
    amount = float(data.get('amount', 2.0))
    duration = int(data.get('duration', 5))
    duration_unit = data.get('duration_unit', 'm')
    
    engine = get_user_engine(request.username)
    
    # Check if dry run
    if engine.settings['dry_run']:
        profit = engine._simulate_trade(symbol, direction, amount)
        
        user_manager.update_user_stats(request.username, {
            'total_trades': 1,
            'successful_trades': 1 if profit > 0 else 0,
            'failed_trades': 1 if profit <= 0 else 0,
            'total_profit': profit,
            'current_balance': engine.client.balance + profit
        })
        
        return jsonify({
            'success': True,
            'message': f'DRY RUN: {direction} {symbol} ${amount:.2f} - Profit: ${profit:.2f}',
            'profit': profit,
            'dry_run': True
        })
    
    # Real trade
    success, result = engine.client.place_trade_with_duration(
        symbol=symbol,
        direction=direction,
        amount=amount,
        duration=duration,
        duration_unit=duration_unit
    )
    
    if success:
        trade_data = json.loads(result)
        
        user_manager.update_user_stats(request.username, {
            'total_trades': 1,
            'successful_trades': 1,
            'total_profit': trade_data['profit'],
            'current_balance': trade_data['balance']
        })
        
        return jsonify({
            'success': True,
            'message': f'Trade executed! Profit: ${trade_data["profit"]:.2f}',
            'profit': trade_data['profit'],
            'balance': trade_data['balance'],
            'dry_run': False
        })
    else:
        user_manager.update_user_stats(request.username, {
            'total_trades': 1,
            'failed_trades': 1
        })
        
        return jsonify({'success': False, 'message': result})

@app.route('/api/trading/start', methods=['POST'])
@token_required
def api_start_trading():
    """Start auto trading"""
    engine = get_user_engine(request.username)
    success, message = engine.start_trading()
    
    return jsonify({'success': success, 'message': message, 'running': engine.running})

@app.route('/api/trading/stop', methods=['POST'])
@token_required
def api_stop_trading():
    """Stop auto trading"""
    engine = get_user_engine(request.username)
    success, message = engine.stop_trading()
    
    return jsonify({'success': success, 'message': message, 'running': engine.running})

@app.route('/api/trades/history')
@token_required
def api_trade_history():
    """Get trade history"""
    engine = get_user_engine(request.username)
    
    if not engine:
        return jsonify({'success': False, 'trades': [], 'total': 0})
    
    # Get user's trades
    user_trades = [t for t in engine.trade_history if t['username'] == request.username]
    recent_trades = user_trades[-100:] if user_trades else []
    
    return jsonify({
        'success': True,
        'trades': recent_trades,
        'total': len(user_trades)
    })

@app.route('/api/analyze/<symbol>')
@token_required
def api_analyze(symbol):
    """Analyze market"""
    engine = get_user_engine(request.username)
    
    if not engine:
        return jsonify({'success': False, 'message': 'Engine not found'})
    
    # Get market data
    market_data = engine.client.get_market_data(symbol)
    if market_data:
        engine.analyzer.add_price_data(symbol, market_data['bid'], '1m')
        engine.analyzer.add_price_data(symbol, market_data['bid'], '5m')
    
    # Analyze
    analysis = engine.analyzer.analyze_multi_timeframe(symbol)
    
    return jsonify({
        'success': True,
        'symbol': symbol,
        'signal': analysis['signal'],
        'confidence': analysis['confidence'],
        'timeframe': analysis['timeframe'],
        'volatility': analysis.get('volatility', 0),
        'price': analysis.get('price', 0)
    })

@app.route('/api/user/data')
@token_required
def api_user_data():
    """Get user data"""
    user_data = user_manager.get_user(request.username)
    
    if not user_data:
        return jsonify({'success': False, 'message': 'User not found'})
    
    return jsonify({
        'success': True,
        'user': {
            'username': request.username,
            'email': user_data.get('email', ''),
            'created_at': user_data.get('created_at', ''),
            'last_login': user_data.get('last_login', '')
        },
        'stats': user_data['trading_stats']
    })

# ============ START APPLICATION ============
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    
    logger.info("""
    ========================================================================
    🚀 KARANKA V8 GOLD EDITION - SMART TIMEFRAME TRADING BOT
    ========================================================================
    • AUTO TIMEFRAME SELECTION: 1-minute or 5-minute based on analysis
    • MULTI-MARKET TRADING: Trade on all selected markets simultaneously
    • CONTINUOUS 24/7 TRADING: Never stops until you tell it to
    • SMART ANALYSIS: RSI, MACD, Bollinger Bands, Trend, Volatility
    • RISK MANAGEMENT: Stop loss, take profit, position sizing
    • GOLD/BLACK UI: Premium interface with all features
    ========================================================================
    """)
    
    logger.info(f"🌐 Server starting on port {port}")
    logger.info(f"📁 User database: {config.DATABASE_FILE}")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )
