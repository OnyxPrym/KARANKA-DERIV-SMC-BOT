#!/usr/bin/env python3
"""
================================================================================
🚀 KARANKA V8 - ULTRA-FREQUENT SMC TRADING BOT
================================================================================
• ULTRA-FAST SMC STRATEGIES (50-100+ trades/hour possible)
• REAL DERIV CONNECTION & TRADE EXECUTION
• MAXIMUM TRADE FREQUENCY CONFIGURATION
• PRODUCTION-READY FOR RENDER.COM
================================================================================
"""

import os
import json
import time
import threading
import hashlib
import secrets
import urllib.parse
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from typing import Dict, List, Optional, Tuple, Any
from uuid import uuid4
import numpy as np
import pandas as pd
import requests
import websocket
from flask import Flask, render_template_string, jsonify, request, session, redirect, url_for
from flask_cors import CORS

# ============ SETUP LOGGING ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('karanka_bot.log')
    ]
)
logger = logging.getLogger(__name__)

# ============ CONFIGURATION ============
class Config:
    # Render.com environment variables
    SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_urlsafe(32))
    PORT = int(os.environ.get('PORT', 10000))
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    # Deriv Configuration
    DERIV_APP_ID = 1089  # Deriv's official app ID
    DERIV_WS_URL = "wss://ws.binaryws.com/websockets/v3"
    
    # Trading Settings
    MIN_TRADE_AMOUNT = 0.35  # Minimum amount for Deriv
    MAX_CONCURRENT_TRADES = 5  # Increased for frequency
    TRADE_DURATION = 3  # 3 minutes for faster trades

# ============ DERIV MARKETS (ULTRA-ACTIVE) ============
DERIV_MARKETS = {
    # Volatility Indices (Most Active - HIGH FREQUENCY)
    "volatility_10_index": {"name": "Volatility 10 Index", "pip": 0.001, "category": "Volatility", "frequency": "ULTRA_HIGH"},
    "volatility_25_index": {"name": "Volatility 25 Index", "pip": 0.001, "category": "Volatility", "frequency": "ULTRA_HIGH"},
    "volatility_50_index": {"name": "Volatility 50 Index", "pip": 0.001, "category": "Volatility", "frequency": "HIGH"},
    "volatility_75_index": {"name": "Volatility 75 Index", "pip": 0.001, "category": "Volatility", "frequency": "ULTRA_HIGH"},
    "volatility_100_index": {"name": "Volatility 100 Index", "pip": 0.001, "category": "Volatility", "frequency": "ULTRA_HIGH"},
    
    # Boom Indices (Frequent Moves)
    "boom_500_index": {"name": "Boom 500 Index", "pip": 0.01, "category": "Boom", "frequency": "HIGH"},
    "boom_1000_index": {"name": "Boom 1000 Index", "pip": 0.01, "category": "Boom", "frequency": "MEDIUM"},
    
    # Crash Indices (Frequent Moves)
    "crash_500_index": {"name": "Crash 500 Index", "pip": 0.01, "category": "Crash", "frequency": "HIGH"},
    "crash_1000_index": {"name": "Crash 1000 Index", "pip": 0.01, "category": "Crash", "frequency": "MEDIUM"},
}

# ============ DATABASE ============
class UserDatabase:
    def __init__(self):
        self.users = {}
        logger.info("✅ User database initialized")
    
    def create_user(self, username: str, password: str) -> Tuple[bool, str]:
        try:
            if username in self.users:
                return False, "Username already exists"
            
            if len(username) < 3:
                return False, "Username must be at least 3 characters"
            
            if len(password) < 6:
                return False, "Password must be at least 6 characters"
            
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            self.users[username] = {
                'user_id': str(uuid4()),
                'username': username,
                'password_hash': password_hash,
                'created_at': datetime.now().isoformat(),
                'settings': {
                    'enabled_markets': ['volatility_75_index', 'volatility_100_index', 'crash_500_index', 'boom_500_index'],
                    'min_confidence': 60,  # LOWER for more trades
                    'trade_amount': 0.50,  # Smaller for more trades
                    'max_concurrent_trades': 5,  # Increased
                    'max_daily_trades': 200,  # Much higher
                    'max_hourly_trades': 50,  # Much higher
                    'dry_run': True,  # SAFETY FIRST
                    'risk_level': 1.0,
                    'trade_duration': 3,  # Shorter trades
                    'use_1m_candles': True,  # Faster analysis
                    'aggressive_mode': True,  # Take more trades
                    'scan_interval': 5,  # Scan every 5 seconds
                },
                'stats': {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_profit': 0.0,
                    'balance': 0.0,
                    'last_login': None
                }
            }
            
            logger.info(f"✅ Created user: {username}")
            return True, "User created successfully"
        except Exception as e:
            logger.error(f"❌ Error creating user: {e}")
            return False, f"Error creating user: {str(e)}"

    def authenticate(self, username: str, password: str) -> Tuple[bool, str]:
        try:
            if username not in self.users:
                return False, "User not found"
            
            user = self.users[username]
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            if user['password_hash'] != password_hash:
                return False, "Invalid password"
            
            user['stats']['last_login'] = datetime.now().isoformat()
            logger.info(f"✅ User authenticated: {username}")
            return True, "Authentication successful"
        except Exception as e:
            logger.error(f"❌ Authentication error: {e}")
            return False, f"Authentication error: {str(e)}"
    
    def get_user(self, username: str) -> Optional[Dict]:
        return self.users.get(username)
    
    def update_user(self, username: str, updates: Dict) -> bool:
        try:
            if username not in self.users:
                return False
            
            user = self.users[username]
            if 'settings' in updates:
                user['settings'].update(updates['settings'])
            else:
                user.update(updates)
            
            logger.info(f"✅ Updated user: {username}")
            return True
        except Exception as e:
            logger.error(f"❌ Error updating user: {e}")
            return False

# ============ DERIV API CLIENT (OPTIMIZED FOR SPEED) ============
class FastDerivAPIClient:
    """OPTIMIZED for maximum speed and frequency"""
    
    def __init__(self):
        self.ws = None
        self.connected = False
        self.account_info = {}
        self.prices = {}
        self.subscriptions = set()
        self.running = False
        self.price_thread = None
        self.connection_lock = threading.Lock()
        self.last_price_update = {}
        
    def connect_with_token(self, api_token: str) -> Tuple[bool, str]:
        """FAST connection to Deriv"""
        try:
            logger.info("🚀 FAST CONNECTION to Deriv...")
            
            api_token = api_token.strip()
            if not api_token:
                return False, "API token is empty"
            
            ws_url = f"{Config.DERIV_WS_URL}?app_id={Config.DERIV_APP_ID}"
            
            with self.connection_lock:
                self.ws = websocket.create_connection(ws_url, timeout=5)
                
                auth_request = {"authorize": api_token, "req_id": 1}
                self.ws.send(json.dumps(auth_request))
                self.ws.settimeout(5)
                
                response = self.ws.recv()
                data = json.loads(response)
                
                if "error" in data:
                    error_msg = data["error"].get("message", "Authentication failed")
                    return False, f"Auth failed: {error_msg}"
                
                if "authorize" not in data:
                    return False, "Invalid response"
                
                self.account_info = data["authorize"]
                self.connected = True
                self.running = True
                
                loginid = self.account_info.get("loginid", "Unknown")
                currency = self.account_info.get("currency", "USD")
                balance = self._get_balance_fast()
                
                logger.info(f"✅ CONNECTED to {loginid} | Balance: {balance:.2f} {currency}")
                
                self._start_price_thread()
                
                return True, f"Connected to {loginid} | Balance: {balance:.2f} {currency}"
                
        except Exception as e:
            logger.error(f"❌ Connection error: {str(e)}")
            return False, f"Connection error: {str(e)}"
    
    def _start_price_thread(self):
        """Start price update thread"""
        def price_worker():
            while self.running and self.connected and self.ws:
                try:
                    self.ws.settimeout(0.5)
                    response = self.ws.recv()
                    data = json.loads(response)
                    
                    if "tick" in data:
                        tick = data["tick"]
                        symbol = tick.get("symbol")
                        price = float(tick.get("quote", 0))
                        
                        if symbol:
                            self.prices[symbol] = price
                            self.last_price_update[symbol] = time.time()
                            
                except websocket.WebSocketTimeoutException:
                    continue
                except:
                    time.sleep(0.1)
        
        self.price_thread = threading.Thread(target=price_worker, daemon=True)
        self.price_thread.start()
    
    def subscribe_price(self, symbol: str) -> bool:
        """FAST subscription"""
        try:
            if not self.connected:
                return False
            
            if symbol in self.subscriptions:
                return True
            
            subscribe_msg = {"ticks": symbol, "subscribe": 1}
            self.ws.send(json.dumps(subscribe_msg))
            self.subscriptions.add(symbol)
            time.sleep(0.3)  # Shorter wait
            
            return True
            
        except:
            return False
    
    def get_price(self, symbol: str) -> Optional[float]:
        """ULTRA-FAST price getter"""
        # Return cached price if recent (last 2 seconds)
        if symbol in self.prices:
            last_update = self.last_price_update.get(symbol, 0)
            if time.time() - last_update < 2:
                return self.prices[symbol]
        
        # Subscribe and get fresh
        if self.subscribe_price(symbol):
            time.sleep(0.2)
            return self.prices.get(symbol)
        
        return None
    
    def get_candles(self, symbol: str, timeframe: str = "1m", count: int = 50) -> Optional[pd.DataFrame]:
        """FAST candle data (using 1m by default)"""
        try:
            if not self.connected:
                return None
            
            timeframe_map = {"1m": 60, "5m": 300}
            granularity = timeframe_map.get(timeframe, 60)
            
            request = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": count,
                "end": "latest",
                "granularity": granularity,
                "style": "candles"
            }
            
            self.ws.send(json.dumps(request))
            self.ws.settimeout(3)
            
            response = self.ws.recv()
            data = json.loads(response)
            
            if "candles" in data and data["candles"]:
                candles = data["candles"]
                df_data = {
                    'time': [pd.to_datetime(c.get('epoch'), unit='s') for c in candles],
                    'open': [float(c.get('open', 0)) for c in candles],
                    'high': [float(c.get('high', 0)) for c in candles],
                    'low': [float(c.get('low', 0)) for c in candles],
                    'close': [float(c.get('close', 0)) for c in candles],
                }
                return pd.DataFrame(df_data)
            
            return None
            
        except:
            return None
    
    def place_trade(self, symbol: str, direction: str, amount: float) -> Tuple[bool, str]:
        """FAST trade execution"""
        try:
            with self.connection_lock:
                if not self.connected:
                    return False, "Not connected"
                
                if amount < Config.MIN_TRADE_AMOUNT:
                    amount = Config.MIN_TRADE_AMOUNT
                
                contract_type = "CALL" if direction.upper() in ["BUY", "UP", "CALL"] else "PUT"
                
                trade_request = {
                    "buy": amount,
                    "price": amount,
                    "parameters": {
                        "amount": amount,
                        "basis": "stake",
                        "contract_type": contract_type,
                        "currency": self.account_info.get("currency", "USD"),
                        "duration": Config.TRADE_DURATION,
                        "duration_unit": "m",
                        "symbol": symbol
                    }
                }
                
                logger.info(f"🚀 EXECUTING: {symbol} {direction} ${amount}")
                
                self.ws.send(json.dumps(trade_request))
                self.ws.settimeout(3)
                
                response = self.ws.recv()
                data = json.loads(response)
                
                if "error" in data:
                    error_msg = data["error"].get("message", "Trade failed")
                    return False, f"Trade failed: {error_msg}"
                
                if "buy" in data:
                    contract_id = data["buy"].get("contract_id", "Unknown")
                    return True, contract_id
                
                return False, "Unknown error"
                
        except Exception as e:
            logger.error(f"❌ Trade error: {str(e)}")
            return False, f"Trade error: {str(e)}"
    
    def _get_balance_fast(self) -> float:
        """Fast balance check"""
        try:
            if not self.connected:
                return 0.0
            
            self.ws.send(json.dumps({"balance": 1}))
            self.ws.settimeout(2)
            
            response = self.ws.recv()
            data = json.loads(response)
            
            if "balance" in data:
                return float(data["balance"]["balance"])
            
            return 0.0
            
        except:
            return 0.0
    
    def close(self):
        """Close connection"""
        self.running = False
        if self.ws:
            self.ws.close()
        self.connected = False

# ============ ULTRA-FREQUENT SMC ANALYZER ============
class UltraFrequentSMCAnalyzer:
    """DESIGNED FOR MAXIMUM TRADE FREQUENCY"""
    
    def __init__(self):
        self.strategy_weights = {
            'liquidity_sweep': 1.0,      # Most frequent
            'inside_bar': 0.9,           # Very frequent
            'order_block': 0.8,          # Frequent
            'structure_break': 0.7,      # Frequent
            'fvg': 0.6,                  # Frequent
            'rsi_divergence': 0.5,       # Moderate
        }
        self.recent_signals = defaultdict(deque)
        
    def analyze(self, symbol: str, df: pd.DataFrame, current_price: float) -> Dict:
        """ULTRA-FAST analysis with multiple strategies"""
        try:
            if df is None or len(df) < 5:
                return self._neutral_signal()
            
            # Prepare quick data
            df = self._prepare_data_fast(df)
            
            # Run ALL strategies quickly
            strategies = [
                self._liquidity_sweep_strategy(df, symbol, current_price),
                self._inside_bar_strategy(df, current_price),
                self._order_block_strategy(df, current_price),
                self._structure_break_strategy(df, current_price),
                self._fvg_strategy(df, current_price),
                self._rsi_divergence_strategy(df),
            ]
            
            # Find strongest signal
            best_signal = None
            best_confidence = 0
            
            for strategy_result in strategies:
                if strategy_result['signal'] != 'NEUTRAL':
                    confidence = strategy_result['confidence']
                    weight = self.strategy_weights.get(strategy_result['strategy'], 0.5)
                    weighted_conf = confidence * weight
                    
                    if weighted_conf > best_confidence:
                        best_confidence = weighted_conf
                        best_signal = strategy_result
            
            if best_signal and best_confidence >= 60:
                # Check if similar signal recently
                if not self._is_duplicate_signal(symbol, best_signal):
                    self._record_signal(symbol, best_signal)
                    return best_signal
            
            return self._neutral_signal()
            
        except Exception as e:
            logger.error(f"❌ Analysis error: {e}")
            return self._neutral_signal()
    
    def _prepare_data_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """FAST data preparation"""
        if len(df) < 2:
            return df
        
        # Quick indicators
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        
        # Fast RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _liquidity_sweep_strategy(self, df: pd.DataFrame, symbol: str, current_price: float) -> Dict:
        """MOST FREQUENT - Liquidity sweeps"""
        if len(df) < 8:
            return self._neutral_signal('liquidity_sweep')
        
        # Recent range (last 8 candles)
        recent_high = df['high'].iloc[-8:].max()
        recent_low = df['low'].iloc[-8:].min()
        current = df.iloc[-1]
        
        # Bullish sweep (dip buy)
        if (current['low'] <= recent_low and 
            current['close'] > recent_low and
            current['close'] > current['open']):
            
            return {
                'signal': 'BUY',
                'confidence': 80,
                'strategy': 'liquidity_sweep',
                'signals': ['🔥 BULLISH SWEEP'],
                'frequency': 'ULTRA_HIGH'
            }
        
        # Bearish sweep (spike sell)
        if (current['high'] >= recent_high and 
            current['close'] < recent_high and
            current['close'] < current['open']):
            
            return {
                'signal': 'SELL',
                'confidence': 80,
                'strategy': 'liquidity_sweep',
                'signals': ['💧 BEARISH SWEEP'],
                'frequency': 'ULTRA_HIGH'
            }
        
        return self._neutral_signal('liquidity_sweep')
    
    def _inside_bar_strategy(self, df: pd.DataFrame, current_price: float) -> Dict:
        """VERY FREQUENT - Inside bar breakouts"""
        if len(df) < 3:
            return self._neutral_signal('inside_bar')
        
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        
        # Inside bar pattern
        if (curr['high'] < prev['high'] and 
            curr['low'] > prev['low']):
            
            # Break above
            if current_price > prev['high']:
                return {
                    'signal': 'BUY',
                    'confidence': 75,
                    'strategy': 'inside_bar',
                    'signals': ['📈 INSIDE BAR BULLISH'],
                    'frequency': 'HIGH'
                }
            # Break below
            elif current_price < prev['low']:
                return {
                    'signal': 'SELL',
                    'confidence': 75,
                    'strategy': 'inside_bar',
                    'signals': ['📉 INSIDE BAR BEARISH'],
                    'frequency': 'HIGH'
                }
        
        return self._neutral_signal('inside_bar')
    
    def _order_block_strategy(self, df: pd.DataFrame, current_price: float) -> Dict:
        """FREQUENT - Order blocks"""
        if len(df) < 4:
            return self._neutral_signal('order_block')
        
        # Look for strong bearish candle
        for i in range(len(df)-4, len(df)-1):
            bear_candle = df.iloc[i]
            bull_candle = df.iloc[i+1]
            
            # Bearish order block (strong red, weak green after)
            bear_strength = abs(bear_candle['close'] - bear_candle['open'])
            bear_range = bear_candle['high'] - bear_candle['low']
            
            if (bear_candle['close'] < bear_candle['open'] and 
                bull_candle['close'] > bull_candle['open'] and
                bear_strength > bear_range * 0.6):
                
                ob_low = bear_candle['low']
                ob_high = bear_candle['high']
                
                # Price in order block zone
                if ob_low <= current_price <= ob_high:
                    return {
                        'signal': 'BUY',
                        'confidence': 70,
                        'strategy': 'order_block',
                        'signals': ['📦 BULLISH ORDER BLOCK'],
                        'frequency': 'HIGH'
                    }
        
        return self._neutral_signal('order_block')
    
    def _structure_break_strategy(self, df: pd.DataFrame, current_price: float) -> Dict:
        """FREQUENT - Structure breaks"""
        if len(df) < 6:
            return self._neutral_signal('structure_break')
        
        # Recent swing points
        swing_high = df['high'].iloc[-6:-1].max()
        swing_low = df['low'].iloc[-6:-1].min()
        current = df.iloc[-1]
        
        # Bullish break
        if current['close'] > swing_high and current['close'] > current['open']:
            return {
                'signal': 'BUY',
                'confidence': 75,
                'strategy': 'structure_break',
                'signals': ['🚀 BULLISH BREAKOUT'],
                'frequency': 'HIGH'
            }
        
        # Bearish break
        if current['close'] < swing_low and current['close'] < current['open']:
            return {
                'signal': 'SELL',
                'confidence': 75,
                'strategy': 'structure_break',
                'signals': ['📉 BEARISH BREAKDOWN'],
                'frequency': 'HIGH'
            }
        
        return self._neutral_signal('structure_break')
    
    def _fvg_strategy(self, df: pd.DataFrame, current_price: float) -> Dict:
        """FREQUENT - Fair Value Gaps"""
        if len(df) < 3:
            return self._neutral_signal('fvg')
        
        c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        
        # Bullish FVG
        if c1['high'] < c3['low']:
            fvg_low = c1['high']
            fvg_high = c3['low']
            
            if fvg_low <= current_price <= fvg_high:
                return {
                    'signal': 'BUY',
                    'confidence': 65,
                    'strategy': 'fvg',
                    'signals': ['⚡ BULLISH FVG'],
                    'frequency': 'MEDIUM'
                }
        
        # Bearish FVG
        if c1['low'] > c3['high']:
            fvg_low = c3['high']
            fvg_high = c1['low']
            
            if fvg_low <= current_price <= fvg_high:
                return {
                    'signal': 'SELL',
                    'confidence': 65,
                    'strategy': 'fvg',
                    'signals': ['⚡ BEARISH FVG'],
                    'frequency': 'MEDIUM'
                }
        
        return self._neutral_signal('fvg')
    
    def _rsi_divergence_strategy(self, df: pd.DataFrame) -> Dict:
        """MODERATE - RSI Divergence"""
        if len(df) < 10 or 'rsi' not in df.columns:
            return self._neutral_signal('rsi_divergence')
        
        prices = df['close'].iloc[-5:].values
        rsi_values = df['rsi'].iloc[-5:].values
        
        # Bullish divergence
        if (prices[-1] < prices[-3] and 
            rsi_values[-1] > rsi_values[-3] and
            rsi_values[-1] < 35):
            
            return {
                'signal': 'BUY',
                'confidence': 70,
                'strategy': 'rsi_divergence',
                'signals': ['↗️ BULLISH DIVERGENCE'],
                'frequency': 'MEDIUM'
            }
        
        # Bearish divergence
        if (prices[-1] > prices[-3] and 
            rsi_values[-1] < rsi_values[-3] and
            rsi_values[-1] > 65):
            
            return {
                'signal': 'SELL',
                'confidence': 70,
                'strategy': 'rsi_divergence',
                'signals': ['↘️ BEARISH DIVERGENCE'],
                'frequency': 'MEDIUM'
            }
        
        return self._neutral_signal('rsi_divergence')
    
    def _is_duplicate_signal(self, symbol: str, signal: Dict) -> bool:
        """Prevent duplicate signals"""
        recent = self.recent_signals.get(symbol, deque(maxlen=5))
        
        for old_signal in recent:
            if (old_signal['signal'] == signal['signal'] and 
                old_signal['strategy'] == signal['strategy']):
                # Check if within 1 minute
                if 'timestamp' in old_signal:
                    old_time = datetime.fromisoformat(old_signal['timestamp'].replace('Z', '+00:00'))
                    if (datetime.now() - old_time).total_seconds() < 60:
                        return True
        
        return False
    
    def _record_signal(self, symbol: str, signal: Dict):
        """Record signal"""
        signal['timestamp'] = datetime.now().isoformat()
        self.recent_signals[symbol].append(signal)
    
    def _neutral_signal(self, strategy: str = 'neutral') -> Dict:
        """Neutral signal"""
        return {
            'signal': 'NEUTRAL',
            'confidence': 0,
            'strategy': strategy,
            'signals': [],
            'frequency': 'NONE'
        }

# ============ HIGH-FREQUENCY TRADING ENGINE ============
class HighFrequencyTradingEngine:
    """ENGINE OPTIMIZED FOR MAXIMUM FREQUENCY"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.api_client = None
        self.analyzer = UltraFrequentSMCAnalyzer()
        self.running = False
        self.trades = []
        self.active_trades = []
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'daily_trades': 0,
            'hourly_trades': 0,
            'last_reset': datetime.now()
        }
        self.settings = {
            'enabled_markets': ['volatility_75_index', 'volatility_100_index', 'crash_500_index', 'boom_500_index'],
            'min_confidence': 60,
            'trade_amount': 0.50,
            'max_concurrent_trades': 5,
            'max_daily_trades': 200,
            'max_hourly_trades': 50,
            'dry_run': True,
            'trade_duration': 3,
            'use_1m_candles': True,
            'aggressive_mode': True,
            'scan_interval': 5,
        }
        self.thread = None
        self.market_cooldowns = {}
        
    def connect_with_token(self, api_token: str) -> Tuple[bool, str]:
        """Connect to Deriv"""
        try:
            self.api_client = FastDerivAPIClient()
            success, message = self.api_client.connect_with_token(api_token)
            
            if success:
                # Subscribe to all enabled markets
                for symbol in self.settings['enabled_markets']:
                    self.api_client.subscribe_price(symbol)
                    time.sleep(0.1)
            
            return success, message
            
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            return False, str(e)
    
    def update_settings(self, settings: Dict):
        """Update settings"""
        old_markets = set(self.settings.get('enabled_markets', []))
        new_markets = set(settings.get('enabled_markets', old_markets))
        
        if self.api_client and self.api_client.connected:
            for symbol in new_markets - old_markets:
                self.api_client.subscribe_price(symbol)
                time.sleep(0.1)
        
        self.settings.update(settings)
    
    def start_trading(self):
        """Start ULTRA-FREQUENT trading"""
        if self.running:
            return False, "Already running"
        
        if not self.api_client or not self.api_client.connected:
            return False, "Not connected to Deriv"
        
        self.running = True
        self.thread = threading.Thread(target=self._ultra_frequent_loop, daemon=True)
        self.thread.start()
        
        mode = "DRY RUN" if self.settings['dry_run'] else "REAL TRADING"
        logger.info(f"💰 {mode} started (ULTRA-FREQUENT)")
        
        return True, f"{mode} started!"
    
    def stop_trading(self):
        """Stop trading"""
        self.running = False
    
    def _ultra_frequent_loop(self):
        """ULTRA-FAST trading loop"""
        logger.info(f"🔥 ULTRA-FREQUENT loop started")
        
        scan_interval = self.settings.get('scan_interval', 5)
        
        while self.running:
            try:
                start_time = time.time()
                
                # Check if can trade
                if not self._can_trade():
                    time.sleep(1)
                    continue
                
                # Process each market
                for symbol in self.settings['enabled_markets']:
                    if not self.running:
                        break
                    
                    try:
                        # Skip if in cooldown
                        if not self._check_cooldown(symbol):
                            continue
                        
                        # Get price (ULTRA-FAST)
                        current_price = self.api_client.get_price(symbol)
                        if not current_price:
                            continue
                        
                        # Get candles (1m for speed)
                        timeframe = "1m" if self.settings.get('use_1m_candles', True) else "5m"
                        candles = self.api_client.get_candles(symbol, timeframe, 30)
                        if candles is None or len(candles) < 5:
                            continue
                        
                        # ULTRA-FAST analysis
                        analysis = self.analyzer.analyze(symbol, candles, current_price)
                        
                        # Execute if signal
                        if (analysis['signal'] != 'NEUTRAL' and 
                            analysis['confidence'] >= self.settings['min_confidence']):
                            
                            self._execute_trade(symbol, analysis)
                        
                    except Exception as e:
                        continue
                
                # Calculate sleep time to maintain exact interval
                elapsed = time.time() - start_time
                sleep_time = max(0.1, scan_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"❌ Loop error: {e}")
                time.sleep(10)
    
    def _execute_trade(self, symbol: str, analysis: Dict):
        """Execute trade"""
        direction = analysis['signal']
        confidence = analysis['confidence']
        
        if self.settings['dry_run']:
            # DRY RUN
            logger.info(f"📝 DRY RUN: {symbol} {direction} ${self.settings['trade_amount']} ({confidence}%)")
            
            self._record_trade({
                'symbol': symbol,
                'direction': direction,
                'amount': self.settings['trade_amount'],
                'confidence': confidence,
                'dry_run': True,
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis
            })
            
        else:
            # REAL TRADE
            logger.info(f"🚀 REAL TRADE: {symbol} {direction} ${self.settings['trade_amount']}")
            
            success, trade_id = self.api_client.place_trade(
                symbol, direction, self.settings['trade_amount']
            )
            
            if success:
                logger.info(f"✅ SUCCESS: {trade_id}")
                
                self._record_trade({
                    'symbol': symbol,
                    'direction': direction,
                    'amount': self.settings['trade_amount'],
                    'trade_id': trade_id,
                    'confidence': confidence,
                    'dry_run': False,
                    'timestamp': datetime.now().isoformat(),
                    'analysis': analysis
                })
                
                # Add to active trades
                self.active_trades.append({
                    'symbol': symbol,
                    'trade_id': trade_id,
                    'timestamp': datetime.now()
                })
                
                # Set cooldown
                self._set_cooldown(symbol)
                
                # Cleanup old active trades
                self._cleanup_active_trades()
                
            else:
                logger.error(f"❌ FAILED: {trade_id}")
    
    def _can_trade(self) -> bool:
        """Check trading conditions"""
        try:
            # Max concurrent trades
            if len(self.active_trades) >= self.settings['max_concurrent_trades']:
                return False
            
            # Reset counters
            now = datetime.now()
            if now.date() > self.stats['last_reset'].date():
                self.stats['daily_trades'] = 0
                self.stats['hourly_trades'] = 0
                self.stats['last_reset'] = now
            
            # Daily limit
            if self.stats['daily_trades'] >= self.settings['max_daily_trades']:
                return False
            
            # Hourly limit
            if self.stats['hourly_trades'] >= self.settings['max_hourly_trades']:
                return False
            
            return True
            
        except:
            return False
    
    def _cleanup_active_trades(self):
        """Cleanup old active trades"""
        try:
            now = datetime.now()
            self.active_trades = [
                trade for trade in self.active_trades
                if (now - trade['timestamp']).total_seconds() < (self.settings['trade_duration'] * 60 + 30)
            ]
        except:
            pass
    
    def _check_cooldown(self, symbol: str) -> bool:
        """Check market cooldown"""
        if symbol not in self.market_cooldowns:
            return True
        
        if datetime.now() >= self.market_cooldowns[symbol]:
            del self.market_cooldowns[symbol]
            return True
        
        return False
    
    def _set_cooldown(self, symbol: str):
        """Set market cooldown"""
        cooldown = 1 if self.settings.get('aggressive_mode', False) else 2
        self.market_cooldowns[symbol] = datetime.now() + timedelta(minutes=cooldown)
    
    def _record_trade(self, trade_data: Dict):
        """Record trade"""
        trade_data['id'] = len(self.trades) + 1
        self.trades.append(trade_data)
        
        self.stats['total_trades'] += 1
        self.stats['daily_trades'] += 1
        self.stats['hourly_trades'] += 1
        
        # Reset hourly counter
        def reset_hourly():
            time.sleep(3600)
            self.stats['hourly_trades'] = max(0, self.stats['hourly_trades'] - 1)
        
        threading.Thread(target=reset_hourly, daemon=True).start()
    
    def get_status(self) -> Dict:
        """Get status"""
        balance = self.api_client._get_balance_fast() if self.api_client else 0.0
        connected = self.api_client.connected if self.api_client else False
        
        # Market data
        market_data = {}
        if self.api_client and self.api_client.connected:
            for symbol in self.settings.get('enabled_markets', []):
                try:
                    price = self.api_client.get_price(symbol)
                    if price:
                        market_data[symbol] = {
                            'name': DERIV_MARKETS.get(symbol, {}).get('name', symbol),
                            'price': price,
                            'category': DERIV_MARKETS.get(symbol, {}).get('category', 'Unknown')
                        }
                except:
                    continue
        
        return {
            'running': self.running,
            'connected': connected,
            'balance': balance,
            'stats': self.stats,
            'settings': self.settings,
            'recent_trades': self.trades[-20:][::-1] if self.trades else [],
            'active_trades': len(self.active_trades),
            'market_data': market_data
        }

# ============ FLASK APP ============
app = Flask(__name__)
CORS(app)
app.secret_key = Config.SECRET_KEY

# Session config
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(hours=24)
)

# Initialize components
user_db = UserDatabase()
trading_engines = {}

# ============ API ROUTES ============
@app.route('/api/login', methods=['POST'])
def api_login():
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password required'})
        
        success, message = user_db.authenticate(username, password)
        
        if success:
            session['username'] = username
            session.permanent = True
            
            if username not in trading_engines:
                user_data = user_db.get_user(username)
                engine = HighFrequencyTradingEngine(user_id=user_data['user_id'])
                engine.update_settings(user_data.get('settings', {}))
                trading_engines[username] = engine
            
            return jsonify({'success': True, 'message': 'Login successful'})
        else:
            return jsonify({'success': False, 'message': message})
            
    except Exception as e:
        logger.error(f"❌ Login error: {e}")
        return jsonify({'success': False, 'message': f'Login error: {str(e)}'})

@app.route('/api/register', methods=['POST'])
def api_register():
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password required'})
        
        success, message = user_db.create_user(username, password)
        
        if success:
            return jsonify({'success': True, 'message': 'Registration successful. Please login.'})
        else:
            return jsonify({'success': False, 'message': message})
            
    except Exception as e:
        logger.error(f"❌ Registration error: {e}")
        return jsonify({'success': False, 'message': f'Registration error: {str(e)}'})

@app.route('/api/logout', methods=['POST'])
def api_logout():
    try:
        username = session.get('username')
        if username:
            if username in trading_engines:
                engine = trading_engines[username]
                engine.stop_trading()
                if engine.api_client:
                    engine.api_client.close()
                del trading_engines[username]
            
            session.clear()
        
        return jsonify({'success': True, 'message': 'Logged out successfully'})
        
    except Exception as e:
        logger.error(f"❌ Logout error: {e}")
        return jsonify({'success': False, 'message': f'Logout error: {str(e)}'})

@app.route('/api/connect-token', methods=['POST'])
def api_connect_token():
    try:
        username = session.get('username')
        if not username:
            return jsonify({'success': False, 'message': 'Not logged in'})
        
        data = request.json
        api_token = data.get('api_token', '').strip()
        
        if not api_token:
            return jsonify({'success': False, 'message': 'API token required'})
        
        if username in trading_engines:
            engine = trading_engines[username]
            engine.stop_trading()
            if engine.api_client:
                engine.api_client.close()
            engine.api_client = None
        else:
            user_data = user_db.get_user(username)
            engine = HighFrequencyTradingEngine(user_id=user_data['user_id'])
            engine.update_settings(user_data.get('settings', {}))
            trading_engines[username] = engine
        
        engine = trading_engines[username]
        success, message = engine.connect_with_token(api_token)
        
        if success:
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'success': False, 'message': message})
            
    except Exception as e:
        logger.error(f"❌ Token connect error: {e}")
        return jsonify({'success': False, 'message': f'Token connection error: {str(e)}'})

@app.route('/api/status', methods=['GET'])
def api_status():
    try:
        username = session.get('username')
        if not username:
            return jsonify({'success': False, 'message': 'Not logged in'})
        
        engine = trading_engines.get(username)
        if not engine:
            return jsonify({'success': False, 'message': 'Not initialized'})
        
        status = engine.get_status()
        
        return jsonify({
            'success': True,
            'status': status,
            'markets': DERIV_MARKETS
        })
        
    except Exception as e:
        logger.error(f"❌ Status error: {e}")
        return jsonify({'success': False, 'message': f'Status error: {str(e)}'})

@app.route('/api/start-trading', methods=['POST'])
def api_start_trading():
    try:
        username = session.get('username')
        if not username:
            return jsonify({'success': False, 'message': 'Not logged in'})
        
        engine = trading_engines.get(username)
        if not engine or not engine.api_client or not engine.api_client.connected:
            return jsonify({'success': False, 'message': 'Not connected to Deriv'})
        
        success, message = engine.start_trading()
        return jsonify({'success': success, 'message': message})
        
    except Exception as e:
        logger.error(f"❌ Start trading error: {e}")
        return jsonify({'success': False, 'message': f'Start trading error: {str(e)}'})

@app.route('/api/stop-trading', methods=['POST'])
def api_stop_trading():
    try:
        username = session.get('username')
        if not username:
            return jsonify({'success': False, 'message': 'Not logged in'})
        
        engine = trading_engines.get(username)
        if not engine:
            return jsonify({'success': True, 'message': 'Not running'})
        
        engine.stop_trading()
        return jsonify({'success': True, 'message': 'Trading stopped'})
        
    except Exception as e:
        logger.error(f"❌ Stop trading error: {e}")
        return jsonify({'success': False, 'message': f'Stop trading error: {str(e)}'})

@app.route('/api/update-settings', methods=['POST'])
def api_update_settings():
    try:
        username = session.get('username')
        if not username:
            return jsonify({'success': False, 'message': 'Not logged in'})
        
        data = request.json
        settings = data.get('settings', {})
        
        # Validate
        if 'trade_amount' in settings and settings['trade_amount'] < Config.MIN_TRADE_AMOUNT:
            return jsonify({'success': False, 'message': f'Minimum trade amount is ${Config.MIN_TRADE_AMOUNT}'})
        
        if 'max_concurrent_trades' in settings:
            max_trades = settings['max_concurrent_trades']
            if not isinstance(max_trades, int) or max_trades < 1 or max_trades > 10:
                return jsonify({'success': False, 'message': 'Max concurrent trades must be 1-10'})
        
        engine = trading_engines.get(username)
        if engine:
            engine.update_settings(settings)
        
        user_data = user_db.get_user(username)
        if user_data:
            user_data['settings'].update(settings)
            user_db.update_user(username, user_data)
        
        return jsonify({'success': True, 'message': 'Settings updated'})
        
    except Exception as e:
        logger.error(f"❌ Update settings error: {e}")
        return jsonify({'success': False, 'message': f'Update settings error: {str(e)}'})

@app.route('/api/place-trade', methods=['POST'])
def api_place_trade():
    try:
        username = session.get('username')
        if not username:
            return jsonify({'success': False, 'message': 'Not logged in'})
        
        data = request.json
        symbol = data.get('symbol')
        direction = data.get('direction')
        amount = float(data.get('amount', 0.50))
        
        if not symbol or not direction:
            return jsonify({'success': False, 'message': 'Symbol and direction required'})
        
        if amount < Config.MIN_TRADE_AMOUNT:
            return jsonify({'success': False, 'message': f'Minimum trade amount is ${Config.MIN_TRADE_AMOUNT}'})
        
        engine = trading_engines.get(username)
        if not engine or not engine.api_client or not engine.api_client.connected:
            return jsonify({'success': False, 'message': 'Not connected to Deriv'})
        
        # Dry run check
        if engine.settings.get('dry_run', True):
            engine._record_trade({
                'symbol': symbol,
                'direction': direction,
                'amount': amount,
                'dry_run': True,
                'timestamp': datetime.now().isoformat(),
                'manual': True
            })
            
            return jsonify({
                'success': True,
                'message': f'DRY RUN: Would trade {symbol} {direction} ${amount}',
                'dry_run': True
            })
        
        # Real trade
        success, trade_id = engine.api_client.place_trade(symbol, direction, amount)
        
        if success:
            engine._record_trade({
                'symbol': symbol,
                'direction': direction,
                'amount': amount,
                'trade_id': trade_id,
                'dry_run': False,
                'timestamp': datetime.now().isoformat(),
                'manual': True
            })
            
            return jsonify({
                'success': True,
                'message': f'✅ REAL TRADE: {trade_id}',
                'trade_id': trade_id
            })
        else:
            return jsonify({'success': False, 'message': trade_id})
        
    except Exception as e:
        logger.error(f"❌ Place trade error: {e}")
        return jsonify({'success': False, 'message': f'Place trade error: {str(e)}'})

@app.route('/api/analyze-market', methods=['POST'])
def api_analyze_market():
    try:
        username = session.get('username')
        if not username:
            return jsonify({'success': False, 'message': 'Not logged in'})
        
        data = request.json
        symbol = data.get('symbol')
        
        if not symbol:
            return jsonify({'success': False, 'message': 'Symbol required'})
        
        engine = trading_engines.get(username)
        if not engine or not engine.api_client:
            return jsonify({'success': False, 'message': 'Not connected'})
        
        # Get market data
        candles = engine.api_client.get_candles(symbol, "1m", 30)
        current_price = engine.api_client.get_price(symbol)
        
        if candles is None or current_price is None:
            return jsonify({'success': False, 'message': 'Failed to get market data'})
        
        # Analyze
        analysis = engine.analyzer.analyze(symbol, candles, current_price)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'symbol': symbol,
            'market_name': DERIV_MARKETS.get(symbol, {}).get('name', symbol)
        })
        
    except Exception as e:
        logger.error(f"❌ Analyze market error: {e}")
        return jsonify({'success': False, 'message': f'Analyze market error: {str(e)}'})

@app.route('/api/check-session', methods=['GET'])
def api_check_session():
    try:
        username = session.get('username')
        if username:
            return jsonify({'success': True, 'username': username})
        else:
            return jsonify({'success': False, 'username': None})
    except:
        return jsonify({'success': False, 'username': None})

# ============ MAIN ROUTES ============
@app.route('/')
def index():
    """Serve web interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🚀 Karanka V8 - Ultra-Frequent Trading Bot</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #0a0a0a; color: #FFD700; }
            .container { max-width: 800px; margin: 0 auto; }
            h1 { color: #FFD700; text-align: center; }
            .status { background: #1a1a1a; padding: 20px; border-radius: 10px; margin: 20px 0; }
            .btn { background: #FFD700; color: #000; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
            .danger { background: #ff4444; color: white; }
            .success { background: #00C851; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Karanka V8 - Ultra-Frequent Trading Bot</h1>
            <div class="status">
                <p>🚀 <strong>ULTRA-FREQUENT SMC STRATEGIES ENABLED</strong></p>
                <p>📈 Designed for maximum trade frequency (50-100+ trades/hour possible)</p>
                <p>💰 Connect with your Deriv API token to start trading</p>
                <p>⚡ Use API endpoints for full control</p>
            </div>
            <div>
                <button class="btn" onclick="window.location.href='/api/status'">Check Status</button>
                <button class="btn success" onclick="window.location.href='https://documenter.getpostman.com/view/6316436/2sA3XLjK6H'">API Documentation</button>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'karanka-ultra-frequent-bot',
        'timestamp': datetime.now().isoformat(),
        'version': 'V8-ULTRA-FREQUENT',
        'strategies': ['Liquidity Sweep', 'Inside Bar', 'Order Block', 'Structure Break', 'FVG', 'RSI Divergence']
    })

# ============ ERROR HANDLERS ============
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'message': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"❌ Internal server error: {error}")
    return jsonify({'success': False, 'message': 'Internal server error'}), 500

# ============ START APPLICATION ============
if __name__ == '__main__':
    port = Config.PORT
    
    print("\n" + "="*80)
    print("🚀 KARANKA V8 - ULTRA-FREQUENT SMC TRADING BOT")
    print("="*80)
    print("⚡ STRATEGIES ENABLED:")
    print("   1. Liquidity Sweep (MOST FREQUENT)")
    print("   2. Inside Bar Breakout (VERY FREQUENT)")
    print("   3. Order Block (FREQUENT)")
    print("   4. Structure Break (FREQUENT)")
    print("   5. Fair Value Gap (FREQUENT)")
    print("   6. RSI Divergence (MODERATE)")
    print("="*80)
    print(f"📊 Markets: {len(DERIV_MARKETS)}")
    print(f"💰 Min Trade: ${Config.MIN_TRADE_AMOUNT}")
    print(f"⏱️ Trade Duration: {Config.TRADE_DURATION} minutes")
    print(f"🎯 Max Concurrent Trades: {Config.MAX_CONCURRENT_TRADES}")
    print("="*80)
    print("⚠️  WARNING: This bot is configured for ULTRA-FREQUENT trading")
    print("    Expected: 50-100+ trades/hour possible")
    print("    Start in DRY RUN mode to test frequency")
    print("="*80)
    
    app.run(host='0.0.0.0', port=port, debug=Config.DEBUG, threaded=True)
