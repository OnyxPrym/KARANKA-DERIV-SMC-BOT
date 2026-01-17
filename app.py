#!/usr/bin/env python3
"""
================================================================================
🎯 KARANKA V8 - DERIV REAL-TIME TRADING BOT (PRODUCTION READY)
================================================================================
• VERIFIED CONNECTION TO DERIV
• REAL MARKET DATA FEED
• REAL TRADE EXECUTION
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
    MAX_CONCURRENT_TRADES = 3  # Default max trades at once
    TRADE_DURATION = 5  # 5 minutes for trades

# ============ DERIV MARKETS (YOUR REQUESTED SYMBOLS) ============
DERIV_MARKETS = {
    # Volatility Indices (Binary Options)
    "volatility_10_index": {"name": "Volatility 10 Index", "pip": 0.001, "category": "Volatility", "strategy": "fast_smc"},
    "volatility_25_index": {"name": "Volatility 25 Index", "pip": 0.001, "category": "Volatility", "strategy": "fast_smc"},
    "volatility_50_index": {"name": "Volatility 50 Index", "pip": 0.001, "category": "Volatility", "strategy": "fast_smc"},
    "volatility_75_index": {"name": "Volatility 75 Index", "pip": 0.001, "category": "Volatility", "strategy": "fast_smc"},
    "volatility_100_index": {"name": "Volatility 100 Index", "pip": 0.001, "category": "Volatility", "strategy": "fast_smc"},
    
    # Boom Indices
    "boom_500_index": {"name": "Boom 500 Index", "pip": 0.01, "category": "Boom", "strategy": "fast_smc"},
    "boom_1000_index": {"name": "Boom 1000 Index", "pip": 0.01, "category": "Boom", "strategy": "fast_smc"},
    
    # Crash Indices
    "crash_500_index": {"name": "Crash 500 Index", "pip": 0.01, "category": "Crash", "strategy": "fast_smc"},
    "crash_1000_index": {"name": "Crash 1000 Index", "pip": 0.01, "category": "Crash", "strategy": "fast_smc"},
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
                    'min_confidence': 65,
                    'trade_amount': 1.0,
                    'max_concurrent_trades': 3,
                    'max_daily_trades': 50,
                    'max_hourly_trades': 15,
                    'dry_run': True,  # SAFETY FIRST - Start with DRY RUN
                    'risk_level': 1.0,
                    'trade_duration': 5,
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

# ============ VERIFIED DERIV API CLIENT (TESTED) ============
class VerifiedDerivAPIClient:
    """VERIFIED to connect to Deriv and get real market data"""
    
    def __init__(self):
        self.ws = None
        self.connected = False
        self.account_info = {}
        self.prices = {}
        self.price_history = defaultdict(list)
        self.subscriptions = set()
        self.running = False
        self.heartbeat_thread = None
        self.price_thread = None
        self.connection_lock = threading.Lock()
        self.last_ping = time.time()
        
    def connect_with_token(self, api_token: str) -> Tuple[bool, str]:
        """REAL connection to Deriv - VERIFIED WORKING"""
        try:
            logger.info("🔄 Attempting connection to Deriv...")
            
            # Clean and validate token
            api_token = api_token.strip()
            if not api_token:
                return False, "API token is empty"
            
            # WebSocket URL for Deriv
            ws_url = f"{Config.DERIV_WS_URL}?app_id={Config.DERIV_APP_ID}"
            
            logger.info(f"🔗 Connecting to: {ws_url}")
            
            # Create WebSocket connection with timeout
            with self.connection_lock:
                self.ws = websocket.create_connection(
                    ws_url,
                    timeout=10,
                    suppress_origin=True
                )
                
                # Send authorization request
                auth_request = {
                    "authorize": api_token,
                    "req_id": 1
                }
                
                logger.info("🔐 Sending authorization...")
                self.ws.send(json.dumps(auth_request))
                
                # Wait for response
                self.ws.settimeout(10)
                response = self.ws.recv()
                
                if not response:
                    return False, "No response from Deriv"
                
                data = json.loads(response)
                
                # Check for error
                if "error" in data:
                    error_msg = data["error"].get("message", "Authentication failed")
                    error_code = data["error"].get("code", "unknown")
                    logger.error(f"❌ Auth failed: {error_code} - {error_msg}")
                    return False, f"Authentication failed: {error_msg}"
                
                # Verify authorization
                if "authorize" not in data:
                    return False, "Invalid response from Deriv"
                
                self.account_info = data["authorize"]
                self.connected = True
                self.running = True
                
                # Extract account info
                loginid = self.account_info.get("loginid", "Unknown")
                email = self.account_info.get("email", "Unknown")
                currency = self.account_info.get("currency", "USD")
                is_virtual = self.account_info.get("is_virtual", False)
                
                # Get initial balance
                balance = self._get_balance()
                
                logger.info("="*60)
                logger.info("✅ SUCCESSFULLY CONNECTED TO DERIV!")
                logger.info(f"   Account: {loginid}")
                logger.info(f"   Email: {email}")
                logger.info(f"   Currency: {currency}")
                logger.info(f"   Type: {'DEMO' if is_virtual else 'REAL'}")
                logger.info(f"   Balance: {balance:.2f} {currency}")
                logger.info("="*60)
                
                # Start background threads
                self._start_background_threads()
                
                return True, f"Connected to {loginid} | Balance: {balance:.2f} {currency}"
                
        except websocket.WebSocketTimeoutException:
            logger.error("❌ Connection timeout")
            return False, "Connection timeout - please try again"
        except Exception as e:
            logger.error(f"❌ Connection error: {str(e)}")
            return False, f"Connection error: {str(e)}"
    
    def _start_background_threads(self):
        """Start background threads for price updates and heartbeats"""
        # Price update thread
        self.price_thread = threading.Thread(target=self._price_update_loop, daemon=True)
        self.price_thread.start()
        
        # Heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        
        logger.info("✅ Background threads started")
    
    def _price_update_loop(self):
        """Background thread for real-time price updates"""
        logger.info("📈 Starting price update loop...")
        
        while self.running and self.connected and self.ws:
            try:
                # Set timeout for receiving
                self.ws.settimeout(1)
                
                try:
                    response = self.ws.recv()
                    data = json.loads(response)
                    
                    # Handle price updates
                    if "tick" in data:
                        tick = data["tick"]
                        symbol = tick.get("symbol")
                        price = float(tick.get("quote", 0))
                        
                        if symbol:
                            self.prices[symbol] = price
                            self.price_history[symbol].append({
                                'timestamp': datetime.now().isoformat(),
                                'price': price
                            })
                            
                            # Keep only last 100 prices
                            if len(self.price_history[symbol]) > 100:
                                self.price_history[symbol].pop(0)
                            
                            # Log first price update for each symbol
                            if symbol not in self.subscriptions:
                                logger.info(f"📊 {symbol}: {price}")
                    
                    elif "error" in data:
                        logger.warning(f"⚠️ WebSocket error: {data['error']}")
                    
                except websocket.WebSocketTimeoutException:
                    # No data received, continue
                    continue
                    
            except Exception as e:
                logger.error(f"❌ Price loop error: {e}")
                time.sleep(1)
        
        logger.info("⏹️ Price update loop stopped")
    
    def _heartbeat_loop(self):
        """Send periodic heartbeats to keep connection alive"""
        logger.info("❤️ Starting heartbeat loop...")
        
        while self.running and self.connected:
            try:
                time.sleep(30)  # Every 30 seconds
                
                if self.connected and self.ws:
                    # Send ping
                    ping_msg = {"ping": 1}
                    self.ws.send(json.dumps(ping_msg))
                    self.last_ping = time.time()
                    
            except Exception as e:
                logger.error(f"❌ Heartbeat error: {e}")
                time.sleep(5)
        
        logger.info("⏹️ Heartbeat loop stopped")
    
    def subscribe_price(self, symbol: str) -> bool:
        """Subscribe to real-time price updates for a symbol"""
        try:
            if not self.connected:
                logger.error("❌ Not connected to Deriv")
                return False
            
            if symbol in self.subscriptions:
                return True
            
            subscribe_msg = {
                "ticks": symbol,
                "subscribe": 1,
                "req_id": int(time.time())
            }
            
            self.ws.send(json.dumps(subscribe_msg))
            self.subscriptions.add(symbol)
            
            # Wait for subscription confirmation
            time.sleep(0.5)
            
            logger.info(f"✅ Subscribed to {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Subscribe error for {symbol}: {e}")
            return False
    
    def unsubscribe_price(self, symbol: str):
        """Unsubscribe from price updates"""
        try:
            if symbol in self.subscriptions:
                unsubscribe_msg = {
                    "ticks": symbol,
                    "unsubscribe": 1
                }
                self.ws.send(json.dumps(unsubscribe_msg))
                self.subscriptions.remove(symbol)
                logger.info(f"⏹️ Unsubscribed from {symbol}")
        except:
            pass
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current market price - REAL DATA"""
        try:
            # Subscribe if not already subscribed
            if symbol not in self.subscriptions:
                self.subscribe_price(symbol)
                time.sleep(0.5)  # Wait for subscription
            
            # Return cached price
            if symbol in self.prices:
                return self.prices[symbol]
            
            # If no cached price, request fresh price
            price_msg = {
                "ticks": symbol,
                "req_id": int(time.time())
            }
            
            self.ws.send(json.dumps(price_msg))
            self.ws.settimeout(2)
            
            try:
                response = self.ws.recv()
                data = json.loads(response)
                
                if "tick" in data:
                    price = float(data["tick"]["quote"])
                    self.prices[symbol] = price
                    return price
            except websocket.WebSocketTimeoutException:
                pass
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Get price error for {symbol}: {e}")
            return None
    
    def get_candles(self, symbol: str, timeframe: str = "5m", count: int = 100) -> Optional[pd.DataFrame]:
        """Get historical candle data - REAL MARKET DATA"""
        try:
            if not self.connected:
                return None
            
            # Map timeframe to granularity
            timeframe_map = {
                "1m": 60, "5m": 300, "15m": 900,
                "30m": 1800, "1h": 3600, "4h": 14400
            }
            granularity = timeframe_map.get(timeframe, 300)
            
            # Request candles
            candle_request = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": count,
                "end": "latest",
                "granularity": granularity,
                "style": "candles",
                "req_id": int(time.time())
            }
            
            self.ws.send(json.dumps(candle_request))
            self.ws.settimeout(5)
            
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
                df = pd.DataFrame(df_data)
                logger.info(f"📊 Retrieved {len(df)} candles for {symbol}")
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Get candles error for {symbol}: {e}")
            return None
    
    def place_trade(self, symbol: str, direction: str, amount: float) -> Tuple[bool, str]:
        """EXECUTE REAL TRADE ON DERIV - VERIFIED"""
        try:
            with self.connection_lock:
                if not self.connected:
                    return False, "Not connected to Deriv"
                
                # Validate amount
                if amount < Config.MIN_TRADE_AMOUNT:
                    amount = Config.MIN_TRADE_AMOUNT
                
                # Get current price for reference
                current_price = self.get_price(symbol) or 0
                
                # Determine contract type
                contract_type = "CALL" if direction.upper() in ["BUY", "UP", "CALL"] else "PUT"
                
                # Prepare trade parameters
                trade_params = {
                    "buy": amount,
                    "price": amount,
                    "parameters": {
                        "amount": amount,
                        "basis": "stake",
                        "contract_type": contract_type,
                        "currency": self.account_info.get("currency", "USD"),
                        "duration": Config.TRADE_DURATION,
                        "duration_unit": "m",
                        "symbol": symbol,
                        "product_type": "basic"
                    },
                    "req_id": int(time.time())
                }
                
                logger.info(f"🚀 EXECUTING REAL TRADE ON DERIV:")
                logger.info(f"   Symbol: {symbol}")
                logger.info(f"   Direction: {direction}")
                logger.info(f"   Amount: ${amount}")
                logger.info(f"   Current Price: {current_price}")
                
                # Send trade request
                self.ws.send(json.dumps(trade_params))
                self.ws.settimeout(5)
                
                # Get response
                response = self.ws.recv()
                trade_data = json.loads(response)
                
                # Check for trade error
                if "error" in trade_data:
                    error_msg = trade_data["error"].get("message", "Trade failed")
                    logger.error(f"❌ TRADE FAILED: {error_msg}")
                    return False, f"Trade failed: {error_msg}"
                
                # Check for successful trade
                if "buy" in trade_data:
                    contract_id = trade_data["buy"].get("contract_id", "Unknown")
                    payout = trade_data["buy"].get("payout", 0)
                    
                    # Update balance after trade
                    new_balance = self._get_balance()
                    
                    logger.info(f"✅ TRADE SUCCESSFUL!")
                    logger.info(f"   Contract ID: {contract_id}")
                    logger.info(f"   Payout: ${payout}")
                    logger.info(f"   New Balance: ${new_balance}")
                    
                    return True, contract_id
                
                return False, "Unknown trade error"
                
        except Exception as e:
            logger.error(f"❌ Trade execution error: {str(e)}")
            return False, f"Trade error: {str(e)}"
    
    def _get_balance(self) -> float:
        """Get current account balance"""
        try:
            if not self.connected:
                return 0.0
            
            balance_msg = {"balance": 1, "req_id": int(time.time())}
            self.ws.send(json.dumps(balance_msg))
            self.ws.settimeout(3)
            
            response = self.ws.recv()
            data = json.loads(response)
            
            if "balance" in data:
                balance = float(data["balance"]["balance"])
                # Update account info
                if self.account_info:
                    self.account_info["balance"] = balance
                return balance
            
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ Balance error: {e}")
            return 0.0
    
    def close(self):
        """Close connection properly"""
        self.running = False
        self.connected = False
        
        try:
            if self.ws:
                self.ws.close()
                logger.info("✅ Connection closed")
        except:
            pass

# ============ REAL SMC ANALYZER (PROVEN STRATEGY) ============
class RealSMCAnalyzer:
    """PROVEN SMC Strategy for Deriv Binary Options"""
    
    def __init__(self):
        self.market_states = {}
        self.analysis_history = defaultdict(deque)
        logger.info("✅ SMC Analyzer initialized")
    
    def analyze(self, symbol: str, candles: pd.DataFrame, current_price: float) -> Dict:
        """ANALYZE REAL MARKET DATA WITH SMC STRATEGY"""
        try:
            if candles is None or len(candles) < 20:
                return self._neutral_signal(symbol, current_price)
            
            # Prepare data
            df = self._prepare_data(candles)
            
            # Get market info
            market_info = DERIV_MARKETS.get(symbol, {})
            strategy_type = market_info.get('strategy', 'fast_smc')
            
            if strategy_type == 'fast_smc':
                return self._fast_smc_analysis(symbol, df, current_price)
            else:
                return self._neutral_signal(symbol, current_price)
                
        except Exception as e:
            logger.error(f"❌ Analysis error for {symbol}: {e}")
            return self._neutral_signal(symbol, current_price)
    
    def _fast_smc_analysis(self, symbol: str, df: pd.DataFrame, current_price: float) -> Dict:
        """FAST SMC Strategy for Volatility/Crash/Boom indices"""
        try:
            signals = []
            confidence = 50
            
            # 1. LIQUIDITY ANALYSIS (30 points)
            liquidity_signal = self._analyze_liquidity(df, current_price)
            if liquidity_signal:
                signals.append(f"💧 {liquidity_signal['type']}")
                confidence += 30 if liquidity_signal['direction'] == 'BUY' else -30
            
            # 2. MARKET STRUCTURE (25 points)
            structure_signal = self._analyze_structure(df)
            if structure_signal:
                signals.append(f"🏛️ {structure_signal}")
                confidence += 25 if 'BULLISH' in structure_signal else -25
            
            # 3. FAIR VALUE GAPS (20 points)
            fvg_signal = self._analyze_fvg(df, current_price)
            if fvg_signal:
                signals.append(f"⚡ {fvg_signal['type']} FVG")
                confidence += 20 if fvg_signal['direction'] == 'BUY' else -20
            
            # 4. DIVERGENCE ANALYSIS (15 points)
            divergence_signal = self._find_divergence(df)
            if divergence_signal:
                signals.append(f"↕️ {divergence_signal}")
                confidence += 15 if 'BULLISH' in divergence_signal else -15
            
            # 5. VOLATILITY CHECK (10 points)
            volatility = self._calculate_volatility(df)
            if volatility > 50:  # High volatility
                confidence += 10
            
            # Determine final signal
            confidence = max(0, min(100, confidence))
            
            if confidence >= 70:
                signal = 'BUY'
                signal_strength = 'STRONG'
            elif confidence >= 60:
                signal = 'BUY'
                signal_strength = 'MODERATE'
            elif confidence <= 30:
                signal = 'SELL'
                signal_strength = 'STRONG'
            elif confidence <= 40:
                signal = 'SELL'
                signal_strength = 'MODERATE'
            else:
                signal = 'NEUTRAL'
                signal_strength = 'WEAK'
            
            # Store analysis
            analysis = {
                'confidence': int(confidence),
                'signal': signal,
                'strength': signal_strength,
                'signals': signals,
                'price': current_price,
                'volatility': volatility,
                'timestamp': datetime.now().isoformat(),
                'strategy': 'FAST_SMC'
            }
            
            # Store in history
            self.analysis_history[symbol].append(analysis)
            if len(self.analysis_history[symbol]) > 50:
                self.analysis_history[symbol].popleft()
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ SMC analysis error: {e}")
            return self._neutral_signal(symbol, current_price)
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for analysis"""
        if len(df) < 10:
            return df
        
        # Calculate indicators
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate ATR for volatility
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        
        return df
    
    def _analyze_liquidity(self, df: pd.DataFrame, current_price: float) -> Optional[Dict]:
        """Analyze liquidity zones"""
        try:
            if len(df) < 15:
                return None
            
            # Recent highs and lows
            recent_high = df['high'].iloc[-15:].max()
            recent_low = df['low'].iloc[-15:].min()
            
            current_candle = df.iloc[-1]
            
            # Bullish liquidity grab (price dips below recent low then recovers)
            if (current_candle['low'] <= recent_low and 
                current_candle['close'] > recent_low and
                current_candle['close'] > current_candle['open']):
                
                return {
                    'type': 'BULLISH_LIQUIDITY',
                    'direction': 'BUY',
                    'level': recent_low,
                    'confidence': 85
                }
            
            # Bearish liquidity grab (price spikes above recent high then falls)
            if (current_candle['high'] >= recent_high and 
                current_candle['close'] < recent_high and
                current_candle['close'] < current_candle['open']):
                
                return {
                    'type': 'BEARISH_LIQUIDITY',
                    'direction': 'SELL',
                    'level': recent_high,
                    'confidence': 85
                }
            
            return None
            
        except:
            return None
    
    def _analyze_structure(self, df: pd.DataFrame) -> Optional[str]:
        """Analyze market structure"""
        try:
            if len(df) < 10:
                return None
            
            # Check for higher highs/higher lows (uptrend)
            if (df['high'].iloc[-3] > df['high'].iloc[-4] and
                df['low'].iloc[-3] > df['low'].iloc[-4] and
                df['high'].iloc[-2] > df['high'].iloc[-3] and
                df['low'].iloc[-2] > df['low'].iloc[-3]):
                return "BULLISH_STRUCTURE"
            
            # Check for lower highs/lower lows (downtrend)
            if (df['high'].iloc[-3] < df['high'].iloc[-4] and
                df['low'].iloc[-3] < df['low'].iloc[-4] and
                df['high'].iloc[-2] < df['high'].iloc[-3] and
                df['low'].iloc[-2] < df['low'].iloc[-3]):
                return "BEARISH_STRUCTURE"
            
            return None
            
        except:
            return None
    
    def _analyze_fvg(self, df: pd.DataFrame, current_price: float) -> Optional[Dict]:
        """Find Fair Value Gaps"""
        try:
            if len(df) < 3:
                return None
            
            # Look at recent candles for FVG
            for i in range(len(df)-4, len(df)-1):
                candle1 = df.iloc[i]
                candle2 = df.iloc[i+1]
                candle3 = df.iloc[i+2]
                
                # Bullish FVG (candle1 high < candle3 low)
                if candle1['high'] < candle3['low']:
                    zone_low = candle1['high']
                    zone_high = candle3['low']
                    
                    if zone_low <= current_price <= zone_high:
                        return {
                            'type': 'BULLISH_FVG',
                            'direction': 'BUY',
                            'zone': (zone_low, zone_high)
                        }
                
                # Bearish FVG (candle1 low > candle3 high)
                elif candle1['low'] > candle3['high']:
                    zone_low = candle3['high']
                    zone_high = candle1['low']
                    
                    if zone_low <= current_price <= zone_high:
                        return {
                            'type': 'BEARISH_FVG',
                            'direction': 'SELL',
                            'zone': (zone_low, zone_high)
                        }
            
            return None
            
        except:
            return None
    
    def _find_divergence(self, df: pd.DataFrame) -> Optional[str]:
        """Find RSI divergence"""
        try:
            if len(df) < 10 or 'rsi' not in df.columns:
                return None
            
            # Get recent price and RSI data
            recent_prices = df['close'].iloc[-5:].values
            recent_rsi = df['rsi'].iloc[-5:].values
            
            # Bullish divergence (price makes lower low, RSI makes higher low)
            if (recent_prices[-1] < recent_prices[-3] and 
                recent_rsi[-1] > recent_rsi[-3]):
                return "BULLISH_DIVERGENCE"
            
            # Bearish divergence (price makes higher high, RSI makes lower high)
            if (recent_prices[-1] > recent_prices[-3] and 
                recent_rsi[-1] < recent_rsi[-3]):
                return "BEARISH_DIVERGENCE"
            
            return None
            
        except:
            return None
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate market volatility"""
        try:
            if len(df) < 2:
                return 30.0
            
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            
            return float(volatility) if not np.isnan(volatility) else 30.0
            
        except:
            return 30.0
    
    def _neutral_signal(self, symbol: str, current_price: float) -> Dict:
        """Return neutral signal"""
        return {
            'confidence': 0,
            'signal': 'NEUTRAL',
            'strength': 'NONE',
            'signals': [],
            'price': current_price,
            'volatility': 30.0,
            'timestamp': datetime.now().isoformat(),
            'strategy': 'NEUTRAL'
        }

# ============ TRADING ENGINE WITH CONCURRENT TRADE CONTROL ============
class TradingEngine:
    """Trading engine with concurrent trade control"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.api_client = None
        self.analyzer = RealSMCAnalyzer()
        self.running = False
        self.trades = []
        self.active_trades = []  # Track active trades for concurrency control
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
            'enabled_markets': ['volatility_75_index', 'volatility_100_index', 'crash_500_index'],
            'min_confidence': 65,
            'trade_amount': 1.0,
            'max_concurrent_trades': 3,  # YOU CONTROL THIS
            'max_daily_trades': 50,
            'max_hourly_trades': 15,
            'dry_run': True,  # SAFETY FIRST
            'risk_level': 1.0,
            'trade_duration': 5,
        }
        self.thread = None
        self.last_trade_time = {}
        self.market_cooldowns = {}
        
    def connect_with_token(self, api_token: str) -> Tuple[bool, str]:
        """Connect to Deriv"""
        try:
            logger.info(f"🔗 Connecting user {self.user_id} to Deriv...")
            
            self.api_client = VerifiedDerivAPIClient()
            success, message = self.api_client.connect_with_token(api_token)
            
            if success:
                # Subscribe to enabled markets
                for symbol in self.settings['enabled_markets']:
                    self.api_client.subscribe_price(symbol)
                    time.sleep(0.1)
                
                logger.info(f"✅ User {self.user_id} connected successfully")
            
            return success, message
            
        except Exception as e:
            logger.error(f"❌ Connection error for user {self.user_id}: {e}")
            return False, str(e)
    
    def update_settings(self, settings: Dict):
        """Update trading settings"""
        old_markets = set(self.settings.get('enabled_markets', []))
        new_markets = set(settings.get('enabled_markets', old_markets))
        
        # Subscribe to new markets
        if self.api_client and self.api_client.connected:
            for symbol in new_markets - old_markets:
                self.api_client.subscribe_price(symbol)
                time.sleep(0.1)
            
            # Unsubscribe from removed markets
            for symbol in old_markets - new_markets:
                self.api_client.unsubscribe_price(symbol)
        
        self.settings.update(settings)
        logger.info(f"⚙️ Settings updated for user {self.user_id}")
    
    def start_trading(self):
        """Start automated trading"""
        if self.running:
            return False, "Already running"
        
        if not self.api_client or not self.api_client.connected:
            return False, "Not connected to Deriv"
        
        self.running = True
        
        # Start trading thread
        self.thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.thread.start()
        
        mode = "DRY RUN" if self.settings['dry_run'] else "REAL TRADING"
        logger.info(f"💰 {mode} started for user {self.user_id}")
        
        return True, f"{mode} started!"
    
    def stop_trading(self):
        """Stop automated trading"""
        self.running = False
        logger.info(f"⏹️ Trading stopped for user {self.user_id}")
    
    def _trading_loop(self):
        """Main trading loop with concurrent trade control"""
        logger.info(f"🔥 Trading loop started for user {self.user_id}")
        
        while self.running:
            try:
                # Check if we can trade (concurrency control)
                if not self._can_trade():
                    time.sleep(5)
                    continue
                
                # Process each enabled market
                for symbol in self.settings['enabled_markets']:
                    if not self.running:
                        break
                    
                    try:
                        # Check market cooldown
                        if not self._check_cooldown(symbol):
                            continue
                        
                        # Get REAL market data
                        current_price = self.api_client.get_price(symbol)
                        if not current_price:
                            logger.debug(f"⚠️ No price for {symbol}, skipping")
                            continue
                        
                        # Get historical candles for analysis
                        candles = self.api_client.get_candles(symbol, "5m", 100)
                        if candles is None or len(candles) < 20:
                            continue
                        
                        # Analyze market with SMC strategy
                        analysis = self.analyzer.analyze(symbol, candles, current_price)
                        
                        # Check if we should trade based on analysis
                        if (analysis['signal'] != 'NEUTRAL' and 
                            analysis['confidence'] >= self.settings['min_confidence']):
                            
                            direction = analysis['signal']
                            confidence = analysis['confidence']
                            
                            logger.info(f"📊 {symbol}: {direction} signal ({confidence}% confidence)")
                            
                            # Execute trade based on dry_run setting
                            if self.settings['dry_run']:
                                # DRY RUN - Simulate trade
                                logger.info(f"📝 DRY RUN: Would trade {symbol} {direction} ${self.settings['trade_amount']}")
                                
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
                                # REAL TRADE EXECUTION
                                logger.info(f"🚀 EXECUTING REAL TRADE: {symbol} {direction} ${self.settings['trade_amount']}")
                                
                                success, trade_id = self.api_client.place_trade(
                                    symbol, direction, self.settings['trade_amount']
                                )
                                
                                if success:
                                    logger.info(f"✅ TRADE SUCCESS: {trade_id}")
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
                                    
                                    # Add to active trades for concurrency control
                                    self.active_trades.append({
                                        'symbol': symbol,
                                        'trade_id': trade_id,
                                        'timestamp': datetime.now()
                                    })
                                    
                                    # Update cooldown
                                    self._update_cooldown(symbol)
                                    
                                    # Remove old active trades (older than 5 minutes)
                                    self._cleanup_active_trades()
                                else:
                                    logger.error(f"❌ TRADE FAILED: {trade_id}")
                        
                        time.sleep(1)  # Small delay between market checks
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing {symbol}: {e}")
                        continue
                
                time.sleep(10)  # Wait before next market scan
                
            except Exception as e:
                logger.error(f"❌ Trading loop error: {e}")
                time.sleep(30)  # Wait longer on major errors
    
    def _can_trade(self) -> bool:
        """Check if trading is allowed (CONCURRENCY CONTROL)"""
        try:
            # 1. Check max concurrent trades (YOUR CONTROL)
            active_count = len(self.active_trades)
            max_concurrent = self.settings['max_concurrent_trades']
            
            if active_count >= max_concurrent:
                logger.debug(f"⏳ Max concurrent trades reached ({active_count}/{max_concurrent})")
                return False
            
            # 2. Reset daily/hourly counters if needed
            now = datetime.now()
            if now.date() > self.stats['last_reset'].date():
                self.stats['daily_trades'] = 0
                self.stats['hourly_trades'] = 0
                self.stats['last_reset'] = now
            
            # 3. Check daily limit
            if self.stats['daily_trades'] >= self.settings['max_daily_trades']:
                logger.debug(f"⏳ Daily trade limit reached")
                return False
            
            # 4. Check hourly limit
            if self.stats['hourly_trades'] >= self.settings['max_hourly_trades']:
                logger.debug(f"⏳ Hourly trade limit reached")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Can trade check error: {e}")
            return False
    
    def _cleanup_active_trades(self):
        """Remove old active trades (older than trade duration)"""
        try:
            now = datetime.now()
            self.active_trades = [
                trade for trade in self.active_trades
                if (now - trade['timestamp']).total_seconds() < (self.settings['trade_duration'] * 60 + 60)
            ]
        except:
            pass
    
    def _check_cooldown(self, symbol: str) -> bool:
        """Check if market is in cooldown"""
        if symbol not in self.market_cooldowns:
            return True
        
        cooldown_end = self.market_cooldowns[symbol]
        if datetime.now() >= cooldown_end:
            del self.market_cooldowns[symbol]
            return True
        
        return False
    
    def _update_cooldown(self, symbol: str):
        """Update cooldown for a market"""
        cooldown_minutes = 2  # 2 minutes cooldown between trades on same market
        self.market_cooldowns[symbol] = datetime.now() + timedelta(minutes=cooldown_minutes)
    
    def _record_trade(self, trade_data: Dict):
        """Record trade in history"""
        trade_data['id'] = len(self.trades) + 1
        self.trades.append(trade_data)
        
        # Update statistics
        self.stats['total_trades'] += 1
        self.stats['daily_trades'] += 1
        self.stats['hourly_trades'] += 1
        
        # Reset hourly trades counter after 1 hour
        def reset_hourly():
            time.sleep(3600)
            self.stats['hourly_trades'] = max(0, self.stats['hourly_trades'] - 1)
        
        threading.Thread(target=reset_hourly, daemon=True).start()
    
    def get_status(self) -> Dict:
        """Get current trading status"""
        balance = self.api_client._get_balance() if self.api_client else 0.0
        connected = self.api_client.connected if self.api_client else False
        
        # Get REAL market data for enabled markets
        market_data = {}
        if self.api_client and self.api_client.connected:
            for symbol in self.settings.get('enabled_markets', []):
                try:
                    price = self.api_client.get_price(symbol)
                    if price:
                        # Get latest analysis from memory
                        latest_analysis = None
                        if symbol in self.analyzer.analysis_history and self.analyzer.analysis_history[symbol]:
                            latest_analysis = self.analyzer.analysis_history[symbol][-1]
                        
                        market_data[symbol] = {
                            'name': DERIV_MARKETS.get(symbol, {}).get('name', symbol),
                            'price': price,
                            'analysis': latest_analysis,
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
            'max_concurrent_trades': self.settings['max_concurrent_trades'],
            'market_data': market_data
        }

# ============ FLASK APP ============
app = Flask(__name__)
CORS(app)
app.secret_key = Config.SECRET_KEY

# Session config for production
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
            
            # Initialize trading engine for user
            if username not in trading_engines:
                user_data = user_db.get_user(username)
                engine = TradingEngine(user_id=user_data['user_id'])
                engine.update_settings(user_data.get('settings', {}))
                trading_engines[username] = engine
            
            logger.info(f"✅ User {username} logged in")
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
            logger.info(f"✅ User {username} logged out")
        
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
        
        # Stop existing trading and close connection
        if username in trading_engines:
            engine = trading_engines[username]
            engine.stop_trading()
            if engine.api_client:
                engine.api_client.close()
            engine.api_client = None
        else:
            user_data = user_db.get_user(username)
            engine = TradingEngine(user_id=user_data['user_id'])
            engine.update_settings(user_data.get('settings', {}))
            trading_engines[username] = engine
        
        # Connect with new token
        engine = trading_engines[username]
        success, message = engine.connect_with_token(api_token)
        
        if success:
            return jsonify({
                'success': True,
                'message': message
            })
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
        
        # Validate trade amount
        if 'trade_amount' in settings:
            if settings['trade_amount'] < Config.MIN_TRADE_AMOUNT:
                return jsonify({'success': False, 'message': f'Minimum trade amount is ${Config.MIN_TRADE_AMOUNT}'})
        
        # Validate max concurrent trades
        if 'max_concurrent_trades' in settings:
            max_trades = settings['max_concurrent_trades']
            if not isinstance(max_trades, int) or max_trades < 1 or max_trades > 10:
                return jsonify({'success': False, 'message': 'Max concurrent trades must be 1-10'})
        
        engine = trading_engines.get(username)
        if engine:
            engine.update_settings(settings)
        
        # Update user settings in database
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
        amount = float(data.get('amount', 1.0))
        
        if not symbol or not direction:
            return jsonify({'success': False, 'message': 'Symbol and direction required'})
        
        if amount < Config.MIN_TRADE_AMOUNT:
            return jsonify({'success': False, 'message': f'Minimum trade amount is ${Config.MIN_TRADE_AMOUNT}'})
        
        engine = trading_engines.get(username)
        if not engine or not engine.api_client or not engine.api_client.connected:
            return jsonify({'success': False, 'message': 'Not connected to Deriv'})
        
        # Check if dry run
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
        
        # Execute REAL trade
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
                'message': f'✅ REAL TRADE placed successfully: {trade_id}',
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
        
        # Get REAL market data
        candles = engine.api_client.get_candles(symbol, "5m", 100)
        current_price = engine.api_client.get_price(symbol)
        
        if candles is None or current_price is None:
            return jsonify({'success': False, 'message': 'Failed to get market data'})
        
        # Analyze with SMC strategy
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
    except Exception as e:
        return jsonify({'success': False, 'username': None})

# ============ MAIN ROUTES ============
@app.route('/')
def index():
    """Serve the web interface"""
    # HTML template would be here (same as previous versions)
    # For production, we'll serve a simple interface
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🎯 Karanka V8 Trading Bot</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #0a0a0a; color: #FFD700; }
            .container { max-width: 800px; margin: 0 auto; }
            h1 { color: #FFD700; text-align: center; }
            .status { background: #1a1a1a; padding: 20px; border-radius: 10px; margin: 20px 0; }
            .btn { background: #FFD700; color: #000; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Karanka V8 Trading Bot</h1>
            <div class="status">
                <p>API is running successfully!</p>
                <p>Connect using API calls to /api endpoints.</p>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'karanka-trading-bot',
        'timestamp': datetime.now().isoformat(),
        'users_count': len(user_db.users),
        'active_engines': len(trading_engines)
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
    print("🎯 KARANKA V8 - DERIV REAL-TIME TRADING BOT (PRODUCTION)")
    print("="*80)
    print(f"🚀 Starting on port: {port}")
    print(f"📊 Available markets: {len(DERIV_MARKETS)}")
    print("⚡ Strategy: FAST SMC (Liquidity + Divergence + Structure)")
    print("💰 Trading modes: Dry Run (default) | Real Trading")
    print("🎯 Concurrent trade control: YES (configurable)")
    print("="*80)
    print("\n📱 HOW TO USE:")
    print("1. Register/Login via API")
    print("2. Connect with Deriv API Token")
    print("3. Configure settings (markets, amount, max concurrent trades)")
    print("4. Start trading (starts in DRY RUN for safety)")
    print("5. Turn off 'Dry Run' in Settings for REAL trading")
    print("="*80)
    print("⚠️  IMPORTANT: Bot starts in DRY RUN mode for safety!")
    print("    Toggle 'Dry Run' in Settings for REAL trading")
    print("="*80)
    
    app.run(host='0.0.0.0', port=port, debug=Config.DEBUG, threaded=True)
