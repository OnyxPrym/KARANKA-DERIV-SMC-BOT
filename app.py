#!/usr/bin/env python3
"""
================================================================================
🎯 KARANKA V8 - DERIV REAL-TIME TRADING BOT (PRODUCTION)
================================================================================
• REAL DERIV API CONNECTION
• LIVE MARKET DATA
• REAL TRADE EXECUTION
• FULL UI WITH ALL FEATURES
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============ DERIV MARKETS ============
DERIV_MARKETS = {
    "frxEURUSD": {"name": "EUR/USD", "pip": 0.0001, "category": "Forex", "strategy_type": "forex"},
    "frxGBPUSD": {"name": "GBP/USD", "pip": 0.0001, "category": "Forex", "strategy_type": "forex"},
    "frxUSDJPY": {"name": "USD/JPY", "pip": 0.01, "category": "Forex", "strategy_type": "forex"},
    "frxAUDUSD": {"name": "AUD/USD", "pip": 0.0001, "category": "Forex", "strategy_type": "forex"},
    "R_25": {"name": "Volatility 25 Index", "pip": 0.001, "category": "Volatility", "strategy_type": "volatility"},
    "R_50": {"name": "Volatility 50 Index", "pip": 0.001, "category": "Volatility", "strategy_type": "volatility"},
    "R_75": {"name": "Volatility 75 Index", "pip": 0.001, "category": "Volatility", "strategy_type": "volatility"},
    "R_100": {"name": "Volatility 100 Index", "pip": 0.001, "category": "Volatility", "strategy_type": "volatility"},
    "CRASH_300": {"name": "Crash 300 Index", "pip": 0.01, "category": "Crash/Boom", "strategy_type": "crash"},
    "CRASH_500": {"name": "Crash 500 Index", "pip": 0.01, "category": "Crash/Boom", "strategy_type": "crash"},
    "BOOM_300": {"name": "Boom 300 Index", "pip": 0.01, "category": "Crash/Boom", "strategy_type": "boom"},
    "BOOM_500": {"name": "Boom 500 Index", "pip": 0.01, "category": "Crash/Boom", "strategy_type": "boom"},
    "cryBTCUSD": {"name": "BTC/USD", "pip": 1.0, "category": "Crypto", "strategy_type": "forex"},
}

# ============ DATABASE ============
class UserDatabase:
    def __init__(self):
        self.users = {}
        logger.info("User database initialized")
    
    def create_user(self, username: str, password: str) -> Tuple[bool, str]:
        try:
            if username in self.users:
                return False, "Username already exists"
            
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            self.users[username] = {
                'user_id': str(uuid4()),
                'username': username,
                'password_hash': password_hash,
                'created_at': datetime.now().isoformat(),
                'settings': {
                    'enabled_markets': ['R_75', 'R_100', 'frxEURUSD', 'frxGBPUSD'],
                    'min_confidence': 65,
                    'trade_amount': 1.0,
                    'max_concurrent_trades': 3,
                    'max_daily_trades': 50,
                    'max_hourly_trades': 15,
                    'dry_run': True,
                    'risk_level': 1.0,
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
            
            logger.info(f"Created user: {username}")
            return True, "User created successfully"
        except Exception as e:
            logger.error(f"Error creating user: {e}")
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
            logger.info(f"User authenticated: {username}")
            return True, "Authentication successful"
        except Exception as e:
            logger.error(f"Authentication error: {e}")
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
            return True
        except Exception as e:
            logger.error(f"Error updating user: {e}")
            return False

# ============ DERIV API CLIENT (REAL CONNECTION) ============
class DerivAPIClient:
    def __init__(self):
        self.ws = None
        self.connected = False
        self.account_info = {}
        self.accounts = []
        self.balance = 0.0
        self.prices = {}
        self.price_subscriptions = {}
        self.last_price_update = {}
        self.reconnect_attempts = 0
        self.max_reconnect = 3
        self.connection_lock = threading.Lock()
        self.running = True
    
    def connect_with_token(self, api_token: str) -> Tuple[bool, str]:
        """Connect using API token"""
        try:
            logger.info("Connecting with API token...")
            success, message = self._connect_websocket(api_token)
            return success, message
        except Exception as e:
            logger.error(f"Token connection error: {e}")
            return False, f"Token error: {str(e)}"
    
    def _connect_websocket(self, token: str) -> Tuple[bool, str]:
        """Connect to Deriv WebSocket"""
        try:
            ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id=1089&l=EN"
            logger.info(f"Connecting to WebSocket: {ws_url}")
            
            self.ws = websocket.create_connection(ws_url, timeout=10)
            
            # Authorize
            auth_request = {"authorize": token}
            self.ws.send(json.dumps(auth_request))
            
            response = self.ws.recv()
            if not response:
                return False, "No response from WebSocket"
            
            response_data = json.loads(response)
            
            if "error" in response_data:
                error_msg = response_data["error"].get("message", "Authentication failed")
                return False, f"Auth failed: {error_msg}"
            
            self.account_info = response_data.get("authorize", {})
            self.connected = True
            
            # Get account details
            loginid = self.account_info.get("loginid", "Unknown")
            is_virtual = self.account_info.get("is_virtual", False)
            currency = self.account_info.get("currency", "USD")
            
            # Get balance
            try:
                self.ws.send(json.dumps({"balance": 1}))
                balance_response = self.ws.recv()
                balance_data = json.loads(balance_response)
                if "balance" in balance_data:
                    self.balance = float(balance_data["balance"]["balance"])
            except:
                self.balance = 0.0
            
            self.accounts = [{
                'loginid': loginid,
                'currency': currency,
                'is_virtual': is_virtual,
                'balance': self.balance,
                'name': f"{'DEMO' if is_virtual else 'REAL'} - {loginid}",
                'type': 'demo' if is_virtual else 'real'
            }]
            
            logger.info(f"✅ Connected to {loginid}")
            return True, f"✅ Connected to {loginid} | Balance: {self.balance:.2f} {currency}"
            
        except Exception as e:
            logger.error(f"WebSocket connection error: {str(e)}")
            return False, f"Connection error: {str(e)}"
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price from Deriv"""
        try:
            if not self.connected or not self.ws:
                logger.warning(f"Not connected for {symbol}")
                return None
            
            # Subscribe if not already
            if symbol not in self.price_subscriptions:
                self.subscribe_price(symbol)
                time.sleep(0.3)
            
            # Request fresh price
            with self.connection_lock:
                price_request = {"ticks": symbol}
                self.ws.send(json.dumps(price_request))
                self.ws.settimeout(3.0)
                
                try:
                    response = self.ws.recv()
                    data = json.loads(response)
                    
                    if "tick" in data:
                        price = float(data["tick"]["quote"])
                        self.prices[symbol] = price
                        self.last_price_update[symbol] = time.time()
                        return price
                except Exception as e:
                    logger.warning(f"Price request timeout for {symbol}: {e}")
                    return self.prices.get(symbol)
            
            return self.prices.get(symbol)
            
        except Exception as e:
            logger.error(f"Get price error for {symbol}: {e}")
            return self.prices.get(symbol)
    
    def get_candles(self, symbol: str, timeframe: str = "5m", count: int = 100) -> Optional[pd.DataFrame]:
        """Get candle data from Deriv"""
        try:
            if not self.connected or not self.ws:
                logger.warning(f"Not connected for {symbol}")
                return None
            
            timeframe_map = {
                "1m": 60, "5m": 300, "15m": 900, 
                "30m": 1800, "1h": 3600, "4h": 14400
            }
            granularity = timeframe_map.get(timeframe, 300)
            
            request = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": count,
                "end": "latest",
                "granularity": granularity,
                "style": "candles"
            }
            
            with self.connection_lock:
                self.ws.send(json.dumps(request))
                self.ws.settimeout(5.0)
                
                try:
                    response = self.ws.recv()
                    data = json.loads(response)
                    
                    if "error" in data:
                        logger.error(f"Candle error for {symbol}: {data['error']}")
                        return None
                    
                    if "candles" in data and data["candles"]:
                        candles = data["candles"]
                        df_data = {
                            'time': [pd.to_datetime(c.get('epoch'), unit='s') for c in candles],
                            'open': [float(c.get('open', 0)) for c in candles],
                            'high': [float(c.get('high', 0)) for c in candles],
                            'low': [float(c.get('low', 0)) for c in candles],
                            'close': [float(c.get('close', 0)) for c in candles],
                            'volume': [float(c.get('volume', 0)) for c in candles]
                        }
                        df = pd.DataFrame(df_data)
                        logger.info(f"📊 Got {len(df)} candles for {symbol}")
                        return df
                    
                    logger.warning(f"No candles data for {symbol}")
                    return None
                    
                except Exception as e:
                    logger.error(f"Candle fetch error for {symbol}: {e}")
                    return None
                
        except Exception as e:
            logger.error(f"Get candles error for {symbol}: {e}")
            return None
    
    def subscribe_price(self, symbol: str):
        """Subscribe to price updates"""
        try:
            if not self.connected or not self.ws:
                return False
            
            if symbol in self.price_subscriptions:
                return True
            
            subscribe_msg = {
                "ticks": symbol,
                "subscribe": 1
            }
            
            self.ws.send(json.dumps(subscribe_msg))
            self.price_subscriptions[symbol] = True
            logger.info(f"✅ Subscribed to {symbol}")
            return True
        except:
            return False
    
    def place_trade(self, symbol: str, direction: str, amount: float) -> Tuple[bool, str]:
        """EXECUTE REAL TRADE on Deriv"""
        try:
            with self.connection_lock:
                if not self.connected or not self.ws:
                    return False, "Not connected to Deriv"
                
                # Minimum amount
                if amount < 0.35:
                    amount = 0.35
                
                # Determine contract type
                contract_type = "CALL" if direction.upper() in ["BUY", "UP", "CALL"] else "PUT"
                currency = self.account_info.get("currency", "USD")
                
                # Get current price for logging
                current_price = self.get_price(symbol) or 0
                
                trade_request = {
                    "buy": amount,
                    "price": amount,
                    "parameters": {
                        "amount": amount,
                        "basis": "stake",
                        "contract_type": contract_type,
                        "currency": currency,
                        "duration": 5,
                        "duration_unit": "m",
                        "symbol": symbol
                    }
                }
                
                logger.info(f"🚀 EXECUTING TRADE: {symbol} {direction} ${amount} at ~{current_price}")
                
                self.ws.send(json.dumps(trade_request))
                response = self.ws.recv()
                data = json.loads(response)
                
                if "error" in data:
                    error_msg = data["error"].get("message", "Trade failed")
                    logger.error(f"❌ Trade failed: {error_msg}")
                    return False, f"Trade failed: {error_msg}"
                
                if "buy" in data:
                    contract_id = data["buy"].get("contract_id", "Unknown")
                    # Update balance
                    self.get_balance()
                    logger.info(f"✅ TRADE SUCCESS: {symbol} {direction} - ID: {contract_id}")
                    return True, contract_id
                
                return False, "Unknown trade error"
                
        except Exception as e:
            logger.error(f"Trade error: {e}")
            return False, f"Trade error: {str(e)}"
    
    def get_balance(self) -> float:
        try:
            if not self.connected or not self.ws:
                return self.balance
            
            self.ws.send(json.dumps({"balance": 1}))
            response = self.ws.recv()
            data = json.loads(response)
            if "balance" in data:
                self.balance = float(data["balance"]["balance"])
            return self.balance
        except:
            return self.balance
    
    def close_connection(self):
        try:
            self.running = False
            if self.ws:
                self.ws.close()
                self.connected = False
                logger.info("WebSocket connection closed")
        except:
            pass

# ============ SMC ANALYZER ============
class SMCAnalyzer:
    def __init__(self):
        self.last_analysis = {}
        logger.info("SMC Analyzer initialized")
    
    def analyze_market(self, df: pd.DataFrame, symbol: str, current_price: float) -> Dict:
        """Simple SMC analysis"""
        try:
            if df is None or len(df) < 20:
                return self._neutral_signal(symbol, current_price)
            
            # Prepare data
            df = self._prepare_data(df)
            
            # Calculate basic indicators
            ema_20 = df['ema_20'].iloc[-1]
            ema_50 = df['ema_50'].iloc[-1]
            
            # Determine trend
            trend = "BULLISH" if ema_20 > ema_50 else "BEARISH"
            
            # Calculate support/resistance
            recent_high = df['high'].iloc[-20:].max()
            recent_low = df['low'].iloc[-20:].min()
            
            # Calculate confidence
            price_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
            
            if trend == "BULLISH" and price_position < 0.4:
                signal = "BUY"
                confidence = 70 + int((0.4 - price_position) * 50)
            elif trend == "BEARISH" and price_position > 0.6:
                signal = "SELL"
                confidence = 70 + int((price_position - 0.6) * 50)
            else:
                signal = "NEUTRAL"
                confidence = 50
            
            confidence = min(90, max(50, confidence))
            
            analysis = {
                "confidence": confidence,
                "signal": signal,
                "price": current_price,
                "trend": trend,
                "ema_20": ema_20,
                "ema_50": ema_50,
                "recent_high": recent_high,
                "recent_low": recent_low,
                "timestamp": datetime.now().isoformat(),
                "strategy": "SMC_CORE"
            }
            
            self.last_analysis[symbol] = analysis
            return analysis
            
        except Exception as e:
            logger.error(f"SMC analysis error: {e}")
            return self._neutral_signal(symbol, current_price)
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        return df
    
    def _neutral_signal(self, symbol: str, current_price: float) -> Dict:
        return {
            "confidence": 50,
            "signal": "NEUTRAL",
            "price": current_price,
            "timestamp": datetime.now().isoformat(),
            "strategy": "NEUTRAL"
        }

# ============ TRADING ENGINE ============
class TradingEngine:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.api_client = None
        self.analyzer = SMCAnalyzer()
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
            'enabled_markets': ['R_75', 'R_100', 'frxEURUSD', 'frxGBPUSD'],
            'min_confidence': 65,
            'trade_amount': 1.0,
            'max_concurrent_trades': 3,
            'max_daily_trades': 50,
            'max_hourly_trades': 15,
            'dry_run': True,
            'risk_level': 1.0,
        }
        self.thread = None
        self.last_trade_time = {}
    
    def connect_with_token(self, api_token: str) -> Tuple[bool, str]:
        try:
            logger.info(f"Connecting with token for user {self.user_id}")
            self.api_client = DerivAPIClient()
            success, message = self.api_client.connect_with_token(api_token)
            return success, message
        except Exception as e:
            logger.error(f"Token connection error: {e}")
            return False, str(e)
    
    def get_market_analysis(self, symbol: str) -> Optional[Dict]:
        """Get real market analysis"""
        try:
            if not self.api_client or not self.api_client.connected:
                logger.error("Not connected to Deriv")
                return None
            
            # Get REAL price from Deriv
            current_price = self.api_client.get_price(symbol)
            if not current_price:
                logger.error(f"No price for {symbol}")
                return None
            
            # Get REAL candles from Deriv
            df = self.api_client.get_candles(symbol, "5m", 100)
            if df is None or len(df) < 20:
                logger.error(f"Insufficient data for {symbol}")
                return None
            
            # Analyze with SMC
            analysis = self.analyzer.analyze_market(df, symbol, current_price)
            
            return {
                'price': current_price,
                'analysis': analysis,
                'market_name': DERIV_MARKETS.get(symbol, {}).get('name', symbol),
                'real_data': True
            }
            
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            return None
    
    def place_trade(self, symbol: str, direction: str, amount: float) -> Tuple[bool, str]:
        """Place a trade (real or dry run)"""
        try:
            if not self.api_client or not self.api_client.connected:
                return False, "Not connected to Deriv"
            
            # Check if dry run
            if self.settings.get('dry_run', True):
                # Simulate trade
                trade_id = f"DRY_{int(time.time())}"
                self._record_trade({
                    'symbol': symbol,
                    'direction': direction,
                    'amount': amount,
                    'trade_id': trade_id,
                    'dry_run': True,
                    'timestamp': datetime.now().isoformat(),
                    'manual': True
                })
                return True, f"DRY RUN: Simulated trade {symbol} {direction} ${amount}"
            
            # Execute REAL trade
            success, trade_id = self.api_client.place_trade(symbol, direction, amount)
            
            if success:
                self._record_trade({
                    'symbol': symbol,
                    'direction': direction,
                    'amount': amount,
                    'trade_id': trade_id,
                    'dry_run': False,
                    'timestamp': datetime.now().isoformat(),
                    'manual': True
                })
                return True, f"✅ REAL TRADE executed: {trade_id}"
            else:
                return False, trade_id
                
        except Exception as e:
            logger.error(f"Place trade error: {e}")
            return False, f"Trade error: {str(e)}"
    
    def _record_trade(self, trade_data: Dict):
        trade_data['id'] = len(self.trades) + 1
        self.trades.append(trade_data)
        
        self.stats['total_trades'] += 1
        self.stats['daily_trades'] += 1
        self.stats['hourly_trades'] += 1
        
        # Add to active trades if real
        if not trade_data.get('dry_run', True):
            self.active_trades.append(trade_data['id'])
    
    def get_status(self) -> Dict:
        balance = self.api_client.get_balance() if self.api_client else 0.0
        connected = self.api_client.connected if self.api_client else False
        
        # Get market data for enabled markets
        market_data = {}
        if self.api_client and self.api_client.connected:
            for symbol in self.settings.get('enabled_markets', []):
                try:
                    price = self.api_client.get_price(symbol)
                    if price:
                        analysis = self.analyzer.last_analysis.get(symbol, {})
                        market_data[symbol] = {
                            'name': DERIV_MARKETS.get(symbol, {}).get('name', symbol),
                            'price': price,
                            'analysis': analysis,
                            'category': DERIV_MARKETS.get(symbol, {}).get('category', 'Unknown')
                        }
                except Exception as e:
                    continue
        
        return {
            'running': self.running,
            'connected': connected,
            'balance': balance,
            'accounts': self.api_client.accounts if self.api_client else [],
            'stats': self.stats,
            'settings': self.settings,
            'recent_trades': self.trades[-10:][::-1] if self.trades else [],
            'active_trades': len(self.active_trades),
            'market_data': market_data
        }
    
    def update_settings(self, settings: Dict):
        self.settings.update(settings)
        logger.info(f"Settings updated for user {self.user_id}")

# ============ FLASK APP ============
app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_urlsafe(32))

# Session config
app.config.update(
    SESSION_COOKIE_SECURE=False,
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
            
            # Create trading engine if not exists
            if username not in trading_engines:
                user_data = user_db.get_user(username)
                engine = TradingEngine(user_id=user_data['user_id'])
                engine.update_settings(user_data.get('settings', {}))
                trading_engines[username] = engine
            
            return jsonify({'success': True, 'message': 'Login successful'})
        else:
            return jsonify({'success': False, 'message': message})
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'success': False, 'message': f'Login error: {str(e)}'})

@app.route('/api/register', methods=['POST'])
def api_register():
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password required'})
        
        if len(username) < 3:
            return jsonify({'success': False, 'message': 'Username must be at least 3 characters'})
        
        if len(password) < 6:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters'})
        
        success, message = user_db.create_user(username, password)
        
        if success:
            return jsonify({'success': True, 'message': 'Registration successful. Please login.'})
        else:
            return jsonify({'success': False, 'message': message})
            
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'success': False, 'message': f'Registration error: {str(e)}'})

@app.route('/api/logout', methods=['POST'])
def api_logout():
    try:
        username = session.get('username')
        if username:
            if username in trading_engines:
                engine = trading_engines[username]
                if engine.api_client:
                    engine.api_client.close_connection()
                del trading_engines[username]
            session.clear()
        
        return jsonify({'success': True, 'message': 'Logged out successfully'})
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
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
        
        engine = trading_engines.get(username)
        if not engine:
            user_data = user_db.get_user(username)
            engine = TradingEngine(user_id=user_data['user_id'])
            engine.update_settings(user_data.get('settings', {}))
            trading_engines[username] = engine
        
        success, message = engine.connect_with_token(api_token)
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'accounts': engine.api_client.accounts if engine.api_client else []
            })
        else:
            return jsonify({'success': False, 'message': message})
            
    except Exception as e:
        logger.error(f"Token connect error: {e}")
        return jsonify({'success': False, 'message': f'Token connection error: {str(e)}'})

@app.route('/api/status', methods=['GET'])
def api_status():
    try:
        username = session.get('username')
        if not username:
            return jsonify({'success': False, 'message': 'Not logged in'})
        
        engine = trading_engines.get(username)
        if not engine:
            return jsonify({'success': False, 'message': 'Not connected'})
        
        status = engine.get_status()
        
        return jsonify({
            'success': True,
            'status': status,
            'markets': DERIV_MARKETS
        })
        
    except Exception as e:
        logger.error(f"Status error: {e}")
        return jsonify({'success': False, 'message': f'Status error: {str(e)}'})

@app.route('/api/update-settings', methods=['POST'])
def api_update_settings():
    try:
        username = session.get('username')
        if not username:
            return jsonify({'success': False, 'message': 'Not logged in'})
        
        data = request.json
        settings = data.get('settings', {})
        
        if 'trade_amount' in settings:
            if settings['trade_amount'] < 0.35:
                return jsonify({'success': False, 'message': 'Minimum trade amount is $0.35'})
        
        engine = trading_engines.get(username)
        if engine:
            engine.update_settings(settings)
        
        user_data = user_db.get_user(username)
        if user_data:
            user_data['settings'].update(settings)
            user_db.update_user(username, user_data)
        
        return jsonify({'success': True, 'message': 'Settings updated'})
        
    except Exception as e:
        logger.error(f"Update settings error: {e}")
        return jsonify({'success': False, 'message': f'Update settings error: {str(e)}'})

@app.route('/api/place-trade', methods=['POST'])
def api_place_trade():
    """Place manual trade - REAL or DRY RUN"""
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
        
        if amount < 0.35:
            return jsonify({'success': False, 'message': 'Minimum trade amount is $0.35'})
        
        engine = trading_engines.get(username)
        if not engine or not engine.api_client or not engine.api_client.connected:
            return jsonify({'success': False, 'message': 'Not connected to Deriv'})
        
        # Place trade
        success, message = engine.place_trade(symbol, direction, amount)
        
        return jsonify({
            'success': success,
            'message': message,
            'dry_run': engine.settings.get('dry_run', True)
        })
        
    except Exception as e:
        logger.error(f"Place trade error: {e}")
        return jsonify({'success': False, 'message': f'Place trade error: {str(e)}'})

@app.route('/api/analyze-market', methods=['POST'])
def api_analyze_market():
    """Analyze market with REAL data"""
    try:
        username = session.get('username')
        if not username:
            return jsonify({'success': False, 'message': 'Not logged in'})
        
        data = request.json
        symbol = data.get('symbol')
        
        if not symbol:
            return jsonify({'success': False, 'message': 'Symbol required'})
        
        engine = trading_engines.get(username)
        if not engine:
            return jsonify({'success': False, 'message': 'Not connected'})
        
        # Get REAL market analysis
        market_data = engine.get_market_analysis(symbol)
        
        if not market_data:
            return jsonify({'success': False, 'message': 'Failed to get market data. Please connect to Deriv first.'})
        
        return jsonify({
            'success': True,
            'analysis': market_data['analysis'],
            'current_price': market_data['price'],
            'symbol': symbol,
            'market_name': market_data['market_name'],
            'real_data': market_data.get('real_data', False)
        })
        
    except Exception as e:
        logger.error(f"Analyze market error: {e}")
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
    """Main route - serve the web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Render"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

# ============ HTML TEMPLATE (SAME UI) ============
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🎯 Karanka V8 - Deriv SMC Trading Bot</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
        :root {
            --black-primary: #0a0a0a;
            --black-secondary: #1a1a1a;
            --black-tertiary: #2a2a2a;
            --gold-primary: #FFD700;
            --gold-secondary: #B8860B;
            --gold-light: #FFF8DC;
            --success: #00C853;
            --warning: #FF9800;
            --danger: #FF5252;
            --info: #2196F3;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            background: var(--black-primary);
            color: var(--gold-light);
            min-height: 100vh;
            padding: 10px;
        }
        
        .app-container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, var(--black-secondary), var(--black-primary));
            border-radius: 15px;
            margin-bottom: 20px;
            border: 2px solid var(--gold-secondary);
            box-shadow: 0 4px 20px rgba(255, 215, 0, 0.1);
        }
        
        .header h1 {
            color: var(--gold-primary);
            font-size: 28px;
            margin-bottom: 15px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.5);
        }
        
        .status-bar {
            display: flex;
            justify-content: center;
            gap: 25px;
            flex-wrap: wrap;
            font-size: 14px;
        }
        
        .status-bar span {
            padding: 8px 16px;
            background: var(--black-tertiary);
            border-radius: 25px;
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 500;
            border: 1px solid var(--gold-secondary);
        }
        
        .tabs-container {
            display: flex;
            overflow-x: auto;
            gap: 8px;
            margin-bottom: 20px;
            padding: 10px;
            background: var(--black-secondary);
            border-radius: 12px;
            border: 1px solid var(--black-tertiary);
        }
        
        .tab {
            padding: 14px 24px;
            background: var(--black-tertiary);
            border-radius: 10px;
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.3s ease;
            border: 1px solid transparent;
            font-weight: 500;
        }
        
        .tab:hover {
            background: var(--gold-secondary);
            color: var(--black-primary);
            transform: translateY(-2px);
        }
        
        .tab.active {
            background: linear-gradient(135deg, var(--gold-primary), var(--gold-secondary));
            color: var(--black-primary);
            font-weight: bold;
            border-color: var(--gold-secondary);
            box-shadow: 0 4px 12px rgba(255, 215, 0, 0.3);
        }
        
        .content-panel {
            display: none;
            padding: 25px;
            background: var(--black-secondary);
            border-radius: 15px;
            margin-bottom: 20px;
            border: 1px solid var(--black-tertiary);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .content-panel.active {
            display: block;
            animation: fadeIn 0.4s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-label {
            display: block;
            margin-bottom: 8px;
            color: var(--gold-secondary);
            font-size: 14px;
            font-weight: 500;
        }
        
        .form-input {
            width: 100%;
            padding: 14px;
            background: var(--black-tertiary);
            border: 1px solid var(--gold-secondary);
            border-radius: 10px;
            color: var(--gold-light);
            font-size: 16px;
            transition: all 0.3s;
        }
        
        .form-input:focus {
            outline: none;
            border-color: var(--gold-primary);
            box-shadow: 0 0 0 2px rgba(255, 215, 0, 0.2);
        }
        
        .btn {
            padding: 14px 28px;
            background: linear-gradient(135deg, var(--gold-primary), var(--gold-secondary));
            color: var(--black-primary);
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
            margin: 8px;
            font-size: 16px;
        }
        
        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(255, 215, 0, 0.4);
        }
        
        .btn-success {
            background: linear-gradient(135deg, var(--success), #00E676);
        }
        
        .btn-warning {
            background: linear-gradient(135deg, var(--warning), #FFB74D);
        }
        
        .btn-danger {
            background: linear-gradient(135deg, var(--danger), #FF8A80);
        }
        
        .btn-info {
            background: linear-gradient(135deg, var(--info), #64B5F6);
        }
        
        .hidden {
            display: none !important;
        }
        
        .alert {
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
            font-size: 14px;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-10px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        .alert-success {
            background: rgba(0, 200, 83, 0.15);
            border: 1px solid var(--success);
            color: #00E676;
        }
        
        .alert-error {
            background: rgba(255, 82, 82, 0.15);
            border: 1px solid var(--danger);
            color: #FF8A80;
        }
        
        .alert-warning {
            background: rgba(255, 152, 0, 0.15);
            border: 1px solid var(--warning);
            color: #FFB74D;
        }
        
        .market-card {
            background: var(--black-tertiary);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid var(--black-secondary);
            transition: all 0.3s;
        }
        
        .market-card:hover {
            border-color: var(--gold-secondary);
            transform: translateY(-2px);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .stat-card {
            background: var(--black-tertiary);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid var(--gold-secondary);
        }
        
        .market-checkbox {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
            padding: 10px;
            background: rgba(255, 215, 0, 0.05);
            border-radius: 8px;
        }
        
        .market-checkbox label {
            cursor: pointer;
            flex: 1;
            color: var(--gold-light);
        }
        
        .market-checkbox input[type="checkbox"] {
            width: 18px;
            height: 18px;
            cursor: pointer;
        }
        
        @media (max-width: 768px) {
            .header h1 {
                font-size: 22px;
            }
            
            .status-bar {
                gap: 10px;
            }
            
            .status-bar span {
                font-size: 12px;
                padding: 6px 12px;
            }
            
            .tabs-container {
                flex-wrap: nowrap;
                overflow-x: auto;
            }
            
            .tab {
                padding: 12px 16px;
                font-size: 14px;
            }
            
            .content-panel {
                padding: 15px;
            }
            
            .btn {
                padding: 12px 20px;
                font-size: 14px;
                margin: 5px;
            }
        }
    </style>
</head>
<body>
    <div class="app-container">
        <div class="header">
            <h1>🎯 KARANKA V8 - DERIV SMC TRADING BOT</h1>
            <div class="status-bar">
                <span id="connection-status">🔴 Disconnected</span>
                <span id="trading-status">❌ Not Trading</span>
                <span id="balance">$0.00</span>
                <span id="username-display">Guest</span>
            </div>
        </div>
        
        <div id="auth-section" class="content-panel active">
            <h2 style="color: var(--gold-primary); margin-bottom: 25px; text-align: center;">🔐 Login / Register</h2>
            <div style="max-width: 400px; margin: 0 auto;">
                <div class="form-group">
                    <label class="form-label">Username</label>
                    <input type="text" id="username" class="form-input" placeholder="Enter username">
                </div>
                <div class="form-group">
                    <label class="form-label">Password</label>
                    <input type="password" id="password" class="form-input" placeholder="Enter password">
                </div>
                <div style="display: flex; gap: 15px; margin-top: 25px;">
                    <button class="btn" onclick="login()" style="flex: 1;">🔑 Login</button>
                    <button class="btn btn-warning" onclick="register()" style="flex: 1;">📝 Register</button>
                </div>
                <div id="auth-message" class="alert" style="display: none;"></div>
            </div>
        </div>
        
        <div id="main-app" class="hidden">
            <div class="tabs-container">
                <div class="tab active" onclick="showTab('dashboard')">📊 Dashboard</div>
                <div class="tab" onclick="showTab('connection')">🔗 Connection</div>
                <div class="tab" onclick="showTab('markets')">📈 Markets</div>
                <div class="tab" onclick="showTab('trading')">⚡ Trading</div>
                <div class="tab" onclick="showTab('settings')">⚙️ Settings</div>
                <div class="tab" onclick="showTab('trades')">💼 Trades</div>
                <div class="tab" onclick="logout()" style="background: var(--danger); color: white;">🚪 Logout</div>
            </div>
            
            <div id="dashboard" class="content-panel active">
                <h2 style="color: var(--gold-primary); margin-bottom: 20px;">📊 Dashboard</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div style="font-size: 12px; color: var(--gold-secondary);">Balance</div>
                        <div style="font-size: 24px; color: var(--gold-primary); font-weight: bold;" id="stat-balance">$0.00</div>
                    </div>
                    <div class="stat-card">
                        <div style="font-size: 12px; color: var(--gold-secondary);">Total Trades</div>
                        <div style="font-size: 24px; color: var(--gold-primary); font-weight: bold;" id="stat-total-trades">0</div>
                    </div>
                    <div class="stat-card">
                        <div style="font-size: 12px; color: var(--gold-secondary);">Active Trades</div>
                        <div style="font-size: 24px; color: var(--gold-primary); font-weight: bold;" id="stat-active-trades">0</div>
                    </div>
                </div>
                <div id="dashboard-alert" class="alert" style="display: none; margin-top: 20px;"></div>
            </div>
            
            <div id="connection" class="content-panel">
                <h2 style="color: var(--gold-primary); margin-bottom: 20px;">🔗 Connect to Deriv</h2>
                <div class="form-group">
                    <label class="form-label">Deriv API Token</label>
                    <input type="text" id="api-token" class="form-input" placeholder="Paste your Deriv API token">
                    <small style="color: var(--gold-secondary); display: block; margin-top: 5px;">
                        Get your token from: <a href="https://app.deriv.com/account/api-token" target="_blank" style="color: var(--gold-primary);">Deriv API Token</a>
                    </small>
                </div>
                <button class="btn btn-success" onclick="connectWithToken()">🔗 Connect with API Token</button>
                <div id="connection-alert" class="alert" style="display: none; margin-top: 20px;"></div>
            </div>
            
            <div id="markets" class="content-panel">
                <h2 style="color: var(--gold-primary); margin-bottom: 20px;">📈 Deriv Markets</h2>
                <div style="display: flex; gap: 15px; margin-bottom: 20px;">
                    <button class="btn" onclick="refreshMarkets()">🔄 Refresh Prices</button>
                    <button class="btn btn-info" onclick="analyzeAllMarkets()">🧠 Analyze All</button>
                </div>
                <div id="markets-grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px;">
                    <!-- Markets loaded here -->
                </div>
                <div id="markets-alert" class="alert" style="display: none; margin-top: 20px;"></div>
            </div>
            
            <div id="trading" class="content-panel">
                <h2 style="color: var(--gold-primary); margin-bottom: 20px;">⚡ Trading</h2>
                <div class="form-group">
                    <label class="form-label">Market</label>
                    <select id="trade-symbol" class="form-input">
                        <option value="">Select market</option>
                    </select>
                </div>
                <div class="form-group">
                    <label class="form-label">Direction</label>
                    <div style="display: flex; gap: 10px;">
                        <button class="btn" onclick="setTradeDirection('BUY')" id="buy-btn">📈 BUY</button>
                        <button class="btn" onclick="setTradeDirection('SELL')" id="sell-btn">📉 SELL</button>
                    </div>
                </div>
                <div class="form-group">
                    <label class="form-label">Amount ($)</label>
                    <input type="number" id="trade-amount" class="form-input" value="1.00" min="0.35" step="0.01">
                </div>
                <button class="btn btn-success" onclick="placeTrade()">🚀 Place Trade</button>
                <button class="btn btn-info" onclick="analyzeTradeMarket()">🧠 Analyze Market</button>
                <div id="trade-analysis" style="margin-top: 20px; padding: 15px; background: var(--black-tertiary); border-radius: 10px; display: none;">
                    <h4 style="color: var(--gold-light); margin-bottom: 10px;">Market Analysis</h4>
                    <div id="analysis-content"></div>
                </div>
                <div id="trading-alert" class="alert" style="display: none; margin-top: 20px;"></div>
            </div>
            
            <div id="settings" class="content-panel">
                <h2 style="color: var(--gold-primary); margin-bottom: 20px;">⚙️ Settings</h2>
                <div class="form-group">
                    <label class="form-label">Trade Amount ($)</label>
                    <input type="number" id="setting-trade-amount" class="form-input" value="1.00" min="0.35" step="0.01">
                </div>
                <div class="form-group">
                    <label class="form-label">Minimum Confidence (%)</label>
                    <input type="number" id="setting-min-confidence" class="form-input" value="65" min="50" max="90">
                </div>
                <div class="form-group">
                    <label class="checkbox-label" style="display: flex; align-items: center; gap: 10px; cursor: pointer; margin-bottom: 15px;">
                        <input type="checkbox" id="setting-dry-run" checked> 
                        <span>Dry Run Mode (Simulate trades only - TURN OFF FOR REAL TRADING)</span>
                    </label>
                </div>
                <button class="btn btn-success" onclick="saveSettings()">💾 Save Settings</button>
                <div id="settings-alert" class="alert" style="display: none; margin-top: 20px;"></div>
            </div>
            
            <div id="trades" class="content-panel">
                <h2 style="color: var(--gold-primary); margin-bottom: 20px;">💼 Trade History</h2>
                <div id="trades-list" style="max-height: 400px; overflow-y: auto;">
                    <div style="text-align: center; padding: 40px; color: var(--gold-secondary);">No trades yet</div>
                </div>
                <button class="btn" onclick="refreshTrades()" style="margin-top: 15px;">🔄 Refresh</button>
            </div>
        </div>
    </div>

    <script>
        let currentUser = null;
        let updateInterval = null;
        
        document.addEventListener('DOMContentLoaded', function() {
            checkSession();
        });
        
        async function checkSession() {
            try {
                const response = await fetch('/api/check-session');
                const data = await response.json();
                if (data.success && data.username) {
                    currentUser = data.username;
                    document.getElementById('username-display').textContent = data.username;
                    document.getElementById('auth-section').classList.add('hidden');
                    document.getElementById('main-app').classList.remove('hidden');
                    loadMarkets();
                    startStatusUpdates();
                }
            } catch (error) {
                console.log('No active session');
            }
        }
        
        async function login() {
            const username = document.getElementById('username').value.trim();
            const password = document.getElementById('password').value.trim();
            if (!username || !password) {
                showAlert('auth-message', 'Please enter username and password', 'error');
                return;
            }
            try {
                const response = await fetch('/api/login', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({username, password})
                });
                const data = await response.json();
                showAlert('auth-message', data.message, data.success ? 'success' : 'error');
                if (data.success) {
                    currentUser = username;
                    document.getElementById('username-display').textContent = username;
                    document.getElementById('auth-section').classList.add('hidden');
                    document.getElementById('main-app').classList.remove('hidden');
                    loadMarkets();
                    startStatusUpdates();
                }
            } catch (error) {
                showAlert('auth-message', 'Network error', 'error');
            }
        }
        
        async function register() {
            const username = document.getElementById('username').value.trim();
            const password = document.getElementById('password').value.trim();
            if (!username || !password) {
                showAlert('auth-message', 'Please enter username and password', 'error');
                return;
            }
            if (username.length < 3) {
                showAlert('auth-message', 'Username must be at least 3 characters', 'error');
                return;
            }
            if (password.length < 6) {
                showAlert('auth-message', 'Password must be at least 6 characters', 'error');
                return;
            }
            try {
                const response = await fetch('/api/register', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({username, password})
                });
                const data = await response.json();
                showAlert('auth-message', data.message, data.success ? 'success' : 'error');
            } catch (error) {
                showAlert('auth-message', 'Network error', 'error');
            }
        }
        
        async function logout() {
            try {
                const response = await fetch('/api/logout', {method: 'POST'});
                const data = await response.json();
                if (data.success) {
                    currentUser = null;
                    document.getElementById('main-app').classList.add('hidden');
                    document.getElementById('auth-section').classList.remove('hidden');
                    document.getElementById('username').value = '';
                    document.getElementById('password').value = '';
                    if (updateInterval) clearInterval(updateInterval);
                    showAlert('auth-message', 'Logged out', 'success');
                }
            } catch (error) {
                console.error('Logout error:', error);
            }
        }
        
        async function connectWithToken() {
            const apiToken = document.getElementById('api-token').value.trim();
            if (!apiToken) {
                showAlert('connection-alert', 'Please enter your API token', 'error');
                return;
            }
            showAlert('connection-alert', 'Connecting...', 'warning');
            try {
                const response = await fetch('/api/connect-token', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({api_token: apiToken})
                });
                const data = await response.json();
                showAlert('connection-alert', data.message, data.success ? 'success' : 'error');
                if (data.success) {
                    document.getElementById('api-token').value = '';
                    startStatusUpdates();
                }
            } catch (error) {
                showAlert('connection-alert', 'Network error', 'error');
            }
        }
        
        async function loadMarkets() {
            try {
                const marketsGrid = document.getElementById('markets-grid');
                const tradeSymbol = document.getElementById('trade-symbol');
                marketsGrid.innerHTML = '';
                tradeSymbol.innerHTML = '<option value="">Select market</option>';
                const response = await fetch('/api/status');
                const data = await response.json();
                if (data.success && data.markets) {
                    for (const [symbol, market] of Object.entries(data.markets)) {
                        const marketCard = document.createElement('div');
                        marketCard.className = 'market-card';
                        marketCard.innerHTML = `
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                <div style="font-weight: bold; color: var(--gold-primary);">${market.name}</div>
                                <div style="font-size: 11px; color: var(--gold-secondary); background: rgba(184,134,11,0.2); padding: 2px 8px; border-radius: 10px;">${symbol}</div>
                            </div>
                            <div style="font-size: 22px; font-weight: bold; color: var(--gold-light); margin-bottom: 10px;" id="price-${symbol}">--.--</div>
                            <div style="margin-bottom: 15px;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 12px; color: var(--gold-secondary);">
                                    <span>SMC Confidence</span>
                                    <span id="confidence-${symbol}">--%</span>
                                </div>
                                <div style="height: 8px; background: var(--black-secondary); border-radius: 4px; overflow: hidden;">
                                    <div id="confidence-bar-${symbol}" style="height: 100%; width: 0%; background: linear-gradient(90deg, var(--info), #64B5F6);"></div>
                                </div>
                            </div>
                            <div style="display: flex; gap: 10px;">
                                <button class="btn" onclick="analyzeMarket('${symbol}')" style="flex: 1; padding: 8px; font-size: 12px;">🧠 Analyze</button>
                            </div>
                        `;
                        marketsGrid.appendChild(marketCard);
                        const option = document.createElement('option');
                        option.value = symbol;
                        option.textContent = `${market.name} (${symbol})`;
                        tradeSymbol.appendChild(option);
                    }
                }
            } catch (error) {
                console.error('Error loading markets:', error);
            }
        }
        
        async function analyzeMarket(symbol) {
            showAlert('markets-alert', `Analyzing ${symbol}...`, 'warning');
            try {
                const response = await fetch('/api/analyze-market', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({symbol})
                });
                const data = await response.json();
                if (data.success) {
                    const analysis = data.analysis;
                    const priceElement = document.getElementById(`price-${symbol}`);
                    const confidenceElement = document.getElementById(`confidence-${symbol}`);
                    const confidenceBar = document.getElementById(`confidence-bar-${symbol}`);
                    if (priceElement) priceElement.textContent = analysis.price.toFixed(5);
                    if (confidenceElement) {
                        confidenceElement.textContent = `${analysis.confidence}%`;
                        confidenceBar.style.width = `${analysis.confidence}%`;
                        if (analysis.confidence >= 70) confidenceBar.style.background = 'linear-gradient(90deg, var(--success), #00E676)';
                        else if (analysis.confidence >= 50) confidenceBar.style.background = 'linear-gradient(90deg, var(--warning), #FFB74D)';
                        else confidenceBar.style.background = 'linear-gradient(90deg, var(--danger), #FF8A80)';
                    }
                    showAlert('markets-alert', `Analysis complete`, 'success');
                } else {
                    showAlert('markets-alert', data.message, 'error');
                }
            } catch (error) {
                showAlert('markets-alert', 'Network error', 'error');
            }
        }
        
        async function analyzeAllMarkets() {
            showAlert('markets-alert', 'Analyzing all markets...', 'warning');
            const marketsGrid = document.getElementById('markets-grid');
            const marketCards = marketsGrid.querySelectorAll('.market-card');
            for (const card of marketCards) {
                const symbolElement = card.querySelector('div:last-child div:last-child');
                if (symbolElement) {
                    const symbol = symbolElement.textContent.trim();
                    await analyzeMarket(symbol);
                    await new Promise(resolve => setTimeout(resolve, 500));
                }
            }
        }
        
        async function analyzeTradeMarket() {
            const symbol = document.getElementById('trade-symbol').value;
            if (!symbol) {
                showAlert('trading-alert', 'Please select a market', 'error');
                return;
            }
            showAlert('trading-alert', 'Analyzing market...', 'warning');
            try {
                const response = await fetch('/api/analyze-market', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({symbol})
                });
                const data = await response.json();
                if (data.success) {
                    const analysis = data.analysis;
                    const analysisDiv = document.getElementById('analysis-content');
                    const tradeAnalysis = document.getElementById('trade-analysis');
                    let signalColor = 'var(--info)';
                    if (analysis.signal === 'BUY') signalColor = 'var(--success)';
                    if (analysis.signal === 'SELL') signalColor = 'var(--danger)';
                    analysisDiv.innerHTML = `
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span>Signal: <strong style="color: ${signalColor}">${analysis.signal}</strong></span>
                            <span>Confidence: <strong>${analysis.confidence}%</strong></span>
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 12px;">
                            <div>Price: ${analysis.price?.toFixed(5) || '--'}</div>
                            <div>Strategy: ${analysis.strategy || '--'}</div>
                        </div>
                    `;
                    tradeAnalysis.style.display = 'block';
                    showAlert('trading-alert', 'Analysis complete', 'success');
                } else {
                    showAlert('trading-alert', data.message, 'error');
                }
            } catch (error) {
                showAlert('trading-alert', 'Network error', 'error');
            }
        }
        
        async function placeTrade() {
            const symbol = document.getElementById('trade-symbol').value;
            const direction = document.getElementById('buy-btn').classList.contains('btn-success') ? 'BUY' : 'SELL';
            const amount = parseFloat(document.getElementById('trade-amount').value);
            if (!symbol) {
                showAlert('trading-alert', 'Please select a market', 'error');
                return;
            }
            if (amount < 0.35) {
                showAlert('trading-alert', 'Minimum trade amount is $0.35', 'error');
                return;
            }
            showAlert('trading-alert', 'Placing trade...', 'warning');
            try {
                const response = await fetch('/api/place-trade', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({symbol, direction, amount})
                });
                const data = await response.json();
                showAlert('trading-alert', data.message, data.success ? 'success' : 'error');
                if (data.success) {
                    refreshTrades();
                    updateStatus();
                }
            } catch (error) {
                showAlert('trading-alert', 'Network error', 'error');
            }
        }
        
        async function saveSettings() {
            const tradeAmount = parseFloat(document.getElementById('setting-trade-amount').value);
            const minConfidence = parseInt(document.getElementById('setting-min-confidence').value);
            const dryRun = document.getElementById('setting-dry-run').checked;
            if (tradeAmount < 0.35) {
                showAlert('settings-alert', 'Minimum trade amount is $0.35', 'error');
                return;
            }
            if (minConfidence < 50 || minConfidence > 90) {
                showAlert('settings-alert', 'Confidence must be between 50-90%', 'error');
                return;
            }
            const settings = {
                trade_amount: tradeAmount,
                min_confidence: minConfidence,
                dry_run: dryRun
            };
            showAlert('settings-alert', 'Saving settings...', 'warning');
            try {
                const response = await fetch('/api/update-settings', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({settings})
                });
                const data = await response.json();
                showAlert('settings-alert', data.message, data.success ? 'success' : 'error');
            } catch (error) {
                showAlert('settings-alert', 'Network error', 'error');
            }
        }
        
        function setTradeDirection(direction) {
            const buyBtn = document.getElementById('buy-btn');
            const sellBtn = document.getElementById('sell-btn');
            if (direction === 'BUY') {
                buyBtn.classList.add('btn-success');
                buyBtn.classList.remove('btn');
                sellBtn.classList.add('btn');
                sellBtn.classList.remove('btn-danger');
            } else {
                sellBtn.classList.add('btn-danger');
                sellBtn.classList.remove('btn');
                buyBtn.classList.add('btn');
                buyBtn.classList.remove('btn-success');
            }
        }
        
        function refreshMarkets() {
            loadMarkets();
            showAlert('markets-alert', 'Markets refreshed', 'success');
        }
        
        async function refreshTrades() {
            await updateStatus();
        }
        
        function showTab(tabName) {
            document.querySelectorAll('.content-panel').forEach(panel => panel.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }
        
        function showAlert(containerId, message, type) {
            const alertDiv = document.getElementById(containerId);
            if (!alertDiv) return;
            alertDiv.textContent = message;
            alertDiv.className = `alert alert-${type}`;
            alertDiv.style.display = 'block';
            setTimeout(() => { alertDiv.style.display = 'none'; }, 5000);
        }
        
        function startStatusUpdates() {
            if (updateInterval) clearInterval(updateInterval);
            updateStatus();
            updateInterval = setInterval(updateStatus, 5000);
        }
        
        async function updateStatus() {
            if (!currentUser) return;
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                if (data.success) {
                    const status = data.status;
                    if (status.connected) {
                        document.getElementById('connection-status').textContent = '🟢 Connected';
                        document.getElementById('connection-status').style.color = 'var(--success)';
                    } else {
                        document.getElementById('connection-status').textContent = '🔴 Disconnected';
                        document.getElementById('connection-status').style.color = 'var(--danger)';
                    }
                    document.getElementById('balance').textContent = `$${status.balance?.toFixed(2) || '0.00'}`;
                    document.getElementById('stat-balance').textContent = `$${status.balance?.toFixed(2) || '0.00'}`;
                    document.getElementById('stat-total-trades').textContent = status.stats?.total_trades || 0;
                    document.getElementById('stat-active-trades').textContent = status.active_trades || 0;
                    if (status.market_data) {
                        for (const [symbol, market] of Object.entries(status.market_data)) {
                            updateMarketCard(symbol, market);
                        }
                    }
                    updateTradesList(status.recent_trades);
                }
            } catch (error) {
                console.error('Status update error:', error);
            }
        }
        
        function updateMarketCard(symbol, market) {
            const priceElement = document.getElementById(`price-${symbol}`);
            const confidenceElement = document.getElementById(`confidence-${symbol}`);
            const confidenceBar = document.getElementById(`confidence-bar-${symbol}`);
            if (priceElement && market.price) {
                priceElement.textContent = market.price.toFixed(5);
                priceElement.style.color = 'var(--gold-light)';
            }
            if (market.analysis) {
                const confidence = market.analysis.confidence || 0;
                if (confidenceElement) {
                    confidenceElement.textContent = `${confidence}%`;
                    if (confidenceBar) confidenceBar.style.width = `${confidence}%`;
                    if (confidenceBar) {
                        if (confidence >= 70) confidenceBar.style.background = 'linear-gradient(90deg, var(--success), #00E676)';
                        else if (confidence >= 50) confidenceBar.style.background = 'linear-gradient(90deg, var(--warning), #FFB74D)';
                        else confidenceBar.style.background = 'linear-gradient(90deg, var(--danger), #FF8A80)';
                    }
                }
            }
        }
        
        function updateTradesList(trades) {
            const tradesList = document.getElementById('trades-list');
            if (!trades || trades.length === 0) {
                tradesList.innerHTML = '<div style="text-align: center; padding: 40px; color: var(--gold-secondary);">No trades yet</div>';
                return;
            }
            tradesList.innerHTML = '';
            trades.forEach(trade => {
                const tradeItem = document.createElement('div');
                tradeItem.style.cssText = `
                    padding: 15px;
                    background: var(--black-tertiary);
                    border-radius: 8px;
                    margin-bottom: 10px;
                    border-left: 5px solid ${trade.direction === 'BUY' ? 'var(--success)' : 'var(--danger)'};
                `;
                const time = new Date(trade.timestamp).toLocaleTimeString();
                const date = new Date(trade.timestamp).toLocaleDateString();
                tradeItem.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-weight: bold; color: var(--gold-primary);">${trade.symbol}</div>
                            <div style="font-size: 11px; color: var(--gold-secondary);">${date} ${time}</div>
                        </div>
                        <div style="display: flex; align-items: center; gap: 15px;">
                            <span style="padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: bold; 
                                  background: ${trade.direction === 'BUY' ? 'var(--success)' : 'var(--danger)'}; 
                                  color: var(--black-primary);">
                                ${trade.direction}
                            </span>
                            <span style="font-weight: bold; color: var(--gold-light);">$${trade.amount?.toFixed(2) || '0.00'}</span>
                            <span style="font-size: 12px; color: ${trade.dry_run ? 'var(--warning)' : 'var(--success)'}">
                                ${trade.dry_run ? 'DRY RUN' : 'REAL'}
                            </span>
                        </div>
                    </div>
                `;
                tradesList.appendChild(tradeItem);
            });
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print("\n" + "="*80)
    print("🎯 KARANKA V8 - DERIV SMART SMC TRADING BOT")
    print("="*80)
    print(f"🚀 Server starting on http://{host}:{port}")
    print("✅ REAL Deriv API Connection")
    print("✅ REAL Market Data")
    print("✅ REAL Trade Execution")
    print("="*80)
    
    app.run(host=host, port=port, debug=debug, threaded=True)
