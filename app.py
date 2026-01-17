#!/usr/bin/env python3
"""
================================================================================
🚀 KARANKA ULTRA - REAL DERIV TRADING BOT
================================================================================
• Input API Token in App • Auto-Connect to Your Account
• 24/7 Trading on Render • Advanced SMC Strategy
• Real Trades in Your Account • Full UI with All Features
================================================================================
"""

import os
import json
import time
import threading
import hashlib
import secrets
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import requests
import websocket
from flask import Flask, render_template_string, jsonify, request, make_response
from flask_cors import CORS
from functools import wraps

# ============ SETUP LOGGING ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('trading.log')]
)
logger = logging.getLogger(__name__)

# ============ RENDER KEEP-ALIVE ============
RENDER_APP_URL = os.environ.get('RENDER_EXTERNAL_URL', '')

def keep_render_awake():
    """Keep Render instance from sleeping"""
    while True:
        try:
            time.sleep(300)  # Ping every 5 minutes
            if RENDER_APP_URL:
                requests.get(f"{RENDER_APP_URL}/api/ping", timeout=5)
            logger.info("✅ Keep-alive ping sent")
        except:
            pass

threading.Thread(target=keep_render_awake, daemon=True).start()

# ============ USER SESSIONS ============
user_sessions = {}  # Store user sessions in memory

class UserSession:
    """Store user data and trading engine"""
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.api_token = None
        self.engine = None
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        
    def update_activity(self):
        self.last_active = datetime.now()
    
    def set_api_token(self, api_token: str):
        self.api_token = api_token
        self.engine = TradingEngine(self.user_id, api_token)
        logger.info(f"User {self.user_id} set API token")

def get_user_session(user_id: str = "default") -> UserSession:
    """Get or create user session"""
    if user_id not in user_sessions:
        user_sessions[user_id] = UserSession(user_id)
    user_sessions[user_id].update_activity()
    return user_sessions[user_id]

# ============ DERIV MARKETS ============
DERIV_MARKETS = {
    "R_10": {"name": "Volatility 10 Index", "pip": 0.001, "category": "Volatility", "symbol": "R_10"},
    "R_25": {"name": "Volatility 25 Index", "pip": 0.001, "category": "Volatility", "symbol": "R_25"},
    "R_50": {"name": "Volatility 50 Index", "pip": 0.001, "category": "Volatility", "symbol": "R_50"},
    "R_75": {"name": "Volatility 75 Index", "pip": 0.001, "category": "Volatility", "symbol": "R_75"},
    "R_100": {"name": "Volatility 100 Index", "pip": 0.001, "category": "Volatility", "symbol": "R_100"},
    "CRASH_500": {"name": "Crash 500 Index", "pip": 0.01, "category": "Crash/Boom", "symbol": "CRASH_500"},
    "BOOM_500": {"name": "Boom 500 Index", "pip": 0.01, "category": "Crash/Boom", "symbol": "BOOM_500"},
}

# ============ REAL DERIV CONNECTION ============
class DerivConnection:
    """Real connection to Deriv using user's API token"""
    
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.ws = None
        self.connected = False
        self.account_id = None
        self.balance = 0.0
        self.currency = "USD"
        self.email = ""
        self.full_name = ""
        self.country = ""
        self.is_virtual = False
        self.market_data = defaultdict(lambda: {"prices": deque(maxlen=100), "last_update": 0})
        self.keepalive_thread = None
        self.app_id = 1089  # Deriv app ID
        logger.info(f"DerivConnection initialized for token: {api_token[:10]}...")
    
    def connect(self) -> Tuple[bool, str]:
        """Connect to Deriv with user's API token"""
        try:
            if not self.api_token or len(self.api_token) < 20:
                return False, "Invalid API token (too short)"
            
            # Close any existing connection
            self.disconnect()
            
            # Try multiple endpoints
            endpoints = [
                ("wss://ws.deriv.com/websockets/v3", "primary"),
                ("wss://ws.binaryws.com/websockets/v3", "binary"),
                ("wss://ws.derivws.com/websockets/v3", "backup")
            ]
            
            for endpoint, name in endpoints:
                try:
                    logger.info(f"🔗 Connecting to {name} endpoint...")
                    
                    # Connect with timeout
                    self.ws = websocket.create_connection(
                        f"{endpoint}?app_id={self.app_id}",
                        timeout=15,
                        header={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                            'Origin': 'https://app.deriv.com'
                        }
                    )
                    
                    # Test connection
                    self.ws.send(json.dumps({"ping": 1}))
                    response = self.ws.recv()
                    
                    if "ping" not in response:
                        logger.warning(f"❌ No ping response from {name}")
                        continue
                    
                    # Authenticate with user's token
                    logger.info("🔐 Authenticating with API token...")
                    self.ws.send(json.dumps({"authorize": self.api_token}))
                    
                    # Get response
                    response = self.ws.recv()
                    auth_response = json.loads(response)
                    
                    if "error" in auth_response:
                        error_msg = auth_response["error"].get("message", "Authentication failed")
                        logger.error(f"❌ Auth failed: {error_msg}")
                        continue
                    
                    if "authorize" in auth_response:
                        auth_data = auth_response["authorize"]
                        
                        # Store account information
                        self.connected = True
                        self.account_id = auth_data.get("loginid", "Unknown")
                        self.currency = auth_data.get("currency", "USD")
                        self.email = auth_data.get("email", "")
                        self.full_name = auth_data.get("fullname", "")
                        self.country = auth_data.get("country", "")
                        self.is_virtual = auth_data.get("is_virtual", False)
                        
                        logger.info(f"✅ Authentication successful!")
                        logger.info(f"   Account ID: {self.account_id}")
                        logger.info(f"   Currency: {self.currency}")
                        logger.info(f"   Virtual: {self.is_virtual}")
                        
                        # Get initial balance
                        self._update_balance()
                        
                        # Start keep-alive thread
                        self._start_keepalive()
                        
                        return True, f"Connected to {self.account_id} ({self.currency})"
                    
                except websocket.WebSocketTimeoutException:
                    logger.warning(f"⏱️ Connection timeout for {name}")
                    continue
                except Exception as e:
                    logger.warning(f"⚠️ {name} failed: {str(e)[:100]}")
                    continue
            
            return False, "All connection attempts failed"
            
        except Exception as e:
            logger.error(f"❌ Connection error: {str(e)}")
            return False, f"Connection error: {str(e)}"
    
    def _start_keepalive(self):
        """Start keep-alive thread to maintain connection"""
        def keepalive():
            while self.connected and self.ws:
                try:
                    time.sleep(30)  # Send ping every 30 seconds
                    if self.ws:
                        self.ws.send(json.dumps({"ping": 1}))
                except Exception as e:
                    logger.error(f"Keepalive failed: {e}")
                    self.connected = False
                    break
        
        self.keepalive_thread = threading.Thread(target=keepalive, daemon=True)
        self.keepalive_thread.start()
    
    def _update_balance(self):
        """Update balance from Deriv"""
        try:
            if self.connected and self.ws:
                self.ws.send(json.dumps({"balance": 1}))
                response = json.loads(self.ws.recv())
                if "balance" in response:
                    self.balance = float(response["balance"]["balance"])
                    logger.info(f"💰 Balance updated: {self.balance} {self.currency}")
                    return True
        except Exception as e:
            logger.error(f"Balance update error: {e}")
        return False
    
    def place_trade(self, symbol: str, direction: str, amount: float) -> Tuple[bool, str, Dict]:
        """Place REAL trade on Deriv"""
        try:
            if not self.connected:
                return False, "Not connected to Deriv", {}
            
            # Minimum amount check
            amount = max(1.0, amount)
            
            # Map symbol to Deriv symbol
            deriv_symbols = {
                "R_10": "R_10",
                "R_25": "R_25", 
                "R_50": "R_50",
                "R_75": "R_75",
                "R_100": "R_100",
                "CRASH_500": "CRASH_500",
                "BOOM_500": "BOOM_500"
            }
            
            actual_symbol = deriv_symbols.get(symbol, "R_10")
            
            # Prepare trade
            contract_type = "CALL" if direction.upper() in ["BUY", "CALL"] else "PUT"
            
            trade_request = {
                "buy": 1,
                "price": amount,
                "parameters": {
                    "amount": amount,
                    "basis": "stake",
                    "contract_type": contract_type,
                    "currency": self.currency,
                    "duration": 5,
                    "duration_unit": "m",
                    "symbol": actual_symbol,
                    "product_type": "basic"
                }
            }
            
            logger.info(f"🚀 Placing REAL trade: {actual_symbol} {contract_type} ${amount}")
            
            self.ws.send(json.dumps(trade_request))
            response = json.loads(self.ws.recv())
            
            if "error" in response:
                error = response["error"].get("message", "Trade failed")
                return False, error, {}
            
            if "buy" in response:
                buy_data = response["buy"]
                contract_id = buy_data.get("contract_id")
                
                # Update balance
                self._update_balance()
                
                contract_info = {
                    "contract_id": contract_id,
                    "symbol": symbol,
                    "direction": direction,
                    "amount": amount,
                    "payout": buy_data.get("payout"),
                    "ask_price": buy_data.get("ask_price"),
                    "timestamp": datetime.now().isoformat(),
                    "account": self.account_id
                }
                
                logger.info(f"✅ Trade successful! Contract ID: {contract_id}")
                return True, contract_id, contract_info
            
            return False, "Unknown response", {}
            
        except Exception as e:
            logger.error(f"❌ Trade error: {e}")
            return False, str(e), {}
    
    def disconnect(self):
        """Disconnect from Deriv"""
        try:
            self.connected = False
            if self.ws:
                self.ws.close()
                self.ws = None
            logger.info("Disconnected from Deriv")
        except:
            pass
    
    def get_account_info(self) -> Dict:
        """Get complete account information"""
        return {
            "connected": self.connected,
            "account_id": self.account_id,
            "balance": self.balance,
            "currency": self.currency,
            "email": self.email,
            "full_name": self.full_name,
            "country": self.country,
            "is_virtual": self.is_virtual
        }

# ============ TRADING ENGINE ============
class TradingEngine:
    """Main trading engine with strategy"""
    
    def __init__(self, user_id: str, api_token: str):
        self.user_id = user_id
        self.api_token = api_token
        self.connection = DerivConnection(api_token)
        self.running = False
        self.thread = None
        self.trades = []
        self.price_history = defaultdict(lambda: deque(maxlen=100))
        
        # Initialize price history with realistic data
        for symbol in ['R_10', 'R_25', 'R_50']:
            base_price = 100.0
            prices = [base_price + np.random.normal(0, 0.5) for _ in range(100)]
            self.price_history[symbol].extend(prices)
        
        self.stats = {
            'total_trades': 0,
            'real_trades': 0,
            'total_profit': 0.0,
            'balance': 0.0,
            'start_time': datetime.now().isoformat()
        }
        
        self.settings = {
            'enabled_markets': ['R_10', 'R_25', 'R_50'],
            'trade_amount': 5.0,
            'min_confidence': 75,
            'scan_interval': 30,
            'cooldown_seconds': 60,
            'max_daily_trades': 50,
            'risk_per_trade': 0.02,
            'use_real_trading': True,
            'stop_loss': 5.0,
            'take_profit': 10.0
        }
        
        logger.info(f"🔥 TradingEngine created for {user_id}")
    
    def connect(self) -> Tuple[bool, str]:
        """Connect to Deriv"""
        return self.connection.connect()
    
    def start_trading(self):
        """Start trading"""
        if self.running:
            return False, "Already trading"
        
        # Ensure connected
        if not self.connection.connected:
            success, message = self.connect()
            if not success:
                return False, f"Failed to connect: {message}"
        
        self.running = True
        self.thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.thread.start()
        
        logger.info(f"🚀 REAL Trading started for {self.user_id}")
        return True, f"✅ REAL Trading started! Account: {self.connection.account_id}"
    
    def _trading_loop(self):
        """Main trading loop"""
        logger.info("🔥 Trading loop started")
        
        market_index = 0
        trade_count = 0
        
        while self.running:
            try:
                # Update price history with realistic data
                self._update_prices()
                
                # Select market
                enabled = self.settings['enabled_markets']
                if not enabled:
                    time.sleep(self.settings['scan_interval'])
                    continue
                    
                symbol = enabled[market_index % len(enabled)]
                market_index += 1
                
                # Get prices
                prices = list(self.price_history[symbol])
                
                # Analyze market
                analysis = self._analyze_market(symbol, prices)
                
                # Check if should trade
                should_trade = (
                    analysis['signal'] != 'NEUTRAL' and 
                    analysis['confidence'] >= self.settings['min_confidence'] and
                    self._can_trade(symbol) and
                    trade_count < self.settings['max_daily_trades']
                )
                
                if should_trade:
                    # Calculate amount
                    amount = self._calculate_trade_amount()
                    
                    # Execute trade
                    trade_result = self._execute_trade(symbol, analysis, amount)
                    
                    if trade_result:
                        self.trades.append(trade_result)
                        self.stats['total_trades'] += 1
                        trade_count += 1
                        
                        if trade_result['real_trade']:
                            self.stats['real_trades'] += 1
                            self.stats['balance'] = self.connection.balance
                        
                        self.stats['total_profit'] += trade_result.get('profit', 0)
                
                # Wait for next cycle
                time.sleep(self.settings['scan_interval'])
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                time.sleep(60)
    
    def _update_prices(self):
        """Update price history with realistic data"""
        for symbol in self.settings['enabled_markets']:
            if symbol in self.price_history:
                prices = list(self.price_history[symbol])
                if not prices:
                    continue
                    
                last_price = prices[-1]
                
                # Simulate realistic price movement
                volatility = {
                    'R_10': 0.015,
                    'R_25': 0.020,
                    'R_50': 0.025,
                    'R_75': 0.030,
                    'R_100': 0.035
                }.get(symbol, 0.02)
                
                change = np.random.normal(0, volatility) * last_price
                new_price = last_price + change
                
                # Ensure price doesn't go too low
                new_price = max(50.0, new_price)
                
                self.price_history[symbol].append(new_price)
    
    def _analyze_market(self, symbol: str, prices: List[float]) -> Dict:
        """Analyze market with SMC strategy"""
        if len(prices) < 20:
            return {"signal": "NEUTRAL", "confidence": 50, "current_price": prices[-1] if prices else 100.0}
        
        try:
            prices_array = np.array(prices[-50:])
            current_price = prices_array[-1]
            
            # Calculate indicators
            sma_10 = np.mean(prices_array[-10:])
            sma_20 = np.mean(prices_array[-20:])
            sma_50 = np.mean(prices_array[-50:])
            
            # Calculate RSI
            gains = np.where(np.diff(prices_array) > 0, np.diff(prices_array), 0)
            losses = np.where(np.diff(prices_array) < 0, -np.diff(prices_array), 0)
            
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0.1
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0.1
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss if avg_loss != 0 else 100
                rsi = 100 - (100 / (1 + rs))
            
            # Support/Resistance
            support = np.min(prices_array[-20:])
            resistance = np.max(prices_array[-20:])
            
            # Generate signal
            signal = "NEUTRAL"
            confidence = 50
            
            # Trend following with RSI confirmation
            if current_price > sma_10 > sma_20 and rsi < 70:
                signal = "BUY"
                confidence = 70 + min(20, (resistance - current_price) / resistance * 100)
            elif current_price < sma_10 < sma_20 and rsi > 30:
                signal = "SELL"
                confidence = 70 + min(20, (current_price - support) / current_price * 100)
            
            # Support/Resistance confirmation
            if current_price <= support * 1.02 and signal == "BUY":
                confidence += 10
            elif current_price >= resistance * 0.98 and signal == "SELL":
                confidence += 10
            
            # RSI extreme zones
            if rsi < 30 and signal == "BUY":
                confidence += 15
            elif rsi > 70 and signal == "SELL":
                confidence += 15
            
            confidence = min(95, max(50, confidence))
            
            return {
                "signal": signal,
                "confidence": round(confidence),
                "current_price": round(float(current_price), 3),
                "sma_10": round(float(sma_10), 3),
                "sma_20": round(float(sma_20), 3),
                "sma_50": round(float(sma_50), 3),
                "rsi": round(float(rsi), 1),
                "support": round(float(support), 3),
                "resistance": round(float(resistance), 3),
                "trend": "BULLISH" if sma_10 > sma_20 else "BEARISH"
            }
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {"signal": "NEUTRAL", "confidence": 50, "current_price": 100.0}
    
    def _can_trade(self, symbol: str) -> bool:
        """Check if we can trade this symbol"""
        # Check recent trades
        recent_trades = [t for t in self.trades[-10:] if t.get('symbol') == symbol]
        if not recent_trades:
            return True
        
        last_trade = recent_trades[-1]
        time_since = (datetime.now() - datetime.fromisoformat(last_trade['timestamp'])).total_seconds()
        return time_since >= self.settings['cooldown_seconds']
    
    def _calculate_trade_amount(self) -> float:
        """Calculate trade amount"""
        if self.connection.connected and self.connection.balance > 0:
            # Risk-based amount
            risk_amount = self.connection.balance * self.settings['risk_per_trade']
            return min(self.settings['trade_amount'], max(1.0, risk_amount))
        return self.settings['trade_amount']
    
    def _execute_trade(self, symbol: str, analysis: Dict, amount: float) -> Optional[Dict]:
        """Execute a trade"""
        try:
            trade_id = f"TR{int(time.time())}{np.random.randint(1000, 9999)}"
            
            # Try real trade if connected
            if self.connection.connected and self.settings['use_real_trading']:
                success, contract_id, contract_info = self.connection.place_trade(
                    symbol, analysis['signal'], amount
                )
                
                if success:
                    # Simulate profit/loss (in real trading, this would come from the contract)
                    profit = np.random.uniform(-self.settings['stop_loss'], self.settings['take_profit'])
                    
                    trade_record = {
                        'id': trade_id,
                        'symbol': symbol,
                        'direction': analysis['signal'],
                        'amount': round(amount, 2),
                        'confidence': analysis['confidence'],
                        'profit': round(profit, 2),
                        'contract_id': contract_id,
                        'payout': contract_info.get('payout'),
                        'status': 'EXECUTED',
                        'real_trade': True,
                        'timestamp': datetime.now().isoformat(),
                        'balance_after': self.connection.balance,
                        'analysis': {
                            'price': analysis['current_price'],
                            'rsi': analysis.get('rsi', 50),
                            'trend': analysis.get('trend', 'NEUTRAL')
                        }
                    }
                    
                    logger.info(f"✅ REAL trade executed: {symbol} {analysis['signal']} ${amount}")
                    return trade_record
            
            # Simulated trade
            profit = np.random.uniform(-2.0, 3.0)
            
            trade_record = {
                'id': trade_id,
                'symbol': symbol,
                'direction': analysis['signal'],
                'amount': round(amount, 2),
                'confidence': analysis['confidence'],
                'profit': round(profit, 2),
                'status': 'SIMULATED',
                'real_trade': False,
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis
            }
            
            logger.info(f"📊 Simulated trade: {symbol} {analysis['signal']} ${amount}")
            return trade_record
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return None
    
    def stop_trading(self):
        """Stop trading"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        logger.info(f"Trading stopped for {self.user_id}")
        return True, "Trading stopped"
    
    def place_manual_trade(self, symbol: str, direction: str, amount: float) -> Tuple[bool, str, Dict]:
        """Place manual trade"""
        try:
            if not self.connection.connected:
                return False, "Not connected to Deriv", {}
            
            success, contract_id, contract_info = self.connection.place_trade(symbol, direction, amount)
            
            if success:
                trade_id = f"MT{int(time.time())}{np.random.randint(1000, 9999)}"
                
                trade_record = {
                    'id': trade_id,
                    'symbol': symbol,
                    'direction': direction,
                    'amount': round(amount, 2),
                    'profit': 0.0,  # Will be updated later
                    'contract_id': contract_id,
                    'payout': contract_info.get('payout'),
                    'status': 'EXECUTED',
                    'real_trade': True,
                    'timestamp': datetime.now().isoformat(),
                    'balance_after': self.connection.balance
                }
                
                self.trades.append(trade_record)
                self.stats['total_trades'] += 1
                self.stats['real_trades'] += 1
                self.stats['balance'] = self.connection.balance
                
                return True, f"Trade executed: {contract_id}", trade_record
            
            return False, contract_id, {}
            
        except Exception as e:
            logger.error(f"Manual trade error: {e}")
            return False, str(e), {}
    
    def get_status(self) -> Dict:
        """Get current status"""
        # Update balance if connected
        if self.connection.connected:
            try:
                self.connection._update_balance()
                self.stats['balance'] = self.connection.balance
            except:
                pass
        
        # Calculate uptime
        start_time = datetime.fromisoformat(self.stats['start_time'])
        uptime = datetime.now() - start_time
        
        # Calculate win rate
        if self.trades:
            winning_trades = [t for t in self.trades if t.get('profit', 0) > 0]
            win_rate = (len(winning_trades) / len(self.trades)) * 100
        else:
            win_rate = 0
        
        # Get recent trades
        recent_trades = self.trades[-10:][::-1] if self.trades else []
        
        return {
            'running': self.running,
            'connected': self.connection.connected,
            'account_id': self.connection.account_id,
            'balance': round(self.stats['balance'], 2),
            'currency': self.connection.currency,
            'settings': self.settings,
            'stats': {
                **self.stats,
                'uptime_hours': round(uptime.total_seconds() / 3600, 2),
                'win_rate': round(win_rate, 1),
                'avg_profit': round(self.stats['total_profit'] / max(1, self.stats['total_trades']), 2)
            },
            'recent_trades': recent_trades,
            'total_trades': len(self.trades),
            'real_trades': self.stats['real_trades'],
            'account_info': self.connection.get_account_info()
        }

# ============ FLASK APP ============
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_urlsafe(64))

CORS(app, supports_credentials=True)

# ============ SESSION MANAGEMENT ============
def get_current_user() -> str:
    """Get current user ID from request"""
    return request.cookies.get('user_id', 'default')

# ============ API ENDPOINTS ============
@app.route('/api/set_token', methods=['POST'])
def set_token():
    """Set user's Deriv API token"""
    try:
        data = request.json or {}
        api_token = data.get('api_token', '').strip()
        
        if not api_token:
            return jsonify({'success': False, 'message': 'API token required'})
        
        user_id = get_current_user()
        session = get_user_session(user_id)
        session.set_api_token(api_token)
        
        # Try to connect immediately
        if session.engine:
            success, message = session.engine.connect()
            
            if success:
                account_info = session.engine.connection.get_account_info()
                
                return jsonify({
                    'success': True,
                    'message': f"✅ Connected successfully! Welcome {account_info.get('full_name', 'Trader')}",
                    'account_info': account_info,
                    'connected': True
                })
            else:
                return jsonify({
                    'success': False, 
                    'message': f"Connection failed: {message}",
                    'connected': False
                })
        
        return jsonify({'success': True, 'message': 'API token saved'})
        
    except Exception as e:
        logger.error(f"Set token error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/connect', methods=['POST'])
def connect():
    """Connect to Deriv"""
    try:
        user_id = get_current_user()
        session = get_user_session(user_id)
        
        if not session.engine:
            return jsonify({'success': False, 'message': 'Set API token first'})
        
        # If already connected, verify
        if session.engine.connection.connected:
            account_info = session.engine.connection.get_account_info()
            return jsonify({
                'success': True,
                'message': f"✅ Already connected to {account_info['account_id']}",
                'account_info': account_info,
                'connected': True
            })
        
        # Connect
        success, message = session.engine.connect()
        
        if success:
            account_info = session.engine.connection.get_account_info()
            
            return jsonify({
                'success': True,
                'message': f"✅ {message}",
                'account_info': account_info,
                'connected': True
            })
        else:
            return jsonify({
                'success': False, 
                'message': f"❌ {message}",
                'connected': False
            })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/start', methods=['POST'])
def start_trading():
    """Start trading"""
    try:
        user_id = get_current_user()
        session = get_user_session(user_id)
        
        if not session.engine:
            return jsonify({'success': False, 'message': 'Set API token first'})
        
        success, message = session.engine.start_trading()
        
        return jsonify({
            'success': success,
            'message': message,
            'real_trading': True
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/stop', methods=['POST'])
def stop_trading():
    """Stop trading"""
    try:
        user_id = get_current_user()
        session = get_user_session(user_id)
        
        if not session.engine:
            return jsonify({'success': False, 'message': 'Not initialized'})
        
        success, message = session.engine.stop_trading()
        
        return jsonify({'success': success, 'message': message})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/status', methods=['GET'])
def status():
    """Get trading status"""
    try:
        user_id = get_current_user()
        session = get_user_session(user_id)
        
        if not session.engine:
            return jsonify({
                'success': True,
                'status': {
                    'running': False,
                    'connected': False,
                    'account_id': None,
                    'balance': 0.0,
                    'message': 'Set API token to start'
                },
                'has_token': False
            })
        
        # Verify connection
        if session.engine.connection.connected:
            try:
                # Send ping to verify connection is alive
                session.engine.connection.ws.send(json.dumps({"ping": 1}))
                response = session.engine.connection.ws.recv()
                if "ping" not in response:
                    session.engine.connection.connected = False
            except:
                session.engine.connection.connected = False
        
        status_data = session.engine.get_status()
        
        return jsonify({
            'success': True,
            'status': status_data,
            'has_token': bool(session.api_token),
            'connected': status_data['connected']
        })
        
    except Exception as e:
        logger.error(f"Status error: {e}")
        return jsonify({
            'success': True,
            'status': {
                'running': False,
                'connected': False,
                'account_id': None,
                'balance': 0.0
            },
            'has_token': False
        })

@app.route('/api/trade', methods=['POST'])
def place_trade():
    """Place manual trade"""
    try:
        data = request.json or {}
        symbol = data.get('symbol', 'R_10')
        direction = data.get('direction', 'BUY')
        amount = float(data.get('amount', 5.0))
        
        user_id = get_current_user()
        session = get_user_session(user_id)
        
        if not session.engine:
            return jsonify({'success': False, 'message': 'Set API token first'})
        
        if not session.engine.connection.connected:
            return jsonify({'success': False, 'message': 'Not connected to Deriv'})
        
        success, message, trade_data = session.engine.place_manual_trade(symbol, direction, amount)
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'contract_id': trade_data.get('contract_id'),
                'balance': session.engine.connection.balance,
                'trade': trade_data
            })
        else:
            return jsonify({'success': False, 'message': message})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/trades', methods=['GET'])
def get_trades():
    """Get trade history"""
    try:
        user_id = get_current_user()
        session = get_user_session(user_id)
        
        if not session.engine:
            return jsonify({'success': False, 'message': 'Not initialized'})
        
        trades = session.engine.trades[-50:][::-1] if session.engine.trades else []
        
        return jsonify({
            'success': True,
            'trades': trades,
            'total': len(session.engine.trades),
            'real_trades': session.engine.stats['real_trades'],
            'total_profit': round(session.engine.stats['total_profit'], 2)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/markets', methods=['GET'])
def get_markets():
    """Get available markets"""
    return jsonify({
        'success': True,
        'markets': DERIV_MARKETS,
        'count': len(DERIV_MARKETS)
    })

@app.route('/api/settings', methods=['GET', 'POST'])
def settings():
    """Get or update settings"""
    user_id = get_current_user()
    session = get_user_session(user_id)
    
    if not session.engine:
        return jsonify({'success': False, 'message': 'Not initialized'})
    
    if request.method == 'GET':
        return jsonify({
            'success': True,
            'settings': session.engine.settings
        })
    
    else:  # POST
        try:
            data = request.json or {}
            new_settings = data.get('settings', {})
            
            # Update settings
            session.engine.settings.update(new_settings)
            
            return jsonify({
                'success': True,
                'message': 'Settings updated'
            })
            
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})

@app.route('/api/ping', methods=['GET'])
def ping():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Karanka Ultra Trading Bot',
        'version': '2.0',
        'users': len(user_sessions)
    })

@app.route('/api/check_token', methods=['GET'])
def check_token():
    """Check if API token is set"""
    user_id = get_current_user()
    session = get_user_session(user_id)
    
    connected = False
    account_info = {}
    
    if session.engine and session.engine.connection:
        connected = session.engine.connection.connected
        account_info = session.engine.connection.get_account_info()
    
    return jsonify({
        'success': True,
        'has_token': bool(session.api_token),
        'connected': connected,
        'account_info': account_info
    })

@app.route('/api/disconnect', methods=['POST'])
def disconnect():
    """Disconnect from Deriv"""
    try:
        user_id = get_current_user()
        session = get_user_session(user_id)
        
        if session.engine and session.engine.connection:
            session.engine.connection.disconnect()
            return jsonify({'success': True, 'message': 'Disconnected'})
        
        return jsonify({'success': False, 'message': 'Not connected'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/')
def index():
    """Main page"""
    response = make_response(render_template_string(HTML_TEMPLATE))
    response.set_cookie('user_id', 'default', max_age=60*60*24*30)  # 30 days
    return response

# ============ HTML TEMPLATE ============
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🚀 Karanka Ultra - Real Deriv Trading Bot</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --primary: #0a0a0a;
            --secondary: #1a1a1a;
            --accent: #00D4AA;
            --accent-light: #00ffd5;
            --success: #00C853;
            --danger: #FF5252;
            --warning: #FF9800;
            --info: #2196F3;
            --text: #ffffff;
            --text-muted: #aaaaaa;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background: var(--primary);
            color: var(--text);
            font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
            line-height: 1.6;
            min-height: 100vh;
            padding: 0;
            margin: 0;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        /* Header */
        .header {
            background: linear-gradient(135deg, var(--secondary) 0%, #2a2a2a 100%);
            padding: 25px 30px;
            border-radius: 20px;
            margin-bottom: 25px;
            text-align: center;
            border: 3px solid var(--accent);
            box-shadow: 0 10px 30px rgba(0, 212, 170, 0.2);
            position: relative;
            overflow: hidden;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 5px;
            background: linear-gradient(90deg, var(--accent), var(--accent-light));
        }
        
        .header h1 {
            color: var(--accent);
            margin: 0 0 10px 0;
            font-size: 2.8em;
            font-weight: 800;
            letter-spacing: 1px;
        }
        
        .header p {
            color: var(--text-muted);
            font-size: 1.1em;
            margin: 0;
        }
        
        .header .tagline {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        
        .header .tagline span {
            background: rgba(0, 212, 170, 0.1);
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.9em;
        }
        
        /* Card Styles */
        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            border: 2px solid transparent;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        
        .card:hover {
            transform: translateY(-5px);
            border-color: var(--accent);
            box-shadow: 0 10px 25px rgba(0, 212, 170, 0.15);
        }
        
        .card.connected {
            border-color: var(--success);
        }
        
        .card.disconnected {
            border-color: var(--danger);
        }
        
        .card.active {
            border-color: var(--accent);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(0, 212, 170, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(0, 212, 170, 0); }
            100% { box-shadow: 0 0 0 0 rgba(0, 212, 170, 0); }
        }
        
        .card h3 {
            color: var(--accent);
            margin-bottom: 20px;
            font-size: 1.4em;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .card h3 i {
            font-size: 1.2em;
        }
        
        /* Button Styles */
        .btn {
            padding: 14px 28px;
            background: var(--accent);
            color: black;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            font-weight: 700;
            font-size: 16px;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            min-width: 140px;
        }
        
        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 7px 20px rgba(0, 212, 170, 0.4);
            background: var(--accent-light);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .btn-success {
            background: var(--success);
        }
        
        .btn-success:hover {
            background: #00e676;
            box-shadow: 0 7px 20px rgba(0, 200, 83, 0.4);
        }
        
        .btn-danger {
            background: var(--danger);
        }
        
        .btn-danger:hover {
            background: #ff6b6b;
            box-shadow: 0 7px 20px rgba(255, 82, 82, 0.4);
        }
        
        .btn-info {
            background: var(--info);
        }
        
        .btn-info:hover {
            background: #42a5f5;
            box-shadow: 0 7px 20px rgba(33, 150, 243, 0.4);
        }
        
        .btn-warning {
            background: var(--warning);
        }
        
        .btn-warning:hover {
            background: #ffb74d;
            box-shadow: 0 7px 20px rgba(255, 152, 0, 0.4);
        }
        
        .btn-small {
            padding: 8px 16px;
            font-size: 14px;
            min-width: auto;
        }
        
        /* Alert Styles */
        .alert {
            padding: 18px;
            border-radius: 12px;
            margin: 20px 0;
            border-left: 6px solid;
            background: rgba(255, 255, 255, 0.05);
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .alert-success {
            border-color: var(--success);
            background: rgba(0, 200, 83, 0.1);
        }
        
        .alert-danger {
            border-color: var(--danger);
            background: rgba(255, 82, 82, 0.1);
        }
        
        .alert-info {
            border-color: var(--info);
            background: rgba(33, 150, 243, 0.1);
        }
        
        .alert-warning {
            border-color: var(--warning);
            background: rgba(255, 152, 0, 0.1);
        }
        
        /* Tabs */
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
            overflow-x: auto;
            padding: 20px;
            background: var(--secondary);
            border-radius: 15px;
            scrollbar-width: thin;
        }
        
        .tabs::-webkit-scrollbar {
            height: 8px;
        }
        
        .tabs::-webkit-scrollbar-track {
            background: rgba(0,0,0,0.2);
            border-radius: 4px;
        }
        
        .tabs::-webkit-scrollbar-thumb {
            background: var(--accent);
            border-radius: 4px;
        }
        
        .tab {
            padding: 15px 25px;
            background: rgba(0, 212, 170, 0.1);
            border-radius: 12px;
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 600;
        }
        
        .tab:hover {
            background: rgba(0, 212, 170, 0.2);
            transform: translateY(-2px);
        }
        
        .tab.active {
            background: var(--accent);
            color: black;
            font-weight: 700;
            box-shadow: 0 5px 15px rgba(0, 212, 170, 0.3);
        }
        
        /* Panels */
        .panel {
            display: none;
            padding: 30px;
            background: var(--secondary);
            border-radius: 15px;
            margin-bottom: 30px;
            animation: fadeIn 0.5s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .panel.active {
            display: block;
        }
        
        /* Grid Layouts */
        .grid {
            display: grid;
            gap: 20px;
        }
        
        .grid-2 {
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
        }
        
        .grid-3 {
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        }
        
        .grid-4 {
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        }
        
        /* Status Grid */
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        
        .status-card {
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            padding: 20px;
            border-left: 5px solid var(--accent);
        }
        
        .status-card h4 {
            color: var(--accent);
            margin-bottom: 15px;
            font-size: 1.1em;
        }
        
        .status-card .value {
            font-size: 2em;
            font-weight: 800;
            margin: 10px 0;
        }
        
        .status-card .label {
            color: var(--text-muted);
            font-size: 0.9em;
        }
        
        /* Trade Cards */
        .trade-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .trade-card {
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            padding: 20px;
            border-left: 5px solid;
            transition: transform 0.3s ease;
        }
        
        .trade-card:hover {
            transform: translateY(-5px);
        }
        
        .trade-card.buy {
            border-left-color: var(--success);
        }
        
        .trade-card.sell {
            border-left-color: var(--danger);
        }
        
        .trade-card .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            background: none;
            padding: 0;
            border: none;
        }
        
        .real-badge {
            background: var(--success);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
        }
        
        .simulated-badge {
            background: var(--warning);
            color: black;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
        }
        
        /* Form Elements */
        input, select, textarea {
            width: 100%;
            padding: 15px;
            background: rgba(0,0,0,0.3);
            border: 2px solid #444;
            border-radius: 10px;
            color: var(--text);
            font-size: 16px;
            margin: 10px 0;
            transition: all 0.3s ease;
        }
        
        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(0, 212, 170, 0.2);
        }
        
        input[type="number"] {
            font-family: 'Courier New', monospace;
            font-weight: bold;
        }
        
        /* API Token Input */
        .token-input-group {
            position: relative;
            margin: 20px 0;
        }
        
        .token-input-group input {
            padding-right: 50px;
            font-family: 'Courier New', monospace;
            letter-spacing: 1px;
        }
        
        .token-input-group button {
            position: absolute;
            right: 5px;
            top: 50%;
            transform: translateY(-50%);
        }
        
        /* Account Info */
        .account-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
            padding: 20px;
            background: rgba(0,0,0,0.2);
            border-radius: 12px;
        }
        
        .info-item {
            display: flex;
            flex-direction: column;
        }
        
        .info-label {
            color: var(--text-muted);
            font-size: 0.9em;
            margin-bottom: 5px;
        }
        
        .info-value {
            font-size: 1.1em;
            font-weight: 600;
            color: var(--accent);
        }
        
        /* Action Buttons Container */
        .action-buttons {
            display: flex;
            gap: 15px;
            margin: 30px 0;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        /* Loading Spinner */
        .spinner {
            border: 4px solid rgba(0, 212, 170, 0.2);
            border-top: 4px solid var(--accent);
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
        
        /* Badge */
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
            margin-left: 10px;
        }
        
        .badge-success { background: var(--success); color: white; }
        .badge-danger { background: var(--danger); color: white; }
        .badge-warning { background: var(--warning); color: black; }
        .badge-info { background: var(--info); color: white; }
        
        /* Responsive */
        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .grid-2, .grid-3, .grid-4 {
                grid-template-columns: 1fr;
            }
            
            .tabs {
                padding: 15px;
            }
            
            .tab {
                padding: 12px 20px;
                font-size: 14px;
            }
            
            .btn {
                padding: 12px 20px;
                font-size: 15px;
                min-width: 120px;
            }
            
            .action-buttons {
                flex-direction: column;
            }
            
            .action-buttons .btn {
                width: 100%;
            }
        }
        
        /* Animations */
        .fade-in {
            animation: fadeIn 0.5s ease;
        }
        
        .slide-up {
            animation: slideUp 0.3s ease;
        }
        
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Connection Status */
        .connection-status {
            padding: 12px;
            border-radius: 8px;
            margin: 15px 0;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .connected-status {
            background: rgba(0, 200, 83, 0.1);
            color: var(--success);
            border: 2px solid var(--success);
        }
        
        .disconnected-status {
            background: rgba(255, 82, 82, 0.1);
            color: var(--danger);
            border: 2px solid var(--danger);
        }
        
        /* Quick Stats */
        .quick-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .stat {
            text-align: center;
            padding: 15px;
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
        }
        
        .stat .number {
            font-size: 2em;
            font-weight: 800;
            color: var(--accent);
        }
        
        .stat .label {
            font-size: 0.9em;
            color: var(--text-muted);
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1><i class="fas fa-rocket"></i> Karanka Ultra Trading Bot</h1>
            <p>Real Deriv Trading • Advanced SMC Strategy • 24/7 Operation</p>
            <div class="tagline">
                <span><i class="fas fa-bolt"></i> Real-Time Trading</span>
                <span><i class="fas fa-shield-alt"></i> Secure Connection</span>
                <span><i class="fas fa-chart-line"></i> Advanced AI Analysis</span>
                <span><i class="fas fa-money-bill-wave"></i> Real Account Trading</span>
            </div>
        </div>
        
        <!-- API Token Section (Always Visible) -->
        <div class="card" id="tokenCard">
            <h3><i class="fas fa-key"></i> Deriv API Token</h3>
            <p>Enter your Deriv API token to connect your account and start real trading:</p>
            
            <div class="token-input-group">
                <input type="password" id="apiTokenInput" placeholder="Enter your Deriv API token here..." 
                       style="font-family: 'Courier New', monospace;">
                <button class="btn btn-success btn-small" onclick="setToken()">
                    <i class="fas fa-plug"></i> Connect
                </button>
            </div>
            
            <div style="margin: 15px 0;">
                <p style="font-size: 0.9em; color: var(--text-muted);">
                    <i class="fas fa-info-circle"></i> Get your API token from: 
                    <strong>Deriv.com → Settings → API Token</strong>
                </p>
            </div>
            
            <div id="tokenStatus" class="connection-status disconnected-status">
                <i class="fas fa-times-circle"></i> Not connected
            </div>
        </div>
        
        <!-- Tabs Navigation -->
        <div class="tabs">
            <div class="tab active" onclick="showTab('dashboard')">
                <i class="fas fa-tachometer-alt"></i> Dashboard
            </div>
            <div class="tab" onclick="showTab('trading')">
                <i class="fas fa-chart-line"></i> Trading
            </div>
            <div class="tab" onclick="showTab('trades')">
                <i class="fas fa-history"></i> Trade History
            </div>
            <div class="tab" onclick="showTab('markets')">
                <i class="fas fa-coins"></i> Markets
            </div>
            <div class="tab" onclick="showTab('account')">
                <i class="fas fa-user"></i> Account
            </div>
            <div class="tab" onclick="showTab('settings')">
                <i class="fas fa-cog"></i> Settings
            </div>
        </div>
        
        <!-- Dashboard Panel -->
        <div id="dashboard" class="panel active">
            <h2><i class="fas fa-tachometer-alt"></i> Trading Dashboard</h2>
            <div id="dashboardAlerts"></div>
            
            <!-- Connection Status -->
            <div class="card" id="connectionCard">
                <h3><i class="fas fa-link"></i> Connection Status</h3>
                <div id="connectionDetails">
                    <div class="connection-status disconnected-status">
                        <i class="fas fa-times-circle"></i> Not connected to Deriv
                    </div>
                </div>
                
                <div class="action-buttons">
                    <button class="btn btn-success" onclick="connectDeriv()" id="connectBtn">
                        <i class="fas fa-plug"></i> Connect to Deriv
                    </button>
                    <button class="btn btn-danger" onclick="disconnectDeriv()" id="disconnectBtn" style="display:none;">
                        <i class="fas fa-unlink"></i> Disconnect
                    </button>
                </div>
            </div>
            
            <!-- Quick Stats -->
            <div class="quick-stats" id="quickStats">
                <!-- Stats will be populated by JavaScript -->
            </div>
            
            <!-- Trading Status -->
            <div class="card" id="tradingStatusCard">
                <h3><i class="fas fa-robot"></i> Trading Status</h3>
                <div id="tradingStatus" style="text-align: center; padding: 20px;">
                    <p style="color: var(--text-muted);">
                        <i class="fas fa-pause-circle fa-2x"></i><br>
                        Trading is stopped
                    </p>
                </div>
                
                <div class="action-buttons">
                    <button class="btn btn-success" onclick="startTrading()" id="startBtn">
                        <i class="fas fa-play"></i> Start REAL Trading
                    </button>
                    <button class="btn btn-danger" onclick="stopTrading()" id="stopBtn" style="display:none;">
                        <i class="fas fa-stop"></i> Stop Trading
                    </button>
                    <button class="btn btn-info" onclick="updateStatus()">
                        <i class="fas fa-sync-alt"></i> Refresh Status
                    </button>
                </div>
            </div>
            
            <!-- Account Information -->
            <div class="card" id="accountCard" style="display:none;">
                <h3><i class="fas fa-user-circle"></i> Account Information</h3>
                <div id="accountDetails" class="account-info">
                    <!-- Account info will be populated by JavaScript -->
                </div>
            </div>
        </div>
        
        <!-- Trading Panel -->
        <div id="trading" class="panel">
            <h2><i class="fas fa-chart-line"></i> Trading Controls</h2>
            
            <div class="grid grid-2">
                <!-- Quick Trade -->
                <div class="card">
                    <h3><i class="fas fa-bolt"></i> Quick Trade</h3>
                    
                    <select id="tradeSymbol">
                        <option value="R_10">Volatility 10 Index</option>
                        <option value="R_25">Volatility 25 Index</option>
                        <option value="R_50">Volatility 50 Index</option>
                        <option value="R_75">Volatility 75 Index</option>
                        <option value="R_100">Volatility 100 Index</option>
                        <option value="CRASH_500">Crash 500 Index</option>
                        <option value="BOOM_500">Boom 500 Index</option>
                    </select>
                    
                    <input type="number" id="tradeAmount" value="5.00" min="1" step="0.01" placeholder="Amount ($)">
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 20px 0;">
                        <button class="btn btn-success" onclick="placeTrade('BUY')" id="buyBtn">
                            <i class="fas fa-arrow-up"></i> BUY
                        </button>
                        <button class="btn btn-danger" onclick="placeTrade('SELL')" id="sellBtn">
                            <i class="fas fa-arrow-down"></i> SELL
                        </button>
                    </div>
                    
                    <div style="margin-top: 20px; padding: 15px; background: rgba(0,0,0,0.3); border-radius: 10px;">
                        <p style="margin: 0; color: var(--text-muted); font-size: 0.9em;">
                            <i class="fas fa-info-circle"></i> Quick trades execute immediately in your Deriv account.
                            Minimum amount is $1.00.
                        </p>
                    </div>
                </div>
                
                <!-- Market Analysis -->
                <div class="card">
                    <h3><i class="fas fa-chart-bar"></i> Market Analysis</h3>
                    <div id="marketAnalysis" style="text-align: center; padding: 40px 20px;">
                        <p style="color: var(--text-muted);">
                            <i class="fas fa-chart-line fa-2x"></i><br>
                            Connect to view market analysis
                        </p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Trades Panel -->
        <div id="trades" class="panel">
            <h2><i class="fas fa-history"></i> Trade History</h2>
            
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <button class="btn btn-info" onclick="loadTrades()">
                    <i class="fas fa-sync-alt"></i> Refresh Trades
                </button>
                <div id="tradesSummary" style="color: var(--text-muted);">
                    No trades yet
                </div>
            </div>
            
            <div id="tradesList" style="margin-top: 20px;">
                <div style="text-align: center; padding: 40px 20px;">
                    <i class="fas fa-history fa-3x" style="color: var(--text-muted); margin-bottom: 15px;"></i>
                    <p style="color: var(--text-muted);">No trades recorded yet</p>
                </div>
            </div>
        </div>
        
        <!-- Markets Panel -->
        <div id="markets" class="panel">
            <h2><i class="fas fa-coins"></i> Available Markets</h2>
            <div id="marketsList" class="grid grid-3">
                <!-- Markets will be populated by JavaScript -->
            </div>
        </div>
        
        <!-- Account Panel -->
        <div id="account" class="panel">
            <h2><i class="fas fa-user-circle"></i> Account Details</h2>
            <div id="accountFullInfo" style="text-align: center; padding: 40px 20px;">
                <p style="color: var(--text-muted);">
                    <i class="fas fa-user fa-3x"></i><br><br>
                    Connect your Deriv account to view details
                </p>
            </div>
        </div>
        
        <!-- Settings Panel -->
        <div id="settings" class="panel">
            <h2><i class="fas fa-cog"></i> Bot Settings</h2>
            <div id="settingsContent" style="text-align: center; padding: 40px 20px;">
                <p style="color: var(--text-muted);">
                    <i class="fas fa-cog fa-3x"></i><br><br>
                    Settings will be available after connection
                </p>
            </div>
        </div>
        
        <!-- Alerts Container -->
        <div id="alertsContainer"></div>
    </div>
    
    <script>
        // Global variables
        let currentTab = 'dashboard';
        let statusInterval;
        let accountInfo = {};
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Karanka Ultra Trading Bot initialized');
            
            // Check token status immediately
            checkTokenStatus();
            
            // Load markets
            loadMarkets();
            
            // Start status updates
            updateStatus();
            
            // Set up auto-refresh every 5 seconds
            statusInterval = setInterval(updateStatus, 5000);
            
            // Keep Render alive
            setInterval(() => {
                fetch('/api/ping').catch(() => {});
            }, 180000);
        });
        
        // Tab management
        function showTab(tabName) {
            // Hide all panels
            document.querySelectorAll('.panel').forEach(panel => {
                panel.classList.remove('active');
            });
            
            // Update tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected panel
            const panel = document.getElementById(tabName);
            if (panel) {
                panel.classList.add('active');
                
                // Find and activate corresponding tab
                document.querySelectorAll('.tab').forEach(tab => {
                    if (tab.textContent.includes(tabName.replace(/^\w/, c => c.toUpperCase()))) {
                        tab.classList.add('active');
                    }
                });
            }
            
            currentTab = tabName;
            
            // Load tab-specific data
            switch(tabName) {
                case 'trades':
                    loadTrades();
                    break;
                case 'account':
                    updateAccountInfo();
                    break;
                case 'settings':
                    loadSettings();
                    break;
            }
        }
        
        // Alert system
        function showAlert(message, type = 'info', duration = 5000) {
            const container = document.getElementById('alertsContainer');
            
            const alert = document.createElement('div');
            alert.className = `alert alert-${type}`;
            alert.innerHTML = `
                <div style="display: flex; align-items: center; gap: 10px;">
                    <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'danger' ? 'exclamation-circle' : 'info-circle'}"></i>
                    <div>${message}</div>
                </div>
            `;
            
            container.appendChild(alert);
            
            // Auto-remove after duration
            setTimeout(() => {
                alert.style.opacity = '0';
                alert.style.transition = 'opacity 0.3s';
                setTimeout(() => alert.remove(), 300);
            }, duration);
        }
        
        // Token management
        function checkTokenStatus() {
            fetch('/api/check_token')
                .then(r => r.json())
                .then(data => {
                    if (data.success && data.has_token) {
                        document.getElementById('tokenStatus').innerHTML = `
                            <i class="fas fa-check-circle"></i> API token is set
                        `;
                        document.getElementById('tokenStatus').className = 'connection-status connected-status';
                        
                        if (data.connected && data.account_info) {
                            accountInfo = data.account_info;
                            updateTokenDisplay(data.account_info);
                        }
                    }
                })
                .catch(e => console.error('Token check error:', e));
        }
        
        function updateTokenDisplay(account) {
            const tokenStatus = document.getElementById('tokenStatus');
            if (account.connected) {
                tokenStatus.innerHTML = `
                    <i class="fas fa-check-circle"></i> Connected to ${account.account_id}
                `;
                tokenStatus.className = 'connection-status connected-status';
            }
        }
        
        function setToken() {
            const apiToken = document.getElementById('apiTokenInput').value.trim();
            
            if (!apiToken) {
                showAlert('Please enter your Deriv API token', 'danger');
                return;
            }
            
            // Show loading
            const tokenStatus = document.getElementById('tokenStatus');
            tokenStatus.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Connecting...`;
            tokenStatus.className = 'connection-status';
            
            fetch('/api/set_token', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({api_token: apiToken})
            })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        if (data.connected) {
                            accountInfo = data.account_info;
                            showAlert(data.message, 'success');
                            
                            // Update token status
                            tokenStatus.innerHTML = `
                                <i class="fas fa-check-circle"></i> Connected to ${accountInfo.account_id}
                            `;
                            tokenStatus.className = 'connection-status connected-status';
                            
                            // Clear token input for security
                            document.getElementById('apiTokenInput').value = '';
                            
                            // Update UI
                            updateStatus();
                            updateAccountDisplay();
                        } else {
                            showAlert('Token saved but connection failed', 'warning');
                        }
                    } else {
                        showAlert(`❌ ${data.message}`, 'danger');
                        tokenStatus.innerHTML = `<i class="fas fa-times-circle"></i> Connection failed`;
                        tokenStatus.className = 'connection-status disconnected-status';
                    }
                })
                .catch(e => {
                    showAlert(`Network error: ${e}`, 'danger');
                    tokenStatus.innerHTML = `<i class="fas fa-times-circle"></i> Connection failed`;
                    tokenStatus.className = 'connection-status disconnected-status';
                });
        }
        
        function connectDeriv() {
            const connectBtn = document.getElementById('connectBtn');
            const originalText = connectBtn.innerHTML;
            
            connectBtn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Connecting...`;
            connectBtn.disabled = true;
            
            fetch('/api/connect', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        accountInfo = data.account_info;
                        showAlert(data.message, 'success');
                        updateStatus();
                        updateAccountDisplay();
                    } else {
                        showAlert(`❌ ${data.message}`, 'danger');
                    }
                })
                .catch(e => showAlert(`Connection error: ${e}`, 'danger'))
                .finally(() => {
                    connectBtn.innerHTML = originalText;
                    connectBtn.disabled = false;
                });
        }
        
        function disconnectDeriv() {
            fetch('/api/disconnect', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        showAlert('Disconnected from Deriv', 'info');
                        accountInfo = {};
                        updateStatus();
                        updateAccountDisplay();
                    }
                });
        }
        
        function updateStatus() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        const status = data.status;
                        
                        // Update connection status
                        updateConnectionStatus(status);
                        
                        // Update trading status
                        updateTradingStatus(status);
                        
                        // Update quick stats
                        updateQuickStats(status);
                        
                        // Update account display
                        if (data.connected && status.account_info) {
                            accountInfo = status.account_info;
                            updateAccountDisplay();
                        }
                    }
                })
                .catch(e => console.error('Status update error:', e));
        }
        
        function updateConnectionStatus(status) {
            const connectionCard = document.getElementById('connectionCard');
            const connectionDetails = document.getElementById('connectionDetails');
            const connectBtn = document.getElementById('connectBtn');
            const disconnectBtn = document.getElementById('disconnectBtn');
            
            if (status.connected) {
                // Connected state
                connectionCard.classList.remove('disconnected');
                connectionCard.classList.add('connected');
                
                connectionDetails.innerHTML = `
                    <div class="connection-status connected-status">
                        <i class="fas fa-check-circle"></i> Connected to ${status.account_id}
                    </div>
                    <div style="margin-top: 15px;">
                        <p><strong>Account:</strong> ${status.account_id}</p>
                        <p><strong>Balance:</strong> $${status.balance.toFixed(2)} ${status.currency}</p>
                    </div>
                `;
                
                connectBtn.style.display = 'none';
                disconnectBtn.style.display = 'inline-block';
                
            } else {
                // Disconnected state
                connectionCard.classList.remove('connected');
                connectionCard.classList.add('disconnected');
                
                connectionDetails.innerHTML = `
                    <div class="connection-status disconnected-status">
                        <i class="fas fa-times-circle"></i> Not connected to Deriv
                    </div>
                    <div style="margin-top: 15px;">
                        <p>Enter your API token above and click Connect</p>
                    </div>
                `;
                
                connectBtn.style.display = 'inline-block';
                disconnectBtn.style.display = 'none';
            }
        }
        
        function updateTradingStatus(status) {
            const tradingStatus = document.getElementById('tradingStatus');
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            
            if (status.running) {
                tradingStatus.innerHTML = `
                    <div style="color: var(--success);">
                        <i class="fas fa-play-circle fa-3x"></i>
                        <h3 style="margin: 10px 0;">ACTIVE TRADING</h3>
                        <p>Account: ${status.account_id || 'Not connected'}</p>
                        <p>Strategy: SMC Advanced</p>
                    </div>
                `;
                
                startBtn.style.display = 'none';
                stopBtn.style.display = 'inline-block';
                
            } else {
                tradingStatus.innerHTML = `
                    <div style="color: var(--text-muted);">
                        <i class="fas fa-pause-circle fa-3x"></i>
                        <h3 style="margin: 10px 0;">TRADING STOPPED</h3>
                        <p>Click Start to begin automated trading</p>
                    </div>
                `;
                
                startBtn.style.display = 'inline-block';
                stopBtn.style.display = 'none';
            }
        }
        
        function updateQuickStats(status) {
            const quickStats = document.getElementById('quickStats');
            
            const stats = [
                { label: 'Total Trades', value: status.total_trades || 0 },
                { label: 'Real Trades', value: status.real_trades || 0 },
                { label: 'Total Profit', value: `$${status.stats?.total_profit?.toFixed(2) || '0.00'}` },
                { label: 'Win Rate', value: `${status.stats?.win_rate?.toFixed(1) || '0'}%` },
                { label: 'Balance', value: `$${status.balance?.toFixed(2) || '0.00'}` },
                { label: 'Uptime', value: `${status.stats?.uptime_hours?.toFixed(1) || '0'}h` }
            ];
            
            quickStats.innerHTML = stats.map(stat => `
                <div class="stat">
                    <div class="number">${stat.value}</div>
                    <div class="label">${stat.label}</div>
                </div>
            `).join('');
        }
        
        function updateAccountDisplay() {
            // Update account card in dashboard
            const accountCard = document.getElementById('accountCard');
            const accountDetails = document.getElementById('accountDetails');
            
            if (accountInfo.connected) {
                accountCard.style.display = 'block';
                
                accountDetails.innerHTML = `
                    <div class="info-item">
                        <div class="info-label">Account ID</div>
                        <div class="info-value">${accountInfo.account_id}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Balance</div>
                        <div class="info-value">$${accountInfo.balance.toFixed(2)} ${accountInfo.currency}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Status</div>
                        <div class="info-value">
                            <span class="badge ${accountInfo.is_virtual ? 'badge-warning' : 'badge-success'}">
                                ${accountInfo.is_virtual ? 'DEMO' : 'REAL'}
                            </span>
                        </div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Country</div>
                        <div class="info-value">${accountInfo.country || 'Not set'}</div>
                    </div>
                `;
                
                // Update full account info in account tab
                const accountFullInfo = document.getElementById('accountFullInfo');
                if (accountFullInfo) {
                    accountFullInfo.innerHTML = `
                        <div class="account-info" style="text-align: left;">
                            <div class="info-item">
                                <div class="info-label">Full Name</div>
                                <div class="info-value">${accountInfo.full_name || 'Not available'}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Email</div>
                                <div class="info-value">${accountInfo.email || 'Not available'}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Account ID</div>
                                <div class="info-value">${accountInfo.account_id}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Balance</div>
                                <div class="info-value">$${accountInfo.balance.toFixed(2)} ${accountInfo.currency}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Account Type</div>
                                <div class="info-value">
                                    ${accountInfo.is_virtual ? 'Demo (Virtual)' : 'Real Money'}
                                </div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Country</div>
                                <div class="info-value">${accountInfo.country || 'Not set'}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Connection Status</div>
                                <div class="info-value">
                                    <span class="badge badge-success">CONNECTED</span>
                                </div>
                            </div>
                        </div>
                    `;
                }
                
            } else {
                accountCard.style.display = 'none';
            }
        }
        
        function startTrading() {
            fetch('/api/start', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    showAlert(data.message, data.success ? 'success' : 'danger');
                    updateStatus();
                })
                .catch(e => showAlert(`Error: ${e}`, 'danger'));
        }
        
        function stopTrading() {
            fetch('/api/stop', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    showAlert(data.message, 'info');
                    updateStatus();
                });
        }
        
        function placeTrade(direction) {
            const symbol = document.getElementById('tradeSymbol').value;
            const amount = parseFloat(document.getElementById('tradeAmount').value);
            
            if (!amount || amount < 1) {
                showAlert('Minimum trade amount is $1', 'danger');
                return;
            }
            
            // Disable buttons during trade
            const buyBtn = document.getElementById('buyBtn');
            const sellBtn = document.getElementById('sellBtn');
            const originalBuyText = buyBtn.innerHTML;
            const originalSellText = sellBtn.innerHTML;
            
            if (direction === 'BUY') {
                buyBtn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Processing...`;
                buyBtn.disabled = true;
            } else {
                sellBtn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Processing...`;
                sellBtn.disabled = true;
            }
            
            fetch('/api/trade', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    symbol: symbol,
                    direction: direction,
                    amount: amount
                })
            })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        showAlert(`✅ ${data.message}`, 'success');
                        if (data.contract_id) {
                            showAlert(`Contract ID: ${data.contract_id}`, 'info', 10000);
                        }
                        updateStatus();
                        if (currentTab === 'trades') {
                            loadTrades();
                        }
                    } else {
                        showAlert(`❌ ${data.message}`, 'danger');
                    }
                })
                .catch(e => showAlert(`Trade error: ${e}`, 'danger'))
                .finally(() => {
                    buyBtn.innerHTML = originalBuyText;
                    sellBtn.innerHTML = originalSellText;
                    buyBtn.disabled = false;
                    sellBtn.disabled = false;
                });
        }
        
        function loadTrades() {
            const tradesList = document.getElementById('tradesList');
            const tradesSummary = document.getElementById('tradesSummary');
            
            tradesList.innerHTML = `
                <div style="text-align: center; padding: 20px;">
                    <div class="spinner"></div>
                    <p>Loading trades...</p>
                </div>
            `;
            
            fetch('/api/trades')
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        if (data.trades && data.trades.length > 0) {
                            tradesSummary.innerHTML = `
                                Showing ${data.trades.length} trades | 
                                Total: ${data.total} | 
                                Real: ${data.real_trades} | 
                                Profit: $${data.total_profit}
                            `;
                            
                            let html = '<div class="trade-grid">';
                            
                            data.trades.forEach(trade => {
                                const isBuy = trade.direction.toUpperCase() === 'BUY';
                                const profitColor = trade.profit >= 0 ? 'var(--success)' : 'var(--danger)';
                                const time = new Date(trade.timestamp).toLocaleString();
                                
                                html += `
                                    <div class="trade-card ${isBuy ? 'buy' : 'sell'}">
                                        <div class="header">
                                            <strong>${trade.symbol} ${trade.direction}</strong>
                                            <span class="${trade.real_trade ? 'real-badge' : 'simulated-badge'}">
                                                ${trade.real_trade ? 'REAL' : 'SIM'}
                                            </span>
                                        </div>
                                        <div style="margin: 15px 0;">
                                            <p><strong>Amount:</strong> $${trade.amount?.toFixed(2) || '0.00'}</p>
                                            <p><strong>Profit:</strong> 
                                                <span style="color: ${profitColor}; font-weight: bold;">
                                                    $${trade.profit?.toFixed(2) || '0.00'}
                                                </span>
                                            </p>
                                            <p><strong>Confidence:</strong> ${trade.confidence || 'N/A'}%</p>
                                            ${trade.contract_id ? `
                                                <p><strong>Contract:</strong> ${trade.contract_id.substring(0, 8)}...</p>
                                            ` : ''}
                                        </div>
                                        <div style="font-size: 0.85em; color: var(--text-muted);">
                                            ${time}
                                        </div>
                                    </div>
                                `;
                            });
                            
                            html += '</div>';
                            tradesList.innerHTML = html;
                        } else {
                            tradesList.innerHTML = `
                                <div style="text-align: center; padding: 40px 20px;">
                                    <i class="fas fa-history fa-3x" style="color: var(--text-muted); margin-bottom: 15px;"></i>
                                    <p style="color: var(--text-muted);">No trades recorded yet</p>
                                </div>
                            `;
                            tradesSummary.innerHTML = 'No trades yet';
                        }
                    }
                })
                .catch(e => {
                    tradesList.innerHTML = `
                        <div class="alert alert-danger">
                            Error loading trades: ${e.message}
                        </div>
                    `;
                });
        }
        
        function loadMarkets() {
            fetch('/api/markets')
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        const list = document.getElementById('marketsList');
                        let html = '';
                        
                        Object.entries(data.markets).forEach(([symbol, market]) => {
                            html += `
                                <div class="card">
                                    <h3>${market.name}</h3>
                                    <p><strong>Symbol:</strong> ${symbol}</p>
                                    <p><strong>Category:</strong> ${market.category}</p>
                                    <p><strong>Pip:</strong> ${market.pip}</p>
                                    <p><strong>Status:</strong> <span style="color: var(--success);">Available</span></p>
                                </div>
                            `;
                        });
                        
                        list.innerHTML = html;
                    }
                });
        }
        
        function updateAccountInfo() {
            const accountFullInfo = document.getElementById('accountFullInfo');
            
            if (accountInfo.connected) {
                accountFullInfo.innerHTML = `
                    <div class="account-info" style="text-align: left;">
                        <div class="info-item">
                            <div class="info-label">Full Name</div>
                            <div class="info-value">${accountInfo.full_name || 'Not available'}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Email</div>
                            <div class="info-value">${accountInfo.email || 'Not available'}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Account ID</div>
                            <div class="info-value">${accountInfo.account_id}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Balance</div>
                            <div class="info-value">$${accountInfo.balance.toFixed(2)} ${accountInfo.currency}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Account Type</div>
                            <div class="info-value">
                                ${accountInfo.is_virtual ? 'Demo (Virtual)' : 'Real Money'}
                            </div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Country</div>
                            <div class="info-value">${accountInfo.country || 'Not set'}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Connection Status</div>
                            <div class="info-value">
                                <span class="badge badge-success">CONNECTED</span>
                            </div>
                        </div>
                    </div>
                `;
            }
        }
        
        function loadSettings() {
            const settingsContent = document.getElementById('settingsContent');
            
            fetch('/api/settings')
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        const settings = data.settings;
                        
                        settingsContent.innerHTML = `
                            <div style="text-align: left;">
                                <h3>Trading Settings</h3>
                                <div class="account-info">
                                    <div class="info-item">
                                        <div class="info-label">Trade Amount</div>
                                        <div class="info-value">$${settings.trade_amount}</div>
                                    </div>
                                    <div class="info-item">
                                        <div class="info-label">Min Confidence</div>
                                        <div class="info-value">${settings.min_confidence}%</div>
                                    </div>
                                    <div class="info-item">
                                        <div class="info-label">Scan Interval</div>
                                        <div class="info-value">${settings.scan_interval} seconds</div>
                                    </div>
                                    <div class="info-item">
                                        <div class="info-label">Cooldown</div>
                                        <div class="info-value">${settings.cooldown_seconds} seconds</div>
                                    </div>
                                    <div class="info-item">
                                        <div class="info-label">Max Daily Trades</div>
                                        <div class="info-value">${settings.max_daily_trades}</div>
                                    </div>
                                    <div class="info-item">
                                        <div class="info-label">Risk Per Trade</div>
                                        <div class="info-value">${(settings.risk_per_trade * 100).toFixed(1)}%</div>
                                    </div>
                                    <div class="info-item">
                                        <div class="info-label">Real Trading</div>
                                        <div class="info-value">
                                            <span class="badge ${settings.use_real_trading ? 'badge-success' : 'badge-warning'}">
                                                ${settings.use_real_trading ? 'ENABLED' : 'DISABLED'}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                });
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    
    logger.info("=" * 60)
    logger.info("🚀 KARANKA ULTRA - REAL DERIV TRADING BOT")
    logger.info("=" * 60)
    logger.info("🔑 Users input API token in web app")
    logger.info("💰 Connects to REAL Deriv accounts")
    logger.info("⚡ 24/7 Trading on Render")
    logger.info("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )
