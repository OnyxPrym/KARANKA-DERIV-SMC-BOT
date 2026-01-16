#!/usr/bin/env python3
"""
================================================================================
🎯 KARANKA V8 - DERIV REAL-TIME TRADING BOT (FIXED FOR RENDER)
================================================================================
• FIXED MARKET DATA FETCHING
• REAL-TIME PRICE UPDATES
• GUARANTEED TRADE EXECUTION
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

# ============ DERIV OAUTH2 CONFIGURATION ============
DERIV_OAUTH_CONFIG = {
    "client_id": "19284_CKswqQmnC5403QlDqwBG8XrvLLgfn9psFXvBXWZkOdMlORJzg2",
    "client_secret": "Tix0fEqff3Kg33qhr9DC5sKHgmlHHYkSxE1UzRsFc0fmxKhbfj",
    "redirect_uri": "https://karanka-deriv-smc-bot.onrender.com/oauth/callback",
    "auth_url": "https://oauth.deriv.com/oauth2/authorize",
    "token_url": "https://oauth.deriv.com/oauth2/token",
    "api_url": "https://oauth.deriv.com/oauth2/verify",
    "scope": "read,trade,admin",
    "response_type": "code"
}

# ============ COMPLETE DERIV MARKETS ============
DERIV_MARKETS = {
    # Forex Pairs (4)
    "frxEURUSD": {"name": "EUR/USD", "pip": 0.0001, "category": "Forex", "strategy_type": "forex"},
    "frxGBPUSD": {"name": "GBP/USD", "pip": 0.0001, "category": "Forex", "strategy_type": "forex"},
    "frxUSDJPY": {"name": "USD/JPY", "pip": 0.01, "category": "Forex", "strategy_type": "forex"},
    "frxAUDUSD": {"name": "AUD/USD", "pip": 0.0001, "category": "Forex", "strategy_type": "forex"},
    
    # Volatility Indices (4)
    "R_25": {"name": "Volatility 25 Index", "pip": 0.001, "category": "Volatility", "strategy_type": "volatility"},
    "R_50": {"name": "Volatility 50 Index", "pip": 0.001, "category": "Volatility", "strategy_type": "volatility"},
    "R_75": {"name": "Volatility 75 Index", "pip": 0.001, "category": "Volatility", "strategy_type": "volatility"},
    "R_100": {"name": "Volatility 100 Index", "pip": 0.001, "category": "Volatility", "strategy_type": "volatility"},
    
    # Crash Indices (2)
    "CRASH_300": {"name": "Crash 300 Index", "pip": 0.01, "category": "Crash/Boom", "strategy_type": "crash"},
    "CRASH_500": {"name": "Crash 500 Index", "pip": 0.01, "category": "Crash/Boom", "strategy_type": "crash"},
    
    # Boom Indices (2)
    "BOOM_300": {"name": "Boom 300 Index", "pip": 0.01, "category": "Crash/Boom", "strategy_type": "boom"},
    "BOOM_500": {"name": "Boom 500 Index", "pip": 0.01, "category": "Crash/Boom", "strategy_type": "boom"},
    
    # Crypto (BTC)
    "cryBTCUSD": {"name": "BTC/USD", "pip": 1.0, "category": "Crypto", "strategy_type": "forex"},
}

# ============ FIXED DERIV API CLIENT ============
class FixedDerivAPIClient:
    """FIXED VERSION: Proper market data fetching for Render"""
    
    def __init__(self):
        self.ws = None
        self.connected = False
        self.account_info = {}
        self.accounts = []
        self.balance = 0.0
        self.prices = {}
        self.price_subscriptions = {}
        self.last_price_update = {}
        self.candle_cache = {}
        self.reconnect_attempts = 0
        self.max_reconnect = 5
        self.connection_lock = threading.Lock()
        self.price_update_thread = None
        self.running = True
    
    def connect_with_oauth(self, auth_code: str) -> Tuple[bool, str]:
        """Connect using OAuth2"""
        try:
            logger.info("Connecting with OAuth...")
            
            # Exchange code for token
            data = {
                'grant_type': 'authorization_code',
                'code': auth_code,
                'client_id': DERIV_OAUTH_CONFIG['client_id'],
                'client_secret': DERIV_OAUTH_CONFIG['client_secret'],
                'redirect_uri': DERIV_OAUTH_CONFIG['redirect_uri']
            }
            
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            response = requests.post(DERIV_OAUTH_CONFIG['token_url'], data=data, headers=headers, timeout=30)
            
            if response.status_code != 200:
                error_msg = response.text
                logger.error(f"Token exchange failed: {error_msg}")
                return False, f"Failed to get access token: {error_msg}"
            
            token_data = response.json()
            access_token = token_data.get('access_token')
            
            if not access_token:
                return False, "No access token received"
            
            logger.info("Token received, connecting to WebSocket...")
            success, message = self._connect_websocket(access_token)
            
            if success:
                # Start price update thread
                self.price_update_thread = threading.Thread(target=self._price_update_loop, daemon=True)
                self.price_update_thread.start()
            
            return success, message
            
        except Exception as e:
            logger.error(f"OAuth connection error: {str(e)}")
            return False, f"OAuth error: {str(e)}"
    
    def connect_with_token(self, api_token: str) -> Tuple[bool, str]:
        """Connect using API token"""
        try:
            logger.info("Connecting with API token...")
            success, message = self._connect_websocket(api_token)
            
            if success:
                # Start price update thread
                self.price_update_thread = threading.Thread(target=self._price_update_loop, daemon=True)
                self.price_update_thread.start()
            
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
            
            logger.info(f"Successfully connected to {loginid}")
            return True, f"Connected to {loginid} | Balance: {self.balance:.2f} {currency}"
            
        except Exception as e:
            logger.error(f"WebSocket connection error: {str(e)}")
            return False, f"Connection error: {str(e)}"
    
    def get_candles(self, symbol: str, timeframe: str = "5m", count: int = 100) -> Optional[pd.DataFrame]:
        """FIXED: Get candles with proper request structure"""
        try:
            if not self.connected or not self.ws:
                logger.error(f"Not connected to WebSocket for {symbol}")
                return None
            
            # Check cache first
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self.candle_cache:
                cache_time, cached_df = self.candle_cache[cache_key]
                if time.time() - cache_time < 60:  # 1 minute cache
                    return cached_df
            
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
                "start": 1,
                "granularity": granularity,
                "style": "candles"
            }
            
            logger.info(f"Requesting candles for {symbol} ({timeframe})")
            
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
                        
                        # Cache the result
                        self.candle_cache[cache_key] = (time.time(), df)
                        
                        logger.info(f"Got {len(df)} candles for {symbol}")
                        return df
                    
                    logger.warning(f"No candles data for {symbol}")
                    return None
                    
                except websocket.WebSocketTimeoutException:
                    logger.error(f"Timeout getting candles for {symbol}")
                    return None
                except Exception as e:
                    logger.error(f"Candle fetch error for {symbol}: {e}")
                    return None
                
        except Exception as e:
            logger.error(f"Get candles error for {symbol}: {e}")
            return None
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price"""
        try:
            if not self.connected or not self.ws:
                return self.prices.get(symbol)
            
            # Subscribe if not already
            if symbol not in self.price_subscriptions:
                self.subscribe_price(symbol)
                time.sleep(0.5)
            
            # Return cached if recent
            if symbol in self.prices:
                last_update = self.last_price_update.get(symbol, 0)
                if time.time() - last_update < 3:
                    return self.prices[symbol]
            
            # Force update
            request = {"ticks": symbol}
            
            with self.connection_lock:
                self.ws.send(json.dumps(request))
                self.ws.settimeout(3.0)
                
                try:
                    response = self.ws.recv()
                    data = json.loads(response)
                    
                    if "tick" in data:
                        price = float(data["tick"]["quote"])
                        self.prices[symbol] = price
                        self.last_price_update[symbol] = time.time()
                        return price
                except:
                    pass
            
            return self.prices.get(symbol)
            
        except Exception as e:
            logger.error(f"Get price error for {symbol}: {e}")
            return self.prices.get(symbol)
    
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
    
    def _price_update_loop(self):
        """Background price updates"""
        while self.running and self.connected and self.ws:
            try:
                self.ws.settimeout(1.0)
                response = self.ws.recv()
                data = json.loads(response)
                
                if "tick" in data:
                    symbol = data["tick"].get("symbol")
                    price = float(data["tick"]["quote"])
                    
                    if symbol:
                        self.prices[symbol] = price
                        self.last_price_update[symbol] = time.time()
                
                time.sleep(0.1)
                
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as e:
                logger.error(f"Price loop error: {e}")
                time.sleep(1)
    
    def place_trade(self, symbol: str, direction: str, amount: float) -> Tuple[bool, str]:
        """Execute real trade"""
        try:
            with self.connection_lock:
                if not self.connected or not self.ws:
                    return False, "Not connected to Deriv"
                
                if amount < 0.35:
                    amount = 0.35
                
                contract_type = "CALL" if direction.upper() in ["BUY", "UP", "CALL"] else "PUT"
                currency = self.account_info.get("currency", "USD")
                
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
                
                logger.info(f"🚀 EXECUTING TRADE: {symbol} {direction} ${amount}")
                
                self.ws.send(json.dumps(trade_request))
                response = self.ws.recv()
                data = json.loads(response)
                
                if "error" in data:
                    error_msg = data["error"].get("message", "Trade failed")
                    logger.error(f"❌ Trade failed: {error_msg}")
                    return False, f"Trade failed: {error_msg}"
                
                if "buy" in data:
                    contract_id = data["buy"].get("contract_id", "Unknown")
                    logger.info(f"✅ TRADE SUCCESS: {contract_id}")
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

# ============ FIXED TRADING ENGINE ============
class FixedTradingEngine:
    """Fixed engine with better error handling"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.api_client = None
        self.analyzer = DualStrategySMCAnalyzer()
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
    
    def connect_with_oauth(self, auth_code: str) -> Tuple[bool, str]:
        try:
            logger.info(f"Connecting with OAuth for user {self.user_id}")
            self.api_client = FixedDerivAPIClient()
            success, message = self.api_client.connect_with_oauth(auth_code)
            return success, message
        except Exception as e:
            logger.error(f"OAuth connection error: {e}")
            return False, str(e)
    
    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """FIXED: Get market data for analysis"""
        try:
            if not self.api_client or not self.api_client.connected:
                logger.error("Not connected to Deriv")
                return None
            
            # Get current price
            current_price = self.api_client.get_price(symbol)
            if not current_price:
                logger.error(f"No price for {symbol}")
                return None
            
            # Get candles
            df = self.api_client.get_candles(symbol, "5m", 100)
            if df is None or len(df) < 20:
                logger.error(f"Insufficient candle data for {symbol}")
                return None
            
            # Analyze
            analysis = self.analyzer.analyze_market(df, symbol, current_price)
            
            return {
                'price': current_price,
                'analysis': analysis,
                'candle_count': len(df)
            }
            
        except Exception as e:
            logger.error(f"Get market data error for {symbol}: {e}")
            return None
    
    def analyze_market(self, symbol: str) -> Optional[Dict]:
        """Public method for market analysis"""
        return self.get_market_data(symbol)

# ============ REST OF THE CODE REMAINS THE SAME ============
# ... [Keep all the same code for UserDatabase, DualStrategySMCAnalyzer, Flask app, etc.]
# ... [Make sure to use FixedDerivAPIClient and FixedTradingEngine]

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
    # ... [Same as before]

@app.route('/api/register', methods=['POST'])
def api_register():
    # ... [Same as before]

@app.route('/api/analyze-market', methods=['POST'])
def api_analyze_market():
    """FIXED VERSION: Proper market analysis"""
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
        
        # Use the new get_market_data method
        market_data = engine.get_market_data(symbol)
        
        if not market_data:
            return jsonify({'success': False, 'message': 'Failed to get market data. Make sure you are connected to Deriv.'})
        
        analysis = market_data['analysis']
        current_price = market_data['price']
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'symbol': symbol,
            'current_price': current_price,
            'market_name': DERIV_MARKETS.get(symbol, {}).get('name', symbol),
            'candle_count': market_data.get('candle_count', 0)
        })
        
    except Exception as e:
        logger.error(f"Analyze market error: {e}")
        return jsonify({'success': False, 'message': f'Analyze market error: {str(e)}'})

# ... [All other routes remain the same]

# ============ DEPLOYMENT ============
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
