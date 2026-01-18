#!/usr/bin/env python3
"""
================================================================================
🚀 KARANKA ULTRA - REAL DERIV TRADING BOT
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

# ============ SETUP LOGGING ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('trading.log')]
)
logger = logging.getLogger(__name__)

# ============ FLASK APP ============
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_urlsafe(64))

CORS(app, supports_credentials=True)

# ============ RENDER KEEP-ALIVE ============
RENDER_APP_URL = os.environ.get('RENDER_EXTERNAL_URL', '')

def keep_render_awake():
    """Keep Render instance from sleeping"""
    while True:
        try:
            time.sleep(300)
            if RENDER_APP_URL:
                requests.get(f"{RENDER_APP_URL}/", timeout=5)
                logger.info("✅ Keep-alive ping sent")
        except Exception as e:
            logger.warning(f"Keep-alive failed: {e}")

if RENDER_APP_URL:
    threading.Thread(target=keep_render_awake, daemon=True).start()

# ============ USER SESSIONS ============
user_sessions = {}

class UserSession:
    """Store user data and trading engine"""
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.api_token = None
        self.engine = None
        self.created_at = datetime.now()
        self.last_active = datetime.now()
    
    def set_api_token(self, api_token: str):
        self.api_token = api_token
        self.engine = TradingEngine(self.user_id, api_token)

def get_user_session(user_id: str = "default") -> UserSession:
    """Get or create user session"""
    if user_id not in user_sessions:
        user_sessions[user_id] = UserSession(user_id)
    user_sessions[user_id].last_active = datetime.now()
    return user_sessions[user_id]

# ============ DERIV MARKETS ============
DERIV_MARKETS = {
    "R_10": {"name": "Volatility 10 Index", "pip": 0.001, "category": "Volatility"},
    "R_25": {"name": "Volatility 25 Index", "pip": 0.001, "category": "Volatility"},
    "R_50": {"name": "Volatility 50 Index", "pip": 0.001, "category": "Volatility"},
    "R_75": {"name": "Volatility 75 Index", "pip": 0.001, "category": "Volatility"},
    "R_100": {"name": "Volatility 100 Index", "pip": 0.001, "category": "Volatility"},
    "CRASH_500": {"name": "Crash 500 Index", "pip": 0.01, "category": "Crash/Boom"},
    "BOOM_500": {"name": "Boom 500 Index", "pip": 0.01, "category": "Crash/Boom"},
}

# ============ DERIV CONNECTION ============
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
        self.app_id = 1089
    
    def connect(self) -> Tuple[bool, str]:
        """Connect to Deriv"""
        try:
            if not self.api_token or len(self.api_token) < 20:
                return False, "Invalid API token"
            
            # Close any existing connection
            self.disconnect()
            
            # Try endpoints
            endpoints = [
                ("wss://ws.deriv.com/websockets/v3", "primary"),
                ("wss://ws.binaryws.com/websockets/v3", "binary"),
                ("wss://ws.derivws.com/websockets/v3", "backup")
            ]
            
            for endpoint, name in endpoints:
                try:
                    logger.info(f"Connecting to {name}...")
                    
                    self.ws = websocket.create_connection(
                        f"{endpoint}?app_id={self.app_id}",
                        timeout=10
                    )
                    
                    # Authenticate
                    self.ws.send(json.dumps({"authorize": self.api_token}))
                    response = json.loads(self.ws.recv())
                    
                    if "error" in response:
                        continue
                    
                    if "authorize" in response:
                        auth = response["authorize"]
                        self.connected = True
                        self.account_id = auth.get("loginid")
                        self.currency = auth.get("currency", "USD")
                        self.email = auth.get("email", "")
                        self.full_name = auth.get("fullname", "")
                        self.country = auth.get("country", "")
                        self.is_virtual = auth.get("is_virtual", False)
                        
                        # Get balance
                        self._update_balance()
                        
                        logger.info(f"✅ Connected to {self.account_id}")
                        return True, f"Connected to {self.account_id}"
                        
                except Exception as e:
                    logger.warning(f"{name} failed: {e}")
                    continue
            
            return False, "Connection failed"
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False, str(e)
    
    def _update_balance(self):
        """Update balance"""
        try:
            if self.connected and self.ws:
                self.ws.send(json.dumps({"balance": 1}))
                response = json.loads(self.ws.recv())
                if "balance" in response:
                    self.balance = float(response["balance"]["balance"])
        except:
            pass
    
    def place_trade(self, symbol: str, direction: str, amount: float) -> Tuple[bool, str, Dict]:
        """Place trade"""
        try:
            if not self.connected:
                return False, "Not connected", {}
            
            amount = max(1.0, amount)
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
                    "symbol": symbol,
                    "product_type": "basic"
                }
            }
            
            self.ws.send(json.dumps(trade_request))
            response = json.loads(self.ws.recv())
            
            if "error" in response:
                return False, response["error"].get("message", "Trade failed"), {}
            
            if "buy" in response:
                contract_id = response["buy"]["contract_id"]
                self._update_balance()
                
                return True, contract_id, {
                    "contract_id": contract_id,
                    "symbol": symbol,
                    "direction": direction,
                    "amount": amount
                }
            
            return False, "Unknown error", {}
            
        except Exception as e:
            return False, str(e), {}
    
    def disconnect(self):
        """Disconnect"""
        try:
            self.connected = False
            if self.ws:
                self.ws.close()
                self.ws = None
        except:
            pass
    
    def get_account_info(self) -> Dict:
        """Get account info"""
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
    """Trading engine"""
    
    def __init__(self, user_id: str, api_token: str):
        self.user_id = user_id
        self.api_token = api_token
        self.connection = DerivConnection(api_token)
        self.running = False
        self.thread = None
        self.trades = []
        
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
            'use_real_trading': True
        }
    
    def connect(self) -> Tuple[bool, str]:
        """Connect"""
        return self.connection.connect()
    
    def start_trading(self):
        """Start trading"""
        if self.running:
            return False, "Already trading"
        
        if not self.connection.connected:
            success, message = self.connect()
            if not success:
                return False, f"Connect failed: {message}"
        
        self.running = True
        self.thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.thread.start()
        return True, f"Started trading on {self.connection.account_id}"
    
    def _trading_loop(self):
        """Trading loop"""
        while self.running:
            try:
                time.sleep(self.settings['scan_interval'])
                
                # Simulate trading
                if np.random.random() > 0.7:  # 30% chance of trade
                    symbol = np.random.choice(self.settings['enabled_markets'])
                    direction = np.random.choice(['BUY', 'SELL'])
                    amount = self.settings['trade_amount']
                    
                    if self.connection.connected and self.settings['use_real_trading']:
                        success, contract_id, trade_info = self.connection.place_trade(
                            symbol, direction, amount
                        )
                        
                        if success:
                            trade = {
                                'id': f"TR{int(time.time())}",
                                'symbol': symbol,
                                'direction': direction,
                                'amount': amount,
                                'contract_id': contract_id,
                                'real_trade': True,
                                'timestamp': datetime.now().isoformat()
                            }
                            self.trades.append(trade)
                            self.stats['total_trades'] += 1
                            self.stats['real_trades'] += 1
                    
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                time.sleep(10)
    
    def stop_trading(self):
        """Stop trading"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        return True, "Stopped trading"
    
    def place_manual_trade(self, symbol: str, direction: str, amount: float) -> Tuple[bool, str, Dict]:
        """Place manual trade"""
        if not self.connection.connected:
            return False, "Not connected", {}
        
        success, message, trade_info = self.connection.place_trade(symbol, direction, amount)
        
        if success:
            trade = {
                'id': f"MT{int(time.time())}",
                'symbol': symbol,
                'direction': direction,
                'amount': amount,
                'contract_id': message,
                'real_trade': True,
                'timestamp': datetime.now().isoformat()
            }
            self.trades.append(trade)
            self.stats['total_trades'] += 1
            self.stats['real_trades'] += 1
            
        return success, message, trade_info
    
    def get_status(self) -> Dict:
        """Get status"""
        if self.connection.connected:
            self.stats['balance'] = self.connection.balance
        
        return {
            'running': self.running,
            'connected': self.connection.connected,
            'account_id': self.connection.account_id,
            'balance': self.stats['balance'],
            'currency': self.connection.currency,
            'settings': self.settings,
            'stats': self.stats,
            'recent_trades': self.trades[-10:][::-1] if self.trades else [],
            'total_trades': len(self.trades),
            'real_trades': self.stats['real_trades'],
            'account_info': self.connection.get_account_info()
        }

# ============ FLASK ROUTES ============
@app.route('/')
def index():
    """Main page"""
    return render_template_string(INDEX_HTML)

@app.route('/api/set_token', methods=['POST'])
def set_token():
    """Set API token"""
    try:
        data = request.json or {}
        api_token = data.get('api_token', '').strip()
        
        if not api_token:
            return jsonify({'success': False, 'message': 'API token required'})
        
        user_id = "default"
        session = get_user_session(user_id)
        session.set_api_token(api_token)
        
        if session.engine:
            success, message = session.engine.connect()
            if success:
                return jsonify({
                    'success': True,
                    'message': f"✅ Connected to {session.engine.connection.account_id}",
                    'account_info': session.engine.connection.get_account_info()
                })
            else:
                return jsonify({'success': False, 'message': message})
        
        return jsonify({'success': False, 'message': 'Failed to initialize'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/connect', methods=['POST'])
def connect():
    """Connect to Deriv"""
    try:
        user_id = "default"
        session = get_user_session(user_id)
        
        if not session.engine:
            return jsonify({'success': False, 'message': 'Set API token first'})
        
        success, message = session.engine.connect()
        
        if success:
            return jsonify({
                'success': True,
                'message': f"✅ {message}",
                'account_info': session.engine.connection.get_account_info()
            })
        else:
            return jsonify({'success': False, 'message': message})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/start', methods=['POST'])
def start_trading():
    """Start trading"""
    try:
        user_id = "default"
        session = get_user_session(user_id)
        
        if not session.engine:
            return jsonify({'success': False, 'message': 'Set API token first'})
        
        success, message = session.engine.start_trading()
        return jsonify({'success': success, 'message': message})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/stop', methods=['POST'])
def stop_trading():
    """Stop trading"""
    try:
        user_id = "default"
        session = get_user_session(user_id)
        
        if not session.engine:
            return jsonify({'success': False, 'message': 'Not initialized'})
        
        success, message = session.engine.stop_trading()
        return jsonify({'success': success, 'message': message})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/status', methods=['GET'])
def status():
    """Get status"""
    try:
        user_id = "default"
        session = get_user_session(user_id)
        
        if not session.engine:
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
        
        status_data = session.engine.get_status()
        return jsonify({
            'success': True,
            'status': status_data,
            'has_token': bool(session.api_token)
        })
        
    except Exception as e:
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
    """Place trade"""
    try:
        data = request.json or {}
        symbol = data.get('symbol', 'R_10')
        direction = data.get('direction', 'BUY')
        amount = float(data.get('amount', 5.0))
        
        user_id = "default"
        session = get_user_session(user_id)
        
        if not session.engine:
            return jsonify({'success': False, 'message': 'Set API token first'})
        
        success, message, trade_data = session.engine.place_manual_trade(symbol, direction, amount)
        return jsonify({'success': success, 'message': message, 'trade': trade_data})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/trades', methods=['GET'])
def get_trades():
    """Get trades"""
    try:
        user_id = "default"
        session = get_user_session(user_id)
        
        if not session.engine:
            return jsonify({'success': False, 'trades': []})
        
        trades = session.engine.trades[-50:][::-1] if session.engine.trades else []
        return jsonify({
            'success': True,
            'trades': trades,
            'total': len(session.engine.trades)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'trades': []})

@app.route('/api/markets', methods=['GET'])
def get_markets():
    """Get markets"""
    return jsonify({
        'success': True,
        'markets': DERIV_MARKETS
    })

@app.route('/api/check_token', methods=['GET'])
def check_token():
    """Check token"""
    user_id = "default"
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

@app.route('/api/ping', methods=['GET'])
def ping():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Karanka Ultra Trading Bot'
    })

@app.route('/health', methods=['GET'])
def health():
    """Health endpoint"""
    return jsonify({'status': 'ok'})

# ============ HTML TEMPLATE ============
INDEX_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>🚀 Karanka Ultra Trading Bot</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background: #0a0a0a;
            color: white;
            font-family: Arial, sans-serif;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 25px;
            text-align: center;
            border: 3px solid #00D4AA;
        }
        
        .header h1 {
            color: #00D4AA;
            margin: 0;
            font-size: 2.5em;
        }
        
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border: 2px solid #333;
        }
        
        .card.connected {
            border-color: #00C853;
        }
        
        .card.disconnected {
            border-color: #FF5252;
        }
        
        .btn {
            padding: 12px 24px;
            background: #00D4AA;
            color: black;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            margin: 5px;
            font-size: 16px;
            transition: all 0.3s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,212,170,0.4);
        }
        
        .btn-success {
            background: #00C853;
        }
        
        .btn-danger {
            background: #FF5252;
        }
        
        .btn-info {
            background: #2196F3;
        }
        
        input, select {
            width: 100%;
            padding: 12px;
            background: rgba(0,0,0,0.3);
            border: 1px solid #444;
            border-radius: 8px;
            color: white;
            margin: 10px 0;
        }
        
        .alert {
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 5px solid;
        }
        
        .alert-success {
            background: rgba(0,200,83,0.1);
            border-color: #00C853;
        }
        
        .alert-danger {
            background: rgba(255,82,82,0.1);
            border-color: #FF5252;
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }
        
        .trade-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .trade-card {
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            padding: 15px;
            border-left: 5px solid;
        }
        
        .trade-card.buy {
            border-left-color: #00C853;
        }
        
        .trade-card.sell {
            border-left-color: #FF5252;
        }
        
        .real-badge {
            background: #00C853;
            color: white;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            margin-left: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Karanka Ultra Trading Bot</h1>
            <p>Real Deriv Trading • 24/7 Operation • Advanced Strategy</p>
        </div>
        
        <!-- API Token Input -->
        <div class="card" id="tokenCard">
            <h3>🔑 Enter Your Deriv API Token</h3>
            <input type="password" id="apiTokenInput" placeholder="Paste your Deriv API token here">
            <button class="btn btn-success" onclick="setToken()">Connect to Deriv</button>
            <div id="tokenStatus" style="margin-top: 10px; padding: 10px; border-radius: 5px; background: rgba(255,82,82,0.1);">
                Not connected
            </div>
        </div>
        
        <!-- Dashboard -->
        <div id="dashboard">
            <div class="card" id="connectionCard">
                <h3>🔗 Connection Status</h3>
                <div id="connectionStatus">Not connected</div>
                <button class="btn btn-success" onclick="connectDeriv()" id="connectBtn">
                    Connect to Deriv
                </button>
            </div>
            
            <div class="status-grid" id="statusGrid">
                <div class="card">
                    <h3>🔄 Trading Status</h3>
                    <p id="tradingStatusText">Stopped</p>
                    <p>Total Trades: <span id="totalTrades">0</span></p>
                    <p>Real Trades: <span id="realTrades">0</span></p>
                </div>
                
                <div class="card">
                    <h3>💰 Account Info</h3>
                    <p>Account: <span id="accountId">Not connected</span></p>
                    <p>Balance: $<span id="balance">0.00</span></p>
                    <p>Currency: <span id="currency">USD</span></p>
                </div>
            </div>
            
            <div style="text-align: center; margin: 20px 0;">
                <button class="btn btn-success" onclick="startTrading()" id="startBtn">
                    ▶️ Start Trading
                </button>
                <button class="btn btn-danger" onclick="stopTrading()" id="stopBtn" style="display:none;">
                    ⏹️ Stop Trading
                </button>
                <button class="btn btn-info" onclick="updateStatus()">
                    🔄 Refresh
                </button>
            </div>
            
            <!-- Quick Trade -->
            <div class="card">
                <h3>🎯 Quick Trade</h3>
                <select id="tradeSymbol">
                    <option value="R_10">Volatility 10 Index</option>
                    <option value="R_25">Volatility 25 Index</option>
                    <option value="R_50">Volatility 50 Index</option>
                </select>
                
                <input type="number" id="tradeAmount" value="5.0" min="1" step="0.1" placeholder="Amount ($)">
                
                <div style="display: flex; gap: 10px; margin: 10px 0;">
                    <button class="btn btn-success" style="flex:1;" onclick="placeTrade('BUY')">📈 BUY</button>
                    <button class="btn btn-danger" style="flex:1;" onclick="placeTrade('SELL')">📉 SELL</button>
                </div>
            </div>
            
            <!-- Recent Trades -->
            <div class="card">
                <h3>📋 Recent Trades</h3>
                <button class="btn btn-info" onclick="loadTrades()" style="margin-bottom: 15px;">
                    Refresh Trades
                </button>
                <div id="tradesList"></div>
            </div>
        </div>
        
        <div id="alerts" style="margin-top: 30px;"></div>
    </div>
    
    <script>
        // Check token on load
        document.addEventListener('DOMContentLoaded', function() {
            checkTokenStatus();
            updateStatus();
            
            // Auto-refresh every 5 seconds
            setInterval(updateStatus, 5000);
            
            // Keep alive
            setInterval(() => {
                fetch('/api/ping').catch(() => {});
            }, 120000);
        });
        
        function showAlert(message, type = 'info') {
            const alerts = document.getElementById('alerts');
            const alert = document.createElement('div');
            alert.className = `alert alert-${type}`;
            alert.innerHTML = message;
            alerts.appendChild(alert);
            
            setTimeout(() => alert.remove(), 5000);
        }
        
        function checkTokenStatus() {
            fetch('/api/check_token')
                .then(r => r.json())
                .then(data => {
                    if (data.success && data.has_token) {
                        document.getElementById('tokenStatus').innerHTML = 
                            '<span style="color: #00C853">✅ API token is set</span>';
                        
                        if (data.connected) {
                            updateTokenDisplay(data.account_info);
                        }
                    }
                });
        }
        
        function updateTokenDisplay(account) {
            const tokenStatus = document.getElementById('tokenStatus');
            if (account.connected) {
                tokenStatus.innerHTML = 
                    `<span style="color: #00C853">✅ Connected to ${account.account_id}</span>`;
                tokenStatus.style.background = 'rgba(0,200,83,0.1)';
            }
        }
        
        function setToken() {
            const apiToken = document.getElementById('apiTokenInput').value.trim();
            
            if (!apiToken) {
                showAlert('Please enter your Deriv API token', 'danger');
                return;
            }
            
            fetch('/api/set_token', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({api_token: apiToken})
            })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        showAlert(data.message, 'success');
                        updateStatus();
                    } else {
                        showAlert(data.message, 'danger');
                    }
                })
                .catch(e => showAlert('Error: ' + e, 'danger'));
        }
        
        function connectDeriv() {
            fetch('/api/connect', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        showAlert(data.message, 'success');
                        updateStatus();
                    } else {
                        showAlert(data.message, 'danger');
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
                        const connectionCard = document.getElementById('connectionCard');
                        const connectionStatus = document.getElementById('connectionStatus');
                        
                        if (status.connected) {
                            connectionCard.className = 'card connected';
                            connectionStatus.innerHTML = 
                                `<span style="color: #00C853">✅ Connected to ${status.account_id}</span>`;
                            document.getElementById('connectBtn').style.display = 'none';
                        } else {
                            connectionCard.className = 'card disconnected';
                            connectionStatus.innerHTML = 
                                '<span style="color: #FF5252">❌ Not connected</span>';
                            document.getElementById('connectBtn').style.display = 'inline-block';
                        }
                        
                        // Update trading status
                        if (status.running) {
                            document.getElementById('tradingStatusText').innerHTML = 
                                '<span style="color: #00C853">✅ ACTIVE</span>';
                            document.getElementById('startBtn').style.display = 'none';
                            document.getElementById('stopBtn').style.display = 'inline-block';
                        } else {
                            document.getElementById('tradingStatusText').innerHTML = 
                                '<span style="color: #FF5252">⏸️ STOPPED</span>';
                            document.getElementById('startBtn').style.display = 'inline-block';
                            document.getElementById('stopBtn').style.display = 'none';
                        }
                        
                        // Update stats
                        document.getElementById('totalTrades').textContent = status.total_trades || 0;
                        document.getElementById('realTrades').textContent = status.real_trades || 0;
                        document.getElementById('accountId').textContent = status.account_id || 'Not connected';
                        document.getElementById('balance').textContent = status.balance?.toFixed(2) || '0.00';
                        document.getElementById('currency').textContent = status.currency || 'USD';
                    }
                });
        }
        
        function startTrading() {
            fetch('/api/start', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    showAlert(data.message, data.success ? 'success' : 'danger');
                    updateStatus();
                });
        }
        
        function stopTrading() {
            fetch('/api/stop', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    showAlert(data.message, 'success');
                    updateStatus();
                });
        }
        
        function placeTrade(direction) {
            const symbol = document.getElementById('tradeSymbol').value;
            const amount = parseFloat(document.getElementById('tradeAmount').value);
            
            if (!amount || amount < 1) {
                showAlert('Minimum amount is $1', 'danger');
                return;
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
                    showAlert(data.message, data.success ? 'success' : 'danger');
                    updateStatus();
                });
        }
        
        function loadTrades() {
            fetch('/api/trades')
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        const list = document.getElementById('tradesList');
                        
                        if (data.trades.length > 0) {
                            let html = '<div class="trade-grid">';
                            
                            data.trades.forEach(trade => {
                                html += `
                                    <div class="trade-card ${trade.direction.toLowerCase()}">
                                        <strong>${trade.symbol} ${trade.direction}</strong>
                                        ${trade.real_trade ? '<span class="real-badge">REAL</span>' : ''}
                                        <p>Amount: $${trade.amount?.toFixed(2) || '0.00'}</p>
                                        <p>Time: ${new Date(trade.timestamp).toLocaleTimeString()}</p>
                                    </div>
                                `;
                            });
                            
                            html += '</div>';
                            list.innerHTML = html;
                        } else {
                            list.innerHTML = '<p>No trades yet</p>';
                        }
                    }
                });
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"Starting Karanka Ultra Trading Bot on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
