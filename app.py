#!/usr/bin/env python3
"""
🎯 KARANKA DERIV SMC BOT V7
📱 Mobile WebApp with REAL Deriv Trading
🚀 Connects to your Deriv account, gets live data, executes real trades
"""

import os
import json
import time
import threading
import hashlib
from datetime import datetime, timedelta
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')

# Flask
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_socketio import SocketIO, emit
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS

# Deriv Trading
import websocket
import requests

# Data
import pandas as pd
import numpy as np
from collections import defaultdict, deque

# ============ FLASK APP ============
app = Flask(__name__)
CORS(app)

# Configure for Render
database_url = os.environ.get('DATABASE_URL', 'sqlite:///karanka.db')
if database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'deriv-bot-secret-key-2025')
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# ============ DATABASE MODELS ============
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    deriv_token = db.Column(db.String(500))
    deriv_account_id = db.Column(db.String(100))
    balance = db.Column(db.Float, default=0.0)
    is_active = db.Column(db.Boolean, default=True)
    settings = db.Column(db.Text, default='{}')
    
    def get_settings(self):
        return json.loads(self.settings) if self.settings else {}
    
    def set_settings(self, settings_dict):
        self.settings = json.dumps(settings_dict)

class Trade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    symbol = db.Column(db.String(20))
    direction = db.Column(db.String(10))
    entry = db.Column(db.Float)
    exit = db.Column(db.Float)
    amount = db.Column(db.Float)
    profit = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(20))
    contract_id = db.Column(db.String(100))
    strategy = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    reason = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ============ DERIV CONFIGURATION ============
DERIV_CONFIG = {
    'app_id': 1089,
    'ws_url': 'wss://ws.derivws.com/websockets/v3',
    'api_url': 'https://api.deriv.com',
    'oauth_url': 'https://oauth.deriv.com/oauth2/authorize'
}

DERIV_MARKETS = {
    # Volatility Indices
    '1HZ10V': {'name': 'Volatility 10 Index', 'pip': 0.001, 'min': 0.35, 'max': 5.00, 'type': 'volatility'},
    '1HZ25V': {'name': 'Volatility 25 Index', 'pip': 0.001, 'min': 0.35, 'max': 5.00, 'type': 'volatility'},
    '1HZ50V': {'name': 'Volatility 50 Index', 'pip': 0.001, 'min': 0.35, 'max': 5.00, 'type': 'volatility'},
    '1HZ75V': {'name': 'Volatility 75 Index', 'pip': 0.001, 'min': 0.35, 'max': 5.00, 'type': 'volatility'},
    '1HZ100V': {'name': 'Volatility 100 Index', 'pip': 0.001, 'min': 0.35, 'max': 5.00, 'type': 'volatility'},
    
    # Boom/Crash
    'BOOM500': {'name': 'Boom 500 Index', 'pip': 0.01, 'min': 0.35, 'max': 5.00, 'type': 'boom'},
    'BOOM1000': {'name': 'Boom 1000 Index', 'pip': 0.01, 'min': 0.35, 'max': 5.00, 'type': 'boom'},
    'CRASH500': {'name': 'Crash 500 Index', 'pip': 0.01, 'min': 0.35, 'max': 5.00, 'type': 'crash'},
    'CRASH1000': {'name': 'Crash 1000 Index', 'pip': 0.01, 'min': 0.35, 'max': 5.00, 'type': 'crash'},
    
    # Forex
    'frxEURUSD': {'name': 'EUR/USD', 'pip': 0.0001, 'min': 1.00, 'max': 50.00, 'type': 'forex'},
    'frxGBPUSD': {'name': 'GBP/USD', 'pip': 0.0001, 'min': 1.00, 'max': 50.00, 'type': 'forex'},
    'frxUSDJPY': {'name': 'USD/JPY', 'pip': 0.01, 'min': 1.00, 'max': 50.00, 'type': 'forex'},
}

# ============ REAL DERIV WEBSOCKET MANAGER ============
class DerivRealTime:
    """Real Deriv WebSocket connection for live market data"""
    
    def __init__(self, token):
        self.token = token
        self.ws = None
        self.connected = False
        self.data = defaultdict(lambda: {'ticks': [], 'candles': defaultdict(list)})
        self.callbacks = []
        
    def connect(self):
        """Connect to real Deriv WebSocket"""
        try:
            ws_url = f"{DERIV_CONFIG['ws_url']}?app_id={DERIV_CONFIG['app_id']}"
            
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Start in background thread
            self.thread = threading.Thread(target=self.ws.run_forever)
            self.thread.daemon = True
            self.thread.start()
            
            # Wait for connection
            for _ in range(10):
                if self.connected:
                    return True
                time.sleep(0.5)
            return False
            
        except Exception as e:
            print(f"WebSocket connection error: {e}")
            return False
    
    def _on_open(self, ws):
        """WebSocket opened"""
        print("✅ Connected to Deriv WebSocket")
        self.connected = True
        self._authorize()
    
    def _authorize(self):
        """Authorize with API token"""
        auth_msg = {
            "authorize": self.token
        }
        self.ws.send(json.dumps(auth_msg))
    
    def _on_message(self, ws, message):
        """Handle incoming messages"""
        try:
            data = json.loads(message)
            
            if 'error' in data:
                print(f"Deriv error: {data['error']}")
                return
            
            if 'authorize' in data:
                print(f"✅ Authorized: {data['authorize']['loginid']}")
                self.account_id = data['authorize']['loginid']
                return
            
            if 'ohlc' in data:
                self._handle_ohlc(data['ohlc'])
            elif 'tick' in data:
                self._handle_tick(data['tick'])
            elif 'candles' in data:
                self._handle_candles(data['candles'])
            elif 'proposal' in data:
                self._handle_proposal(data['proposal'])
            elif 'buy' in data:
                self._handle_buy(data['buy'])
            
            # Notify callbacks
            for callback in self.callbacks:
                callback(data)
                
        except Exception as e:
            print(f"Message processing error: {e}")
    
    def _handle_ohlc(self, ohlc):
        """Handle OHLC data"""
        symbol = ohlc.get('symbol')
        if symbol:
            candle = {
                'time': ohlc.get('epoch'),
                'open': float(ohlc.get('open', 0)),
                'high': float(ohlc.get('high', 0)),
                'low': float(ohlc.get('low', 0)),
                'close': float(ohlc.get('close', 0))
            }
            self.data[symbol]['candles']['60'].append(candle)
            
            # Keep last 200 candles
            if len(self.data[symbol]['candles']['60']) > 200:
                self.data[symbol]['candles']['60'] = self.data[symbol]['candles']['60'][-200:]
    
    def _handle_tick(self, tick):
        """Handle tick data"""
        symbol = tick.get('symbol')
        if symbol:
            tick_data = {
                'time': tick.get('epoch'),
                'price': float(tick.get('quote', 0))
            }
            self.data[symbol]['ticks'].append(tick_data)
            
            # Keep last 500 ticks
            if len(self.data[symbol]['ticks']) > 500:
                self.data[symbol]['ticks'] = self.data[symbol]['ticks'][-500:]
    
    def _handle_candles(self, candles):
        """Handle candle batch"""
        pass
    
    def _handle_proposal(self, proposal):
        """Handle proposal response"""
        pass
    
    def _handle_buy(self, buy):
        """Handle buy response - REAL TRADE EXECUTED"""
        print(f"✅ REAL TRADE EXECUTED: {buy}")
        if 'contract_id' in buy:
            # Emit to all connected clients
            socketio.emit('trade_executed', {
                'contract_id': buy['contract_id'],
                'buy_price': buy.get('buy_price', 0),
                'payout': buy.get('payout', 0),
                'symbol': buy.get('symbol', 'Unknown')
            })
    
    def _on_error(self, ws, error):
        """WebSocket error"""
        print(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status, close_msg):
        """WebSocket closed"""
        self.connected = False
        print("WebSocket closed")
    
    def subscribe_ticks(self, symbol):
        """Subscribe to symbol ticks"""
        if not self.connected:
            return False
        
        subscribe_msg = {
            "ticks": symbol,
            "subscribe": 1
        }
        
        try:
            self.ws.send(json.dumps(subscribe_msg))
            return True
        except:
            return False
    
    def get_current_price(self, symbol):
        """Get current price for symbol"""
        if symbol in self.data and self.data[symbol]['ticks']:
            return self.data[symbol]['ticks'][-1]['price']
        return None
    
    def get_candles(self, symbol, count=100):
        """Get candles for symbol"""
        if symbol in self.data and self.data[symbol]['candles']['60']:
            candles = self.data[symbol]['candles']['60'][-count:]
            if candles:
                df = pd.DataFrame(candles)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                return df
        return None
    
    def buy_contract(self, symbol, direction, amount, duration=5):
        """BUY REAL CONTRACT on Deriv"""
        if not self.connected:
            return None
        
        contract_type = "CALL" if direction.upper() == "BUY" else "PUT"
        
        buy_request = {
            "buy": 1,
            "price": amount,
            "parameters": {
                "amount": amount,
                "basis": "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "duration": duration,
                "duration_unit": "t",
                "symbol": symbol,
                "barrier": "+0.1",
                "barrier2": "-0.1"
            }
        }
        
        try:
            self.ws.send(json.dumps(buy_request))
            return True
        except Exception as e:
            print(f"Buy request error: {e}")
            return False

# ============ TRADING ENGINE ============
class TradingEngine:
    """Main trading engine with real Deriv connection"""
    
    def __init__(self):
        self.user_bots = {}  # user_id -> bot data
        self.deriv_connections = {}  # user_id -> DerivRealTime
        self.active_trades = defaultdict(list)
        
    def start_bot_for_user(self, user_id, token, settings):
        """Start trading bot with real Deriv connection"""
        if user_id in self.user_bots:
            return False, "Bot already running"
        
        print(f"🚀 Starting bot for user {user_id}")
        
        # Create Deriv connection
        deriv = DerivRealTime(token)
        if not deriv.connect():
            return False, "Failed to connect to Deriv"
        
        # Store connection
        self.deriv_connections[user_id] = deriv
        
        # Create bot
        bot = {
            'running': True,
            'stop_flag': threading.Event(),
            'settings': settings,
            'trades_today': 0,
            'last_trade_time': None,
            'thread': None
        }
        
        self.user_bots[user_id] = bot
        
        # Start trading thread
        thread = threading.Thread(target=self._trading_loop, args=(user_id,))
        thread.daemon = True
        thread.start()
        bot['thread'] = thread
        
        return True, "Bot started successfully"
    
    def stop_bot_for_user(self, user_id):
        """Stop trading bot"""
        if user_id in self.user_bots:
            self.user_bots[user_id]['running'] = False
            self.user_bots[user_id]['stop_flag'].set()
            
            # Close Deriv connection
            if user_id in self.deriv_connections:
                del self.deriv_connections[user_id]
            
            del self.user_bots[user_id]
            return True, "Bot stopped"
        return False, "Bot not running"
    
    def _trading_loop(self, user_id):
        """Main trading loop - analyzes real market data and trades"""
        bot = self.user_bots.get(user_id)
        if not bot:
            return
        
        deriv = self.deriv_connections.get(user_id)
        if not deriv or not deriv.connected:
            bot['running'] = False
            return
        
        settings = bot['settings']
        
        while bot['running'] and not bot['stop_flag'].is_set():
            try:
                # Check trade limits
                if bot['trades_today'] >= settings.get('max_daily_trades', 100):
                    print(f"Daily limit reached for user {user_id}")
                    time.sleep(60)
                    continue
                
                # Analyze enabled markets
                signals = []
                
                for symbol in settings.get('markets', []):
                    if not bot['running']:
                        break
                    
                    # Subscribe to ticks
                    deriv.subscribe_ticks(symbol)
                    
                    # Get real market data
                    df = deriv.get_candles(symbol, count=50)
                    current_price = deriv.get_current_price(symbol)
                    
                    if df is None or current_price is None:
                        continue
                    
                    # Analyze with SMC strategies
                    analysis = self._analyze_with_smc(df, symbol, current_price, settings)
                    
                    if analysis['confidence'] >= settings.get('min_confidence', 55):
                        signals.append(analysis)
                
                # Execute best signal
                if signals and len(self.active_trades[user_id]) < settings.get('max_concurrent', 3):
                    # Sort by confidence
                    signals.sort(key=lambda x: x['confidence'], reverse=True)
                    best_signal = signals[0]
                    
                    # Execute REAL trade on Deriv
                    success = self._execute_real_trade(
                        user_id, 
                        best_signal['symbol'],
                        best_signal['direction'],
                        best_signal['amount'],
                        deriv
                    )
                    
                    if success:
                        bot['trades_today'] += 1
                        bot['last_trade_time'] = datetime.utcnow()
                        time.sleep(2)  # Wait between trades
                
                # Wait for next scan
                time.sleep(settings.get('scan_interval', 10))
                
            except Exception as e:
                print(f"Trading loop error for user {user_id}: {e}")
                time.sleep(10)
    
    def _analyze_with_smc(self, df, symbol, current_price, settings):
        """Analyze with SMC strategies - REAL ANALYSIS"""
        # Liquidity Grab Strategy
        liquidity_signal = self._liquidity_grab_strategy(df, symbol, current_price)
        
        # FVG Strategy
        fvg_signal = self._fvg_strategy(df, symbol, current_price)
        
        # Order Block Strategy
        ob_signal = self._order_block_strategy(df, symbol, current_price)
        
        # Choose best signal
        signals = [s for s in [liquidity_signal, fvg_signal, ob_signal] if s]
        
        if signals:
            best = max(signals, key=lambda x: x['confidence'])
            
            # Calculate amount based on settings
            amount = settings.get('trade_amount', 0.35)
            if settings.get('risk_level', 1.0) > 1:
                amount *= settings['risk_level']
            
            best['amount'] = amount
            return best
        
        # Default neutral signal
        return {
            'symbol': symbol,
            'direction': 'NEUTRAL',
            'confidence': 0,
            'amount': 0,
            'strategy': 'None',
            'reason': 'No clear signal'
        }
    
    def _liquidity_grab_strategy(self, df, symbol, current_price):
        """Liquidity Grab + Reversal Strategy"""
        if len(df) < 20:
            return None
        
        # Find recent highs/lows
        recent_high = df['high'].iloc[-20:-1].max()
        recent_low = df['low'].iloc[-20:-1].min()
        
        last_candle = df.iloc[-1]
        
        # Bullish setup: Sweep low, close above
        if (last_candle['low'] <= recent_low and 
            last_candle['close'] > recent_low):
            return {
                'symbol': symbol,
                'direction': 'BUY',
                'confidence': 75,
                'strategy': 'Liquidity Grab',
                'reason': f'Bullish reversal after liquidity sweep at {recent_low:.5f}'
            }
        
        # Bearish setup: Sweep high, close below
        if (last_candle['high'] >= recent_high and 
            last_candle['close'] < recent_high):
            return {
                'symbol': symbol,
                'direction': 'SELL',
                'confidence': 75,
                'strategy': 'Liquidity Grab',
                'reason': f'Bearish reversal after liquidity sweep at {recent_high:.5f}'
            }
        
        return None
    
    def _fvg_strategy(self, df, symbol, current_price):
        """Fair Value Gap Strategy"""
        if len(df) < 10:
            return None
        
        # Look for FVG in last 10 candles
        for i in range(len(df) - 10, len(df) - 2):
            candle1 = df.iloc[i]
            candle3 = df.iloc[i + 2]
            
            # Bullish FVG
            if candle1['high'] < candle3['low']:
                fvg_low = candle1['high']
                fvg_high = candle3['low']
                
                if fvg_low <= current_price <= fvg_high:
                    return {
                        'symbol': symbol,
                        'direction': 'BUY',
                        'confidence': 80,
                        'strategy': 'FVG Retest',
                        'reason': f'Bullish FVG retest at {current_price:.5f}'
                    }
            
            # Bearish FVG
            if candle3['high'] < candle1['low']:
                fvg_high = candle1['low']
                fvg_low = candle3['high']
                
                if fvg_low <= current_price <= fvg_high:
                    return {
                        'symbol': symbol,
                        'direction': 'SELL',
                        'confidence': 80,
                        'strategy': 'FVG Retest',
                        'reason': f'Bearish FVG retest at {current_price:.5f}'
                    }
        
        return None
    
    def _order_block_strategy(self, df, symbol, current_price):
        """Order Block Strategy"""
        if len(df) < 15:
            return None
        
        # Look for displacement candles
        for i in range(len(df) - 15, len(df) - 1):
            candle = df.iloc[i]
            prev_candle = df.iloc[i - 1] if i > 0 else candle
            
            body = abs(candle['close'] - candle['open'])
            candle_range = candle['high'] - candle['low']
            
            if candle_range == 0:
                continue
            
            body_percent = (body / candle_range) * 100
            
            # Strong bullish candle
            if (candle['close'] > candle['open'] and 
                body_percent > 70):
                
                # Check if price is in order block zone
                ob_low = prev_candle['low']
                ob_high = prev_candle['high']
                
                if ob_low <= current_price <= ob_high:
                    return {
                        'symbol': symbol,
                        'direction': 'BUY',
                        'confidence': 85,
                        'strategy': 'Order Block',
                        'reason': 'Bullish order block retest'
                    }
            
            # Strong bearish candle
            if (candle['close'] < candle['open'] and 
                body_percent > 70):
                
                ob_low = prev_candle['low']
                ob_high = prev_candle['high']
                
                if ob_low <= current_price <= ob_high:
                    return {
                        'symbol': symbol,
                        'direction': 'SELL',
                        'confidence': 85,
                        'strategy': 'Order Block',
                        'reason': 'Bearish order block retest'
                    }
        
        return None
    
    def _execute_real_trade(self, user_id, symbol, direction, amount, deriv):
        """Execute REAL trade on Deriv"""
        try:
            # Buy contract on Deriv
            success = deriv.buy_contract(symbol, direction, amount, duration=5)
            
            if success:
                # Record trade in database
                trade = Trade(
                    user_id=user_id,
                    symbol=symbol,
                    direction=direction,
                    entry=deriv.get_current_price(symbol) or 0,
                    amount=amount,
                    status='open',
                    strategy='SMC',
                    confidence=75,
                    reason='Real trade executed on Deriv'
                )
                db.session.add(trade)
                db.session.commit()
                
                # Add to active trades
                self.active_trades[user_id].append({
                    'trade_id': trade.id,
                    'symbol': symbol,
                    'direction': direction,
                    'amount': amount,
                    'time': datetime.utcnow()
                })
                
                # Emit socket event
                socketio.emit('trade_update', {
                    'user_id': user_id,
                    'trade': {
                        'id': trade.id,
                        'symbol': symbol,
                        'direction': direction,
                        'amount': amount,
                        'status': 'open',
                        'time': datetime.utcnow().isoformat()
                    }
                })
                
                return True
            
            return False
            
        except Exception as e:
            print(f"Trade execution error: {e}")
            return False

# Global trading engine
engine = TradingEngine()

# ============ ROUTES ============
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        
        if not username or not email or not password:
            flash('All fields are required', 'error')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return redirect(url_for('register'))
        
        # Default settings
        default_settings = {
            'min_confidence': 55,
            'trade_amount': 0.35,
            'max_daily_trades': 100,
            'max_concurrent': 3,
            'scan_interval': 10,
            'risk_level': 1.0,
            'markets': ['1HZ10V', '1HZ25V', '1HZ50V'],
            'dry_run': False
        }
        
        user = User(
            username=username,
            email=email,
            password=generate_password_hash(password),
            settings=json.dumps(default_settings)
        )
        
        db.session.add(user)
        db.session.commit()
        
        login_user(user, remember=True)
        flash('Account created successfully!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user, remember=True)
            return redirect(url_for('dashboard'))
        
        flash('Invalid email or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    # Stop bot if running
    engine.stop_bot_for_user(current_user.id)
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get user stats
    trades = Trade.query.filter_by(user_id=current_user.id).order_by(Trade.created_at.desc()).limit(10).all()
    
    # Check bot status
    bot_running = current_user.id in engine.user_bots
    
    # Get balance from Deriv if connected
    balance = current_user.balance
    
    return render_template('dashboard.html', 
                         user=current_user,
                         trades=trades,
                         bot_running=bot_running,
                         balance=balance,
                         markets=DERIV_MARKETS)

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if request.method == 'POST':
        settings = current_user.get_settings()
        
        # Update trading settings
        settings['min_confidence'] = float(request.form.get('min_confidence', 55))
        settings['trade_amount'] = float(request.form.get('trade_amount', 0.35))
        settings['max_daily_trades'] = int(request.form.get('max_daily_trades', 100))
        settings['max_concurrent'] = int(request.form.get('max_concurrent', 3))
        settings['scan_interval'] = int(request.form.get('scan_interval', 10))
        settings['risk_level'] = float(request.form.get('risk_level', 1.0))
        settings['dry_run'] = request.form.get('dry_run') == 'on'
        
        # Update markets
        selected_markets = request.form.getlist('markets')
        settings['markets'] = selected_markets
        
        # Update Deriv token
        deriv_token = request.form.get('deriv_token', '').strip()
        if deriv_token:
            current_user.deriv_token = deriv_token
        
        current_user.set_settings(settings)
        db.session.commit()
        
        flash('Settings saved successfully!', 'success')
        return redirect(url_for('settings'))
    
    settings_data = current_user.get_settings()
    return render_template('settings.html',
                         settings=settings_data,
                         markets=DERIV_MARKETS,
                         deriv_token=current_user.deriv_token or '')

@app.route('/api/start_bot', methods=['POST'])
@login_required
def start_bot():
    if not current_user.deriv_token:
        return jsonify({'success': False, 'message': 'Please set your Deriv API token first'})
    
    settings = current_user.get_settings()
    success, message = engine.start_bot_for_user(
        current_user.id,
        current_user.deriv_token,
        settings
    )
    
    return jsonify({'success': success, 'message': message})

@app.route('/api/stop_bot', methods=['POST'])
@login_required
def stop_bot():
    success, message = engine.stop_bot_for_user(current_user.id)
    return jsonify({'success': success, 'message': message})

@app.route('/api/bot_status')
@login_required
def bot_status():
    running = current_user.id in engine.user_bots
    active_trades = len(engine.active_trades.get(current_user.id, []))
    
    return jsonify({
        'running': running,
        'active_trades': active_trades,
        'user_id': current_user.id
    })

@app.route('/api/trades')
@login_required
def get_trades():
    trades = Trade.query.filter_by(user_id=current_user.id).order_by(Trade.created_at.desc()).limit(20).all()
    
    trades_data = []
    for trade in trades:
        trades_data.append({
            'id': trade.id,
            'symbol': trade.symbol,
            'direction': trade.direction,
            'entry': trade.entry,
            'amount': trade.amount,
            'profit': trade.profit,
            'status': trade.status,
            'strategy': trade.strategy,
            'reason': trade.reason,
            'time': trade.created_at.strftime('%H:%M:%S'),
            'date': trade.created_at.strftime('%Y-%m-%d')
        })
    
    return jsonify(trades_data)

@app.route('/api/market_data/<symbol>')
@login_required
def market_data(symbol):
    """Get real market data for a symbol"""
    if current_user.id in engine.deriv_connections:
        deriv = engine.deriv_connections[current_user.id]
        
        # Get candles
        df = deriv.get_candles(symbol, count=50)
        current_price = deriv.get_current_price(symbol)
        
        if df is not None and current_price is not None:
            return jsonify({
                'success': True,
                'symbol': symbol,
                'current_price': current_price,
                'candles': df.to_dict('records') if not df.empty else [],
                'timestamp': datetime.utcnow().isoformat()
            })
    
    return jsonify({
        'success': False,
        'message': 'No data available'
    })

@app.route('/api/account_balance')
@login_required
def account_balance():
    """Get account balance"""
    return jsonify({
        'balance': current_user.balance,
        'currency': 'USD'
    })

# ============ SOCKET EVENTS ============
@socketio.on('connect')
def handle_connect():
    if current_user.is_authenticated:
        emit('connected', {'user_id': current_user.id, 'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# ============ INITIALIZE ============
@app.before_first_request
def setup():
    db.create_all()
    print("✅ Database initialized")

# ============ RUN ============
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
