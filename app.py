#!/usr/bin/env python3
"""
================================================================================
🎯 KARANKA V8 - DERIV REAL-TIME TRADING BOT (PRODUCTION READY)
================================================================================
• FIXED SESSION MANAGEMENT
• PRODUCTION-READY CONFIG
• REAL TRADE EXECUTION
• ALL UI WORKING
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
from uuid import uuid4
import numpy as np
import pandas as pd
import requests
import websocket
from flask import Flask, render_template_string, jsonify, request, session, redirect, url_for
from flask_cors import CORS
from flask_session import Session
import redis

# ============ SETUP LOGGING ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot.log')
    ]
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

# ============ DUAL STRATEGY SMC ANALYZER ============
class DualStrategySMCAnalyzer:
    """Dual SMC Strategy: Fast logic for indices, Original for Forex/BTC"""
    
    def __init__(self):
        self.memory = defaultdict(lambda: deque(maxlen=100))
        self.prices = {}
        self.last_analysis = {}
        logger.info("Dual Strategy SMC Engine initialized")
    
    def analyze_market(self, df: pd.DataFrame, symbol: str, current_price: float) -> Dict:
        """Analyze market with appropriate strategy"""
        try:
            if df is None or len(df) < 20:
                return self._neutral_signal(symbol, current_price)
            
            market_info = DERIV_MARKETS.get(symbol, {})
            strategy_type = market_info.get('strategy_type', 'forex')
            
            # Store current price
            self.prices[symbol] = current_price
            
            # Choose strategy based on market type
            if strategy_type in ['volatility', 'crash', 'boom']:
                # FAST LOGIC FOR INDICES
                analysis = self._fast_indices_strategy(df, symbol, current_price)
            else:
                # ORIGINAL STRATEGY FOR FOREX/BTC
                analysis = self._original_forex_strategy(df, symbol, current_price)
            
            # Store in memory
            self.memory[symbol].append(analysis)
            self.last_analysis[symbol] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"SMC analysis error for {symbol}: {e}")
            return self._neutral_signal(symbol, current_price)
    
    def _fast_indices_strategy(self, df: pd.DataFrame, symbol: str, current_price: float) -> Dict:
        """FAST LIQUIDITY-BASED STRATEGY FOR INDICES"""
        
        df = self._prepare_data(df)
        confidence = 50
        signal = "NEUTRAL"
        signals = []
        
        # 1. LIQUIDITY SWEEP DETECTION (40 points)
        sweep_signal = self._detect_liquidity_sweep(df, symbol, current_price)
        if sweep_signal:
            confidence += 40
            signal = sweep_signal['direction']
            signals.append(f"🎯 {sweep_signal['type']}")
        
        # 2. DISPLACEMENT DETECTION (30 points)
        displacement = self._detect_displacement(df, symbol)
        if displacement['valid']:
            if signal == "NEUTRAL":
                signal = 'BUY' if displacement['direction'] == 'UP' else 'SELL'
                confidence += 30
            elif signal == displacement['direction']:
                confidence += 25  # Confirming signal
            signals.append(f"📈 Displacement")
        
        # 3. QUICK FVG DETECTION (20 points)
        fvg = self._find_quick_fvg(df, symbol, current_price)
        if fvg and fvg['type'].lower() == signal.lower():
            confidence += 20
            signals.append(f"⚡ Quick FVG")
        
        # 4. VOLATILITY ADJUSTMENT
        volatility = self._calculate_volatility(df)
        if volatility > 50 and signal != "NEUTRAL":
            confidence += 10  # Bonus for high volatility
        
        # Cap confidence
        confidence = min(95, max(50, confidence))
        
        return {
            "confidence": int(confidence),
            "signal": signal,
            "strength": min(95, abs(confidence - 50) * 2),
            "price": current_price,
            "signals": signals,
            "volatility": volatility,
            "timestamp": datetime.now().isoformat(),
            "strategy": "FAST_INDICES"
        }
    
    def _original_forex_strategy(self, df: pd.DataFrame, symbol: str, current_price: float) -> Dict:
        """ORIGINAL SMC STRATEGY FOR FOREX/BTC"""
        
        df = self._prepare_data(df)
        
        # 1. MARKET STRUCTURE (40 points)
        structure_score = self._analyze_market_structure(df)
        
        # 2. ORDER BLOCKS (30 points)
        order_block_score = self._analyze_order_blocks(df)
        
        # 3. FAIR VALUE GAPS (15 points)
        fvg_score = self._analyze_fair_value_gaps(df)
        
        # 4. LIQUIDITY ANALYSIS (15 points)
        liquidity_score = self._analyze_liquidity(df, current_price)
        
        # Total confidence
        total_confidence = 50 + structure_score + order_block_score + fvg_score + liquidity_score
        total_confidence = max(0, min(100, total_confidence))
        
        # Determine signal
        signal = "NEUTRAL"
        if total_confidence >= 65:
            signal = "BUY" if structure_score > 0 else "SELL"
        elif total_confidence <= 35:
            signal = "SELL" if structure_score < 0 else "BUY"
        
        return {
            "confidence": int(total_confidence),
            "signal": signal,
            "strength": min(95, abs(total_confidence - 50) * 2),
            "price": current_price,
            "structure_score": structure_score,
            "order_block_score": order_block_score,
            "fvg_score": fvg_score,
            "liquidity_score": liquidity_score,
            "volatility": self._calculate_volatility(df),
            "timestamp": datetime.now().isoformat(),
            "strategy": "ORIGINAL_FOREX"
        }
    
    def _detect_liquidity_sweep(self, df: pd.DataFrame, symbol: str, current_price: float) -> Optional[Dict]:
        """Detect liquidity sweeps for indices"""
        try:
            # Look for recent swing points
            lookback = 15 if 'CRASH' in symbol or 'BOOM' in symbol else 25
            
            recent_high = df['high'].iloc[-lookback:-1].max()
            recent_low = df['low'].iloc[-lookback:-1].min()
            
            current_candle = df.iloc[-1]
            
            # BULLISH SWEEP: Price dips below recent low then closes higher
            if (current_candle['low'] <= recent_low and
                current_candle['close'] > recent_low and
                current_candle['close'] > current_candle['open']):
                
                return {
                    'type': 'BULLISH_SWEEP',
                    'direction': 'BUY',
                    'level': recent_low,
                    'confidence': 85
                }
            
            # BEARISH SWEEP: Price spikes above recent high then closes lower
            if (current_candle['high'] >= recent_high and
                current_candle['close'] < recent_high and
                current_candle['close'] < current_candle['open']):
                
                return {
                    'type': 'BEARISH_SWEEP',
                    'direction': 'SELL',
                    'level': recent_high,
                    'confidence': 85
                }
            
            return None
        except:
            return None
    
    def _detect_displacement(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Detect strong price displacement"""
        try:
            # Look at last 3 candles
            closes = df['close'].values[-3:]
            
            # Calculate total move
            total_move = abs(closes[-1] - closes[0])
            
            # Calculate average candle body strength
            body_strengths = []
            for i in range(-3, 0):
                body = abs(df['close'].iloc[i] - df['open'].iloc[i])
                total_range = df['high'].iloc[i] - df['low'].iloc[i]
                if total_range > 0:
                    body_percent = (body / total_range) * 100
                    body_strengths.append(body_percent)
            
            avg_body_strength = np.mean(body_strengths) if body_strengths else 0
            
            # Determine thresholds based on symbol
            thresholds = {
                'R_25': 0.05, 'R_50': 0.08, 'R_75': 0.12, 'R_100': 0.15,
                'CRASH_300': 0.25, 'CRASH_500': 0.30,
                'BOOM_300': 0.25, 'BOOM_500': 0.30
            }
            
            threshold = thresholds.get(symbol, 0.10)
            pip_value = DERIV_MARKETS.get(symbol, {}).get('pip', 0.001)
            pip_move = total_move / pip_value
            
            is_valid = (
                total_move >= threshold and
                avg_body_strength >= 60 and
                pip_move >= 8
            )
            
            direction = 'UP' if closes[-1] > closes[0] else 'DOWN'
            
            return {
                'valid': is_valid,
                'direction': direction,
                'strength': avg_body_strength,
                'move_pips': pip_move
            }
        except:
            return {'valid': False, 'direction': 'NONE', 'strength': 0, 'move_pips': 0}
    
    def _find_quick_fvg(self, df: pd.DataFrame, symbol: str, current_price: float) -> Optional[Dict]:
        """Find recent FVGs"""
        try:
            for i in range(len(df)-5, len(df)-2):
                c1 = df.iloc[i]
                c3 = df.iloc[i+2]
                
                # Bullish FVG
                if c1['high'] < c3['low']:
                    zone_low = c1['high']
                    zone_high = c3['low']
                    
                    if zone_low <= current_price <= zone_high:
                        return {'type': 'BULLISH', 'zone': (zone_low, zone_high)}
                
                # Bearish FVG
                elif c1['low'] > c3['high']:
                    zone_low = c3['high']
                    zone_high = c1['low']
                    
                    if zone_low <= current_price <= zone_high:
                        return {'type': 'BEARISH', 'zone': (zone_low, zone_high)}
            
            return None
        except:
            return None
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for analysis"""
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['higher_high'] = (df['high'] > df['high'].shift(1))
        df['lower_low'] = (df['low'] < df['low'].shift(1))
        return df
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> float:
        """Original market structure analysis"""
        try:
            score = 0
            if df['ema_20'].iloc[-1] > df['ema_50'].iloc[-1]:
                score += 15
            else:
                score -= 15
            
            recent_higher_highs = df['higher_high'].tail(5).sum()
            recent_lower_lows = df['lower_low'].tail(5).sum()
            
            if recent_higher_highs >= 2:
                score += 15
            elif recent_lower_lows >= 2:
                score -= 15
            
            return score / 40 * 100
        except:
            return 0
    
    def _analyze_order_blocks(self, df: pd.DataFrame) -> float:
        """Original order block analysis"""
        try:
            score = 0
            for i in range(len(df)-5, len(df)):
                if i > 0 and df['close'].iloc[i] < df['open'].iloc[i]:
                    if df['close'].iloc[i+1] > df['open'].iloc[i+1]:
                        score += 5
                if i > 0 and df['close'].iloc[i] > df['open'].iloc[i]:
                    if df['close'].iloc[i+1] < df['open'].iloc[i+1]:
                        score -= 5
            return max(-15, min(15, score))
        except:
            return 0
    
    def _analyze_fair_value_gaps(self, df: pd.DataFrame) -> float:
        """Original FVG analysis"""
        try:
            score = 0
            for i in range(2, len(df)-1):
                if df['low'].iloc[i] > df['high'].iloc[i-1]:
                    score += 2
                elif df['high'].iloc[i] < df['low'].iloc[i-1]:
                    score -= 2
            return max(-7.5, min(7.5, score))
        except:
            return 0
    
    def _analyze_liquidity(self, df: pd.DataFrame, current_price: float) -> float:
        """Original liquidity analysis"""
        try:
            score = 0
            for i in range(len(df)-10, len(df)):
                if abs(df['high'].iloc[i] - current_price) < (df['high'].iloc[i] - df['low'].iloc[i]) * 0.1:
                    score -= 3
                if abs(df['low'].iloc[i] - current_price) < (df['high'].iloc[i] - df['low'].iloc[i]) * 0.1:
                    score += 3
            return max(-7.5, min(7.5, score))
        except:
            return 0
    
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
            "confidence": 0,
            "signal": "NEUTRAL",
            "strength": 0,
            "price": current_price,
            "timestamp": datetime.now().isoformat(),
            "strategy": "NEUTRAL"
        }

# ============ DERIV API CLIENT ============
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
        self.candle_cache = {}
        self.reconnect_attempts = 0
        self.max_reconnect = 3
        self.connection_lock = threading.Lock()
        self.running = True
        self.app_id = 1089  # Deriv app ID
    
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
            ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}&l=EN"
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
                except:
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
            
            # Check cache (1 minute)
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self.candle_cache:
                cache_time, cached_df = self.candle_cache[cache_key]
                if time.time() - cache_time < 60:
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
                        
                        # Cache for 1 minute
                        self.candle_cache[cache_key] = (time.time(), df)
                        
                        logger.info(f"📊 Got {len(df)} candles for {symbol}")
                        return df
                    
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
                
                # Proper trade request for Deriv API
                trade_request = {
                    "buy": 1,
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

# ============ TRADING ENGINE ============
class TradingEngine:
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
    
    def connect_with_token(self, api_token: str) -> Tuple[bool, str]:
        try:
            logger.info(f"Connecting with token for user {self.user_id}")
            self.api_client = DerivAPIClient()
            success, message = self.api_client.connect_with_token(api_token)
            return success, message
        except Exception as e:
            logger.error(f"Token connection error: {e}")
            return False, str(e)
    
    def update_settings(self, settings: Dict):
        """Update settings including market selection"""
        old_markets = set(self.settings.get('enabled_markets', []))
        new_markets = set(settings.get('enabled_markets', old_markets))
        
        # Update enabled markets subscriptions
        if self.api_client and self.api_client.connected:
            # Subscribe to new markets
            for symbol in new_markets - old_markets:
                self.api_client.subscribe_price(symbol)
                time.sleep(0.1)
        
        self.settings.update(settings)
        logger.info(f"Settings updated for user {self.user_id}")
    
    def start_trading(self):
        """Start the trading bot"""
        if self.running:
            return False, "Already running"
        
        if not self.api_client or not self.api_client.connected:
            return False, "Not connected to Deriv"
        
        # Subscribe to all enabled markets
        for symbol in self.settings['enabled_markets']:
            self.api_client.subscribe_price(symbol)
            time.sleep(0.1)
        
        self.running = True
        
        # Start trading thread
        self.thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.thread.start()
        
        mode = "DRY RUN" if self.settings['dry_run'] else "REAL TRADING"
        logger.info(f"💰 {mode} started for user {self.user_id}")
        return True, f"{mode} started!"
    
    def stop_trading(self):
        """Stop the trading bot"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        logger.info(f"Trading stopped for user {self.user_id}")
        return True, "Trading stopped"
    
    def _trading_loop(self):
        """Main trading loop"""
        logger.info(f"🔥 Trading loop started for user {self.user_id}")
        
        while self.running:
            try:
                # Check if we can trade
                if not self._can_trade():
                    time.sleep(5)
                    continue
                
                # Scan enabled markets
                for symbol in self.settings['enabled_markets']:
                    if not self.running:
                        break
                    
                    try:
                        # Check cooldown
                        if not self._check_cooldown(symbol):
                            continue
                        
                        # Get real-time price
                        current_price = self.api_client.get_price(symbol)
                        if not current_price:
                            continue
                        
                        # Get candles for analysis
                        df = self.api_client.get_candles(symbol, "5m", 100)
                        if df is None or len(df) < 20:
                            continue
                        
                        # Analyze market with DUAL SMC
                        analysis = self.analyzer.analyze_market(df, symbol, current_price)
                        
                        # Check if we should trade
                        if (analysis['signal'] != 'NEUTRAL' and 
                            analysis['confidence'] >= self.settings['min_confidence']):
                            
                            direction = analysis['signal']
                            confidence = analysis['confidence']
                            
                            # Execute trade based on dry_run setting
                            if self.settings['dry_run']:
                                logger.info(f"📝 DRY RUN: Would trade {symbol} {direction} at ${self.settings['trade_amount']} (Confidence: {confidence}%)")
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
                                logger.info(f"🚀 EXECUTING REAL TRADE: {symbol} {direction} ${self.settings['trade_amount']} (Confidence: {confidence}%)")
                                
                                success, trade_id = self.api_client.place_trade(
                                    symbol, direction, self.settings['trade_amount']
                                )
                                
                                if success:
                                    logger.info(f"✅ TRADE SUCCESS: {symbol} {direction} - ID: {trade_id}")
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
                                    
                                    # Update cooldown
                                    self.last_trade_time[symbol] = datetime.now()
                                
                        time.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        continue
                
                # Wait before next scan
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                time.sleep(30)
    
    def _check_cooldown(self, symbol: str) -> bool:
        """Check if we should wait before trading this symbol again"""
        try:
            if symbol not in self.last_trade_time:
                return True
            
            last_trade = self.last_trade_time[symbol]
            time_since_last = (datetime.now() - last_trade).total_seconds()
            
            # 5 minute cooldown for all markets
            if time_since_last < 300:
                return False
            
            return True
        except:
            return True
    
    def _can_trade(self) -> bool:
        try:
            # Check max concurrent trades
            if len(self.active_trades) >= self.settings['max_concurrent_trades']:
                return False
            
            # Reset daily/hourly counters if needed
            now = datetime.now()
            if now.date() > self.stats['last_reset'].date():
                self.stats['daily_trades'] = 0
                self.stats['hourly_trades'] = 0
                self.stats['last_reset'] = now
            
            # Check daily limit
            if self.stats['daily_trades'] >= self.settings['max_daily_trades']:
                return False
            
            # Check hourly limit
            if self.stats['hourly_trades'] >= self.settings['max_hourly_trades']:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Can trade check error: {e}")
            return False
    
    def _record_trade(self, trade_data: Dict):
        trade_data['id'] = len(self.trades) + 1
        self.trades.append(trade_data)
        
        self.stats['total_trades'] += 1
        self.stats['daily_trades'] += 1
        self.stats['hourly_trades'] += 1
        
        # Add to active trades if real
        if not trade_data.get('dry_run', True):
            self.active_trades.append(trade_data['id'])
        
        # Reset hourly trades after 1 hour
        def reset_hourly():
            time.sleep(3600)
            self.stats['hourly_trades'] = max(0, self.stats['hourly_trades'] - 1)
        
        threading.Thread(target=reset_hourly, daemon=True).start()
    
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
            
            # Analyze with DUAL SMC STRATEGY
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
    
    def place_manual_trade(self, symbol: str, direction: str, amount: float) -> Tuple[bool, str]:
        """Place a manual trade (real or dry run)"""
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

# ============ FLASK APP ============
app = Flask(__name__)

# Production-ready session configuration
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_urlsafe(32))
app.config['SESSION_TYPE'] = 'filesystem'  # Use filesystem for simplicity
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
app.config['SESSION_COOKIE_SECURE'] = True  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_REFRESH_EACH_REQUEST'] = True

# CORS configuration for production
CORS(app, 
     supports_credentials=True,
     resources={r"/api/*": {
         "origins": ["https://*.onrender.com", "http://localhost:5000", "http://127.0.0.1:5000"],
         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization"],
         "expose_headers": ["Content-Type"],
         "supports_credentials": True,
         "max_age": 3600
     }}
)

# Initialize components
user_db = UserDatabase()
trading_engines = {}

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    # Allow credentials
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    
    # Set allowed origins for production
    allowed_origins = [
        'https://*.onrender.com',
        'http://localhost:5000',
        'http://127.0.0.1:5000'
    ]
    
    origin = request.headers.get('Origin')
    if origin:
        # Check if origin is allowed
        for allowed in allowed_origins:
            if allowed.startswith('*'):
                if origin.endswith(allowed[2:]):  # Remove '*.' from start
                    response.headers.add('Access-Control-Allow-Origin', origin)
                    break
            elif origin == allowed:
                response.headers.add('Access-Control-Allow-Origin', origin)
                break
    
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Expose-Headers', 'Content-Type')
    
    return response

# ============ API ROUTES ============
@app.route('/api/login', methods=['POST', 'OPTIONS'])
def api_login():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password required'})
        
        success, message = user_db.authenticate(username, password)
        
        if success:
            session['username'] = username
            session['user_id'] = user_db.get_user(username)['user_id']
            session.permanent = True
            
            # Create trading engine if not exists
            if username not in trading_engines:
                user_data = user_db.get_user(username)
                engine = TradingEngine(user_id=user_data['user_id'])
                engine.update_settings(user_data.get('settings', {}))
                trading_engines[username] = engine
            
            logger.info(f"User {username} logged in successfully")
            return jsonify({'success': True, 'message': 'Login successful'})
        else:
            return jsonify({'success': False, 'message': message})
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'success': False, 'message': f'Login error: {str(e)}'})

@app.route('/api/register', methods=['POST', 'OPTIONS'])
def api_register():
    if request.method == 'OPTIONS':
        return '', 200
    
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

@app.route('/api/logout', methods=['POST', 'OPTIONS'])
def api_logout():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        username = session.get('username')
        if username:
            if username in trading_engines:
                engine = trading_engines[username]
                engine.stop_trading()
                if engine.api_client:
                    engine.api_client.close_connection()
                del trading_engines[username]
            session.clear()
            logger.info(f"User {username} logged out")
        
        return jsonify({'success': True, 'message': 'Logged out successfully'})
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return jsonify({'success': False, 'message': f'Logout error: {str(e)}'})

@app.route('/api/connect-token', methods=['POST', 'OPTIONS'])
def api_connect_token():
    if request.method == 'OPTIONS':
        return '', 200
    
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

@app.route('/api/status', methods=['GET', 'OPTIONS'])
def api_status():
    if request.method == 'OPTIONS':
        return '', 200
    
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

@app.route('/api/start-trading', methods=['POST', 'OPTIONS'])
def api_start_trading():
    if request.method == 'OPTIONS':
        return '', 200
    
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
        logger.error(f"Start trading error: {e}")
        return jsonify({'success': False, 'message': f'Start trading error: {str(e)}'})

@app.route('/api/stop-trading', methods=['POST', 'OPTIONS'])
def api_stop_trading():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        username = session.get('username')
        if not username:
            return jsonify({'success': False, 'message': 'Not logged in'})
        
        engine = trading_engines.get(username)
        if not engine:
            return jsonify({'success': True, 'message': 'Not running'})
        
        success, message = engine.stop_trading()
        return jsonify({'success': success, 'message': message})
        
    except Exception as e:
        logger.error(f"Stop trading error: {e}")
        return jsonify({'success': False, 'message': f'Stop trading error: {str(e)}'})

@app.route('/api/update-settings', methods=['POST', 'OPTIONS'])
def api_update_settings():
    if request.method == 'OPTIONS':
        return '', 200
    
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

@app.route('/api/place-trade', methods=['POST', 'OPTIONS'])
def api_place_trade():
    """Place manual trade"""
    if request.method == 'OPTIONS':
        return '', 200
    
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
        success, message = engine.place_manual_trade(symbol, direction, amount)
        
        return jsonify({
            'success': success,
            'message': message,
            'dry_run': engine.settings.get('dry_run', True)
        })
        
    except Exception as e:
        logger.error(f"Place trade error: {e}")
        return jsonify({'success': False, 'message': f'Place trade error: {str(e)}'})

@app.route('/api/analyze-market', methods=['POST', 'OPTIONS'])
def api_analyze_market():
    """Analyze market with REAL data"""
    if request.method == 'OPTIONS':
        return '', 200
    
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

@app.route('/api/check-session', methods=['GET', 'OPTIONS'])
def api_check_session():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        username = session.get('username')
        if username:
            return jsonify({'success': True, 'username': username})
        else:
            return jsonify({'success': False, 'username': None})
    except Exception as e:
        return jsonify({'success': False, 'username': None})

# ============ DEBUG ENDPOINTS ============
@app.route('/api/debug-session', methods=['GET'])
def debug_session():
    """Debug endpoint to check session state"""
    return jsonify({
        'username': session.get('username'),
        'session_id': session.sid if hasattr(session, 'sid') else 'N/A',
        'logged_in': 'username' in session,
        'session_keys': list(session.keys())
    })

@app.route('/api/debug-engines', methods=['GET'])
def debug_engines():
    """Debug endpoint to check trading engines"""
    return jsonify({
        'total_engines': len(trading_engines),
        'engines': list(trading_engines.keys()),
        'users_in_session': session.get('username')
    })

# ============ MAIN ROUTES ============
@app.route('/')
def index():
    """Main route - serve the web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Render"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'engines_active': len(trading_engines),
        'users_total': len(user_db.users)
    })

# ============ HTML TEMPLATE (FIXED) ============
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
            border: 1px solid var(--gold-secondary);
        }
        
        .market-checkbox:hover {
            background: rgba(255, 215, 0, 0.1);
        }
        
        .market-checkbox label {
            cursor: pointer;
            flex: 1;
            color: var(--gold-light);
            font-size: 14px;
        }
        
        .market-checkbox input[type="checkbox"] {
            width: 18px;
            height: 18px;
            cursor: pointer;
            accent-color: var(--gold-primary);
        }
        
        .market-category {
            margin-bottom: 20px;
            padding: 15px;
            background: var(--black-tertiary);
            border-radius: 10px;
            border: 1px solid var(--gold-secondary);
        }
        
        .market-category-title {
            font-size: 16px;
            color: var(--gold-primary);
            margin-bottom: 10px;
            font-weight: bold;
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
                <h2 style="color: var(--gold-primary); margin-bottom: 20px;">📊 Trading Dashboard</h2>
                
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
                
                <div style="display: flex; gap: 15px; margin-top: 20px;">
                    <button class="btn btn-success" onclick="startTrading()" id="start-btn">🚀 Start Trading</button>
                    <button class="btn btn-danger" onclick="stopTrading()" id="stop-btn">⏹️ Stop Trading</button>
                </div>
                
                <div style="margin-top: 30px; padding: 20px; background: var(--black-tertiary); border-radius: 10px;">
                    <h3 style="color: var(--gold-light); margin-bottom: 15px;">📈 Strategy Status</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px;">
                        <div style="background: rgba(0, 200, 83, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid var(--success);">
                            <div style="font-weight: bold; color: var(--success);">Forex/BTC Strategy</div>
                            <div style="font-size: 12px; color: var(--gold-secondary); margin-top: 5px;">Original SMC Logic</div>
                        </div>
                        <div style="background: rgba(33, 150, 243, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid var(--info);">
                            <div style="font-weight: bold; color: var(--info);">Indices Strategy</div>
                            <div style="font-size: 12px; color: var(--gold-secondary); margin-top: 5px;">Fast Liquidity Logic</div>
                        </div>
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
                
                <div class="form-group">
                    <label class="form-label">Enabled Markets</label>
                    <div id="market-selection" style="max-height: 300px; overflow-y: auto; padding: 15px; background: var(--black-tertiary); border-radius: 10px;">
                        <!-- Market checkboxes loaded here -->
                    </div>
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
        let allMarkets = {};
        let currentSettings = {};
        
        document.addEventListener('DOMContentLoaded', function() {
            checkSession();
        });
        
        async function checkSession() {
            try {
                const response = await fetch('/api/check-session', {
                    credentials: 'include'  // CRITICAL FOR SESSION
                });
                const data = await response.json();
                if (data.success && data.username) {
                    currentUser = data.username;
                    document.getElementById('username-display').textContent = data.username;
                    document.getElementById('auth-section').classList.add('hidden');
                    document.getElementById('main-app').classList.remove('hidden');
                    loadMarkets();
                    startStatusUpdates();
                    loadSettings();
                }
            } catch (error) {
                console.log('No active session:', error);
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
                    credentials: 'include',  // CRITICAL FOR SESSION
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
                    loadSettings();
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
                const response = await fetch('/api/logout', {
                    method: 'POST',
                    credentials: 'include'
                });
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
                    credentials: 'include',
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
        
        async function startTrading() {
            showAlert('dashboard-alert', 'Starting trading...', 'warning');
            try {
                const response = await fetch('/api/start-trading', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    credentials: 'include',
                    body: JSON.stringify({})
                });
                const data = await response.json();
                showAlert('dashboard-alert', data.message, data.success ? 'success' : 'error');
            } catch (error) {
                showAlert('dashboard-alert', 'Network error', 'error');
            }
        }
        
        async function stopTrading() {
            showAlert('dashboard-alert', 'Stopping trading...', 'warning');
            try {
                const response = await fetch('/api/stop-trading', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    credentials: 'include',
                    body: JSON.stringify({})
                });
                const data = await response.json();
                showAlert('dashboard-alert', data.message, data.success ? 'success' : 'error');
            } catch (error) {
                showAlert('dashboard-alert', 'Network error', 'error');
            }
        }
        
        async function loadMarkets() {
            try {
                const response = await fetch('/api/status', {
                    credentials: 'include'
                });
                const data = await response.json();
                if (data.success && data.markets) {
                    allMarkets = data.markets;
                    createMarketCards();
                    createMarketSelection();
                    updateTradeSymbols();
                }
            } catch (error) {
                console.error('Error loading markets:', error);
            }
        }
        
        function createMarketCards() {
            const marketsGrid = document.getElementById('markets-grid');
            marketsGrid.innerHTML = '';
            for (const [symbol, market] of Object.entries(allMarkets)) {
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
            }
        }
        
        function createMarketSelection() {
            const marketSelection = document.getElementById('market-selection');
            marketSelection.innerHTML = '';
            
            // Group markets by category
            const categories = {};
            for (const [symbol, market] of Object.entries(allMarkets)) {
                const category = market.category || 'Other';
                if (!categories[category]) categories[category] = [];
                categories[category].push({symbol, name: market.name});
            }
            
            // Create checkboxes for each category
            for (const [category, markets] of Object.entries(categories)) {
                const categoryDiv = document.createElement('div');
                categoryDiv.className = 'market-category';
                
                const categoryTitle = document.createElement('div');
                categoryTitle.className = 'market-category-title';
                categoryTitle.textContent = `${category} (${markets.length})`;
                categoryDiv.appendChild(categoryTitle);
                
                markets.forEach(market => {
                    const checkboxDiv = document.createElement('div');
                    checkboxDiv.className = 'market-checkbox';
                    
                    const checkbox = document.createElement('input');
                    checkbox.type = 'checkbox';
                    checkbox.id = `market-${market.symbol}`;
                    checkbox.value = market.symbol;
                    checkbox.checked = currentSettings.enabled_markets?.includes(market.symbol) || false;
                    
                    const label = document.createElement('label');
                    label.htmlFor = `market-${market.symbol}`;
                    label.textContent = market.name;
                    
                    checkboxDiv.appendChild(checkbox);
                    checkboxDiv.appendChild(label);
                    categoryDiv.appendChild(checkboxDiv);
                });
                
                marketSelection.appendChild(categoryDiv);
            }
        }
        
        function updateTradeSymbols() {
            const tradeSymbol = document.getElementById('trade-symbol');
            tradeSymbol.innerHTML = '<option value="">Select market</option>';
            for (const [symbol, market] of Object.entries(allMarkets)) {
                const option = document.createElement('option');
                option.value = symbol;
                option.textContent = `${market.name} (${symbol})`;
                tradeSymbol.appendChild(option);
            }
        }
        
        async function analyzeMarket(symbol) {
            showAlert('markets-alert', `Analyzing ${symbol}...`, 'warning');
            try {
                const response = await fetch('/api/analyze-market', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    credentials: 'include',
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
            for (const symbol in allMarkets) {
                await analyzeMarket(symbol);
                await new Promise(resolve => setTimeout(resolve, 300));
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
                    credentials: 'include',
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
                    credentials: 'include',
                    body: JSON.stringify({symbol, direction, amount})
                });
                const data = await response.json();
                showAlert('trading-alert', data.message, data.success ? 'success' : 'error');
                if (data.success) refreshTrades();
            } catch (error) {
                showAlert('trading-alert', 'Network error', 'error');
            }
        }
        
        async function loadSettings() {
            try {
                const response = await fetch('/api/status', {
                    credentials: 'include'
                });
                const data = await response.json();
                if (data.success && data.status && data.status.settings) {
                    currentSettings = data.status.settings;
                    document.getElementById('setting-trade-amount').value = currentSettings.trade_amount || 1.0;
                    document.getElementById('setting-min-confidence').value = currentSettings.min_confidence || 65;
                    document.getElementById('setting-dry-run').checked = currentSettings.dry_run !== false;
                    createMarketSelection();
                }
            } catch (error) {
                console.error('Error loading settings:', error);
            }
        }
        
        async function saveSettings() {
            const tradeAmount = parseFloat(document.getElementById('setting-trade-amount').value);
            const minConfidence = parseInt(document.getElementById('setting-min-confidence').value);
            const dryRun = document.getElementById('setting-dry-run').checked;
            
            // Get selected markets
            const enabledMarkets = [];
            document.querySelectorAll('#market-selection input[type="checkbox"]:checked').forEach(checkbox => {
                enabledMarkets.push(checkbox.value);
            });
            
            if (tradeAmount < 0.35) {
                showAlert('settings-alert', 'Minimum trade amount is $0.35', 'error');
                return;
            }
            if (minConfidence < 50 || minConfidence > 90) {
                showAlert('settings-alert', 'Confidence must be between 50-90%', 'error');
                return;
            }
            if (enabledMarkets.length === 0) {
                showAlert('settings-alert', 'Please select at least one market', 'error');
                return;
            }
            
            const settings = {
                trade_amount: tradeAmount,
                min_confidence: minConfidence,
                dry_run: dryRun,
                enabled_markets: enabledMarkets
            };
            
            showAlert('settings-alert', 'Saving settings...', 'warning');
            try {
                const response = await fetch('/api/update-settings', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    credentials: 'include',
                    body: JSON.stringify({settings})
                });
                const data = await response.json();
                showAlert('settings-alert', data.message, data.success ? 'success' : 'error');
                if (data.success) loadSettings();
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
                const response = await fetch('/api/status', {
                    credentials: 'include'
                });
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
                    if (status.running) {
                        document.getElementById('trading-status').textContent = '🟢 Trading';
                        document.getElementById('trading-status').style.color = 'var(--success)';
                    } else {
                        document.getElementById('trading-status').textContent = '❌ Not Trading';
                        document.getElementById('trading-status').style.color = 'var(--danger)';
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
                            ${trade.confidence ? `<div style="font-size: 11px; color: var(--info);">Confidence: ${trade.confidence}%</div>` : ''}
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
    
    print("\n" + "="*80)
    print("🎯 KARANKA V8 - DERIV SMART SMC TRADING BOT (PRODUCTION READY)")
    print("="*80)
    print(f"🚀 Server starting on http://{host}:{port}")
    print("✅ FIXED: Session management for Render.com deployment")
    print("✅ FIXED: CORS configuration for all tabs")
    print("✅ FIXED: WebSocket connections with proper credentials")
    print("✅ READY: Real trade execution when Dry Run is OFF")
    print("="*80)
    
    app.run(host=host, port=port, debug=False, threaded=True)
