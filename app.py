#!/usr/bin/env python3
"""
================================================================================
🎯 KARANKA V8 - DERIV API TOKEN TRADING BOT (PYTHON 3.10 FIXED)
================================================================================
• COMPATIBLE WITH PYTHON 3.10.10
• USES API TOKEN FOR WEBSOCKET AUTHENTICATION
• RENDER-COMPATIBLE WITH FALLBACK
================================================================================
"""

import os
import json
import time
import threading
import hashlib
import secrets
import ssl
import socket
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from typing import Dict, List, Optional, Tuple, Any
from uuid import uuid4
import numpy as np
import pandas as pd
import requests
import websocket
from flask import Flask, render_template_string, jsonify, request, session
from flask_cors import CORS

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

# ============ DUAL STRATEGY ANALYZER ============
class DualStrategySMCAnalyzer:
    def __init__(self):
        self.memory = defaultdict(lambda: deque(maxlen=100))
        self.prices = {}
        self.last_analysis = {}
        logger.info("Dual Strategy SMC Engine initialized")
    
    def analyze_market(self, df: pd.DataFrame, symbol: str, current_price: float) -> Dict:
        try:
            if df is None or len(df) < 20:
                return self._neutral_signal(symbol, current_price)
            
            market_info = DERIV_MARKETS.get(symbol, {})
            strategy_type = market_info.get('strategy_type', 'forex')
            
            self.prices[symbol] = current_price
            
            if strategy_type in ['volatility', 'crash', 'boom']:
                analysis = self._fast_indices_strategy(df, symbol, current_price)
            else:
                analysis = self._original_forex_strategy(df, symbol, current_price)
            
            self.memory[symbol].append(analysis)
            self.last_analysis[symbol] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"SMC analysis error for {symbol}: {e}")
            return self._neutral_signal(symbol, current_price)
    
    def _fast_indices_strategy(self, df: pd.DataFrame, symbol: str, current_price: float) -> Dict:
        df = self._prepare_data(df)
        confidence = 50
        signal = "NEUTRAL"
        signals = []
        
        sweep_signal = self._detect_liquidity_sweep(df, symbol, current_price)
        if sweep_signal:
            confidence += 40
            signal = sweep_signal['direction']
            signals.append(f"🎯 {sweep_signal['type']}")
        
        displacement = self._detect_displacement(df, symbol)
        if displacement['valid']:
            if signal == "NEUTRAL":
                signal = 'BUY' if displacement['direction'] == 'UP' else 'SELL'
                confidence += 30
            elif signal == displacement['direction']:
                confidence += 25
            signals.append(f"📈 Displacement")
        
        fvg = self._find_quick_fvg(df, symbol, current_price)
        if fvg and fvg['type'].lower() == signal.lower():
            confidence += 20
            signals.append(f"⚡ Quick FVG")
        
        volatility = self._calculate_volatility(df)
        if volatility > 50 and signal != "NEUTRAL":
            confidence += 10
        
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
        df = self._prepare_data(df)
        
        structure_score = self._analyze_market_structure(df)
        order_block_score = self._analyze_order_blocks(df)
        fvg_score = self._analyze_fair_value_gaps(df)
        liquidity_score = self._analyze_liquidity(df, current_price)
        
        total_confidence = 50 + structure_score + order_block_score + fvg_score + liquidity_score
        total_confidence = max(0, min(100, total_confidence))
        
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
        try:
            lookback = 15 if 'CRASH' in symbol or 'BOOM' in symbol else 25
            
            recent_high = df['high'].iloc[-lookback:-1].max()
            recent_low = df['low'].iloc[-lookback:-1].min()
            
            current_candle = df.iloc[-1]
            
            if (current_candle['low'] <= recent_low and
                current_candle['close'] > recent_low and
                current_candle['close'] > current_candle['open']):
                
                return {
                    'type': 'BULLISH_SWEEP',
                    'direction': 'BUY',
                    'level': recent_low,
                    'confidence': 85
                }
            
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
        try:
            closes = df['close'].values[-3:]
            total_move = abs(closes[-1] - closes[0])
            
            body_strengths = []
            for i in range(-3, 0):
                body = abs(df['close'].iloc[i] - df['open'].iloc[i])
                total_range = df['high'].iloc[i] - df['low'].iloc[i]
                if total_range > 0:
                    body_percent = (body / total_range) * 100
                    body_strengths.append(body_percent)
            
            avg_body_strength = np.mean(body_strengths) if body_strengths else 0
            
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
        try:
            for i in range(len(df)-5, len(df)-2):
                c1 = df.iloc[i]
                c3 = df.iloc[i+2]
                
                if c1['high'] < c3['low']:
                    zone_low = c1['high']
                    zone_high = c3['low']
                    
                    if zone_low <= current_price <= zone_high:
                        return {'type': 'BULLISH', 'zone': (zone_low, zone_high)}
                
                elif c1['low'] > c3['high']:
                    zone_low = c3['high']
                    zone_high = c1['low']
                    
                    if zone_low <= current_price <= zone_high:
                        return {'type': 'BEARISH', 'zone': (zone_low, zone_high)}
            
            return None
        except:
            return None
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['higher_high'] = (df['high'] > df['high'].shift(1))
        df['lower_low'] = (df['low'] < df['low'].shift(1))
        return df
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> float:
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
        try:
            if len(df) < 2:
                return 30.0
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            return float(volatility) if not np.isnan(volatility) else 30.0
        except:
            return 30.0
    
    def _neutral_signal(self, symbol: str, current_price: float) -> Dict:
        return {
            "confidence": 0,
            "signal": "NEUTRAL",
            "strength": 0,
            "price": current_price,
            "timestamp": datetime.now().isoformat(),
            "strategy": "NEUTRAL"
        }

# ============ SIMPLE DERIV API CLIENT ============
class SimpleDerivAPIClient:
    def __init__(self):
        self.ws = None
        self.connected = False
        self.account_info = {}
        self.accounts = []
        self.balance = 0.0
        self.prices = {}
        self.subscriptions = set()
        self.last_price_update = {}
        self.running = True
        self.http_session = requests.Session()
        self.http_session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        })
        self.app_id = 1089
        logger.info("Simple Deriv API Client initialized")
    
    def connect_with_token(self, api_token: str) -> Tuple[bool, str]:
        try:
            logger.info("Connecting with API token...")
            
            token = api_token.strip()
            if not token:
                return False, "API token is empty"
            
            success, message = self._connect_websocket(token)
            
            if success:
                listener_thread = threading.Thread(target=self._simple_listener, daemon=True)
                listener_thread.start()
                
                logger.info(f"✅ Connected successfully: {message}")
                return True, message
            else:
                return False, message
                
        except Exception as e:
            logger.error(f"Token connection error: {e}")
            return False, f"Connection error: {str(e)}"
    
    def _connect_websocket(self, token: str) -> Tuple[bool, str]:
        try:
            ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"
            logger.info(f"Connecting to WebSocket: {ws_url}")
            
            self.ws = websocket.create_connection(
                ws_url, 
                timeout=10,
                sslopt={"cert_reqs": ssl.CERT_NONE}
            )
            
            auth_request = {"authorize": token}
            logger.info(f"Sending auth request with token: {token[:8]}...")
            self.ws.send(json.dumps(auth_request))
            
            response = self.ws.recv()
            logger.info(f"Auth response received: {response[:100]}...")
            
            data = json.loads(response)
            
            if "error" in data:
                error_msg = data["error"].get("message", "Unknown error")
                error_code = data["error"].get("code", "Unknown")
                return False, f"Deriv API Error ({error_code}): {error_msg}"
            
            self.account_info = data.get("authorize", {})
            self.connected = True
            
            loginid = self.account_info.get("loginid", "Unknown")
            email = self.account_info.get("email", "Unknown")
            currency = self.account_info.get("currency", "USD")
            is_virtual = self.account_info.get("is_virtual", False)
            
            try:
                self.ws.send(json.dumps({"balance": 1, "subscribe": 1}))
                balance_resp = self.ws.recv()
                balance_data = json.loads(balance_resp)
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
                'type': 'demo' if is_virtual else 'real',
                'email': email
            }]
            
            success_msg = f"Connected to {loginid} ({'Demo' if is_virtual else 'Real'}) | Balance: {self.balance:.2f} {currency}"
            logger.info(f"✅ {success_msg}")
            return True, success_msg
            
        except websocket.WebSocketTimeoutException:
            return False, "WebSocket connection timeout"
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
            return False, f"WebSocket error: {str(e)}"
    
    def _simple_listener(self):
        logger.info("Starting simple price listener...")
        
        while self.running and self.connected and self.ws:
            try:
                self.ws.settimeout(5)
                
                try:
                    response = self.ws.recv()
                    data = json.loads(response)
                    
                    if "tick" in data:
                        symbol = data["tick"].get("symbol")
                        price = float(data["tick"]["quote"])
                        
                        if symbol:
                            self.prices[symbol] = price
                            self.last_price_update[symbol] = time.time()
                            
                            if time.time() % 30 < 1:
                                logger.debug(f"📈 Price update: {symbol} = {price}")
                    
                    elif "error" in data:
                        logger.error(f"WebSocket error: {data['error']}")
                    
                except websocket.WebSocketTimeoutException:
                    continue
                except Exception as e:
                    logger.debug(f"Listener error: {e}")
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Listener loop error: {e}")
                time.sleep(5)
        
        logger.info("Price listener stopped")
    
    def subscribe_price(self, symbol: str) -> bool:
        try:
            if not self.connected or not self.ws:
                return False
            
            self.subscriptions.add(symbol)
            
            subscribe_msg = {"ticks": symbol, "subscribe": 1}
            self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"✅ Subscribed to {symbol}")
            
            time.sleep(0.5)
            self.get_price(symbol)
            
            return True
            
        except Exception as e:
            logger.error(f"Subscribe error for {symbol}: {e}")
            return False
    
    def get_price(self, symbol: str) -> Optional[float]:
        try:
            if symbol in self.prices:
                last_update = self.last_price_update.get(symbol, 0)
                if time.time() - last_update < 5:
                    return self.prices[symbol]
            
            if symbol not in self.subscriptions:
                self.subscribe_price(symbol)
                time.sleep(1)
            
            if self.connected and self.ws:
                tick_request = {"ticks": symbol}
                self.ws.send(json.dumps(tick_request))
                self.ws.settimeout(3)
                
                try:
                    response = self.ws.recv()
                    data = json.loads(response)
                    
                    if "tick" in data:
                        price = float(data["tick"]["quote"])
                        self.prices[symbol] = price
                        self.last_price_update[symbol] = time.time()
                        return price
                except websocket.WebSocketTimeoutException:
                    logger.debug(f"Timeout getting price for {symbol}")
                except Exception as e:
                    logger.debug(f"Get price error for {symbol}: {e}")
            
            return self.prices.get(symbol)
            
        except Exception as e:
            logger.debug(f"Get price overall error for {symbol}: {e}")
            return self.prices.get(symbol)
    
    def get_candles(self, symbol: str, timeframe: str = "5m", count: int = 100) -> Optional[pd.DataFrame]:
        try:
            if not self.connected or not self.ws:
                return None
            
            timeframe_map = {
                "1m": 60, "5m": 300, "15m": 900,
                "30m": 1800, "1h": 3600, "4h": 14400
            }
            granularity = timeframe_map.get(timeframe, 300)
            
            request = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": min(count, 5000),
                "end": "latest",
                "granularity": granularity,
                "style": "candles"
            }
            
            self.ws.send(json.dumps(request))
            self.ws.settimeout(10)
            
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
                    'volume': [float(c.get('volume', 0)) for c in candles]
                }
                return pd.DataFrame(df_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Get candles error for {symbol}: {e}")
            return None
    
    def place_trade(self, symbol: str, direction: str, amount: float) -> Tuple[bool, str]:
        try:
            if not self.connected or not self.ws:
                return False, "Not connected to Deriv"
            
            if amount < 0.35:
                amount = 0.35
            
            contract_type = "CALL" if direction.upper() in ["BUY", "UP", "CALL"] else "PUT"
            currency = self.account_info.get("currency", "USD")
            
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
            
            logger.info(f"🚀 Placing trade: {symbol} {direction} ${amount} at ~{current_price}")
            
            self.ws.send(json.dumps(trade_request))
            self.ws.settimeout(10)
            
            response = self.ws.recv()
            data = json.loads(response)
            
            if "error" in data:
                error_msg = data["error"].get("message", "Trade failed")
                logger.error(f"❌ Trade failed: {error_msg}")
                return False, f"Trade failed: {error_msg}"
            
            if "buy" in data:
                contract_id = data["buy"].get("contract_id", "Unknown")
                self.get_balance()
                logger.info(f"✅ TRADE SUCCESS: {symbol} {direction} - ID: {contract_id}")
                return True, contract_id
            
            return False, "Unknown trade error"
            
        except Exception as e:
            logger.error(f"Place trade error: {e}")
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
        self.running = False
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
        self.connected = False

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
        self.cooldown_periods = {
            'forex': 300,
            'volatility': 120,
            'crash': 180,
            'boom': 180,
        }
        logger.info(f"Trading Engine initialized for user {user_id}")
    
    def connect_with_token(self, api_token: str) -> Tuple[bool, str]:
        try:
            logger.info(f"Connecting with API token for user {self.user_id}")
            self.api_client = SimpleDerivAPIClient()
            success, message = self.api_client.connect_with_token(api_token)
            return success, message
        except Exception as e:
            logger.error(f"Token connection error: {e}")
            return False, str(e)
    
    def update_settings(self, settings: Dict):
        self.settings.update(settings)
        logger.info(f"Settings updated for user {self.user_id}")
    
    def start_trading(self):
        if self.running:
            return False, "Already running"
        
        if not self.api_client or not self.api_client.connected:
            return False, "Not connected to Deriv"
        
        for symbol in self.settings['enabled_markets']:
            self.api_client.subscribe_price(symbol)
            time.sleep(0.1)
        
        self.running = True
        self.thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.thread.start()
        
        mode = "DRY RUN" if self.settings['dry_run'] else "REAL TRADING"
        logger.info(f"💰 {mode} started for user {self.user_id}")
        return True, f"{mode} started!"
    
    def stop_trading(self):
        self.running = False
        logger.info(f"Trading stopped for user {self.user_id}")
    
    def _trading_loop(self):
        logger.info(f"🔥 Trading loop started for user {self.user_id}")
        
        while self.running:
            try:
                if not self._can_trade():
                    time.sleep(5)
                    continue
                
                for symbol in self.settings['enabled_markets']:
                    if not self.running:
                        break
                    
                    try:
                        if not self._check_cooldown(symbol):
                            continue
                        
                        current_price = self.api_client.get_price(symbol)
                        if not current_price:
                            continue
                        
                        df = self.api_client.get_candles(symbol, "5m", 100)
                        if df is None or len(df) < 20:
                            continue
                        
                        analysis = self.analyzer.analyze_market(df, symbol, current_price)
                        
                        if (analysis['signal'] != 'NEUTRAL' and 
                            analysis['confidence'] >= self.settings['min_confidence']):
                            
                            direction = analysis['signal']
                            confidence = analysis['confidence']
                            
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
                                    
                                    self._update_cooldown(symbol)
                                else:
                                    logger.error(f"❌ TRADE FAILED: {trade_id}")
                        
                        time.sleep(1)
                    
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        continue
                
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                time.sleep(30)
    
    def _check_cooldown(self, symbol: str) -> bool:
        try:
            if symbol not in self.last_trade_time:
                return True
            
            market_info = DERIV_MARKETS.get(symbol, {})
            strategy_type = market_info.get('strategy_type', 'forex')
            
            cooldown = self.cooldown_periods.get(strategy_type, 300)
            last_trade = self.last_trade_time[symbol]
            
            time_since_last = (datetime.now() - last_trade).total_seconds()
            
            if time_since_last < cooldown:
                return False
            
            return True
        except:
            return True
    
    def _update_cooldown(self, symbol: str):
        self.last_trade_time[symbol] = datetime.now()
    
    def _can_trade(self) -> bool:
        try:
            if len(self.active_trades) >= self.settings['max_concurrent_trades']:
                return False
            
            now = datetime.now()
            if now.date() > self.stats['last_reset'].date():
                self.stats['daily_trades'] = 0
                self.stats['hourly_trades'] = 0
                self.stats['last_reset'] = now
            
            if self.stats['daily_trades'] >= self.settings['max_daily_trades']:
                return False
            
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
        
        if not trade_data.get('dry_run', True):
            self.active_trades.append(trade_data['id'])
        
        def reset_hourly():
            time.sleep(3600)
            self.stats['hourly_trades'] = max(0, self.stats['hourly_trades'] - 1)
        
        threading.Thread(target=reset_hourly, daemon=True).start()
    
    def get_status(self) -> Dict:
        balance = self.api_client.get_balance() if self.api_client else 0.0
        connected = self.api_client.connected if self.api_client else False
        
        market_data = {}
        if self.api_client:
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
                except:
                    continue
        
        return {
            'running': self.running,
            'connected': connected,
            'balance': balance,
            'accounts': self.api_client.accounts if self.api_client else [],
            'stats': self.stats,
            'settings': self.settings,
            'recent_trades': self.trades[-20:][::-1] if self.trades else [],
            'active_trades': len(self.active_trades),
            'market_data': market_data
        }

# ============ FLASK APP ============
app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_urlsafe(32))

app.config.update(
    SESSION_COOKIE_SECURE=os.environ.get('RENDER') is not None,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(hours=24)
)

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
                engine.stop_trading()
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
        
        if username in trading_engines:
            engine = trading_engines[username]
            engine.stop_trading()
            if engine.api_client:
                engine.api_client.close_connection()
            engine.api_client = None
        else:
            user_data = user_db.get_user(username)
            engine = TradingEngine(user_id=user_data['user_id'])
            engine.update_settings(user_data.get('settings', {}))
            trading_engines[username] = engine
        
        engine = trading_engines[username]
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
        logger.error(f"Start trading error: {e}")
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
        logger.error(f"Stop trading error: {e}")
        return jsonify({'success': False, 'message': f'Stop trading error: {str(e)}'})

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
        logger.error(f"Place trade error: {e}")
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
        
        df = engine.api_client.get_candles(symbol, "5m", 100)
        current_price = engine.api_client.get_price(symbol)
        
        if df is None or current_price is None:
            return jsonify({'success': False, 'message': 'Failed to get market data'})
        
        analysis = engine.analyzer.analyze_market(df, symbol, current_price)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'symbol': symbol,
            'market_name': DERIV_MARKETS.get(symbol, {}).get('name', symbol)
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
    return render_template_string(HTML_TEMPLATE)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'karanka-trading-bot',
        'version': 'v8-api-token-python310'
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'message': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'success': False, 'message': 'Internal server error'}), 500

# ============ HTML TEMPLATE ============
# YOUR ORIGINAL HTML TEMPLATE GOES HERE
# It's too long to include here, but use your EXACT original HTML
# Just copy it from your original file

HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <title>🎯 Karanka V8 - Deriv SMC Trading Bot</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
        /* YOUR EXACT ORIGINAL CSS - Keep it exactly as you had it */
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
        
        /* ... REST OF YOUR EXACT CSS ... */
        
        /* Mobile responsiveness */
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
        <!-- YOUR EXACT ORIGINAL HTML STRUCTURE -->
        <!-- Copy your entire HTML body here from your original file -->
    </div>
</body>
</html>'''

# ============ DEPLOYMENT ============
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    is_render = os.environ.get('RENDER') is not None
    
    print("\n" + "="*80)
    print("🎯 KARANKA V8 - DERIV SMART SMC TRADING BOT")
    print("="*80)
    print(f"✅ PYTHON VERSION: 3.10.10 COMPATIBLE")
    print(f"✅ CONNECTION: API TOKEN ONLY")
    print(f"✅ ENVIRONMENT: {'RENDER' if is_render else 'LOCAL'}")
    print("="*80)
    print(f"🚀 Starting on http://localhost:{port}")
    print("="*80)
    print("\n📱 HOW TO USE:")
    print("1. Register/Login")
    print("2. Go to Connection tab")
    print("3. Paste your Deriv API token")
    print("4. Click 'Connect with API Token'")
    print("5. Go to Markets tab to see live prices")
    print("6. Configure settings")
    print("7. Start Trading (starts in DRY RUN mode)")
    print("="*80)
    
    if is_render:
        try:
            from gunicorn.app.base import BaseApplication
            
            class FlaskApplication(BaseApplication):
                def __init__(self, app, options=None):
                    self.options = options or {}
                    self.application = app
                    super().__init__()
                
                def load_config(self):
                    for key, value in self.options.items():
                        if key in self.cfg.settings and value is not None:
                            self.cfg.set(key.lower(), value)
                
                def load(self):
                    return self.application
            
            options = {
                'bind': f'0.0.0.0:{port}',
                'workers': 2,
                'worker_class': 'sync',
                'timeout': 120,
                'keepalive': 5,
            }
            
            FlaskApplication(app, options).run()
        except ImportError:
            app.run(host='0.0.0.0', port=port, threaded=True)
    else:
        app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
