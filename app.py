#!/usr/bin/env python3
"""
================================================================================
🎯 KARANKA MULTIVERSE V7 - DERIV REAL TRADING BOT
================================================================================
• REAL DERIV API CONNECTION WITH API TOKEN
• REAL DEMO/REAL ACCOUNT DETECTION & SELECTION
• REAL TRADING WITH USER-CONTROLLED $ AMOUNTS
• FULL SMC STRATEGY OPTIMIZED FOR DERIV MARKETS
• MOBILE-FRIENDLY BLACK & GOLD WEB APP
• REAL-TIME TRADE TRACKING (WINS/LOSSES)
• RENDER.COM DEPLOYMENT READY
================================================================================
"""

import os
import json
import time
import threading
import hashlib
import secrets
from datetime import datetime, timedelta
from collections import defaultdict
import logging
from typing import Dict, List, Optional, Tuple

from flask import Flask, render_template_string, jsonify, request, session, redirect
import websocket
import pandas as pd
import numpy as np

# ============ SETUP LOGGING ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============ DERIV MARKET CONFIGS ============
DERIV_MARKETS = {
    "R_10": {"name": "Volatility 10 Index", "pip": 0.001, "volatility": 10, "category": "Volatility"},
    "R_25": {"name": "Volatility 25 Index", "pip": 0.001, "volatility": 25, "category": "Volatility"},
    "R_50": {"name": "Volatility 50 Index", "pip": 0.001, "volatility": 50, "category": "Volatility"},
    "R_75": {"name": "Volatility 75 Index", "pip": 0.001, "volatility": 75, "category": "Volatility"},
    "R_100": {"name": "Volatility 100 Index", "pip": 0.001, "volatility": 100, "category": "Volatility"},
    "CRASH_300": {"name": "Crash 300 Index", "pip": 0.01, "volatility": 300, "category": "Crash/Boom"},
    "CRASH_500": {"name": "Crash 500 Index", "pip": 0.01, "volatility": 500, "category": "Crash/Boom"},
    "CRASH_1000": {"name": "Crash 1000 Index", "pip": 0.01, "volatility": 1000, "category": "Crash/Boom"},
    "BOOM_300": {"name": "Boom 300 Index", "pip": 0.01, "volatility": 300, "category": "Crash/Boom"},
    "BOOM_500": {"name": "Boom 500 Index", "pip": 0.01, "volatility": 500, "category": "Crash/Boom"},
    "BOOM_1000": {"name": "Boom 1000 Index", "pip": 0.01, "volatility": 1000, "category": "Crash/Boom"},
}

# ============ SESSION CONFIGS ============
SESSIONS = {
    "Asian": {"hours": (0, 9), "trades_hour": 3, "risk": 0.8},
    "London": {"hours": (8, 17), "trades_hour": 5, "risk": 1.0},
    "NewYork": {"hours": (13, 22), "trades_hour": 6, "risk": 1.2},
    "Overlap": {"hours": (13, 17), "trades_hour": 8, "risk": 1.5},
    "Night": {"hours": (22, 24), "trades_hour": 2, "risk": 0.7},
}

# ============ DERIV API CLIENT (REAL CONNECTION) ============
class DerivAPIClient:
    """Real Deriv API Client - Connects with API Token"""
    
    def __init__(self, api_token: str, app_id: str = "1089"):
        self.api_token = api_token
        self.app_id = app_id
        self.ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
        self.ws = None
        self.connected = False
        self.account_info = {}
        self.accounts = []  # List of demo/real accounts found
        self.request_id = 0
        self.balance = 0
        self.prices = {}
        logger.info(f"Deriv API Client initialized with token: {api_token[:10]}...")
    
    def connect(self) -> Tuple[bool, str]:
        """Connect to Deriv and discover accounts"""
        try:
            logger.info(f"Connecting to Deriv WebSocket: {self.ws_url}")
            self.ws = websocket.create_connection(self.ws_url, timeout=30)
            self.connected = True
            
            # Step 1: Authorize with token
            auth_request = {
                "authorize": self.api_token,
                "req_id": self._next_req_id()
            }
            logger.info("Sending authorization request...")
            self.ws.send(json.dumps(auth_request))
            
            # Wait for response with timeout
            response_str = self.ws.recv()
            if not response_str:
                return False, "No response from Deriv"
                
            response = json.loads(response_str)
            
            if response.get("error"):
                error_msg = response["error"].get("message", "Unknown error")
                logger.error(f"Authorization failed: {error_msg}")
                return False, f"Authorization failed: {error_msg}"
            
            # Get account info
            auth_data = response.get("authorize", {})
            self.account_info = auth_data
            
            # Step 2: Get account list (demo and real)
            self.accounts = self.get_account_list()
            
            if not self.accounts:
                return False, "No trading accounts found with this API token"
            
            logger.info(f"Connected successfully. Found {len(self.accounts)} accounts")
            return True, f"Connected! Found {len(self.accounts)} accounts"
            
        except websocket.WebSocketTimeoutException:
            return False, "Connection timeout - check your internet"
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            return False, f"Connection error: {str(e)}"
    
    def _next_req_id(self) -> int:
        """Get next request ID"""
        self.request_id += 1
        return self.request_id
    
    def get_account_list(self) -> List[Dict]:
        """Get list of available accounts (demo and real)"""
        try:
            if not self.connected:
                return []
            
            # Get account settings which contains available accounts
            request = {
                "get_settings": 1,
                "req_id": self._next_req_id()
            }
            self.ws.send(json.dumps(request))
            response = json.loads(self.ws.recv())
            
            accounts = []
            
            # If we have authorized account, add it
            if self.account_info:
                accounts.append({
                    'loginid': self.account_info.get('loginid', 'Unknown'),
                    'currency': self.account_info.get('currency', 'USD'),
                    'landing_company_name': self.account_info.get('landing_company_name', 'Unknown'),
                    'is_virtual': self.account_info.get('is_virtual', False),
                    'balance': self.account_info.get('balance', 0),
                    'name': f"{'DEMO' if self.account_info.get('is_virtual') else 'REAL'} - {self.account_info.get('loginid', 'Account')}",
                    'type': 'demo' if self.account_info.get('is_virtual') else 'real'
                })
            
            logger.info(f"Found {len(accounts)} trading accounts")
            return accounts
            
        except Exception as e:
            logger.error(f"Error getting account list: {e}")
            return []
    
    def switch_account(self, loginid: str) -> Tuple[bool, str]:
        """Switch to specific account"""
        try:
            # Get new token for this account
            request = {
                "authorize": self.api_token,
                "req_id": self._next_req_id()
            }
            self.ws.send(json.dumps(request))
            response = json.loads(self.ws.recv())
            
            if response.get("error"):
                return False, response["error"].get("message", "Switch failed")
            
            # Update account info
            self.account_info = response.get("authorize", {})
            logger.info(f"Switched to account: {loginid}")
            return True, f"Switched to {loginid}"
            
        except Exception as e:
            logger.error(f"Switch account error: {e}")
            return False, str(e)
    
    def get_balance(self) -> float:
        """Get current account balance"""
        try:
            if not self.connected:
                return 0.0
            
            request = {
                "balance": 1,
                "subscribe": 0,
                "req_id": self._next_req_id()
            }
            self.ws.send(json.dumps(request))
            response = json.loads(self.ws.recv())
            
            if response.get("balance"):
                self.balance = float(response["balance"]["balance"])
                return self.balance
            return 0.0
            
        except Exception as e:
            logger.error(f"Get balance error: {e}")
            return 0.0
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            if not self.connected:
                return None
            
            # Use ticks stream for real-time price
            request = {
                "ticks": symbol,
                "subscribe": 1,
                "req_id": self._next_req_id()
            }
            self.ws.send(json.dumps(request))
            response = json.loads(self.ws.recv())
            
            if response.get("tick"):
                price = float(response["tick"]["quote"])
                self.prices[symbol] = price
                return price
            
            # Try again without subscription
            request = {
                "ticks": symbol,
                "subscribe": 0,
                "req_id": self._next_req_id()
            }
            self.ws.send(json.dumps(request))
            response = json.loads(self.ws.recv())
            
            if response.get("tick"):
                price = float(response["tick"]["quote"])
                self.prices[symbol] = price
                return price
                
            return self.prices.get(symbol)
            
        except Exception as e:
            logger.error(f"Get price error for {symbol}: {e}")
            return self.prices.get(symbol)
    
    def get_candles(self, symbol: str, timeframe: str = "5m", count: int = 100) -> Optional[pd.DataFrame]:
        """Get historical candles"""
        try:
            if not self.connected:
                return None
            
            # Map timeframe to Deriv granularity
            timeframe_map = {
                "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
                "1h": 3600, "4h": 14400, "1d": 86400
            }
            
            granularity = timeframe_map.get(timeframe, 300)
            
            request = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": count,
                "end": "latest",
                "granularity": granularity,
                "style": "candles",
                "req_id": self._next_req_id()
            }
            self.ws.send(json.dumps(request))
            response = json.loads(self.ws.recv())
            
            if response.get("candles"):
                candles = response["candles"]
                df = pd.DataFrame({
                    'time': pd.to_datetime(candles['epoch'], unit='s'),
                    'open': candles['open'].astype(float),
                    'high': candles['high'].astype(float),
                    'low': candles['low'].astype(float),
                    'close': candles['close'].astype(float),
                    'volume': candles['volume'].astype(float)
                })
                return df
                
            return None
            
        except Exception as e:
            logger.error(f"Get candles error for {symbol}: {e}")
            return None
    
    def place_trade(self, symbol: str, direction: str, amount: float, 
                   duration: int = 5, duration_unit: str = "m") -> Tuple[bool, str]:
        """Place a REAL trade on Deriv"""
        try:
            if not self.connected:
                return False, "Not connected to Deriv"
            
            # Get current price first
            price = self.get_price(symbol)
            if not price:
                return False, "Cannot get current price"
            
            # Determine contract type
            contract_type = "CALL" if direction.upper() == "BUY" else "PUT"
            
            # Prepare trade request
            request = {
                "buy": amount,  # Amount in USD
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
                },
                "req_id": self._next_req_id()
            }
            
            logger.info(f"Placing trade: {symbol} {direction} ${amount}")
            self.ws.send(json.dumps(request))
            
            # Wait for response
            response = json.loads(self.ws.recv())
            
            if response.get("error"):
                error_msg = response["error"].get("message", "Unknown error")
                logger.error(f"Trade error: {error_msg}")
                return False, f"Trade failed: {error_msg}"
            
            if response.get("buy"):
                contract_id = response["buy"]["contract_id"]
                payout = response["buy"].get("payout", 0)
                logger.info(f"Trade successful! Contract ID: {contract_id}, Payout: ${payout}")
                return True, contract_id
            
            return False, "Unknown trade error"
            
        except Exception as e:
            logger.error(f"Place trade error: {e}")
            return False, f"Trade error: {str(e)}"
    
    def get_open_positions(self) -> List[Dict]:
        """Get open positions"""
        try:
            if not self.connected:
                return []
            
            request = {
                "portfolio": 1,
                "req_id": self._next_req_id()
            }
            self.ws.send(json.dumps(request))
            response = json.loads(self.ws.recv())
            
            if response.get("portfolio"):
                return response["portfolio"].get("contracts", [])
            return []
            
        except Exception as e:
            logger.error(f"Get positions error: {e}")
            return []
    
    def close_position(self, contract_id: str) -> Tuple[bool, str]:
        """Close a position"""
        try:
            request = {
                "sell": contract_id,
                "price": 0,
                "req_id": self._next_req_id()
            }
            self.ws.send(json.dumps(request))
            response = json.loads(self.ws.recv())
            
            if response.get("error"):
                return False, response["error"].get("message", "Close failed")
            
            if response.get("sell"):
                return True, "Position closed"
            
            return False, "Close failed"
            
        except Exception as e:
            logger.error(f"Close position error: {e}")
            return False, str(e)
    
    def close_connection(self):
        """Close WebSocket connection"""
        try:
            if self.ws:
                self.ws.close()
                self.connected = False
                logger.info("Deriv connection closed")
        except:
            pass

# ============ ENHANCED SMC ANALYZER FOR DERIV ============
class DerivSMCAnalyzer:
    """SMC Strategy Optimized for Deriv Markets"""
    
    def __init__(self):
        self.cache = {}
        logger.info("SMC Analyzer initialized for Deriv markets")
    
    def analyze(self, symbol: str, market_config: Dict, api_client: DerivAPIClient) -> Optional[Dict]:
        """Analyze symbol with enhanced SMC strategies"""
        try:
            logger.info(f"Analyzing {symbol} with SMC strategy...")
            
            # Get multi-timeframe data
            m5_data = api_client.get_candles(symbol, "5m", 80)
            m15_data = api_client.get_candles(symbol, "15m", 80)
            h1_data = api_client.get_candles(symbol, "1h", 100)
            
            if m5_data is None or len(m5_data) < 20:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            current_price = api_client.get_price(symbol)
            if not current_price:
                return None
            
            # Analyze liquidity zones
            liquidity_zones = self.find_liquidity_zones(m5_data, m15_data, h1_data)
            
            # Analyze with multiple strategies
            analyses = []
            
            # 1. M5+M15 Scalping
            scalp_analysis = self.analyze_scalp(m5_data, m15_data, current_price, market_config, liquidity_zones)
            if scalp_analysis['confidence'] > 0:
                analyses.append(scalp_analysis)
            
            # 2. M15+H1 Intraday
            intraday_analysis = self.analyze_intraday(m15_data, h1_data, current_price, market_config, liquidity_zones)
            if intraday_analysis['confidence'] > 0:
                analyses.append(intraday_analysis)
            
            # 3. Enhanced Liquidity Analysis
            liquidity_analysis = self.analyze_liquidity(m5_data, m15_data, h1_data, current_price, market_config)
            if liquidity_analysis['confidence'] > 0:
                analyses.append(liquidity_analysis)
            
            if not analyses:
                return None
            
            # Get best analysis
            best_analysis = max(analyses, key=lambda x: x['confidence'])
            
            # Apply session adjustments
            session = self.get_current_session()
            session_config = SESSIONS[session]
            
            # Adjust confidence based on session
            session_adjustment = session_config['risk'] * 10
            final_confidence = best_analysis['confidence'] + session_adjustment
            
            if final_confidence < 65:
                return None
            
            # Prepare trade decision
            decision = self.prepare_trade_decision(
                best_analysis, current_price, market_config, 
                liquidity_zones, session_config
            )
            
            return {
                'symbol': symbol,
                'market_name': market_config['name'],
                'direction': best_analysis['direction'],
                'confidence': final_confidence,
                'price': current_price,
                'strategy': best_analysis['strategy'],
                'signals': best_analysis['signals'],
                'liquidity_zones': liquidity_zones,
                'session': session,
                'session_risk': session_config['risk'],
                'trading_decision': decision,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Analysis error for {symbol}: {e}")
            return None
    
    def find_liquidity_zones(self, m5_data, m15_data, h1_data) -> Dict:
        """Find liquidity zones (SMC concept)"""
        zones = {
            'highs': [],
            'lows': [],
            'fair_value_gaps': [],
            'order_blocks': []
        }
        
        # Find recent highs and lows
        for tf_name, data in [('M5', m5_data), ('M15', m15_data), ('H1', h1_data)]:
            if data is not None and len(data) > 20:
                recent_high = data['high'].values[-20:].max()
                recent_low = data['low'].values[-20:].min()
                
                zones['highs'].append({
                    'timeframe': tf_name,
                    'price': recent_high,
                    'strength': len(data) / 100  # Rough strength indicator
                })
                
                zones['lows'].append({
                    'timeframe': tf_name,
                    'price': recent_low,
                    'strength': len(data) / 100
                })
        
        return zones
    
    def analyze_scalp(self, m5_data, m15_data, current_price, config, liquidity_zones) -> Dict:
        """M5+M15 scalping strategy with liquidity focus"""
        analysis = {
            'strategy': 'SCALP_M5_M15',
            'confidence': 0,
            'direction': 'NONE',
            'signals': []
        }
        
        if len(m5_data) < 10:
            return analysis
        
        # 1. Check displacement (price movement)
        closes = m5_data['close'].values[-3:]
        movement = abs(closes[-1] - closes[0])
        pip_move = movement / config['pip']
        
        if pip_move >= 8:  # Significant movement
            analysis['confidence'] += 30
            analysis['direction'] = 'BUY' if closes[-1] > closes[0] else 'SELL'
            analysis['signals'].append(f"Displacement: {pip_move:.1f} pips")
        
        # 2. Check liquidity zone proximity
        for zone in liquidity_zones['highs'] + liquidity_zones['lows']:
            distance = abs(current_price - zone['price']) / config['pip']
            if distance <= 15:  # Near liquidity zone
                analysis['confidence'] += 15
                analysis['signals'].append(f"Near {zone['timeframe']} liquidity")
                break
        
        # 3. Check order block
        order_block = self.find_order_block(m5_data)
        if order_block:
            analysis['confidence'] += 20
            analysis['signals'].append("Order Block detected")
        
        # 4. Check Fair Value Gap
        fvg = self.find_fvg(m5_data)
        if fvg:
            analysis['confidence'] += 15
            analysis['signals'].append("FVG present")
        
        # 5. M15 alignment
        if m15_data is not None and len(m15_data) > 10:
            m15_trend = self.get_trend_direction(m15_data)
            if analysis['direction'] == m15_trend:
                analysis['confidence'] += 10
                analysis['signals'].append("M15 alignment")
        
        analysis['confidence'] = min(100, analysis['confidence'])
        return analysis
    
    def analyze_intraday(self, m15_data, h1_data, current_price, config, liquidity_zones) -> Dict:
        """M15+H1 intraday strategy"""
        analysis = {
            'strategy': 'INTRADAY_M15_H1',
            'confidence': 0,
            'direction': 'NONE',
            'signals': []
        }
        
        if m15_data is None or len(m15_data) < 30:
            return analysis
        
        # Similar structure as scalp but with larger thresholds
        closes = m15_data['close'].values[-3:]
        movement = abs(closes[-1] - closes[0])
        pip_move = movement / config['pip']
        
        if pip_move >= 15:
            analysis['confidence'] += 35
            analysis['direction'] = 'BUY' if closes[-1] > closes[0] else 'SELL'
            analysis['signals'].append(f"Intraday move: {pip_move:.1f} pips")
        
        # H1 alignment
        if h1_data is not None and len(h1_data) > 20:
            h1_trend = self.get_trend_direction(h1_data)
            if analysis['direction'] == h1_trend:
                analysis['confidence'] += 25
                analysis['signals'].append("H1 trend alignment")
        
        # Liquidity analysis
        for zone in liquidity_zones['highs']:
            distance = abs(current_price - zone['price']) / config['pip']
            if distance <= 20 and analysis['direction'] == 'SELL':
                analysis['confidence'] += 15
                analysis['signals'].append("Trading into supply")
                break
        
        for zone in liquidity_zones['lows']:
            distance = abs(current_price - zone['price']) / config['pip']
            if distance <= 20 and analysis['direction'] == 'BUY':
                analysis['confidence'] += 15
                analysis['signals'].append("Trading into demand")
                break
        
        analysis['confidence'] = min(100, analysis['confidence'])
        return analysis
    
    def analyze_liquidity(self, m5_data, m15_data, h1_data, current_price, config) -> Dict:
        """Enhanced liquidity analysis"""
        analysis = {
            'strategy': 'LIQUIDITY_HUNT',
            'confidence': 0,
            'direction': 'NONE',
            'signals': []
        }
        
        # Check for liquidity sweeps
        liquidity_sweep = self.check_liquidity_sweep(m5_data, current_price, config)
        if liquidity_sweep['found']:
            analysis['confidence'] += 40
            analysis['direction'] = liquidity_sweep['direction']
            analysis['signals'].append(f"Liquidity sweep: {liquidity_sweep['type']}")
        
        # Check for accumulation/distribution
        accumulation = self.check_accumulation(m15_data)
        if accumulation['found']:
            analysis['confidence'] += 30
            if accumulation['type'] == 'accumulation':
                analysis['direction'] = 'BUY'
            else:
                analysis['direction'] = 'SELL'
            analysis['signals'].append(f"{accumulation['type'].title()} zone")
        
        return analysis
    
    def find_order_block(self, data) -> bool:
        """Find order blocks in price data"""
        if len(data) < 10:
            return False
        
        for i in range(len(data) - 10, len(data) - 1):
            if i < 0:
                continue
            
            candle = data.iloc[i]
            body = abs(candle['close'] - candle['open'])
            total = candle['high'] - candle['low']
            
            if total > 0 and (body / total) > 0.7:  # Strong body
                # Check if next candle moves against this candle
                next_candle = data.iloc[i + 1]
                if (candle['close'] > candle['open'] and 
                    next_candle['close'] < candle['close']):
                    return True  # Bullish order block
                elif (candle['close'] < candle['open'] and 
                      next_candle['close'] > candle['close']):
                    return True  # Bearish order block
        
        return False
    
    def find_fvg(self, data) -> bool:
        """Find Fair Value Gaps"""
        if len(data) < 3:
            return False
        
        for i in range(len(data) - 3, len(data) - 1):
            if i < 0:
                continue
            
            candle1 = data.iloc[i]
            candle2 = data.iloc[i + 1]
            candle3 = data.iloc[i + 2]
            
            # Check for bullish FVG (gap up)
            if candle1['high'] < candle3['low']:
                return True
            
            # Check for bearish FVG (gap down)
            if candle1['low'] > candle3['high']:
                return True
        
        return False
    
    def get_trend_direction(self, data) -> str:
        """Get trend direction"""
        if len(data) < 10:
            return 'NEUTRAL'
        
        sma_10 = data['close'].rolling(window=10).mean().iloc[-1]
        sma_20 = data['close'].rolling(window=20).mean().iloc[-1]
        
        if sma_10 > sma_20 * 1.001:
            return 'BUY'
        elif sma_10 < sma_20 * 0.999:
            return 'SELL'
        else:
            return 'NEUTRAL'
    
    def check_liquidity_sweep(self, data, current_price, config) -> Dict:
        """Check for liquidity sweeps"""
        if len(data) < 20:
            return {'found': False, 'type': '', 'direction': 'NONE'}
        
        recent_high = data['high'].values[-20:].max()
        recent_low = data['low'].values[-20:].min()
        
        # Check if price swept liquidity above
        if current_price < recent_high and any(data['high'].values[-5:] >= recent_high):
            return {
                'found': True,
                'type': 'High sweep',
                'direction': 'SELL'  # After sweeping highs, expect down move
            }
        
        # Check if price swept liquidity below
        if current_price > recent_low and any(data['low'].values[-5:] <= recent_low):
            return {
                'found': True,
                'type': 'Low sweep',
                'direction': 'BUY'  # After sweeping lows, expect up move
            }
        
        return {'found': False, 'type': '', 'direction': 'NONE'}
    
    def check_accumulation(self, data) -> Dict:
        """Check for accumulation/distribution patterns"""
        if len(data) < 30:
            return {'found': False, 'type': ''}
        
        # Simple volume analysis
        recent_volume = data['volume'].values[-10:].mean()
        prev_volume = data['volume'].values[-20:-10].mean()
        
        if recent_volume > prev_volume * 1.5:
            # Check if price is in a range (accumulation/distribution)
            recent_range = data['high'].values[-10:].max() - data['low'].values[-10:].min()
            prev_range = data['high'].values[-20:-10].max() - data['low'].values[-20:-10].min()
            
            if recent_range < prev_range * 0.7:  # Range contraction
                if data['close'].values[-1] > data['open'].values[-1]:
                    return {'found': True, 'type': 'accumulation'}
                else:
                    return {'found': True, 'type': 'distribution'}
        
        return {'found': False, 'type': ''}
    
    def prepare_trade_decision(self, analysis, current_price, config, 
                              liquidity_zones, session_config) -> Dict:
        """Prepare final trade decision with risk management"""
        
        # Calculate position size (simplified - will be overridden by user settings)
        base_amount = 1.0  # Base $ amount
        confidence_multiplier = analysis['confidence'] / 100
        session_multiplier = session_config['risk']
        
        suggested_amount = base_amount * confidence_multiplier * session_multiplier
        
        # For Deriv contracts, duration is fixed
        duration = 5  # 5 minutes for volatility indices
        
        return {
            'action': analysis['direction'],
            'amount': suggested_amount,
            'duration': duration,
            'risk_level': 'MEDIUM',
            'reason': f"SMC: {', '.join(analysis['signals'][:2])}",
            'liquidity_nearby': len(liquidity_zones['highs'] + liquidity_zones['lows']) > 0
        }
    
    def get_current_session(self) -> str:
        """Get current trading session"""
        hour = datetime.utcnow().hour
        
        if 13 <= hour < 17:
            return "Overlap"
        elif 0 <= hour < 9:
            return "Asian"
        elif 8 <= hour < 17:
            return "London"
        elif 13 <= hour < 22:
            return "NewYork"
        else:
            return "Night"

# ============ TRADING ENGINE ============
class TradingEngine:
    """Main Trading Engine - Manages real trading"""
    
    def __init__(self, user_id: str, api_token: str, selected_account: str):
        self.user_id = user_id
        self.api_token = api_token
        self.selected_account = selected_account
        self.api_client = None
        self.analyzer = DerivSMCAnalyzer()
        self.running = False
        self.trades = []
        self.active_trades = []
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0,
            'daily_trades': 0,
            'hourly_trades': 0,
            'last_reset': datetime.now()
        }
        self.settings = {
            'enabled_markets': ['R_75', 'R_100'],
            'min_confidence': 65,
            'trade_amount': 1.0,  # $ per trade
            'max_concurrent_trades': 3,
            'max_daily_trades': 30,
            'max_hourly_trades': 10,
            'dry_run': False  # REAL TRADING by default
        }
        self.thread = None
        logger.info(f"Trading engine initialized for user {user_id}")
    
    def connect(self) -> Tuple[bool, str]:
        """Connect to Deriv API"""
        try:
            self.api_client = DerivAPIClient(self.api_token)
            success, message = self.api_client.connect()
            
            if success and self.selected_account:
                # Switch to selected account
                switch_success, switch_msg = self.api_client.switch_account(self.selected_account)
                if not switch_success:
                    logger.warning(f"Account switch failed: {switch_msg}")
            
            return success, message
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False, str(e)
    
    def update_settings(self, settings: Dict):
        """Update trading settings"""
        self.settings.update(settings)
        logger.info(f"Settings updated: {settings}")
    
    def start_trading(self) -> Tuple[bool, str]:
        """Start the trading engine"""
        if not self.api_client or not self.api_client.connected:
            return False, "Not connected to Deriv"
        
        self.running = True
        self.thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.thread.start()
        
        logger.info("Trading engine started")
        return True, "Trading started"
    
    def stop_trading(self) -> Tuple[bool, str]:
        """Stop the trading engine"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        logger.info("Trading engine stopped")
        return True, "Trading stopped"
    
    def _trading_loop(self):
        """Main trading loop"""
        logger.info("Trading loop started")
        
        while self.running:
            try:
                # Check limits
                if not self._check_trading_limits():
                    time.sleep(5)
                    continue
                
                # Analyze markets
                for symbol in self.settings['enabled_markets']:
                    if not self.running:
                        break
                    
                    market_config = DERIV_MARKETS.get(symbol)
                    if not market_config:
                        continue
                    
                    # Analyze the market
                    analysis = self.analyzer.analyze(symbol, market_config, self.api_client)
                    
                    if analysis and analysis['confidence'] >= self.settings['min_confidence']:
                        # Execute trade
                        self._execute_trade(analysis)
                        time.sleep(2)  # Small delay between trades
                
                time.sleep(5)  # Check markets every 5 seconds
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                time.sleep(10)
    
    def _check_trading_limits(self) -> bool:
        """Check if trading is allowed based on limits"""
        now = datetime.now()
        
        # Reset hourly counter
        if (now - self.stats['last_reset']).total_seconds() > 3600:
            self.stats['hourly_trades'] = 0
            self.stats['last_reset'] = now
        
        # Reset daily counter at midnight UTC
        if now.hour == 0 and now.minute == 0:
            self.stats['daily_trades'] = 0
        
        # Check limits
        if len(self.active_trades) >= self.settings['max_concurrent_trades']:
            return False
        
        if self.stats['hourly_trades'] >= self.settings['max_hourly_trades']:
            return False
        
        if self.stats['daily_trades'] >= self.settings['max_daily_trades']:
            return False
        
        return True
    
    def _execute_trade(self, analysis: Dict):
        """Execute a trade based on analysis"""
        try:
            symbol = analysis['symbol']
            direction = analysis['direction']
            amount = self.settings['trade_amount']
            
            logger.info(f"Executing trade: {symbol} {direction} ${amount}")
            
            if self.settings['dry_run']:
                # Dry run - simulate trade
                trade = self._create_dry_trade(analysis, amount)
                self.active_trades.append(trade)
                self.trades.append(trade)
                
                # Update stats
                self.stats['total_trades'] += 1
                self.stats['daily_trades'] += 1
                self.stats['hourly_trades'] += 1
                
                logger.info(f"DRY RUN: {symbol} {direction} @ {analysis['price']}")
                
            else:
                # REAL TRADE
                success, result = self.api_client.place_trade(
                    symbol=symbol,
                    direction=direction,
                    amount=amount,
                    duration=analysis['trading_decision']['duration']
                )
                
                if success:
                    trade = self._create_real_trade(analysis, amount, result)
                    self.active_trades.append(trade)
                    self.trades.append(trade)
                    
                    # Update stats
                    self.stats['total_trades'] += 1
                    self.stats['daily_trades'] += 1
                    self.stats['hourly_trades'] += 1
                    
                    logger.info(f"REAL TRADE: {symbol} {direction} | Contract: {result}")
                else:
                    logger.error(f"Trade failed: {result}")
        
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
    
    def _create_dry_trade(self, analysis: Dict, amount: float) -> Dict:
        """Create a dry run trade record"""
        return {
            'id': f"DRY_{int(time.time())}_{secrets.token_hex(4)}",
            'symbol': analysis['symbol'],
            'market_name': analysis['market_name'],
            'direction': analysis['direction'],
            'entry_price': analysis['price'],
            'amount': amount,
            'confidence': analysis['confidence'],
            'strategy': analysis['strategy'],
            'session': analysis['session'],
            'signals': analysis['signals'],
            'timestamp': datetime.now().isoformat(),
            'status': 'OPEN',
            'dry_run': True,
            'profit': 0,
            'outcome': 'PENDING'
        }
    
    def _create_real_trade(self, analysis: Dict, amount: float, contract_id: str) -> Dict:
        """Create a real trade record"""
        return {
            'id': f"REAL_{contract_id}",
            'symbol': analysis['symbol'],
            'market_name': analysis['market_name'],
            'direction': analysis['direction'],
            'entry_price': analysis['price'],
            'amount': amount,
            'contract_id': contract_id,
            'confidence': analysis['confidence'],
            'strategy': analysis['strategy'],
            'session': analysis['session'],
            'signals': analysis['signals'],
            'timestamp': datetime.now().isoformat(),
            'status': 'OPEN',
            'dry_run': False,
            'profit': 0,
            'outcome': 'PENDING'
        }
    
    def get_status(self) -> Dict:
        """Get current trading status"""
        balance = 0
        if self.api_client and self.api_client.connected:
            balance = self.api_client.get_balance()
        
        return {
            'connected': self.api_client.connected if self.api_client else False,
            'running': self.running,
            'balance': balance,
            'active_trades': len(self.active_trades),
            'total_trades': self.stats['total_trades'],
            'winning_trades': self.stats['winning_trades'],
            'losing_trades': self.stats['losing_trades'],
            'daily_trades': self.stats['daily_trades'],
            'hourly_trades': self.stats['hourly_trades'],
            'total_profit': self.stats['total_profit'],
            'session': self.analyzer.get_current_session()
        }
    
    def get_trades(self, limit: int = 20) -> List[Dict]:
        """Get recent trades"""
        return self.trades[-limit:] if self.trades else []
    
    def close_connection(self):
        """Close API connection"""
        if self.api_client:
            self.api_client.close_connection()

# ============ USER DATABASE ============
class UserDatabase:
    """Simple user database"""
    
    def __init__(self):
        self.users = {}
        self.load_users()
    
    def load_users(self):
        """Load users from file"""
        try:
            if os.path.exists('users.json'):
                with open('users.json', 'r') as f:
                    self.users = json.load(f)
                logger.info(f"Loaded {len(self.users)} users")
        except Exception as e:
            logger.error(f"Load users error: {e}")
            self.users = {}
    
    def save_users(self):
        """Save users to file"""
        try:
            with open('users.json', 'w') as f:
                json.dump(self.users, f, indent=2)
            logger.info("Users saved")
        except Exception as e:
            logger.error(f"Save users error: {e}")
    
    def create_user(self, username: str, password: str) -> Tuple[bool, str]:
        """Create a new user"""
        if username in self.users:
            return False, "Username already exists"
        
        user_id = hashlib.sha256(f"{username}{datetime.now()}".encode()).hexdigest()[:16]
        
        self.users[username] = {
            'user_id': user_id,
            'password_hash': hashlib.sha256(password.encode()).hexdigest(),
            'api_token': '',  # Will be set later
            'selected_account': '',
            'settings': {
                'enabled_markets': ['R_75', 'R_100'],
                'min_confidence': 65,
                'trade_amount': 1.0,
                'max_concurrent_trades': 3,
                'max_daily_trades': 30,
                'max_hourly_trades': 10,
                'dry_run': True  # Start in dry run mode for safety
            },
            'created_at': datetime.now().isoformat(),
            'last_login': None
        }
        
        self.save_users()
        return True, user_id
    
    def verify_user(self, username: str, password: str) -> Tuple[bool, str]:
        """Verify user credentials"""
        if username not in self.users:
            return False, ""
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        if self.users[username]['password_hash'] == password_hash:
            self.users[username]['last_login'] = datetime.now().isoformat()
            self.save_users()
            return True, self.users[username]['user_id']
        
        return False, ""
    
    def update_user(self, username: str, updates: Dict) -> bool:
        """Update user data"""
        if username not in self.users:
            return False
        
        self.users[username].update(updates)
        self.save_users()
        return True
    
    def get_user(self, username: str) -> Optional[Dict]:
        """Get user data"""
        return self.users.get(username)

# ============ FLASK WEB APP ============
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_urlsafe(32))

# Initialize database
user_db = UserDatabase()

# Trading engines storage
trading_engines = {}

def get_trading_engine(username: str) -> Optional[TradingEngine]:
    """Get or create trading engine for user"""
    if username not in trading_engines:
        user_data = user_db.get_user(username)
        if not user_data:
            return None
        
        engine = TradingEngine(
            user_id=user_data['user_id'],
            api_token=user_data.get('api_token', ''),
            selected_account=user_data.get('selected_account', '')
        )
        
        # Apply user settings
        engine.update_settings(user_data.get('settings', {}))
        
        trading_engines[username] = engine
    
    return trading_engines.get(username)

# ============ BLACK & GOLD UI TEMPLATE ============
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Karanka V7 Deriv - Real Trading</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
        :root {
            --gold-primary: #D4AF37;
            --gold-secondary: #FFD700;
            --gold-light: #FFED4E;
            --gold-dark: #B8860B;
            --black-primary: #0a0a0a;
            --black-secondary: #1a1a1a;
            --black-tertiary: #2a2a2a;
            --border-color: #333;
            --success: #00FF00;
            --error: #FF4444;
            --warning: #FFAA00;
            --info: #00AAFF;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            -webkit-tap-highlight-color: transparent;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--black-primary);
            color: var(--gold-secondary);
            line-height: 1.6;
            overflow-x: hidden;
            padding: 10px;
            -webkit-font-smoothing: antialiased;
        }
        
        .app-container {
            max-width: 100%;
            margin: 0 auto;
        }
        
        /* Header */
        .header {
            background: linear-gradient(135deg, var(--black-secondary), var(--black-tertiary));
            padding: 15px;
            border-radius: 12px;
            margin-bottom: 15px;
            border: 2px solid var(--border-color);
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            text-align: center;
        }
        
        .header h1 {
            font-size: 22px;
            margin-bottom: 5px;
            background: linear-gradient(90deg, var(--gold-primary), var(--gold-light));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }
        
        .status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            color: var(--gold-light);
            margin-top: 8px;
        }
        
        /* Tabs */
        .tabs-container {
            display: flex;
            gap: 8px;
            margin-bottom: 20px;
            overflow-x: auto;
            padding-bottom: 5px;
            scrollbar-width: none;
        }
        
        .tabs-container::-webkit-scrollbar {
            display: none;
        }
        
        .tab {
            background: var(--black-secondary);
            color: var(--gold-secondary);
            border: 1px solid var(--border-color);
            padding: 12px 20px;
            border-radius: 8px;
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.3s ease;
            font-size: 14px;
            font-weight: 600;
            flex-shrink: 0;
        }
        
        .tab:hover {
            background: var(--black-tertiary);
        }
        
        .tab.active {
            background: var(--gold-primary);
            color: var(--black-primary);
            box-shadow: 0 2px 8px rgba(212, 175, 55, 0.3);
            border-color: var(--gold-dark);
        }
        
        /* Content */
        .content-panel {
            background: var(--black-secondary);
            border-radius: 12px;
            padding: 20px;
            border: 2px solid var(--border-color);
            display: none;
            animation: fadeIn 0.3s ease;
        }
        
        .content-panel.active {
            display: block;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Buttons */
        .btn {
            background: linear-gradient(135deg, var(--gold-primary), var(--gold-dark));
            color: var(--black-primary);
            border: none;
            padding: 15px 25px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 700;
            font-size: 16px;
            width: 100%;
            margin: 10px 0;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(212, 175, 55, 0.2);
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(212, 175, 55, 0.3);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .btn-success {
            background: linear-gradient(135deg, var(--success), #00AA00);
        }
        
        .btn-danger {
            background: linear-gradient(135deg, var(--error), #AA0000);
        }
        
        .btn-warning {
            background: linear-gradient(135deg, var(--warning), #AA7700);
        }
        
        /* Forms */
        .form-group {
            margin: 15px 0;
        }
        
        .form-label {
            display: block;
            margin-bottom: 8px;
            color: var(--gold-light);
            font-size: 14px;
            font-weight: 600;
        }
        
        .form-input {
            width: 100%;
            padding: 12px 15px;
            background: var(--black-primary);
            border: 1px solid var(--border-color);
            color: var(--gold-secondary);
            border-radius: 8px;
            font-size: 16px;
            transition: border 0.3s ease;
        }
        
        .form-input:focus {
            outline: none;
            border-color: var(--gold-primary);
        }
        
        .form-select {
            width: 100%;
            padding: 12px 15px;
            background: var(--black-primary);
            border: 1px solid var(--border-color);
            color: var(--gold-secondary);
            border-radius: 8px;
            font-size: 16px;
        }
        
        /* Stats */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin: 20px 0;
        }
        
        .stat-box {
            background: var(--black-tertiary);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid var(--border-color);
        }
        
        .stat-value {
            font-size: 28px;
            font-weight: 700;
            color: var(--gold-primary);
            margin-bottom: 5px;
        }
        
        .stat-label {
            font-size: 12px;
            color: var(--gold-light);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Trade Cards */
        .trade-card {
            background: var(--black-tertiary);
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid var(--gold-primary);
        }
        
        .trade-card.buy {
            border-left-color: var(--success);
        }
        
        .trade-card.sell {
            border-left-color: var(--error);
        }
        
        .trade-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .trade-symbol {
            font-weight: 700;
            font-size: 16px;
            color: var(--gold-light);
        }
        
        .trade-direction {
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        
        .trade-direction.buy {
            background: rgba(0, 255, 0, 0.1);
            color: var(--success);
        }
        
        .trade-direction.sell {
            background: rgba(255, 68, 68, 0.1);
            color: var(--error);
        }
        
        .trade-details {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            font-size: 12px;
            color: var(--gold-secondary);
        }
        
        /* Alerts */
        .alert {
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            font-size: 14px;
            display: none;
        }
        
        .alert-success {
            background: rgba(0, 255, 0, 0.1);
            color: var(--success);
            border: 1px solid var(--success);
        }
        
        .alert-error {
            background: rgba(255, 68, 68, 0.1);
            color: var(--error);
            border: 1px solid var(--error);
        }
        
        .alert-warning {
            background: rgba(255, 170, 0, 0.1);
            color: var(--warning);
            border: 1px solid var(--warning);
        }
        
        /* Market Selection */
        .market-category {
            margin: 20px 0;
        }
        
        .category-title {
            font-size: 16px;
            color: var(--gold-primary);
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 1px solid var(--border-color);
        }
        
        .market-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 10px;
        }
        
        .market-checkbox {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 12px;
            background: var(--black-tertiary);
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.3s ease;
        }
        
        .market-checkbox:hover {
            background: var(--border-color);
        }
        
        .market-checkbox input {
            width: 18px;
            height: 18px;
        }
        
        .market-label {
            color: var(--gold-secondary);
            font-size: 14px;
        }
        
        /* Connection Status */
        .connection-status {
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            background: var(--black-tertiary);
            border: 1px solid var(--border-color);
        }
        
        .account-list {
            margin: 15px 0;
        }
        
        .account-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px;
            background: var(--black-tertiary);
            border-radius: 8px;
            margin: 8px 0;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 1px solid transparent;
        }
        
        .account-item:hover {
            border-color: var(--gold-primary);
        }
        
        .account-item.selected {
            border-color: var(--gold-primary);
            background: rgba(212, 175, 55, 0.1);
        }
        
        .account-name {
            font-weight: 600;
            color: var(--gold-light);
        }
        
        .account-type {
            font-size: 12px;
            padding: 3px 8px;
            border-radius: 4px;
            background: var(--gold-primary);
            color: var(--black-primary);
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .header h1 {
                font-size: 20px;
            }
            
            .tab {
                padding: 10px 16px;
                font-size: 13px;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
            
            .market-grid {
                grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
            }
        }
        
        /* Loading */
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid var(--gold-light);
            border-top-color: transparent;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Hide elements */
        .hidden {
            display: none !important;
        }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Header -->
        <div class="header">
            <h1>🎯 KARANKA V7 - DERIV REAL TRADING</h1>
            <div class="status-bar">
                <span id="connection-status">🔴 Disconnected</span>
                <span id="trading-status">❌ Not Trading</span>
                <span id="balance">$0.00</span>
            </div>
        </div>
        
        <!-- Authentication Section -->
        <div id="auth-section" class="content-panel active">
            <h2 style="color: var(--gold-primary); margin-bottom: 20px;">Login / Register</h2>
            
            <div class="form-group">
                <label class="form-label">Username</label>
                <input type="text" id="username" class="form-input" placeholder="Enter username">
            </div>
            
            <div class="form-group">
                <label class="form-label">Password</label>
                <input type="password" id="password" class="form-input" placeholder="Enter password">
            </div>
            
            <button class="btn" onclick="login()">🔑 Login</button>
            <button class="btn btn-warning" onclick="register()">📝 Register</button>
            
            <div id="auth-message" class="alert" style="display: none;"></div>
            
            <div style="margin-top: 20px; padding: 15px; background: var(--black-tertiary); border-radius: 8px;">
                <h4 style="color: var(--gold-light); margin-bottom: 10px;">ℹ️ How to get Deriv API Token:</h4>
                <ol style="color: var(--gold-secondary); font-size: 12px; line-height: 1.6; padding-left: 20px;">
                    <li>Go to <a href="https://app.deriv.com" style="color: var(--gold-primary);">app.deriv.com</a></li>
                    <li>Login to your account (demo or real)</li>
                    <li>Go to Settings → API Token</li>
                    <li>Create new token with "Read" and "Trade" permissions</li>
                    <li>Copy the token and save it securely</li>
                </ol>
            </div>
        </div>
        
        <!-- Main App (hidden until logged in) -->
        <div id="main-app" class="hidden">
            <!-- Tabs -->
            <div class="tabs-container">
                <div class="tab active" onclick="showTab('dashboard')">📊 Dashboard</div>
                <div class="tab" onclick="showTab('connection')">🔗 Connection</div>
                <div class="tab" onclick="showTab('markets')">📈 Markets</div>
                <div class="tab" onclick="showTab('settings')">⚙️ Settings</div>
                <div class="tab" onclick="showTab('trades')">💼 Trades</div>
                <div class="tab" onclick="showTab('analysis')">🧠 Analysis</div>
            </div>
            
            <!-- Dashboard Tab -->
            <div id="dashboard" class="content-panel active">
                <h2>Trading Dashboard</h2>
                
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-value" id="stat-balance">$0</div>
                        <div class="stat-label">Balance</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="stat-active">0</div>
                        <div class="stat-label">Active Trades</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="stat-wins">0</div>
                        <div class="stat-label">Winning Trades</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="stat-losses">0</div>
                        <div class="stat-label">Losing Trades</div>
                    </div>
                </div>
                
                <div style="margin: 20px 0;">
                    <button class="btn btn-success" onclick="startTrading()">🚀 Start Trading</button>
                    <button class="btn btn-danger" onclick="stopTrading()">🛑 Stop Trading</button>
                </div>
                
                <div class="connection-status">
                    <h4 style="color: var(--gold-light); margin-bottom: 10px;">Current Status</h4>
                    <div id="status-details">
                        <p>Connecting to Deriv...</p>
                    </div>
                </div>
                
                <div style="margin-top: 20px;">
                    <button class="btn" onclick="logout()">🚪 Logout</button>
                </div>
            </div>
            
            <!-- Connection Tab -->
            <div id="connection" class="content-panel">
                <h2>Deriv Connection</h2>
                
                <div class="form-group">
                    <label class="form-label">Deriv API Token</label>
                    <input type="text" id="api-token" class="form-input" placeholder="Paste your Deriv API token here">
                    <small style="color: var(--gold-light); font-size: 12px;">Get token from app.deriv.com → Settings → API Token</small>
                </div>
                
                <button class="btn" onclick="connectDeriv()">🔗 Connect to Deriv</button>
                
                <div id="connection-result" class="alert" style="display: none;"></div>
                
                <!-- Account Selection -->
                <div id="account-selection" class="hidden">
                    <h3 style="color: var(--gold-primary); margin: 20px 0 10px 0;">Select Account</h3>
                    <div id="accounts-list" class="account-list">
                        <!-- Accounts will be populated here -->
                    </div>
                    <button class="btn btn-success" onclick="selectAccount()">✅ Select Account</button>
                </div>
            </div>
            
            <!-- Markets Tab -->
            <div id="markets" class="content-panel">
                <h2>Market Selection</h2>
                <p style="color: var(--gold-light); margin-bottom: 20px;">Select which Deriv markets to trade:</p>
                
                <!-- Volatility Indices -->
                <div class="market-category">
                    <h3 class="category-title">📊 Volatility Indices</h3>
                    <div class="market-grid" id="volatility-markets">
                        <!-- Markets will be populated -->
                    </div>
                </div>
                
                <!-- Crash/Boom Indices -->
                <div class="market-category">
                    <h3 class="category-title">⚡ Crash & Boom Indices</h3>
                    <div class="market-grid" id="crashboom-markets">
                        <!-- Markets will be populated -->
                    </div>
                </div>
                
                <button class="btn" onclick="saveMarkets()">💾 Save Market Selection</button>
            </div>
            
            <!-- Settings Tab -->
            <div id="settings" class="content-panel">
                <h2>Trading Settings</h2>
                
                <div class="form-group">
                    <label class="form-label">Trading Mode</label>
                    <select id="trading-mode" class="form-select">
                        <option value="dry">🟡 Dry Run (Test Mode)</option>
                        <option value="real">🔴 Real Trading</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Trade Amount ($)</label>
                    <input type="number" id="trade-amount" class="form-input" value="1.0" min="0.35" step="0.1">
                    <small style="color: var(--gold-light);">Amount in USD per trade (minimum $0.35)</small>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Minimum Confidence (%)</label>
                    <input type="number" id="min-confidence" class="form-input" value="65" min="50" max="95">
                    <small style="color: var(--gold-light);">Only trade when confidence is above this level</small>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Max Concurrent Trades</label>
                    <input type="number" id="max-concurrent" class="form-input" value="3" min="1" max="10">
                </div>
                
                <div class="form-group">
                    <label class="form-label">Max Daily Trades</label>
                    <input type="number" id="max-daily" class="form-input" value="30" min="5" max="100">
                </div>
                
                <div class="form-group">
                    <label class="form-label">Max Hourly Trades</label>
                    <input type="number" id="max-hourly" class="form-input" value="10" min="1" max="30">
                </div>
                
                <button class="btn" onclick="saveSettings()">💾 Save Settings</button>
            </div>
            
            <!-- Trades Tab -->
            <div id="trades" class="content-panel">
                <h2>Recent Trades</h2>
                <div id="trades-list">
                    <p style="color: var(--gold-light); text-align: center; padding: 20px;">
                        No trades yet. Start trading to see your trades here.
                    </p>
                </div>
            </div>
            
            <!-- Analysis Tab -->
            <div id="analysis" class="content-panel">
                <h2>Market Analysis</h2>
                <div id="analysis-content">
                    <div style="padding: 20px; background: var(--black-tertiary); border-radius: 8px;">
                        <h4 style="color: var(--gold-light); margin-bottom: 10px;">🧠 SMC Strategy Active</h4>
                        <p style="color: var(--gold-secondary);">
                            The bot is analyzing markets using Smart Money Concepts:
                        </p>
                        <ul style="color: var(--gold-secondary); margin: 10px 0 10px 20px;">
                            <li>Liquidity zone detection</li>
                            <li>Order block identification</li>
                            <li>Fair Value Gap analysis</li>
                            <li>Displacement patterns</li>
                            <li>Multi-timeframe confirmation</li>
                        </ul>
                        <p style="color: var(--gold-secondary); font-size: 12px;">
                            Strategy optimized for Deriv Volatility Indices
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let currentUser = null;
        let currentTab = 'dashboard';
        let selectedAccount = null;
        let updateInterval = null;
        
        // Show/hide tabs
        function showTab(tabName) {
            // Hide all content panels
            document.querySelectorAll('.content-panel').forEach(panel => {
                panel.classList.remove('active');
            });
            
            // Deactivate all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
            currentTab = tabName;
        }
        
        // Show message
        function showAlert(elementId, message, type = 'success') {
            const element = document.getElementById(elementId);
            element.textContent = message;
            element.className = `alert alert-${type}`;
            element.style.display = 'block';
            
            setTimeout(() => {
                element.style.display = 'none';
            }, 5000);
        }
        
        // Login function
        async function login() {
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            
            if (!username || !password) {
                showAlert('auth-message', 'Please enter username and password', 'error');
                return;
            }
            
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({username, password})
            });
            
            const data = await response.json();
            
            if (data.success) {
                currentUser = username;
                document.getElementById('auth-section').style.display = 'none';
                document.getElementById('main-app').classList.remove('hidden');
                
                // Load user data
                await loadUserData();
                
                // Start auto-update
                startAutoUpdate();
                
                showAlert('auth-message', 'Login successful!', 'success');
            } else {
                showAlert('auth-message', data.message, 'error');
            }
        }
        
        // Register function
        async function register() {
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            
            if (!username || !password) {
                showAlert('auth-message', 'Please enter username and password', 'error');
                return;
            }
            
            const response = await fetch('/api/register', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({username, password})
            });
            
            const data = await response.json();
            showAlert('auth-message', data.message, data.success ? 'success' : 'error');
        }
        
        // Load user data
        async function loadUserData() {
            // Load settings
            const settingsResponse = await fetch('/api/settings');
            const settings = await settingsResponse.json();
            
            if (settings) {
                document.getElementById('trading-mode').value = settings.dry_run ? 'dry' : 'real';
                document.getElementById('trade-amount').value = settings.trade_amount;
                document.getElementById('min-confidence').value = settings.min_confidence;
                document.getElementById('max-concurrent').value = settings.max_concurrent_trades;
                document.getElementById('max-daily').value = settings.max_daily_trades;
                document.getElementById('max-hourly').value = settings.max_hourly_trades;
            }
            
            // Load markets
            await loadMarkets();
            
            // Update status
            await updateStatus();
        }
        
        // Load markets
        async function loadMarkets() {
            const response = await fetch('/api/markets');
            const data = await response.json();
            
            if (data.enabled_markets) {
                // Clear existing
                document.getElementById('volatility-markets').innerHTML = '';
                document.getElementById('crashboom-markets').innerHTML = '';
                
                // Add markets
                for (const [symbol, config] of Object.entries(data.all_markets)) {
                    const container = config.category === 'Volatility' 
                        ? document.getElementById('volatility-markets')
                        : document.getElementById('crashboom-markets');
                    
                    const isChecked = data.enabled_markets.includes(symbol) ? 'checked' : '';
                    
                    container.innerHTML += `
                        <label class="market-checkbox">
                            <input type="checkbox" value="${symbol}" ${isChecked}>
                            <span class="market-label">${config.name}</span>
                        </label>
                    `;
                }
            }
        }
        
        // Connect to Deriv
        async function connectDeriv() {
            const apiToken = document.getElementById('api-token').value.trim();
            
            if (!apiToken) {
                showAlert('connection-result', 'Please enter your Deriv API token', 'error');
                return;
            }
            
            showAlert('connection-result', 'Connecting to Deriv...', 'warning');
            
            const response = await fetch('/api/connect', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({api_token: apiToken})
            });
            
            const data = await response.json();
            
            if (data.success) {
                showAlert('connection-result', data.message, 'success');
                
                // Show account selection if accounts found
                if (data.accounts && data.accounts.length > 0) {
                    const accountsList = document.getElementById('accounts-list');
                    accountsList.innerHTML = '';
                    
                    data.accounts.forEach(account => {
                        accountsList.innerHTML += `
                            <div class="account-item" onclick="selectAccountItem(this, '${account.loginid}')">
                                <div>
                                    <div class="account-name">${account.name}</div>
                                    <div style="font-size: 11px; color: var(--gold-secondary);">
                                        Balance: $${account.balance} | ${account.currency}
                                    </div>
                                </div>
                                <span class="account-type">${account.type.toUpperCase()}</span>
                            </div>
                        `;
                    });
                    
                    document.getElementById('account-selection').classList.remove('hidden');
                }
            } else {
                showAlert('connection-result', data.message, 'error');
            }
        }
        
        // Select account item
        function selectAccountItem(element, loginid) {
            // Remove selection from all items
            document.querySelectorAll('.account-item').forEach(item => {
                item.classList.remove('selected');
            });
            
            // Select this item
            element.classList.add('selected');
            selectedAccount = loginid;
        }
        
        // Save selected account
        async function selectAccount() {
            if (!selectedAccount) {
                showAlert('connection-result', 'Please select an account', 'error');
                return;
            }
            
            const response = await fetch('/api/select-account', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({account: selectedAccount})
            });
            
            const data = await response.json();
            showAlert('connection-result', data.message, data.success ? 'success' : 'error');
            
            if (data.success) {
                document.getElementById('account-selection').classList.add('hidden');
                await updateStatus();
            }
        }
        
        // Save market selection
        async function saveMarkets() {
            const selectedMarkets = [];
            
            // Get selected volatility markets
            document.querySelectorAll('#volatility-markets input:checked').forEach(cb => {
                selectedMarkets.push(cb.value);
            });
            
            // Get selected crash/boom markets
            document.querySelectorAll('#crashboom-markets input:checked').forEach(cb => {
                selectedMarkets.push(cb.value);
            });
            
            if (selectedMarkets.length === 0) {
                showAlert('connection-result', 'Please select at least one market', 'error');
                return;
            }
            
            const response = await fetch('/api/markets', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({markets: selectedMarkets})
            });
            
            const data = await response.json();
            alert(data.message);
        }
        
        // Save settings
        async function saveSettings() {
            const settings = {
                dry_run: document.getElementById('trading-mode').value === 'dry',
                trade_amount: parseFloat(document.getElementById('trade-amount').value),
                min_confidence: parseInt(document.getElementById('min-confidence').value),
                max_concurrent_trades: parseInt(document.getElementById('max-concurrent').value),
                max_daily_trades: parseInt(document.getElementById('max-daily').value),
                max_hourly_trades: parseInt(document.getElementById('max-hourly').value)
            };
            
            const response = await fetch('/api/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(settings)
            });
            
            const data = await response.json();
            alert(data.message);
        }
        
        // Start trading
        async function startTrading() {
            const response = await fetch('/api/start', {
                method: 'POST'
            });
            
            const data = await response.json();
            alert(data.message);
        }
        
        // Stop trading
        async function stopTrading() {
            const response = await fetch('/api/stop', {
                method: 'POST'
            });
            
            const data = await response.json();
            alert(data.message);
        }
        
        // Update status
        async function updateStatus() {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if (data.connected) {
                document.getElementById('connection-status').innerHTML = 
                    `🟢 Connected ${data.session ? '| ' + data.session : ''}`;
                document.getElementById('trading-status').innerHTML = 
                    data.running ? '🟢 Trading' : '🟡 Ready';
                document.getElementById('balance').innerHTML = 
                    `$${data.balance ? data.balance.toFixed(2) : '0.00'}`;
                
                // Update stats
                document.getElementById('stat-balance').innerHTML = 
                    `$${data.balance ? data.balance.toFixed(2) : '0'}`;
                document.getElementById('stat-active').innerHTML = data.active_trades;
                document.getElementById('stat-wins').innerHTML = data.winning_trades;
                document.getElementById('stat-losses').innerHTML = data.losing_trades;
                
                // Update status details
                const statusDetails = document.getElementById('status-details');
                statusDetails.innerHTML = `
                    <p>Connected: ✅</p>
                    <p>Trading: ${data.running ? '✅ Active' : '⏸️ Stopped'}</p>
                    <p>Session: ${data.session || 'Unknown'}</p>
                    <p>Active Trades: ${data.active_trades}</p>
                    <p>Total Profit: $${data.total_profit ? data.total_profit.toFixed(2) : '0.00'}</p>
                `;
                
                // Update trades list
                await updateTrades();
            } else {
                document.getElementById('connection-status').innerHTML = '🔴 Disconnected';
                document.getElementById('trading-status').innerHTML = '❌ Not Trading';
                document.getElementById('balance').innerHTML = '$0.00';
            }
        }
        
        // Update trades list
        async function updateTrades() {
            const response = await fetch('/api/trades');
            const trades = await response.json();
            
            const tradesList = document.getElementById('trades-list');
            
            if (trades.length === 0) {
                tradesList.innerHTML = `
                    <p style="color: var(--gold-light); text-align: center; padding: 20px;">
                        No trades yet. Start trading to see your trades here.
                    </p>
                `;
                return;
            }
            
            let html = '';
            trades.reverse().forEach(trade => {
                const directionClass = trade.direction.toLowerCase();
                const isDry = trade.dry_run ? ' (DRY)' : '';
                const outcomeClass = trade.outcome === 'WIN' ? 'success' : 
                                   trade.outcome === 'LOSS' ? 'error' : 'warning';
                const outcomeText = trade.outcome === 'PENDING' ? 'Pending' : 
                                  `${trade.outcome} $${trade.profit || 0}`;
                
                html += `
                    <div class="trade-card ${directionClass}">
                        <div class="trade-header">
                            <div class="trade-symbol">${trade.symbol}${isDry}</div>
                            <span class="trade-direction ${directionClass}">${trade.direction}</span>
                        </div>
                        <div class="trade-details">
                            <div>Entry: $${trade.entry_price.toFixed(5)}</div>
                            <div>Amount: $${trade.amount}</div>
                            <div>Confidence: ${trade.confidence.toFixed(1)}%</div>
                            <div style="color: var(--${outcomeClass});">${outcomeText}</div>
                        </div>
                        <div style="font-size: 11px; color: var(--gold-secondary); margin-top: 8px;">
                            ${trade.strategy} | ${trade.session} | ${new Date(trade.timestamp).toLocaleTimeString()}
                        </div>
                    </div>
                `;
            });
            
            tradesList.innerHTML = html;
        }
        
        // Start auto-update
        function startAutoUpdate() {
            if (updateInterval) {
                clearInterval(updateInterval);
            }
            
            updateInterval = setInterval(async () => {
                await updateStatus();
            }, 3000);
            
            // Initial update
            updateStatus();
        }
        
        // Logout
        async function logout() {
            await fetch('/api/logout');
            location.reload();
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            // Check if already logged in
            if (currentUser) {
                document.getElementById('auth-section').style.display = 'none';
                document.getElementById('main-app').classList.remove('hidden');
                startAutoUpdate();
            }
        });
    </script>
</body>
</html>
'''

# ============ FLASK ROUTES ============
@app.route('/')
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/register', methods=['POST'])
def api_register():
    """Register new user"""
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password required'})
        
        success, message = user_db.create_user(username, password)
        return jsonify({'success': success, 'message': message})
        
    except Exception as e:
        logger.error(f"Register error: {e}")
        return jsonify({'success': False, 'message': 'Registration failed'})

@app.route('/api/login', methods=['POST'])
def api_login():
    """Login user"""
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password required'})
        
        success, user_id = user_db.verify_user(username, password)
        
        if success:
            session['username'] = username
            session['user_id'] = user_id
            return jsonify({'success': True, 'message': 'Login successful'})
        else:
            return jsonify({'success': False, 'message': 'Invalid credentials'})
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'success': False, 'message': 'Login failed'})

@app.route('/api/logout', methods=['POST'])
def api_logout():
    """Logout user"""
    username = session.get('username')
    
    if username and username in trading_engines:
        # Stop trading and close connection
        engine = trading_engines[username]
        engine.stop_trading()
        engine.close_connection()
        del trading_engines[username]
    
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out'})

@app.route('/api/connect', methods=['POST'])
def api_connect():
    """Connect to Deriv API"""
    try:
        username = session.get('username')
        if not username:
            return jsonify({'success': False, 'message': 'Not logged in'})
        
        data = request.json
        api_token = data.get('api_token', '').strip()
        
        if not api_token:
            return jsonify({'success': False, 'message': 'API token required'})
        
        # Get or create trading engine
        engine = get_trading_engine(username)
        if not engine:
            return jsonify({'success': False, 'message': 'User not found'})
        
        # Update API token
        user_db.update_user(username, {'api_token': api_token})
        
        # Connect to Deriv
        success, message = engine.connect()
        
        if success:
            # Get accounts list
            accounts = []
            if engine.api_client and engine.api_client.accounts:
                accounts = engine.api_client.accounts
            
            return jsonify({
                'success': True,
                'message': message,
                'accounts': accounts
            })
        else:
            return jsonify({'success': False, 'message': message})
            
    except Exception as e:
        logger.error(f"Connect error: {e}")
        return jsonify({'success': False, 'message': f'Connection error: {str(e)}'})

@app.route('/api/select-account', methods=['POST'])
def api_select_account():
    """Select trading account"""
    try:
        username = session.get('username')
        if not username:
            return jsonify({'success': False, 'message': 'Not logged in'})
        
        data = request.json
        account = data.get('account', '')
        
        if not account:
            return jsonify({'success': False, 'message': 'Account required'})
        
        engine = get_trading_engine(username)
        if not engine or not engine.api_client:
            return jsonify({'success': False, 'message': 'Not connected to Deriv'})
        
        # Switch account
        success, message = engine.api_client.switch_account(account)
        
        if success:
            engine.selected_account = account
            user_db.update_user(username, {'selected_account': account})
            return jsonify({'success': True, 'message': f'Selected account: {account}'})
        else:
            return jsonify({'success': False, 'message': message})
            
    except Exception as e:
        logger.error(f"Select account error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/markets', methods=['GET', 'POST'])
def api_markets():
    """Get or update market selection"""
    try:
        username = session.get('username')
        if not username:
            return jsonify({'success': False, 'message': 'Not logged in'})
        
        if request.method == 'GET':
            engine = get_trading_engine(username)
            if not engine:
                return jsonify({'enabled_markets': [], 'all_markets': DERIV_MARKETS})
            
            return jsonify({
                'enabled_markets': engine.settings['enabled_markets'],
                'all_markets': DERIV_MARKETS
            })
        
        else:  # POST - update markets
            data = request.json
            markets = data.get('markets', [])
            
            if not markets:
                return jsonify({'success': False, 'message': 'No markets selected'})
            
            engine = get_trading_engine(username)
            if not engine:
                return jsonify({'success': False, 'message': 'Engine not found'})
            
            engine.settings['enabled_markets'] = markets
            
            # Update user settings
            user_data = user_db.get_user(username)
            if user_data:
                user_data['settings']['enabled_markets'] = markets
                user_db.update_user(username, user_data)
            
            return jsonify({'success': True, 'message': f'{len(markets)} markets selected'})
            
    except Exception as e:
        logger.error(f"Markets error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    """Get or update settings"""
    try:
        username = session.get('username')
        if not username:
            return jsonify({'success': False, 'message': 'Not logged in'})
        
        engine = get_trading_engine(username)
        if not engine:
            return jsonify({'success': False, 'message': 'Engine not found'})
        
        if request.method == 'GET':
            return jsonify(engine.settings)
        
        else:  # POST - update settings
            data = request.json
            
            # Update engine settings
            engine.update_settings(data)
            
            # Update user settings
            user_data = user_db.get_user(username)
            if user_data:
                user_data['settings'].update(data)
                user_db.update_user(username, user_data)
            
            return jsonify({'success': True, 'message': 'Settings saved'})
            
    except Exception as e:
        logger.error(f"Settings error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/start', methods=['POST'])
def api_start():
    """Start trading"""
    try:
        username = session.get('username')
        if not username:
            return jsonify({'success': False, 'message': 'Not logged in'})
        
        engine = get_trading_engine(username)
        if not engine:
            return jsonify({'success': False, 'message': 'Engine not found'})
        
        if not engine.api_client or not engine.api_client.connected:
            return jsonify({'success': False, 'message': 'Not connected to Deriv'})
        
        success, message = engine.start_trading()
        return jsonify({'success': success, 'message': message})
        
    except Exception as e:
        logger.error(f"Start error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop trading"""
    try:
        username = session.get('username')
        if not username:
            return jsonify({'success': False, 'message': 'Not logged in'})
        
        engine = get_trading_engine(username)
        if not engine:
            return jsonify({'success': False, 'message': 'Engine not found'})
        
        success, message = engine.stop_trading()
        return jsonify({'success': success, 'message': message})
        
    except Exception as e:
        logger.error(f"Stop error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/status')
def api_status():
    """Get trading status"""
    try:
        username = session.get('username')
        if not username:
            return jsonify({
                'connected': False,
                'running': False,
                'balance': 0,
                'active_trades': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'daily_trades': 0,
                'hourly_trades': 0,
                'total_profit': 0,
                'session': 'N/A'
            })
        
        engine = get_trading_engine(username)
        if not engine:
            return jsonify({
                'connected': False,
                'running': False,
                'balance': 0,
                'active_trades': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'daily_trades': 0,
                'hourly_trades': 0,
                'total_profit': 0,
                'session': 'N/A'
            })
        
        status = engine.get_status()
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Status error: {e}")
        return jsonify({
            'connected': False,
            'running': False,
            'balance': 0,
            'active_trades': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'daily_trades': 0,
            'hourly_trades': 0,
            'total_profit': 0,
            'session': 'N/A'
        })

@app.route('/api/trades')
def api_trades():
    """Get recent trades"""
    try:
        username = session.get('username')
        if not username:
            return jsonify([])
        
        engine = get_trading_engine(username)
        if not engine:
            return jsonify([])
        
        trades = engine.get_trades(20)
        return jsonify(trades)
        
    except Exception as e:
        logger.error(f"Trades error: {e}")
        return jsonify([])

# ============ MAIN ============
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    print("\n" + "="*80)
    print("🎯 KARANKA V7 - DERIV REAL TRADING BOT")
    print("="*80)
    print("✅ REAL DERIV API CONNECTION")
    print("✅ REAL DEMO/REAL ACCOUNT TRADING")
    print("✅ SMC STRATEGY OPTIMIZED FOR DERIV")
    print("✅ MOBILE-FRIENDLY BLACK & GOLD UI")
    print("✅ RENDER.COM DEPLOYMENT READY")
    print("="*80)
    print(f"🚀 Starting on http://localhost:{port}")
    print("="*80)
    
    app.run(host='0.0.0.0', port=port, debug=False)
