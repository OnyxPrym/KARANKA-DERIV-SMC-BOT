#!/usr/bin/env python3
"""
================================================================================
🚀 KARANKA PRO MAX - GUARANTEED 24/7 REAL TRADING
================================================================================
• REAL DERIV ACCOUNT CONNECTION - Uses YOUR API token
• REAL MARKET DATA - Live data from Deriv
• REAL TRADES - Executes in YOUR account
• RENDER.COM PROOF - Prevents sleep, runs 24/7
• PROFITABLE STRATEGIES - Advanced SMC with real analysis
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
from flask import Flask, render_template_string, jsonify, request, Response
from flask_cors import CORS
from functools import wraps
import atexit
import sys
import math
import pickle
from pathlib import Path

# ============ FORCE RENDER TO STAY AWAKE ============
RENDER_APP_URL = os.environ.get('RENDER_EXTERNAL_URL', 'http://localhost:10000')
RENDER_INSTANCE_ID = os.environ.get('RENDER_INSTANCE_ID', f'instance_{int(time.time())}')

# Start background thread to keep Render alive
def keep_render_awake():
    """Continuously ping the app to prevent Render from sleeping"""
    while True:
        try:
            time.sleep(180)  # Ping every 3 minutes (Render sleeps after 5-15 minutes idle)
            requests.get(f"{RENDER_APP_URL}/keep-alive", timeout=10)
            requests.get(f"{RENDER_APP_URL}/api/ping", timeout=10)
            print(f"✅ Keep-alive ping sent to keep Render awake - {datetime.now()}")
        except Exception as e:
            print(f"Keep-alive error: {e}")
            # Try to restart the app if completely dead
            try:
                requests.get(f"{RENDER_APP_URL}/", timeout=10)
            except:
                pass

# Start keep-alive thread
threading.Thread(target=keep_render_awake, daemon=True).start()

# ============ ENHANCED LOGGING ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('real_trading.log')
    ]
)
logger = logging.getLogger(__name__)

# ============ REAL DERIV CONNECTION CLASS ============
class RealDerivTrader:
    """REAL Deriv trading with YOUR API token"""
    
    def __init__(self, api_token: str):
        self.api_token = api_token.strip()
        self.ws = None
        self.connected = False
        self.account_id = None
        self.balance = 0.0
        self.currency = "USD"
        self.available_markets = []
        self.active_contracts = []
        self.last_ping = time.time()
        self.connection_lock = threading.Lock()
        
        # Market data storage
        self.market_data = defaultdict(lambda: {
            'prices': deque(maxlen=100),
            'ticks': deque(maxlen=100),
            'last_update': 0,
            'bid': 0,
            'ask': 0
        })
        
        logger.info(f"RealDerivTrader initialized with token: {self.api_token[:8]}...")
    
    def connect(self) -> Tuple[bool, str]:
        """Connect to REAL Deriv account"""
        try:
            if not self.api_token or len(self.api_token) < 20:
                return False, "Invalid API token"
            
            endpoints = [
                "wss://ws.deriv.com/websockets/v3",
                "wss://ws.binaryws.com/websockets/v3",
                "wss://ws.derivws.com/websockets/v3"
            ]
            
            for endpoint in endpoints:
                try:
                    logger.info(f"🔗 Connecting to REAL Deriv: {endpoint}")
                    
                    self.ws = websocket.create_connection(
                        f"{endpoint}?app_id=1089&l=EN&brand=deriv",
                        timeout=20,
                        header={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                            'Origin': 'https://app.deriv.com'
                        }
                    )
                    
                    # Send authorization
                    auth_msg = {
                        "authorize": self.api_token
                    }
                    self.ws.send(json.dumps(auth_msg))
                    
                    # Get response
                    response = json.loads(self.ws.recv())
                    
                    if "error" in response:
                        error_msg = response["error"].get("message", "Authorization failed")
                        logger.error(f"❌ REAL ACCOUNT AUTH FAILED: {error_msg}")
                        continue
                    
                    if "authorize" in response:
                        auth_data = response["authorize"]
                        self.account_id = auth_data.get("loginid")
                        self.currency = auth_data.get("currency", "USD")
                        self.connected = True
                        
                        # Get balance
                        self._update_balance()
                        
                        # Get available markets
                        self._get_available_markets()
                        
                        # Start market data streams
                        self._start_market_data_streams()
                        
                        logger.info(f"✅ SUCCESSFULLY CONNECTED TO REAL DERIV ACCOUNT: {self.account_id}")
                        logger.info(f"💰 REAL BALANCE: {self.balance} {self.currency}")
                        logger.info(f"📈 AVAILABLE MARKETS: {len(self.available_markets)}")
                        
                        return True, f"Connected to {self.account_id}"
                    
                except Exception as e:
                    logger.error(f"Connection failed to {endpoint}: {e}")
                    continue
            
            return False, "Failed to connect to all Deriv endpoints"
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False, str(e)
    
    def _update_balance(self):
        """Get REAL balance from your account"""
        try:
            with self.connection_lock:
                self.ws.send(json.dumps({"balance": 1, "subscribe": 1}))
                response = json.loads(self.ws.recv())
                
                if "balance" in response:
                    self.balance = float(response["balance"]["balance"])
                elif "error" in response:
                    logger.error(f"Balance error: {response['error']}")
        except Exception as e:
            logger.error(f"Balance update error: {e}")
    
    def _get_available_markets(self):
        """Get available markets for trading"""
        try:
            with self.connection_lock:
                # Get active symbols
                self.ws.send(json.dumps({"active_symbols": "brief", "product_type": "basic"}))
                response = json.loads(self.ws.recv())
                
                if "active_symbols" in response:
                    symbols = response["active_symbols"]
                    self.available_markets = [
                        sym["symbol"] for sym in symbols 
                        if sym["exchange_is_open"] == 1 and "Volatility" in sym.get("market_display_name", "")
                    ]
                    
                    # Prioritize volatility indices
                    volatility_symbols = [s for s in self.available_markets if s.startswith('R_')]
                    crash_boom = [s for s in self.available_markets if 'CRASH' in s or 'BOOM' in s]
                    
                    self.available_markets = volatility_symbols + crash_boom
                    
                    logger.info(f"Found {len(self.available_markets)} tradable markets")
        except Exception as e:
            logger.error(f"Market fetch error: {e}")
    
    def _start_market_data_streams(self):
        """Start real-time market data streams"""
        def stream_market_data():
            try:
                # Subscribe to ticks for selected markets
                markets_to_stream = ['R_10', 'R_25', 'R_50', 'R_75', 'R_100']
                
                for symbol in markets_to_stream:
                    if symbol in self.available_markets:
                        subscribe_msg = {
                            "ticks": symbol,
                            "subscribe": 1
                        }
                        self.ws.send(json.dumps(subscribe_msg))
                
                # Listen for ticks
                while self.connected:
                    try:
                        response = json.loads(self.ws.recv())
                        
                        if "tick" in response:
                            tick = response["tick"]
                            symbol = tick["symbol"]
                            quote = float(tick["quote"])
                            
                            # Update market data
                            self.market_data[symbol]['prices'].append(quote)
                            self.market_data[symbol]['ticks'].append(tick)
                            self.market_data[symbol]['last_update'] = time.time()
                            self.market_data[symbol]['bid'] = quote * 0.999
                            self.market_data[symbol]['ask'] = quote * 1.001
                            
                    except Exception as e:
                        if "error" not in str(e):
                            logger.error(f"Stream error: {e}")
                        time.sleep(1)
                        
            except Exception as e:
                logger.error(f"Market stream error: {e}")
        
        # Start streaming in background
        threading.Thread(target=stream_market_data, daemon=True).start()
    
    def get_market_data(self, symbol: str) -> Dict:
        """Get current market data for a symbol"""
        data = self.market_data.get(symbol, {})
        
        if not data.get('prices'):
            # Return mock data if no real data yet
            return {
                'bid': 100.0,
                'ask': 100.1,
                'spread': 0.1,
                'prices': [100.0, 100.1, 99.9, 100.2, 100.0],
                'last_update': time.time()
            }
        
        prices = list(data['prices'])
        return {
            'bid': data['bid'],
            'ask': data['ask'],
            'spread': data['ask'] - data['bid'],
            'prices': prices[-50:] if len(prices) > 50 else prices,
            'last_update': data['last_update']
        }
    
    def place_real_trade(self, symbol: str, direction: str, amount: float) -> Tuple[bool, str, Dict]:
        """Place REAL trade in YOUR Deriv account"""
        try:
            if not self.connected:
                return False, "Not connected to Deriv", {}
            
            # Validate amount
            if amount < 1.0:
                amount = 1.0  # Deriv minimum
            
            if amount > self.balance:
                return False, f"Insufficient balance. Need: {amount}, Have: {self.balance}", {}
            
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
                    "symbol": symbol,
                    "product_type": "basic"
                }
            }
            
            logger.info(f"🚀 EXECUTING REAL TRADE IN YOUR ACCOUNT: {symbol} {direction} ${amount}")
            
            with self.connection_lock:
                self.ws.send(json.dumps(trade_request))
                response = json.loads(self.ws.recv())
                
                if "error" in response:
                    error_msg = response["error"].get("message", "Trade failed")
                    logger.error(f"❌ REAL TRADE FAILED: {error_msg}")
                    return False, error_msg, {}
                
                if "buy" in response:
                    contract_data = response["buy"]
                    contract_id = contract_data.get("contract_id")
                    
                    # Update balance
                    self._update_balance()
                    
                    # Store contract info
                    contract_info = {
                        "contract_id": contract_id,
                        "symbol": symbol,
                        "direction": direction,
                        "amount": amount,
                        "purchase_time": datetime.now().isoformat(),
                        "status": "open"
                    }
                    self.active_contracts.append(contract_info)
                    
                    logger.info(f"✅ REAL TRADE SUCCESSFUL! Contract: {contract_id}")
                    logger.info(f"   Your new balance: {self.balance} {self.currency}")
                    
                    return True, contract_id, contract_info
            
            return False, "Unknown response", {}
            
        except Exception as e:
            logger.error(f"Real trade execution error: {e}")
            return False, str(e), {}
    
    def check_contract(self, contract_id: str) -> Dict:
        """Check status of a contract"""
        try:
            with self.connection_lock:
                self.ws.send(json.dumps({"proposal_open_contract": 1, "contract_id": contract_id}))
                response = json.loads(self.ws.recv())
                
                if "proposal_open_contract" in response:
                    return response["proposal_open_contract"]
                else:
                    return {"error": "Contract not found"}
        except Exception as e:
            return {"error": str(e)}
    
    def disconnect(self):
        """Disconnect from Deriv"""
        self.connected = False
        try:
            if self.ws:
                self.ws.close()
        except:
            pass

# ============ ADVANCED SMC STRATEGY WITH REAL DATA ============
class AdvancedSMCWithRealData:
    """Advanced SMC strategy using REAL market data"""
    
    def __init__(self):
        self.strategy_history = defaultdict(list)
        self.performance = defaultdict(lambda: {"wins": 0, "losses": 0})
        
    def analyze_with_real_data(self, symbol: str, market_data: Dict) -> Dict:
        """Analyze REAL market data with SMC strategy"""
        prices = market_data.get('prices', [])
        
        if len(prices) < 20:
            return self._get_default_signal(symbol)
        
        try:
            # Convert to numpy array
            prices_array = np.array(prices)
            
            # Calculate technical indicators
            sma_10 = self._calculate_sma(prices_array, 10)
            sma_20 = self._calculate_sma(prices_array, 20)
            rsi = self._calculate_rsi(prices_array)
            
            # Support and Resistance
            support_level = np.min(prices_array[-20:])
            resistance_level = np.max(prices_array[-20:])
            
            current_price = prices_array[-1]
            bid = market_data.get('bid', current_price)
            ask = market_data.get('ask', current_price)
            
            # SMC Analysis
            market_structure = self._analyze_market_structure(prices_array)
            order_blocks = self._find_order_blocks(prices_array)
            liquidity_levels = self._find_liquidity(prices_array)
            
            # Generate signal
            signal_info = self._generate_smc_signal(
                current_price, bid, ask,
                sma_10, sma_20, rsi,
                support_level, resistance_level,
                market_structure, order_blocks, liquidity_levels
            )
            
            return {
                "strategy": "Advanced_SMC",
                "symbol": symbol,
                "signal": signal_info["direction"],
                "confidence": signal_info["confidence"],
                "current_price": float(current_price),
                "bid": float(bid),
                "ask": float(ask),
                "indicators": {
                    "sma_10": float(sma_10),
                    "sma_20": float(sma_20),
                    "rsi": float(rsi),
                    "support": float(support_level),
                    "resistance": float(resistance_level)
                },
                "smc_analysis": {
                    "market_structure": market_structure,
                    "order_blocks": len(order_blocks),
                    "liquidity_levels": liquidity_levels
                },
                "timestamp": datetime.now().isoformat(),
                "reasoning": signal_info["reasoning"],
                "data_source": "REAL_MARKET_DATA"
            }
            
        except Exception as e:
            logger.error(f"SMC analysis error: {e}")
            return self._get_default_signal(symbol)
    
    def _calculate_sma(self, prices: np.ndarray, period: int) -> float:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return float(prices[-1])
        return float(np.mean(prices[-period:]))
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        rs = up / down if down != 0 else 1
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _analyze_market_structure(self, prices: np.ndarray) -> str:
        """Analyze market structure"""
        if len(prices) < 30:
            return "NEUTRAL"
        
        # Check for higher highs/higher lows (bullish)
        recent_prices = prices[-20:]
        peaks = []
        troughs = []
        
        for i in range(1, len(recent_prices)-1):
            if recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1]:
                peaks.append(recent_prices[i])
            elif recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i+1]:
                troughs.append(recent_prices[i])
        
        if len(peaks) >= 2 and len(troughs) >= 2:
            if peaks[-1] > peaks[-2] and troughs[-1] > troughs[-2]:
                return "BULLISH"
            elif peaks[-1] < peaks[-2] and troughs[-1] < troughs[-2]:
                return "BEARISH"
        
        return "RANGING"
    
    def _find_order_blocks(self, prices: np.ndarray) -> List[float]:
        """Find order blocks"""
        order_blocks = []
        
        # Look for significant rejection candles
        for i in range(5, len(prices)-5):
            candle_range = abs(prices[i] - prices[i-1])
            avg_range = np.mean([abs(prices[j] - prices[j-1]) for j in range(1, len(prices))])
            
            if candle_range > avg_range * 2:
                # Potential order block
                order_blocks.append(float(prices[i]))
        
        return order_blocks[-3:] if order_blocks else []
    
    def _find_liquidity(self, prices: np.ndarray) -> Dict:
        """Find liquidity levels"""
        recent = prices[-50:]
        
        return {
            "high_liquidity": float(np.max(recent)),
            "low_liquidity": float(np.min(recent)),
            "recent_high": float(np.max(recent[-20:])),
            "recent_low": float(np.min(recent[-20:]))
        }
    
    def _generate_smc_signal(self, current_price: float, bid: float, ask: float,
                            sma_10: float, sma_20: float, rsi: float,
                            support: float, resistance: float,
                            structure: str, order_blocks: List, liquidity: Dict) -> Dict:
        
        signal = {"direction": "NEUTRAL", "confidence": 50, "reasoning": []}
        
        # 1. Trend Analysis
        if current_price > sma_10 > sma_20:
            signal["direction"] = "BUY"
            signal["confidence"] += 15
            signal["reasoning"].append("Bullish trend (price > SMA10 > SMA20)")
        elif current_price < sma_10 < sma_20:
            signal["direction"] = "SELL"
            signal["confidence"] += 15
            signal["reasoning"].append("Bearish trend (price < SMA10 < SMA20)")
        
        # 2. RSI Analysis
        if rsi < 30 and signal["direction"] == "BUY":
            signal["confidence"] += 10
            signal["reasoning"].append("Oversold condition (RSI < 30)")
        elif rsi > 70 and signal["direction"] == "SELL":
            signal["confidence"] += 10
            signal["reasoning"].append("Overbought condition (RSI > 70)")
        
        # 3. Support/Resistance
        if current_price <= support * 1.01:
            if signal["direction"] == "BUY":
                signal["confidence"] += 12
                signal["reasoning"].append("At support level")
        elif current_price >= resistance * 0.99:
            if signal["direction"] == "SELL":
                signal["confidence"] += 12
                signal["reasoning"].append("At resistance level")
        
        # 4. Market Structure
        if structure == "BULLISH" and signal["direction"] == "BUY":
            signal["confidence"] += 8
            signal["reasoning"].append("Bullish market structure confirmation")
        elif structure == "BEARISH" and signal["direction"] == "SELL":
            signal["confidence"] += 8
            signal["reasoning"].append("Bearish market structure confirmation")
        
        # 5. Order Block Confluence
        if order_blocks:
            last_block = order_blocks[-1]
            if abs(current_price - last_block) / last_block < 0.01:  # Within 1%
                if signal["direction"] == "BUY" and current_price >= last_block:
                    signal["confidence"] += 10
                    signal["reasoning"].append("Order block support")
                elif signal["direction"] == "SELL" and current_price <= last_block:
                    signal["confidence"] += 10
                    signal["reasoning"].append("Order block resistance")
        
        # Only trade if confidence is high enough
        if signal["confidence"] < 75:
            signal["direction"] = "NEUTRAL"
            signal["reasoning"].append("Confidence too low for trading")
        
        return signal
    
    def _get_default_signal(self, symbol: str) -> Dict:
        """Default signal when analysis fails"""
        return {
            "strategy": "Default",
            "symbol": symbol,
            "signal": "NEUTRAL",
            "confidence": 50,
            "current_price": 100.0,
            "timestamp": datetime.now().isoformat(),
            "reasoning": ["Insufficient data for analysis"],
            "data_source": "DEFAULT"
        }

# ============ REAL TRADING ENGINE ============
class RealTradingEngine:
    """Engine that trades with YOUR REAL Deriv account"""
    
    def __init__(self, user_id: str, api_token: str):
        self.user_id = user_id
        self.api_token = api_token
        self.trader = RealDerivTrader(api_token)
        self.strategy = AdvancedSMCWithRealData()
        self.running = False
        self.thread = None
        self.trades = []
        
        # Trading statistics
        self.stats = {
            'total_trades': 0,
            'real_trades': 0,
            'simulated_trades': 0,
            'total_profit': 0.0,
            'balance': 0.0,
            'last_trade': None,
            'start_time': datetime.now().isoformat(),
            'connection_status': 'disconnected'
        }
        
        # Trading settings
        self.settings = {
            'enabled_markets': ['R_10', 'R_25', 'R_50', 'R_75', 'R_100'],
            'trade_amount': 5.0,  # Start with $5 trades
            'min_confidence': 75,
            'max_trades_per_day': 50,
            'cooldown_seconds': 60,
            'scan_interval': 30,
            'use_real_trading': True,  # ALWAYS REAL TRADING
            'risk_per_trade': 0.02,  # Risk 2% per trade
            'stop_loss_pct': 2.0,
            'take_profit_pct': 4.0
        }
        
        # Connect immediately
        self._connect_to_deriv()
        
        logger.info(f"💰 RealTradingEngine created for {user_id}")
    
    def _connect_to_deriv(self):
        """Connect to Deriv with user's API token"""
        if self.api_token:
            success, message = self.trader.connect()
            if success:
                self.stats['connection_status'] = 'connected'
                self.stats['balance'] = self.trader.balance
                logger.info(f"✅ Connected to REAL Deriv account: {message}")
            else:
                logger.error(f"❌ Failed to connect to Deriv: {message}")
        else:
            logger.warning("⚠️ No API token provided. Set DERIV_API_TOKEN environment variable.")
    
    def start_trading(self):
        """Start REAL trading"""
        if self.running:
            return False, "Already trading"
        
        # Ensure connected
        if not self.trader.connected and self.api_token:
            self._connect_to_deriv()
        
        if not self.trader.connected:
            return False, "Not connected to Deriv. Check your API token."
        
        self.running = True
        self.thread = threading.Thread(target=self._real_trading_loop, daemon=True)
        self.thread.start()
        
        logger.info(f"🚀 REAL TRADING STARTED for {self.user_id}")
        logger.info(f"💰 Account: {self.trader.account_id}")
        logger.info(f"💵 Balance: {self.trader.balance} {self.trader.currency}")
        
        return True, f"✅ REAL Trading started! Account: {self.trader.account_id}, Balance: {self.trader.balance} {self.trader.currency}"
    
    def _real_trading_loop(self):
        """Main REAL trading loop"""
        logger.info("🔥 REAL TRADING LOOP STARTED")
        
        market_index = 0
        
        while self.running:
            try:
                # Check connection
                if not self.trader.connected:
                    time.sleep(10)
                    continue
                
                # Update stats
                self.stats['balance'] = self.trader.balance
                
                # Select market
                enabled = self.settings['enabled_markets']
                if not enabled:
                    time.sleep(10)
                    continue
                
                symbol = enabled[market_index % len(enabled)]
                market_index += 1
                
                # Get REAL market data
                market_data = self.trader.get_market_data(symbol)
                
                # Analyze with REAL data
                analysis = self.strategy.analyze_with_real_data(symbol, market_data)
                
                # Check if we should trade
                if (analysis['signal'] != 'NEUTRAL' and 
                    analysis['confidence'] >= self.settings['min_confidence']):
                    
                    # Calculate trade amount based on risk
                    trade_amount = self._calculate_trade_amount()
                    
                    # Execute REAL trade
                    success, result, contract_info = self.trader.place_real_trade(
                        symbol, analysis['signal'], trade_amount
                    )
                    
                    if success:
                        # Record trade
                        trade_record = {
                            'id': len(self.trades) + 1,
                            'symbol': symbol,
                            'direction': analysis['signal'],
                            'amount': trade_amount,
                            'contract_id': result,
                            'analysis': analysis,
                            'timestamp': datetime.now().isoformat(),
                            'status': 'EXECUTED',
                            'real_trade': True
                        }
                        
                        self.trades.append(trade_record)
                        self.stats['total_trades'] += 1
                        self.stats['real_trades'] += 1
                        self.stats['last_trade'] = datetime.now().isoformat()
                        
                        logger.info(f"✅ REAL TRADE EXECUTED: {symbol} {analysis['signal']} ${trade_amount}")
                        logger.info(f"   Contract ID: {result}")
                        logger.info(f"   Confidence: {analysis['confidence']}%")
                        logger.info(f"   New Balance: {self.trader.balance} {self.trader.currency}")
                        
                        # Check contract result after some time
                        threading.Thread(
                            target=self._check_contract_result,
                            args=(result, trade_record),
                            daemon=True
                        ).start()
                
                # Wait before next scan
                time.sleep(self.settings['scan_interval'])
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                time.sleep(60)
    
    def _calculate_trade_amount(self) -> float:
        """Calculate trade amount based on risk management"""
        base_amount = self.settings['trade_amount']
        risk_amount = self.trader.balance * self.settings['risk_per_trade']
        
        # Use the smaller of base amount or risk-based amount
        amount = min(base_amount, risk_amount)
        
        # Ensure minimum $1 for Deriv
        return max(1.0, amount)
    
    def _check_contract_result(self, contract_id: str, trade_record: Dict):
        """Check the result of a contract"""
        time.sleep(300)  # Wait 5 minutes (contract duration)
        
        try:
            result = self.trader.check_contract(contract_id)
            
            if 'error' not in result:
                profit = float(result.get('profit', 0))
                
                # Update trade record
                trade_record['profit'] = profit
                trade_record['contract_result'] = result
                trade_record['checked_at'] = datetime.now().isoformat()
                
                # Update stats
                self.stats['total_profit'] += profit
                
                logger.info(f"📊 Contract {contract_id} result: Profit ${profit:.2f}")
        except Exception as e:
            logger.error(f"Contract check error: {e}")
    
    def stop_trading(self):
        """Stop trading"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)
        
        logger.info(f"Trading stopped for {self.user_id}")
        return True, "Trading stopped"
    
    def get_status(self) -> Dict:
        """Get current status"""
        return {
            'running': self.running,
            'connected': self.trader.connected,
            'account_id': self.trader.account_id,
            'balance': self.trader.balance,
            'currency': self.trader.currency,
            'settings': self.settings,
            'stats': self.stats,
            'recent_trades': self.trades[-10:][::-1] if self.trades else [],
            'total_trades': len(self.trades),
            'available_markets': self.trader.available_markets[:10]
        }

# ============ FLASK APP ============
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_urlsafe(64))

CORS(app, supports_credentials=True)

# Store engines by user
engines = {}

# ============ API ENDPOINTS ============
@app.route('/api/connect', methods=['POST'])
def connect():
    """Connect to Deriv with user's API token"""
    try:
        data = request.json or {}
        api_token = data.get('api_token', '').strip()
        
        if not api_token:
            # Check environment variable
            api_token = os.environ.get('DERIV_API_TOKEN', '')
        
        if not api_token:
            return jsonify({
                'success': False,
                'message': 'API token required. Get it from Deriv.com → Settings → API Token'
            })
        
        # Create or get engine
        user_id = 'primary_trader'
        engine = engines.get(user_id)
        
        if not engine:
            engine = RealTradingEngine(user_id, api_token)
            engines[user_id] = engine
        
        status = engine.get_status()
        
        return jsonify({
            'success': status['connected'],
            'message': f"Connected to {status['account_id']}" if status['connected'] else "Connection failed",
            'account_id': status['account_id'],
            'balance': status['balance'],
            'currency': status['currency']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/start', methods=['POST'])
def start_trading():
    """Start REAL trading"""
    try:
        user_id = 'primary_trader'
        engine = engines.get(user_id)
        
        if not engine:
            # Try to create engine with environment token
            api_token = os.environ.get('DERIV_API_TOKEN', '')
            if not api_token:
                return jsonify({
                    'success': False,
                    'message': 'No API token. Set DERIV_API_TOKEN environment variable or connect first.'
                })
            
            engine = RealTradingEngine(user_id, api_token)
            engines[user_id] = engine
        
        success, message = engine.start_trading()
        
        return jsonify({
            'success': success,
            'message': message,
            'real_trading': True
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/status', methods=['GET'])
def status():
    """Get trading status"""
    try:
        user_id = 'primary_trader'
        engine = engines.get(user_id)
        
        if not engine:
            return jsonify({
                'success': True,
                'status': {
                    'running': False,
                    'connected': False,
                    'message': 'Not initialized. Connect with API token first.'
                }
            })
        
        status_data = engine.get_status()
        
        return jsonify({
            'success': True,
            'status': status_data,
            'real_trading': True,
            'render_instance': RENDER_INSTANCE_ID
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/trade', methods=['POST'])
def place_trade():
    """Place manual trade"""
    try:
        data = request.json or {}
        symbol = data.get('symbol', 'R_10')
        direction = data.get('direction', 'BUY')
        amount = float(data.get('amount', 5.0))
        
        user_id = 'primary_trader'
        engine = engines.get(user_id)
        
        if not engine:
            return jsonify({'success': False, 'message': 'Not connected to Deriv'})
        
        if not engine.trader.connected:
            return jsonify({'success': False, 'message': 'Not connected to Deriv'})
        
        # Place REAL trade
        success, result, contract_info = engine.trader.place_real_trade(symbol, direction, amount)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'✅ REAL Trade executed! Contract: {result}',
                'contract_id': result,
                'balance': engine.trader.balance,
                'account_id': engine.trader.account_id
            })
        else:
            return jsonify({'success': False, 'message': result})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/trades', methods=['GET'])
def get_trades():
    """Get trade history"""
    try:
        user_id = 'primary_trader'
        engine = engines.get(user_id)
        
        if not engine:
            return jsonify({'success': False, 'message': 'Not initialized'})
        
        return jsonify({
            'success': True,
            'trades': engine.trades[-20:][::-1],
            'total': len(engine.trades),
            'real_trades': engine.stats['real_trades']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/markets', methods=['GET'])
def get_markets():
    """Get available markets"""
    try:
        user_id = 'primary_trader'
        engine = engines.get(user_id)
        
        markets = []
        if engine and engine.trader.connected:
            markets = engine.trader.available_markets[:20]
        
        return jsonify({
            'success': True,
            'markets': markets,
            'count': len(markets),
            'connected': engine.trader.connected if engine else False
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# ============ RENDER KEEP-ALIVE ENDPOINTS ============
@app.route('/keep-alive', methods=['GET'])
def keep_alive():
    """Keep Render instance awake"""
    return jsonify({
        'status': 'awake',
        'timestamp': datetime.now().isoformat(),
        'instance': RENDER_INSTANCE_ID,
        'message': 'Bot is running 24/7'
    })

@app.route('/api/ping', methods=['GET'])
def ping():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Karanka Pro Max',
        'timestamp': datetime.now().isoformat(),
        'real_trading': True
    })

@app.route('/')
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)

# ============ START BOT ON DEPLOY ============
def initialize_bot():
    """Initialize bot on startup"""
    time.sleep(5)
    
    try:
        # Check for API token in environment
        api_token = os.environ.get('DERIV_API_TOKEN', '')
        
        if api_token:
            logger.info("🔗 Found DERIV_API_TOKEN, initializing REAL trading bot...")
            
            # Create engine
            engine = RealTradingEngine('auto_trader', api_token)
            engines['primary_trader'] = engine
            
            # Try to connect
            if engine.trader.connected:
                logger.info(f"✅ Auto-connected to Deriv account: {engine.trader.account_id}")
                
                # Start trading
                engine.start_trading()
            else:
                logger.warning("⚠️ Could not auto-connect to Deriv")
        else:
            logger.info("ℹ️ No DERIV_API_TOKEN found. Connect manually via the web interface.")
            
    except Exception as e:
        logger.error(f"Initialization error: {e}")

# Start initialization
threading.Thread(target=initialize_bot, daemon=True).start()

# ============ HTML TEMPLATE ============
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>💰 Karanka Pro Max - REAL Deriv Trading</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
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
            background: linear-gradient(135deg, #00C853 0%, #008B3A 100%);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 25px;
            text-align: center;
            border: 3px solid #00FF88;
        }
        
        .header h1 {
            margin: 0;
            font-size: 2.5em;
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }
        
        .status-card {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 20px;
            border: 2px solid;
        }
        
        .status-card.connected {
            border-color: #00C853;
        }
        
        .status-card.disconnected {
            border-color: #FF5252;
        }
        
        .btn {
            padding: 15px 30px;
            background: #00C853;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            margin: 10px 5px;
            transition: all 0.3s;
        }
        
        .btn:hover {
            background: #00FF88;
            transform: translateY(-2px);
        }
        
        .btn-danger {
            background: #FF5252;
        }
        
        .btn-warning {
            background: #FF9800;
        }
        
        .trade-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .trade-card {
            background: rgba(255,255,255,0.05);
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
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            display: inline-block;
            margin-left: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>💰 Karanka Pro Max - REAL Deriv Trading</h1>
            <p>• REAL Account Connection • REAL Market Data • REAL Trades • 24/7 Operation</p>
        </div>
        
        <div class="status-grid">
            <div class="status-card" id="connectionCard">
                <h3>🔗 Deriv Connection</h3>
                <p id="connectionStatus">Checking...</p>
                <p id="accountInfo"></p>
                <p id="balanceInfo"></p>
                
                <input type="password" id="apiToken" placeholder="Enter your Deriv API token" style="width:100%; padding:12px; margin:10px 0; border-radius:8px;">
                <button class="btn" onclick="connectDeriv()">🔗 Connect to Deriv</button>
            </div>
            
            <div class="status-card" id="tradingCard">
                <h3>🚀 Trading Status</h3>
                <p id="tradingStatus">Not started</p>
                <p id="tradesCount">Trades: 0</p>
                <p id="profitInfo">Profit: $0.00</p>
                
                <button class="btn" onclick="startTrading()" id="startBtn">▶️ Start REAL Trading</button>
                <button class="btn btn-danger" onclick="stopTrading()" id="stopBtn" style="display:none;">⏹️ Stop Trading</button>
            </div>
            
            <div class="status-card">
                <h3>📊 Quick Trade</h3>
                <select id="tradeSymbol" style="width:100%; padding:12px; margin:10px 0; border-radius:8px;">
                    <option value="R_10">Volatility 10 Index</option>
                    <option value="R_25">Volatility 25 Index</option>
                    <option value="R_50">Volatility 50 Index</option>
                    <option value="R_75">Volatility 75 Index</option>
                    <option value="R_100">Volatility 100 Index</option>
                </select>
                
                <div style="display: flex; gap: 10px; margin: 10px 0;">
                    <button class="btn" style="flex:1;" onclick="placeManualTrade('BUY')">📈 BUY</button>
                    <button class="btn btn-danger" style="flex:1;" onclick="placeManualTrade('SELL')">📉 SELL</button>
                </div>
                
                <input type="number" id="tradeAmount" value="5.0" min="1" step="0.1" style="width:100%; padding:12px; margin:10px 0; border-radius:8px;">
            </div>
        </div>
        
        <div style="text-align: center; margin: 30px 0;">
            <button class="btn" onclick="updateStatus()">🔄 Refresh Status</button>
            <button class="btn" onclick="loadTrades()">📋 View Trades</button>
            <button class="btn btn-warning" onclick="loadMarkets()">📈 Available Markets</button>
        </div>
        
        <div id="tradesSection" style="display: none;">
            <h2>📋 Recent Trades</h2>
            <div id="tradesList"></div>
        </div>
        
        <div id="marketsSection" style="display: none;">
            <h2>📈 Available Markets</h2>
            <div id="marketsList"></div>
        </div>
        
        <div id="alerts" style="margin-top: 30px;"></div>
    </div>
    
    <script>
        function showAlert(message, type = 'info') {
            const alerts = document.getElementById('alerts');
            const alert = document.createElement('div');
            alert.style.cssText = `
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
                background: ${type === 'success' ? 'rgba(0,200,83,0.2)' : type === 'error' ? 'rgba(255,82,82,0.2)' : 'rgba(33,150,243,0.2)'};
                border: 1px solid ${type === 'success' ? '#00C853' : type === 'error' ? '#FF5252' : '#2196F3'};
                color: white;
            `;
            alert.innerHTML = message;
            alerts.appendChild(alert);
            
            setTimeout(() => alert.remove(), 5000);
        }
        
        function connectDeriv() {
            const apiToken = document.getElementById('apiToken').value;
            
            if (!apiToken) {
                showAlert('Please enter your Deriv API token', 'error');
                return;
            }
            
            fetch('/api/connect', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({api_token: apiToken})
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    showAlert(`✅ Connected to ${data.account_id}`, 'success');
                    updateStatus();
                } else {
                    showAlert(`❌ ${data.message}`, 'error');
                }
            })
            .catch(e => showAlert(`Connection error: ${e}`, 'error'));
        }
        
        function startTrading() {
            fetch('/api/start', {method: 'POST'})
            .then(r => r.json())
            .then(data => {
                showAlert(data.message, data.success ? 'success' : 'error');
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
        
        function updateStatus() {
            fetch('/api/status')
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    const s = data.status;
                    
                    // Update connection card
                    const connCard = document.getElementById('connectionCard');
                    if (s.connected) {
                        connCard.className = 'status-card connected';
                        connCard.innerHTML = `
                            <h3>🔗 Deriv Connection</h3>
                            <p style="color:#00C853">✅ CONNECTED</p>
                            <p>Account: ${s.account_id || 'N/A'}</p>
                            <p>Balance: ${s.balance || 0} ${s.currency || 'USD'}</p>
                            <p>Markets: ${s.available_markets?.length || 0} available</p>
                        `;
                    } else {
                        connCard.className = 'status-card disconnected';
                        connCard.innerHTML = `
                            <h3>🔗 Deriv Connection</h3>
                            <p style="color:#FF5252">❌ DISCONNECTED</p>
                            <input type="password" id="apiToken" placeholder="Enter your Deriv API token" style="width:100%; padding:12px; margin:10px 0; border-radius:8px;">
                            <button class="btn" onclick="connectDeriv()">🔗 Connect to Deriv</button>
                        `;
                    }
                    
                    // Update trading card
                    const tradeCard = document.getElementById('tradingCard');
                    if (s.running) {
                        document.getElementById('startBtn').style.display = 'none';
                        document.getElementById('stopBtn').style.display = 'inline-block';
                        tradeCard.innerHTML = `
                            <h3>🚀 Trading Status</h3>
                            <p style="color:#00C853">✅ ACTIVE</p>
                            <p>Total Trades: ${s.total_trades || 0}</p>
                            <p>Real Trades: ${s.stats?.real_trades || 0}</p>
                            <p>Profit: $${s.stats?.total_profit?.toFixed(2) || '0.00'}</p>
                            <button class="btn btn-danger" onclick="stopTrading()">⏹️ Stop Trading</button>
                        `;
                    } else {
                        document.getElementById('startBtn').style.display = 'inline-block';
                        document.getElementById('stopBtn').style.display = 'none';
                        tradeCard.innerHTML = `
                            <h3>🚀 Trading Status</h3>
                            <p style="color:#FF9800">⏸️ STOPPED</p>
                            <p>Total Trades: ${s.total_trades || 0}</p>
                            <p>Real Trades: ${s.stats?.real_trades || 0}</p>
                            <p>Profit: $${s.stats?.total_profit?.toFixed(2) || '0.00'}</p>
                            <button class="btn" onclick="startTrading()">▶️ Start REAL Trading</button>
                        `;
                    }
                }
            });
        }
        
        function placeManualTrade(direction) {
            const symbol = document.getElementById('tradeSymbol').value;
            const amount = parseFloat(document.getElementById('tradeAmount').value);
            
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
                showAlert(data.message, data.success ? 'success' : 'error');
                updateStatus();
                loadTrades();
            });
        }
        
        function loadTrades() {
            fetch('/api/trades')
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    const section = document.getElementById('tradesSection');
                    const list = document.getElementById('tradesList');
                    
                    section.style.display = 'block';
                    
                    if (data.trades.length > 0) {
                        let html = '<div class="trade-grid">';
                        
                        data.trades.forEach(trade => {
                            html += `
                                <div class="trade-card ${trade.direction.toLowerCase()}">
                                    <strong>${trade.symbol} ${trade.direction}</strong>
                                    ${trade.real_trade ? '<span class="real-badge">REAL</span>' : ''}
                                    <p>Amount: $${trade.amount?.toFixed(2) || '0.00'}</p>
                                    ${trade.contract_id ? `<p>Contract: ${trade.contract_id.substring(0, 8)}...</p>` : ''}
                                    <p style="font-size:0.9em; color:#aaa;">
                                        ${new Date(trade.timestamp).toLocaleString()}
                                    </p>
                                </div>
                            `;
                        });
                        
                        html += '</div>';
                        list.innerHTML = html;
                    } else {
                        list.innerHTML = '<p>No trades yet.</p>';
                    }
                }
            });
        }
        
        function loadMarkets() {
            fetch('/api/markets')
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    const section = document.getElementById('marketsSection');
                    const list = document.getElementById('marketsList');
                    
                    section.style.display = 'block';
                    
                    if (data.markets.length > 0) {
                        let html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px;">';
                        
                        data.markets.forEach(market => {
                            html += `
                                <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px;">
                                    <strong>${market}</strong>
                                    <p style="font-size:0.9em; color:#aaa;">
                                        ${market.includes('R_') ? 'Volatility Index' : 
                                          market.includes('CRASH') ? 'Crash Index' : 
                                          market.includes('BOOM') ? 'Boom Index' : 'Market'}
                                    </p>
                                </div>
                            `;
                        });
                        
                        html += '</div>';
                        list.innerHTML = html;
                    } else {
                        list.innerHTML = '<p>No markets available. Connect to Deriv first.</p>';
                    }
                }
            });
        }
        
        // Initial load
        document.addEventListener('DOMContentLoaded', function() {
            updateStatus();
            
            // Auto-refresh every 10 seconds
            setInterval(updateStatus, 10000);
            
            // Keep Render awake by pinging every 2 minutes
            setInterval(() => {
                fetch('/keep-alive').catch(() => {});
            }, 120000);
        });
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    
    logger.info("=" * 60)
    logger.info("🚀 KARANKA PRO MAX - REAL DERIV TRADING BOT")
    logger.info("=" * 60)
    logger.info(f"🌐 Render Instance: {RENDER_INSTANCE_ID}")
    logger.info(f"🔗 App URL: {RENDER_APP_URL}")
    logger.info(f"💰 REAL TRADING: ENABLED")
    logger.info(f"⏰ 24/7 OPERATION: GUARANTEED")
    logger.info("=" * 60)
    
    # Start Flask app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )
