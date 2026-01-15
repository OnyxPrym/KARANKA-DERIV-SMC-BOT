#!/usr/bin/env python3
"""
================================================================================
🎯 KARANKA MULTIVERSE V7 - 24/7 DERIV SMC BOT (OPTIMIZED FOR DERIV)
================================================================================
• ENHANCED FOR DERIV VOLATILITY INDICES
• ATR-ADAPTIVE SL/TP (No tight SLs on synthetic spikes)
• SHORTER TP LOGIC (0.5R-1.2R optimal for Deriv)
• VOLATILITY REACTION PRIORITY (not institutional OB)
• MICRO-STRUCTURE DOMINANCE (HTF bias minimized)
================================================================================
"""

import sys
import os
import json
import time
import asyncio
import traceback
import warnings
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# ============ FASTAPI WEB FRAMEWORK ============
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form, Depends
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from pydantic import BaseModel
    import uvicorn
    import aiofiles
    import requests
    import websockets
    print("✅ Web dependencies loaded")
except ImportError as e:
    print(f"❌ Import error: {e}")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "fastapi", "uvicorn", "jinja2", "python-multipart",
                          "pandas", "numpy", "scipy", "aiofiles", "requests", "websockets"])
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form, Depends
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from pydantic import BaseModel
    import uvicorn
    import aiofiles
    import requests
    import websockets

# ============ FIXED: CREATE APP FIRST ============
app = FastAPI(title="Karanka Multiverse V7 - Deriv SMC Bot", 
              description="24/7 Professional SMC Bot for Deriv Volatility Indices")

# ============ FOLDERS & PATHS ============
def ensure_data_folder():
    """Create all necessary folders"""
    app_data_dir = os.path.join(os.getcwd(), "karanka_deriv_data")
    folders = ["logs", "settings", "cache", "market_data", "structure_analysis", 
               "trade_analysis", "backups", "strategies", "performance", "static", "templates"]
    
    for folder in folders:
        folder_path = os.path.join(app_data_dir, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    
    return app_data_dir

APP_DATA_DIR = ensure_data_folder()
SETTINGS_FILE = os.path.join(APP_DATA_DIR, "settings", "deriv_settings_v7.json")
TRADES_LOG_FILE = os.path.join(APP_DATA_DIR, "logs", "trades_log.txt")
STRUCTURE_LOG_FILE = os.path.join(APP_DATA_DIR, "logs", "structure_log.txt")
ANALYSIS_FILE = os.path.join(APP_DATA_DIR, "trade_analysis", "analysis.json")
PERFORMANCE_FILE = os.path.join(APP_DATA_DIR, "performance", "performance.json")

# Mount static files AFTER app is created
static_dir = os.path.join(APP_DATA_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Templates
templates_dir = os.path.join(APP_DATA_DIR, "templates")
os.makedirs(templates_dir, exist_ok=True)
templates = Jinja2Templates(directory=templates_dir)

# ============ DERIV API CONFIGURATION ============
class DerivAPI:
    """Deriv API integration for Volatility Indices"""
    
    def __init__(self):
        self.app_id = 1089
        self.url = "wss://ws.binaryws.com/websockets/v3"
        self.websocket = None
        self.connected = False
        self.token = None
        self.account_id = None
        self.balance = 0
        self.positions = []
        
    async def connect(self, token: str):
        """Connect to Deriv API"""
        try:
            self.token = token
            self.websocket = await websockets.connect(self.url)
            
            auth_request = {"authorize": token}
            await self.websocket.send(json.dumps(auth_request))
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if 'error' in data:
                print(f"❌ Deriv auth error: {data['error']['message']}")
                return False
            
            self.connected = True
            self.account_id = data['authorize']['loginid']
            self.balance = float(data['authorize']['balance'])
            
            print(f"✅ Connected to Deriv: Account {self.account_id}, Balance: {self.balance}")
            return True
            
        except Exception as e:
            print(f"❌ Deriv connection error: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Deriv"""
        if self.websocket:
            await self.websocket.close()
        self.connected = False
    
    async def get_candles(self, symbol: str, timeframe: str = "60", count: int = 100):
        """Get candle history for Deriv"""
        if not self.connected:
            # Return simulated data for demo
            return self.get_simulated_candles(symbol, count)
        
        try:
            resolution_map = {"M5": "60", "M15": "300", "H1": "3600", "H4": "14400"}
            resolution = resolution_map.get(timeframe, "60")
            
            request = {
                "ticks_history": symbol,
                "end": "latest",
                "count": count * 2,
                "style": "candles",
                "granularity": int(resolution)
            }
            
            await self.websocket.send(json.dumps(request))
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if 'error' in data:
                print(f"❌ Candles error: {data['error']['message']}")
                return self.get_simulated_candles(symbol, count)
            
            candles = data.get('candles', [])
            if candles:
                df_data = []
                for candle in candles[-count:]:
                    df_data.append({
                        'time': datetime.fromtimestamp(candle['epoch']),
                        'open': float(candle['open']),
                        'high': float(candle['high']),
                        'low': float(candle['low']),
                        'close': float(candle['close']),
                        'volume': 0
                    })
                return pd.DataFrame(df_data)
            
            return self.get_simulated_candles(symbol, count)
            
        except Exception as e:
            print(f"❌ Get candles error: {e}")
            return self.get_simulated_candles(symbol, count)
    
    def get_simulated_candles(self, symbol: str, count: int):
        """Generate realistic simulated candles for Deriv markets"""
        np.random.seed(int(time.time()))
        
        # Base price based on symbol
        if 'R_75' in symbol:
            base_price = 10000
            volatility = 0.0075
        elif 'R_100' in symbol:
            base_price = 10000
            volatility = 0.0100
        elif 'CRASH' in symbol or 'BOOM' in symbol:
            base_price = 1000
            volatility = 0.05
        else:
            base_price = 10000
            volatility = 0.005
        
        df_data = []
        current_price = base_price
        
        for i in range(count):
            time_point = datetime.now() - timedelta(minutes=5*(count-i))
            
            # Realistic price movement
            change = np.random.normal(0, volatility)
            current_price = current_price * (1 + change)
            
            # Generate OHLC
            open_price = current_price
            high_price = open_price * (1 + abs(np.random.normal(0, volatility/2)))
            low_price = open_price * (1 - abs(np.random.normal(0, volatility/2)))
            close_price = open_price * (1 + np.random.normal(0, volatility/3))
            
            # Ensure high > low
            high_price = max(open_price, close_price, high_price)
            low_price = min(open_price, close_price, low_price)
            
            df_data.append({
                'time': time_point,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.randint(100, 1000)
            })
        
        return pd.DataFrame(df_data)
    
    async def place_trade(self, symbol: str, direction: str, amount: float, 
                         duration: int = 300, duration_unit: str = "s"):
        """Place a trade on Deriv"""
        if not self.connected:
            # Simulate trade for demo
            return {
                'buy': {
                    'contract_id': f"SIM_{int(time.time())}",
                    'price': amount,
                    'payout': amount * 1.8
                }
            }
        
        try:
            contract_type = "CALL" if direction == "BUY" else "PUT"
            
            request = {
                "buy": 1,
                "subscribe": 1,
                "price": amount,
                "parameters": {
                    "amount": amount,
                    "basis": "stake",
                    "contract_type": contract_type,
                    "currency": "USD",
                    "duration": duration,
                    "duration_unit": duration_unit,
                    "symbol": symbol
                }
            }
            
            await self.websocket.send(json.dumps(request))
            response = await self.websocket.recv()
            return json.loads(response)
            
        except Exception as e:
            print(f"❌ Place trade error: {e}")
            return None

# ============ DERIV VOLATILITY INDICES CONFIG ============
DERIV_MARKETS = {
    "volatility_75": {
        "symbol": "R_75",
        "display_name": "Volatility 75 Index",
        "pip_size": 0.01,
        "digits": 2,
        "avg_daily_range": 0.75,
        "optimal_time": "ALL",
        "description": "75% annual volatility - Perfect for SMC strategies",
        "category": "Volatility"
    },
    "volatility_100": {
        "symbol": "R_100",
        "display_name": "Volatility 100 Index",
        "pip_size": 0.01,
        "digits": 2,
        "avg_daily_range": 1.00,
        "optimal_time": "ALL",
        "description": "100% annual volatility - Higher risk/reward",
        "category": "Volatility"
    },
    "crash_1000": {
        "symbol": "CRASH1000",
        "display_name": "Crash 1000 Index",
        "pip_size": 0.001,
        "digits": 3,
        "avg_daily_range": 5.0,
        "optimal_time": "HIGH_VOL",
        "description": "Crash indices - Fast moves, high volatility",
        "category": "Crash/Boom"
    },
    "boom_1000": {
        "symbol": "BOOM1000",
        "display_name": "Boom 1000 Index",
        "pip_size": 0.001,
        "digits": 3,
        "avg_daily_range": 5.0,
        "optimal_time": "HIGH_VOL",
        "description": "Boom indices - Trend continuation patterns",
        "category": "Crash/Boom"
    }
}

# ============ ENHANCED LOGGER ============
class EnhancedLogger:
    """Enhanced logging with performance tracking"""
    
    def __init__(self):
        self.performance_stats = self.load_performance()
    
    async def log_trade(self, action, symbol, direction, entry, sl, tp, amount, comment=""):
        """Log trade execution"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {action} | {symbol} {direction} | Entry: {entry:.5f} | SL: {sl:.5f} | TP: {tp:.5f} | Amount: ${amount:.2f} | {comment}\n"
        
        async with aiofiles.open(TRADES_LOG_FILE, 'a', encoding='utf-8') as f:
            await f.write(log_entry)
        
        print(f"📝 Trade: {log_entry.strip()}")
        return log_entry

# ============ ENHANCED DERIV-OPTIMIZED SMC STRATEGY ============
class DerivOptimizedSMCStrategy:
    """SMC strategy OPTIMIZED for Deriv Volatility Indices"""
    
    def __init__(self, symbol, market_config):
        self.symbol = symbol
        self.config = market_config
        self.pip_size = market_config['pip_size']
        
    def analyze_market_structure(self, df):
        """Enhanced structure analysis for Deriv"""
        if df is None or len(df) < 20:
            return {'valid': False, 'direction': 'NEUTRAL', 'confidence': 0}
        
        # 1. VOLATILITY REACTION (Primary for Deriv)
        volatility_score = self.analyze_volatility_reaction(df)
        
        # 2. MICRO-STRUCTURE (Not institutional OB)
        structure_score = self.analyze_micro_structure(df)
        
        # 3. LIQUIDITY SWEEP DETECTION
        liquidity_score = self.detect_liquidity_sweep(df)
        
        # 4. TREND STRENGTH (Short-term only)
        trend_score = self.analyze_trend_strength(df)
        
        # Combine scores (Volatility is PRIMARY for Deriv)
        total_score = (
            volatility_score * 0.40 +      # 40% weight to volatility
            structure_score * 0.30 +       # 30% to micro-structure
            liquidity_score * 0.20 +       # 20% to liquidity
            trend_score * 0.10             # 10% to trend
        )
        
        # Determine direction
        direction = self.determine_direction(df, volatility_score, structure_score)
        
        return {
            'valid': total_score > 50,
            'direction': direction,
            'confidence': min(100, total_score),
            'volatility_score': volatility_score,
            'structure_score': structure_score,
            'liquidity_score': liquidity_score,
            'trend_score': trend_score
        }
    
    def analyze_volatility_reaction(self, df):
        """ANALYZE VOLATILITY EXPANSION/CONTRACTION (Primary for Deriv)"""
        if len(df) < 20:
            return 0
        
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().iloc[-1]
        
        # Recent volatility vs historical
        recent_atr = tr.iloc[-5:].mean()
        historical_atr = tr.iloc[-20:].mean()
        
        # Check for volatility expansion
        if recent_atr > historical_atr * 1.5:
            return 80  # High confidence on expansion
        elif recent_atr > historical_atr * 1.2:
            return 60  # Moderate expansion
        
        # Check for volatility contraction (range-bound)
        if recent_atr < historical_atr * 0.8:
            return 40  # Range-bound, lower confidence
        
        return 30  # Normal volatility
    
    def analyze_micro_structure(self, df):
        """Analyze MICRO-structure (not institutional OB)"""
        if len(df) < 10:
            return 0
        
        # 1. Check for recent highs/lows
        recent_high = df['high'].iloc[-5:].max()
        recent_low = df['low'].iloc[-5:].min()
        current_price = df['close'].iloc[-1]
        
        # 2. Check break of structure (BOS)
        bos_score = 0
        if current_price > recent_high:
            bos_score = 70  # Bullish BOS
        elif current_price < recent_low:
            bos_score = 70  # Bearish BOS
        
        # 3. Check for change of character (CHOCH)
        choch_score = 0
        if len(df) >= 15:
            prev_high = df['high'].iloc[-10:-5].max()
            prev_low = df['low'].iloc[-10:-5].min()
            
            if current_price > prev_high and df['close'].iloc[-5] < prev_low:
                choch_score = 60  # Strong CHOCH
        
        return max(bos_score, choch_score)
    
    def detect_liquidity_sweep(self, df):
        """Detect liquidity sweeps (Deriv engines love these)"""
        if len(df) < 10:
            return 0
        
        last_candle = df.iloc[-1]
        prev_high = df['high'].iloc[-10:-1].max()
        prev_low = df['low'].iloc[-10:-1].min()
        
        # Bullish liquidity sweep (wick above then close below)
        if last_candle['high'] > prev_high and last_candle['close'] < prev_high:
            return 70
        
        # Bearish liquidity sweep (wick below then close above)
        if last_candle['low'] < prev_low and last_candle['close'] > prev_low:
            return 70
        
        return 0
    
    def analyze_trend_strength(self, df):
        """Analyze SHORT-TERM trend only (HTF bias minimized)"""
        if len(df) < 20:
            return 0
        
        # Use EMA cross for short-term trend
        ema_fast = df['close'].ewm(span=9).mean().iloc[-1]
        ema_slow = df['close'].ewm(span=21).mean().iloc[-1()
        
        if ema_fast > ema_slow:
            direction = 'UP'
        else:
            direction = 'DOWN'
        
        # Measure trend strength using ADX
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
        minus_di = 100 * (abs(minus_dm).rolling(window=14).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=14).mean().iloc[-1]
        
        if adx > 25:
            return 60  # Strong trend
        elif adx > 20:
            return 40  # Moderate trend
        
        return 20  # Weak trend
    
    def determine_direction(self, df, volatility_score, structure_score):
        """Determine trade direction"""
        if len(df) < 5:
            return 'NEUTRAL'
        
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        
        # Price action direction
        price_direction = 'UP' if current_price > prev_price else 'DOWN'
        
        # Volatility direction (expansion favors continuation)
        if volatility_score > 60:
            return price_direction  # Follow volatility expansion
        
        # Structure direction
        if structure_score > 50:
            recent_high = df['high'].iloc[-5:].max()
            recent_low = df['low'].iloc[-5:].min()
            
            if current_price > recent_high:
                return 'UP'
            elif current_price < recent_low:
                return 'DOWN'
        
        return price_direction
    
    def calculate_atr_sl_tp(self, df, entry_price, direction, confidence):
        """ATR-ADAPTIVE SL/TP for Deriv (NO TIGHT SLs)"""
        if len(df) < 14:
            # Default conservative values
            return {
                'sl_pips': 30 if 'R_75' in self.symbol else 40,
                'tp_pips': 45 if 'R_75' in self.symbol else 60,
                'risk_reward': 1.5
            }
        
        # Calculate ATR
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().iloc[-1]
        
        # Convert ATR to pips
        atr_pips = atr / self.pip_size
        
        # ATR-based SL (LARGER for Deriv synthetic spikes)
        # Minimum 1.5 ATR, maximum 3 ATR
        base_sl_atr = 1.8  # Conservative for Deriv
        
        # Adjust based on volatility
        recent_vol = tr.iloc[-5:].mean() / self.pip_size
        if recent_vol > atr_pips * 1.3:
            base_sl_atr = 2.2  # Larger SL during high volatility
        
        sl_pips = atr_pips * base_sl_atr
        
        # Minimum SL limits
        if 'R_75' in self.symbol or 'R_100' in self.symbol:
            sl_pips = max(sl_pips, 25)  # Minimum 25 pips for volatility indices
            sl_pips = min(sl_pips, 80)  # Maximum 80 pips
        elif 'CRASH' in self.symbol or 'BOOM' in self.symbol:
            sl_pips = max(sl_pips, 15)  # Minimum 15 pips for crash/boom
            sl_pips = min(sl_pips, 50)  # Maximum 50 pips
        
        # SHORTER TP for Deriv (0.8R to 1.5R optimal)
        # Deriv engines snap back, so smaller TPs work better
        rr_ratio = np.random.uniform(0.8, 1.5)  # Dynamic R:R
        
        # Higher confidence = slightly larger TP
        if confidence > 70:
            rr_ratio = min(rr_ratio * 1.2, 1.8)
        
        tp_pips = sl_pips * rr_ratio
        
        # Ensure minimum TP
        tp_pips = max(tp_pips, sl_pips * 0.8)  # At least 0.8:1 R:R
        
        return {
            'sl_pips': round(sl_pips, 1),
            'tp_pips': round(tp_pips, 1),
            'risk_reward': round(rr_ratio, 2),
            'atr_value': round(atr_pips, 1)
        }

# ============ TRADING ENGINE ============
class DerivTradingEngine:
    """Trading engine optimized for Deriv"""
    
    def __init__(self):
        self.api = DerivAPI()
        self.logger = EnhancedLogger()
        self.settings = self.load_settings()
        self.active_trades = []
        self.trade_history = []
        self.running = False
        
    def load_settings(self):
        """Load settings"""
        return {
            'dry_run': True,
            'trade_amount': 1.0,
            'min_confidence': 60,
            'enabled_markets': list(DERIV_MARKETS.keys())
        }
    
    async def analyze_and_trade(self):
        """Main trading loop"""
        if not self.running:
            return
        
        for market_key in self.settings['enabled_markets']:
            try:
                market = DERIV_MARKETS[market_key]
                symbol = market['symbol']
                
                # Get market data
                df_m5 = await self.api.get_candles(symbol, "M5", 100)
                df_m15 = await self.api.get_candles(symbol, "M15", 100)
                
                if df_m5 is None or len(df_m5) < 20:
                    continue
                
                # Analyze with Deriv-optimized strategy
                strategy = DerivOptimizedSMCStrategy(symbol, market)
                analysis = strategy.analyze_market_structure(df_m5)
                
                if analysis['valid'] and analysis['confidence'] >= self.settings['min_confidence']:
                    await self.execute_trade(symbol, market, df_m5, analysis)
                    
            except Exception as e:
                print(f"❌ Error analyzing {market_key}: {e}")
    
    async def execute_trade(self, symbol, market, df, analysis):
        """Execute a trade with ATR-adaptive SL/TP"""
        current_price = df['close'].iloc[-1]
        direction = analysis['direction']
        confidence = analysis['confidence']
        
        # Calculate ATR-adaptive SL/TP
        strategy = DerivOptimizedSMCStrategy(symbol, market)
        sl_tp = strategy.calculate_atr_sl_tp(df, current_price, direction, confidence)
        
        # Calculate SL/TP prices
        if direction == 'BUY':
            sl_price = current_price - (market['pip_size'] * sl_tp['sl_pips'])
            tp_price = current_price + (market['pip_size'] * sl_tp['tp_pips'])
        else:
            sl_price = current_price + (market['pip_size'] * sl_tp['sl_pips'])
            tp_price = current_price - (market['pip_size'] * sl_tp['tp_pips'])
        
        # Round to correct digits
        sl_price = round(sl_price, market['digits'])
        tp_price = round(tp_price, market['digits'])
        
        # Execute trade
        trade_amount = self.settings['trade_amount']
        
        if self.settings['dry_run']:
            print(f"✅ DRY RUN: {symbol} {direction} at {current_price:.5f}")
            print(f"   SL: {sl_price:.5f} ({sl_tp['sl_pips']} pips)")
            print(f"   TP: {tp_price:.5f} ({sl_tp['tp_pips']} pips)")
            print(f"   R:R: {sl_tp['risk_reward']}:1 | Confidence: {confidence:.1f}%")
            
            trade = {
                'id': f"DRY_{int(time.time())}",
                'symbol': symbol,
                'direction': direction,
                'entry': current_price,
                'sl': sl_price,
                'tp': tp_price,
                'amount': trade_amount,
                'sl_pips': sl_tp['sl_pips'],
                'tp_pips': sl_tp['tp_pips'],
                'rr_ratio': sl_tp['risk_reward'],
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'status': 'OPEN'
            }
            self.active_trades.append(trade)
            
        else:
            # Real trade
            result = await self.api.place_trade(symbol, direction, trade_amount)
            if result:
                print(f"✅ LIVE TRADE: {symbol} {direction} at {current_price:.5f}")
                print(f"   Contract: {result.get('buy', {}).get('contract_id', 'Unknown')}")

# ============ GLOBAL ENGINE INSTANCE ============
trading_engine = DerivTradingEngine()

# ============ WEB ROUTES ============
@app.get("/")
async def root():
    return {"message": "Karanka V7 Deriv SMC Bot", "status": "running"}

@app.get("/dashboard")
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/status")
async def get_status():
    return {
        "status": "running",
        "connected": trading_engine.api.connected,
        "active_trades": len(trading_engine.active_trades),
        "dry_run": trading_engine.settings['dry_run']
    }

@app.post("/api/trading/start")
async def start_trading():
    trading_engine.running = True
    asyncio.create_task(trading_engine.analyze_and_trade())
    return {"success": True, "message": "Trading started"}

@app.post("/api/trading/stop")
async def stop_trading():
    trading_engine.running = False
    return {"success": True, "message": "Trading stopped"}

@app.post("/api/deriv/connect")
async def connect_deriv(token: str):
    success = await trading_engine.api.connect(token)
    return {"success": success, "message": "Connected" if success else "Failed"}

# ============ CREATE SIMPLE TEMPLATE ============
def create_templates():
    """Create basic HTML template"""
    template_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Karanka V7 - Deriv SMC Bot</title>
        <style>
            body {
                background: #0a0a0a;
                color: #FFD700;
                font-family: Arial;
                padding: 20px;
            }
            .card {
                background: #1a1a1a;
                border: 1px solid #333;
                padding: 20px;
                margin: 10px;
                border-radius: 5px;
            }
            .btn {
                background: #D4AF37;
                color: #000;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <h1>🎯 Karanka V7 - Deriv SMC Bot</h1>
        
        <div class="card">
            <h2>Status</h2>
            <p id="status">Loading...</p>
            <p>Active Trades: <span id="trades">0</span></p>
            <p>Mode: <span id="mode">Dry Run</span></p>
        </div>
        
        <div class="card">
            <h2>Controls</h2>
            <button class="btn" onclick="startTrading()">Start Trading</button>
            <button class="btn" onclick="stopTrading()">Stop Trading</button>
        </div>
        
        <script>
            async function updateStatus() {
                const response = await fetch('/api/status');
                const data = await response.json();
                document.getElementById('status').textContent = data.connected ? 'Connected' : 'Disconnected';
                document.getElementById('trades').textContent = data.active_trades;
                document.getElementById('mode').textContent = data.dry_run ? 'Dry Run' : 'Live Trading';
            }
            
            async function startTrading() {
                await fetch('/api/trading/start', {method: 'POST'});
                alert('Trading started!');
            }
            
            async function stopTrading() {
                await fetch('/api/trading/stop', {method: 'POST'});
                alert('Trading stopped!');
            }
            
            setInterval(updateStatus, 3000);
            updateStatus();
        </script>
    </body>
    </html>
    """
    
    os.makedirs(templates_dir, exist_ok=True)
    with open(os.path.join(templates_dir, "dashboard.html"), "w") as f:
        f.write(template_content)

# ============ STARTUP ============
@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    create_templates()
    print("✅ Karanka V7 Deriv Bot Started")
    print("✅ FastAPI App Ready")
    print("✅ Access at: http://localhost:10000")

# ============ MAIN ============
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"\n🚀 Starting Karanka V7 Deriv Bot on port {port}")
    print("✅ App is defined and ready")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
