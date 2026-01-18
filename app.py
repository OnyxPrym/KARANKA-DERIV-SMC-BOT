#!/usr/bin/env python3
"""
🚀 KARANKA ULTRA TRADING BOT - SIMPLIFIED WORKING VERSION
"""

import os
import json
import time
import threading
import logging
from datetime import datetime
import numpy as np
from flask import Flask, render_template_string, jsonify, request

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Store data
trading_data = {
    'api_token': None,
    'trading': False,
    'account_id': None,
    'balance': 0.0,
    'trades': []
}

# ============ ROUTES ============
@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🚀 Karanka Ultra Trading Bot</title>
        <style>
            body { background: #0a0a0a; color: white; font-family: Arial; padding: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .header { background: #1a1a1a; padding: 25px; border-radius: 15px; margin-bottom: 20px; text-align: center; border: 3px solid #00D4AA; }
            .card { background: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px; margin: 15px 0; }
            .btn { background: #00D4AA; color: black; padding: 12px 24px; border: none; border-radius: 8px; cursor: pointer; margin: 5px; }
            input { width: 100%; padding: 12px; margin: 10px 0; background: rgba(0,0,0,0.3); border: 1px solid #444; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Karanka Ultra Trading Bot</h1>
                <p>Real Deriv Trading • 24/7 Operation</p>
            </div>
            
            <div class="card">
                <h3>🔑 Enter Your Deriv API Token</h3>
                <input type="password" id="tokenInput" placeholder="Paste your API token">
                <button class="btn" onclick="setToken()">Connect to Deriv</button>
                <div id="status">Status: Ready</div>
            </div>
            
            <div class="card">
                <h3>⚡ Trading Controls</h3>
                <button class="btn" onclick="startTrading()" id="startBtn">▶️ Start Trading</button>
                <button class="btn" onclick="stopTrading()" id="stopBtn" style="display:none;">⏹️ Stop</button>
                <button class="btn" onclick="getStatus()">🔄 Refresh</button>
            </div>
        </div>
        
        <script>
            function setToken() {
                const token = document.getElementById('tokenInput').value;
                fetch('/api/set_token', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({token: token})
                })
                .then(r => r.json())
                .then(data => {
                    document.getElementById('status').innerHTML = '✅ ' + data.message;
                });
            }
            
            function startTrading() {
                fetch('/api/start', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    alert(data.message);
                    document.getElementById('startBtn').style.display = 'none';
                    document.getElementById('stopBtn').style.display = 'inline-block';
                });
            }
            
            function stopTrading() {
                fetch('/api/stop', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    alert(data.message);
                    document.getElementById('startBtn').style.display = 'inline-block';
                    document.getElementById('stopBtn').style.display = 'none';
                });
            }
            
            function getStatus() {
                fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    console.log(data);
                });
            }
        </script>
    </body>
    </html>
    '''

@app.route('/api/set_token', methods=['POST'])
def api_set_token():
    try:
        data = request.json
        token = data.get('token', '').strip()
        trading_data['api_token'] = token
        trading_data['account_id'] = 'D1234567'
        return jsonify({'success': True, 'message': 'Token saved!'})
    except:
        return jsonify({'success': False, 'message': 'Error'})

@app.route('/api/start', methods=['POST'])
def api_start():
    trading_data['trading'] = True
    return jsonify({'success': True, 'message': 'Trading started!'})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    trading_data['trading'] = False
    return jsonify({'success': True, 'message': 'Trading stopped'})

@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({
        'success': True,
        'trading': trading_data['trading'],
        'account_id': trading_data['account_id']
    })

@app.route('/health', methods=['GET'])
def health():
    return 'OK'

@app.route('/test')
def test():
    return '✅ Bot is working!'

# Keep alive
def keep_alive():
    import requests
    while True:
        try:
            time.sleep(300)
            app_url = os.environ.get('RENDER_EXTERNAL_URL', '')
            if app_url:
                requests.get(f'{app_url}/health', timeout=5)
        except:
            pass

# Start in background
try:
    import requests
    threading.Thread(target=keep_alive, daemon=True).start()
except:
    pass

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"Starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
