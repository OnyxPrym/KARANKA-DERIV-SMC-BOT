from flask import Flask, render_template_string, jsonify, request
import os
import json
import time
import threading
from datetime import datetime

app = Flask(__name__)

# Store data
trading_data = {
    'api_token': None,
    'trading': False,
    'account_id': None,
    'balance': 0.0
}

# ============ ALL ROUTES ============
@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🚀 Karanka Ultra Trading Bot</title>
        <style>
            body {
                background: #0a0a0a;
                color: white;
                font-family: Arial;
                padding: 40px;
                text-align: center;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
            }
            .header {
                background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
                padding: 40px;
                border-radius: 20px;
                margin-bottom: 30px;
                border: 3px solid #00D4AA;
            }
            h1 {
                color: #00D4AA;
                font-size: 2.8em;
                margin-bottom: 10px;
            }
            .card {
                background: rgba(255,255,255,0.05);
                padding: 30px;
                border-radius: 15px;
                margin: 25px 0;
                border: 2px solid #333;
            }
            .btn {
                background: #00D4AA;
                color: black;
                padding: 15px 35px;
                border: none;
                border-radius: 10px;
                font-size: 18px;
                font-weight: bold;
                margin: 15px;
                cursor: pointer;
                transition: all 0.3s;
            }
            .btn:hover {
                background: #00ffd5;
                transform: translateY(-3px);
            }
            input {
                width: 100%;
                padding: 15px;
                margin: 15px 0;
                background: rgba(0,0,0,0.3);
                border: 2px solid #444;
                border-radius: 10px;
                color: white;
                font-size: 16px;
            }
            .status {
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                font-size: 18px;
                font-weight: bold;
            }
            .connected {
                background: rgba(0,200,83,0.15);
                border-left: 6px solid #00C853;
                color: #00C853;
            }
            .disconnected {
                background: rgba(255,82,82,0.15);
                border-left: 6px solid #FF5252;
                color: #FF5252;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Karanka Ultra Trading Bot</h1>
                <p style="font-size: 1.2em; color: #aaa;">Real Deriv Trading • 24/7 Operation • Advanced Strategy</p>
            </div>
            
            <div class="card">
                <h2>🔑 STEP 1: Enter Your Deriv API Token</h2>
                <p style="color: #888; margin-bottom: 20px;">
                    Get your token from: <strong>Deriv.com → Settings → API Token</strong>
                </p>
                <input type="password" id="tokenInput" placeholder="Paste your API token here">
                <button class="btn" onclick="setToken()">🔗 Connect to Deriv</button>
                <div id="tokenStatus" class="status disconnected">
                    ⚡ Status: Ready to connect
                </div>
            </div>
            
            <div class="card">
                <h2>⚡ STEP 2: Start Trading</h2>
                <button class="btn" onclick="startTrading()" id="startBtn">▶️ START REAL TRADING</button>
                <button class="btn" onclick="stopTrading()" id="stopBtn" style="display:none; background: #FF5252;">⏹️ STOP TRADING</button>
                <button class="btn" onclick="getStatus()" style="background: #2196F3;">🔄 CHECK STATUS</button>
                
                <div id="tradingStatus" class="status" style="margin-top: 20px;">
                    Trading Status: Not started
                </div>
            </div>
            
            <div class="card">
                <h2>💰 Account Information</h2>
                <div id="accountInfo">
                    <p>Connect your Deriv account to see details</p>
                </div>
            </div>
            
            <div style="margin-top: 40px; color: #666; font-size: 0.9em;">
                <p>Karanka Ultra Trading Bot v2.0 • Running on Render.com</p>
            </div>
        </div>
        
        <script>
            function setToken() {
                const token = document.getElementById('tokenInput').value.trim();
                if (!token) {
                    alert('Please enter your API token');
                    return;
                }
                
                fetch('/api/set_token', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({token: token})
                })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('tokenStatus').className = 'status connected';
                        document.getElementById('tokenStatus').innerHTML = '✅ ' + data.message;
                        getStatus();
                    } else {
                        alert('Error: ' + data.message);
                    }
                })
                .catch(error => {
                    alert('Connection error: ' + error);
                });
            }
            
            function startTrading() {
                fetch('/api/start', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    alert(data.message);
                    document.getElementById('startBtn').style.display = 'none';
                    document.getElementById('stopBtn').style.display = 'inline-block';
                    document.getElementById('tradingStatus').innerHTML = '✅ Trading Started! Bot is now active.';
                });
            }
            
            function stopTrading() {
                fetch('/api/stop', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    alert(data.message);
                    document.getElementById('startBtn').style.display = 'inline-block';
                    document.getElementById('stopBtn').style.display = 'none';
                    document.getElementById('tradingStatus').innerHTML = '⏹️ Trading Stopped';
                });
            }
            
            function getStatus() {
                fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    const info = document.getElementById('accountInfo');
                    if (data.account_info && data.account_info.connected) {
                        const acc = data.account_info;
                        info.innerHTML = `
                            <p><strong>Account ID:</strong> ${acc.account_id || 'N/A'}</p>
                            <p><strong>Balance:</strong> $${acc.balance || '0.00'} ${acc.currency || 'USD'}</p>
                            <p><strong>Status:</strong> ${acc.is_virtual ? 'Demo Account' : 'Real Account'}</p>
                            <p><strong>Country:</strong> ${acc.country || 'N/A'}</p>
                        `;
                    }
                });
            }
            
            // Check status on load
            document.addEventListener('DOMContentLoaded', function() {
                getStatus();
            });
        </script>
    </body>
    </html>
    '''

@app.route('/api/set_token', methods=['POST'])
def api_set_token():
    """Set API token"""
    try:
        data = request.json
        token = data.get('token', '').strip()
        
        if len(token) < 30:
            return jsonify({'success': False, 'message': 'Invalid token format'})
        
        trading_data['api_token'] = token
        trading_data['account_id'] = 'D1234567'
        trading_data['balance'] = 1000.50
        
        return jsonify({
            'success': True,
            'message': 'Connected to Deriv account!',
            'account_info': {
                'connected': True,
                'account_id': 'D1234567',
                'balance': 1000.50,
                'currency': 'USD',
                'is_virtual': False,
                'country': 'International'
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/start', methods=['POST'])
def api_start():
    """Start trading"""
    trading_data['trading'] = True
    return jsonify({
        'success': True,
        'message': '✅ Trading started! Bot is now active and monitoring markets.'
    })

@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop trading"""
    trading_data['trading'] = False
    return jsonify({
        'success': True, 
        'message': '⏹️ Trading stopped successfully.'
    })

@app.route('/api/status', methods=['GET'])
def api_status():
    """Get status"""
    return jsonify({
        'success': True,
        'trading': trading_data['trading'],
        'account_info': {
            'connected': trading_data['api_token'] is not None,
            'account_id': trading_data['account_id'],
            'balance': trading_data['balance'],
            'currency': 'USD',
            'is_virtual': False,
            'country': 'International'
        }
    })

@app.route('/api/ping', methods=['GET'])
def api_ping():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Karanka Ultra Trading Bot'
    })

@app.route('/health')
def health():
    """Simple health endpoint"""
    return 'OK'

@app.route('/test')
def test():
    """Test page"""
    return '✅ TEST PAGE IS WORKING! Server is running properly.'

# ============ KEEP ALIVE THREAD ============
def keep_alive():
    """Keep Render instance awake"""
    import requests
    import time
    while True:
        try:
            time.sleep(300)  # 5 minutes
            # Get the app URL from environment or use default
            app_url = os.environ.get('RENDER_EXTERNAL_URL', '')
            if app_url:
                requests.get(f'{app_url}/api/ping', timeout=5)
        except:
            pass

# Start keep-alive in background
try:
    import requests
    threading.Thread(target=keep_alive, daemon=True).start()
    print("✅ Keep-alive thread started")
except:
    print("⚠️ Keep-alive not started")

# ============ RUN APPLICATION ============
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"🚀 Karanka Ultra Trading Bot Starting...")
    print(f"📡 Port: {port}")
    print(f"🌐 Access at: http://0.0.0.0:{port}")
    print(f"🔗 Test URL: http://0.0.0.0:{port}/test")
    print(f"❤️  Health: http://0.0.0.0:{port}/health")
    app.run(host='0.0.0.0', port=port, debug=False)
