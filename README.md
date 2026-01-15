# 🎯 Karanka V7 - Deriv Real Trading Bot

A professional trading bot for Deriv.com with real API connection, SMC strategies, and mobile web interface.

## Features

✅ **Real Deriv API Connection**
- Connect with your Deriv API token
- Auto-detect demo and real accounts
- Real trading with actual money

✅ **Smart Money Concepts (SMC) Strategy**
- Optimized for Deriv Volatility Indices
- Liquidity zone detection
- Order block analysis
- Multi-timeframe confirmation

✅ **User Controls**
- Select specific markets to trade
- Set $ amount per trade
- Control max concurrent trades
- Set daily/hourly limits

✅ **Real-Time Tracking**
- Live balance updates
- Trade win/loss tracking
- Performance statistics
- Session awareness

## Deployment on Render.com

1. **Fork/Create Repository**
   - Create a new GitHub repository
   - Upload all files from this project

2. **Connect to Render.com**
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

3. **Configure Service**
   - **Name:** `karanka-deriv-bot`
   - **Environment:** `Docker`
   - **Branch:** `main`
   - **Region:** `Oregon (US West)` (recommended)
   - **Instance Type:** `Starter ($7/month)` or higher
   - **Auto-Deploy:** Enable

4. **Deploy**
   - Click "Create Web Service"
   - Wait for build to complete (2-3 minutes)

5. **Get Your URL**
   - Once deployed, you'll get a URL like: `https://karanka-deriv-bot.onrender.com`

## Getting Started

1. **Get Deriv API Token:**
   - Login to [app.deriv.com](https://app.deriv.com)
   - Go to Settings → API Token
   - Create token with "Read" and "Trade" permissions
   - Copy the token

2. **Use the Bot:**
   - Go to your deployed URL
   - Register/Login
   - Go to Connection tab
   - Paste your API token
   - Select account (demo or real)
   - Configure markets and settings
   - Start trading!

## Safety Features

🔒 **Dry Run Mode:** Test without real money
📊 **Risk Controls:** Set max trades and amounts
⚡ **Auto-Stop:** Stops at daily/hourly limits
📱 **Mobile Optimized:** Works on all devices

## Support

For issues or questions:
1. Check browser console for errors
2. Ensure Deriv account has funds
3. Verify API token permissions
4. Contact: [Your contact info]

## Disclaimer

⚠️ **Trading involves risk. Only trade with money you can afford to lose.**
This bot is a tool, not financial advice. Past performance doesn't guarantee future results.
