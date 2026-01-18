# 🎯 Karanka Deriv Trading Bot V7

🚀 **Mobile WebApp for Automated Deriv Trading with SMC Strategies**

## 🌟 Features

✅ **REAL Deriv Trading** - Connects to your actual Deriv account  
✅ **Live Market Data** - Real-time prices from Deriv  
✅ **4 SMC Strategies** - Liquidity Grab, FVG, Order Block, BOS  
✅ **Mobile Responsive** - Works perfectly on phone/tablet  
✅ **User Authentication** - Secure login/registration  
✅ **Risk Management** - Control trades per day, concurrent trades, amounts  
✅ **Real-time Updates** - WebSocket notifications  
✅ **Deploy on Render.com** - One-click deployment  

## 📱 Mobile App Features

- 📊 Dashboard with real-time stats
- ⚙️ Complete settings control
- 🔔 Push notifications
- 📱 Add to homescreen (PWA)
- 📈 Live trade updates
- 🎯 Strategy performance tracking

## 🚀 Quick Deploy on Render.com

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

### Deployment Steps:

1. **Click "Deploy to Render" button above**
2. **Connect your GitHub repository**
3. **Name your service:** `karanka-deriv`
4. **Wait for build to complete** (5-10 minutes)
5. **Add PostgreSQL database** (auto-provisioned)
6. **Set environment variables:**
   - `SECRET_KEY`: Click "Generate"
   - `DATABASE_URL`: Auto-filled from database
7. **Access your bot:** `https://your-service.onrender.com`

## 🔧 Local Development

```bash
# Clone repository
git clone https://github.com/yourusername/karanka-deriv.git
cd karanka-deriv

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "SECRET_KEY=your-secret-key" > .env
echo "DATABASE_URL=sqlite:///karanka.db" >> .env

# Run the app
python app.py

# Open in browser
# http://localhost:5000
