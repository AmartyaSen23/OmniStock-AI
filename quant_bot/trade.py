# =============================================================================
# 📊 CELL 2: Live Execution & NLP Fusion
# =============================================================================

import pandas as pd
import numpy as np
import yfinance as yf
import os
import time
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

APCA_API_KEY_ID = os.environ.get('APCA_API_KEY_ID')
APCA_API_SECRET_KEY = os.environ.get('APCA_API_SECRET_KEY')
APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'
# =============================================================================
# 1. The Bridge: Live Probability Prediction (MACRO-UPGRADED)
# =============================================================================
def apply_technical_features(df):
    df_feat = df.copy()

    df_feat['Close_EMA7'] = df_feat['Close'].ewm(span=7, adjust=False).mean()
    df_feat['Close_SMA7'] = df_feat['Close'].rolling(window=7).mean()
    df_feat['Close_STD7'] = df_feat['Close'].rolling(window=7).std()

    df_feat['BB_Middle'] = df_feat['Close'].rolling(window=20).mean()
    df_feat['BB_Upper'] = df_feat['BB_Middle'] + 2 * df_feat['Close'].rolling(window=20).std()
    df_feat['BB_Lower'] = df_feat['BB_Middle'] - 2 * df_feat['Close'].rolling(window=20).std()

    df_feat['Momentum'] = df_feat['Close'] - df_feat['Close'].shift(10)

    delta = df_feat['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df_feat['RSI'] = 100 - (100 / (1 + rs))

    ema_12 = df_feat['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df_feat['Close'].ewm(span=26, adjust=False).mean()
    df_feat['MACD'] = ema_12 - ema_26

    df_feat['HL'] = df_feat['High'] - df_feat['Low']
    df_feat['HC'] = np.abs(df_feat['High'] - df_feat['Close'].shift())
    df_feat['LC'] = np.abs(df_feat['Low'] - df_feat['Close'].shift())
    df_feat['TR'] = df_feat[['HL', 'HC', 'LC']].max(axis=1)
    df_feat['ATR'] = df_feat['TR'].rolling(window=14).mean()

    df_feat = df_feat.dropna().reset_index(drop=True)
    return df_feat

def predict_live_probability(model, scaler, feature_cols, ticker="GOOG"):
    print(f"🔄 Fetching latest 60 days of {ticker} & Macro market data...")
    
    # 1. Fetch Live Data
    df = yf.download(ticker, period="60d", interval="1d", progress=False, auto_adjust=True)
    df_qqq = yf.download("QQQ", period="60d", interval="1d", progress=False, auto_adjust=True)
    df_vix = yf.download("^VIX", period="60d", interval="1d", progress=False, auto_adjust=True)
    
    # Clean headers and strip timezones to ensure a perfect merge
    for d in [df, df_qqq, df_vix]:
        d.reset_index(inplace=True)
        d.columns = [col[0] if isinstance(col, tuple) else col for col in d.columns]
        d['Date'] = pd.to_datetime(d['Date']).dt.tz_localize(None)
        
    # Merge MACRO data
    df = df.merge(df_qqq[['Date', 'Close']], on='Date', suffixes=('', '_QQQ'))
    df = df.merge(df_vix[['Date', 'Close']], on='Date', suffixes=('', '_VIX'))
    
    # Build Macro Features
    df['QQQ_Momentum'] = df['Close_QQQ'] - df['Close_QQQ'].shift(10)
    df['VIX_Level'] = df['Close_VIX']
    df['GOOG_vs_QQQ_Rel'] = (df['Close'] / df['Close'].shift(10)) - (df['Close_QQQ'] / df['Close_QQQ'].shift(10))
    
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Apply standard technical features (This uses your helper function from Cell 1)
    df_feat = apply_technical_features(df)
    
    # Get the data for right NOW
    latest_data = df_feat.iloc[-1:].copy()
    current_price = latest_data['Close'].values[0]
    
    # Isolate the exact 13 features the XGBoost model expects
    X_live = latest_data[feature_cols]
    X_live_scaled = scaler.transform(X_live)
    
    # 🔄 Ask XGBoost for the PROBABILITY it goes UP in the next 3 days
    probability_up = model.predict_proba(X_live_scaled)[0][1]
    
    return probability_up, current_price

# =============================================================================
# 2. Live NLP Pipeline (VADER + Time Decay) - FIXED
# =============================================================================
def get_live_nlp_sentiment(ticker="GOOG"):
    print(f"📰 Scanning live headlines for {ticker}...")
    analyzer = SentimentIntensityAnalyzer()
    
    stock = yf.Ticker(ticker)
    live_news = stock.news
    
    if not live_news:
        print("⚠️ No live news found.")
        return 0
        
    relevance_keywords = [
        'earnings', 'guidance', 'lawsuit', 'acquisition', 'downgrade', 
        'upgrade', 'ai', 'tech', 'market', 'revenue', 'growth', 'google', 'alphabet'
    ]
    lambda_decay = 0.1
    
    weighted_scores = []
    total_weights = 0
    current_unix_time = int(time.time())
    
    for item in live_news:
        # 🛡️ SAFE EXTRACTION: APIs change constantly and inject ads/malformed data.
        # .get() prevents the KeyError from crashing the system.
        headline = item.get('title', '')
        
        # Fallback for newer yfinance nesting structures
        if not headline and 'content' in item:
            headline = item['content'].get('title', '')
            
        # If no headline is found, skip this item
        if not headline:
            continue
            
        headline_lower = headline.lower()
        
        # Relevance filter
        if not any(kw in headline_lower for kw in relevance_keywords):
            continue
            
        # VADER Score
        vader_scores = analyzer.polarity_scores(headline)
        raw_score = vader_scores['compound'] 
        
        # 🛡️ SAFE TIME EXTRACTION
        publish_time = item.get('providerPublishTime')
        if publish_time:
            hours_old = (current_unix_time - publish_time) / 3600
        else:
            hours_old = 2 # Fallback if Yahoo changes their time format
            
        weight = np.exp(-lambda_decay * hours_old)
        
        weighted_scores.append(raw_score * weight)
        total_weights += weight
        
        print(f"   ↳ [{hours_old:.1f}h old] {headline[:50]}... | Score: {raw_score:.2f}")
        
    final_sentiment = sum(weighted_scores) / total_weights if total_weights > 0 else 0
    return final_sentiment
# =============================================================================
# 🔌 PHASE 3: Alpaca Live Order Execution
# =============================================================================
import alpaca_trade_api as tradeapi

# 1. Your Paper Trading Credentials (You'll get these from Alpaca's dashboard)
# IMPORTANT: Ensure it says 'paper'

# 2. Initialize the API Connection
api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, api_version='v2')

def execute_bracket_order(symbol, qty, side, take_profit_price, stop_loss_price):
    """
    Fires an OCO (One Cancels the Other) Bracket Order to the exchange.
    Side must be 'buy' or 'sell'.
    """
    print(f"\n📡 Transmitting {side.upper()} order to Alpaca servers...")
    
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side.lower(),       # 'buy' or 'sell'
            type='market',           # Execute the master order right now
            time_in_force='gtc',     # Good 'Til Canceled
            order_class='bracket',   # Tells the broker to link the next two orders
            take_profit=dict(
                limit_price=round(take_profit_price, 2),
            ),
            stop_loss=dict(
                stop_price=round(stop_loss_price, 2),
            )
        )
        print(f"✅ SUCCESS: {qty} shares of {symbol} secured.")
        print(f"   ↳ Stop-Loss resting at: ${round(stop_loss_price, 2)}")
        print(f"   ↳ Take-Profit resting at: ${round(take_profit_price, 2)}")
        return order
        
    except Exception as e:
        print(f"❌ ORDER FAILED: {e}")
        return None
# =============================================================================
# 3. Fusion & Strict Risk Management Execution
# =============================================================================
def execute_trading_system(model, scaler, feature_cols):
    CAPITAL = 100000.00
    RISK_PER_TRADE = 0.02
    
    print("\n" + "="*50)
    print("🚀 INITIALIZING QUANT EXECUTION SYSTEM")
    print("="*50)
    
    buy_probability, current_price = predict_live_probability(model, scaler, feature_cols, "GOOG")
    sentiment_score = get_live_nlp_sentiment("GOOG")
    
    print(f"\n📊 Model BUY Confidence   : {buy_probability*100:.1f}%")
    print(f"🧠 NLP Weighted Sentiment : {sentiment_score:.2f}")

    PROB_THRESH = 0.55       # Model must be > 55% confident
    SENTIMENT_THRESH = 0.05  # News must be positive
    
    trade_decision = "HOLD"
    
    if buy_probability > PROB_THRESH and sentiment_score > SENTIMENT_THRESH:
        trade_decision = "BUY"
    elif buy_probability < (1 - PROB_THRESH) and sentiment_score < -SENTIMENT_THRESH:
        trade_decision = "SELL"

    # =============================================================================
    # 📝 THE LEDGER: Silently Logging the Data
    # =============================================================================
    log_filename = "algo_trading_log.csv"
    
    # Package today's metrics into a clean dictionary
    log_data = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Ticker": "GOOG",
        "Current_Price": round(current_price, 2),
        "Model_Confidence": round(buy_probability * 100, 2),
        "NLP_Sentiment": round(sentiment_score, 2),
        "Decision": trade_decision
    }
    
    # Convert to DataFrame and append to the CSV
    log_df = pd.DataFrame([log_data])
    file_exists = os.path.isfile(log_filename)
    
    # mode='a' means append. If the file doesn't exist, it creates it and adds the header.
    log_df.to_csv(log_filename, mode='a', header=not file_exists, index=False)
    print(f"💾 Run data silently archived to {log_filename}")
    
    # trade_decision = "BUY" # ⚠️ TEMPORARY OVERRIDE FOR API TEST
    print(f"\n⚡ SYSTEM DECISION: {trade_decision}")

    if trade_decision in ["BUY", "SELL"]:
        # 🔄 UPDATED: Dynamic Sizing based on Classification Probability
        # If model is 75% confident, we scale position size linearly
        base_confidence = buy_probability if trade_decision == "BUY" else (1 - buy_probability)
        
        # Blend model confidence with NLP sentiment impact
        combined_confidence = min(1.0, base_confidence + abs(sentiment_score))
        
        base_position_size = CAPITAL * RISK_PER_TRADE
        scaled_position_size = base_position_size * combined_confidence
        shares_to_trade = int(scaled_position_size / current_price)
        
        # STRICT 1:2 R:R BRACKET MATH
        if trade_decision == "BUY":
            stop_loss = current_price * 0.98   # 2% Risk
            take_profit = current_price * 1.04 # 4% Reward
        else: # SELL
            stop_loss = current_price * 1.02   # 2% Risk
            take_profit = current_price * 0.96 # 4% Reward

        if shares_to_trade > 0:
            print("\n🛡️ TRADE EXECUTION PARAMETERS (1:2 R:R) 🛡️")
            print(f"Entry Price:    ${current_price:.2f}")
            print(f"Confidence:     {confidence*100:.1f}%")
            print(f"Shares to Trade:{shares_to_trade}")
            print(f"Stop Loss:      ${stop_loss:.2f} (Max Loss: 2%)")
            print(f"Take Profit:    ${take_profit:.2f} (Target Gain: 4%)")
        else:
            print("\n⚠️ Signal generated, but confidence too low to buy 1 full share.")

        # If the system generated a valid trade, pull the trigger!
        if shares_to_trade > 0:
            execute_bracket_order(
                symbol="GOOG", 
                qty=shares_to_trade, 
                side=trade_decision, 
                take_profit_price=take_profit, 
                stop_loss_price=stop_loss
            )

# =============================================================================
# RUN LIVE SYSTEM (CLOUD FIX)
# =============================================================================
import joblib

# 1. Load the pre-trained brain from your repository files
master_model = joblib.load('master_xgb_model.pkl')
master_scaler = joblib.load('master_scaler.pkl')

# 2. Define the exact features the model is expecting
feature_cols = [
    'Close_EMA7', 'Close_SMA7', 'Close_STD7', 'BB_Lower', 'Momentum', 
    'Low', 'BB_Middle', 'ATR', 'RSI', 'MACD', 'QQQ_Momentum', 'VIX_Level', 'GOOG_vs_QQQ_Rel'
]

# 3. Execute!
execute_trading_system(master_model, master_scaler, feature_cols)