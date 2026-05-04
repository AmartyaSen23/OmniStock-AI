import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
import gc
import urllib.request
import xml.etree.ElementTree as ET
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

# ==========================================
# 1. GLOBAL INITIALIZATION (Crucial for API speed)
# ==========================================
# Models load ONCE when the server boots, not on every API call.
# This makes your endpoint lightning fast.
nltk.download('vader_lexicon', quiet=True)
print("Loading VADER...")   
vader = SentimentIntensityAnalyzer()

print("Loading FinBERT... (This might take a moment)")
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
print("✅ Models loaded. AI Engine is LIVE.")
# ==========================================
# 1.5 GLOBAL LSTM INITIALIZATION
# ==========================================
print("Booting Neural Cores (LSTM)...")
lstm_model = Sequential()
lstm_model.add(Input(shape=(60, 12))) 
lstm_model.add(LSTM(units=64, return_sequences=True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=64, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1))

# Make sure 'OmniStock_LSTM_Best.h5' is in the exact same folder!
try:
    lstm_model.load_weights('OmniStock_LSTM_Best.h5')
    print("✅ LSTM Weights loaded.")
except Exception as e:
    print(f"⚠️ Warning: LSTM weights not found. {e}")

# ==========================================
# 2. THE CORE ENGINE
# ==========================================
def analyze_sentiment(ticker: str) -> dict:
    # We use a specific Chrome/Mac User-Agent to trick the firewall
    url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        response = urllib.request.urlopen(req, timeout=10) # Added timeout to prevent hanging
        root = ET.fromstring(response.read())
        headlines = [item.find('title').text for item in root.findall('./channel/item') if item.find('title').text]
        
        # If Google STILL blocks us, we use Yahoo as a "Safety Net"
        if not headlines:
            stock = yf.Ticker(ticker)
            headlines = [item.get('title', '') for item in stock.news if item.get('title')]
            
        if not headlines:
            return {
                "ticker": ticker.upper(),
                "ensemble_score": 0.0,
                "confidence": 100.0,
                "trading_signal": "⚪ NEUTRAL",
                "headlines_analyzed": 0,
                "top_headlines": []
            }
            
        total_score, penalties = 0, 0
        
        # Analyze each headline
        for text in headlines:
            v_score = vader.polarity_scores(text)['compound']
            f_res = finbert(text)[0]
            f_score = f_res['score'] if f_res['label'] == 'positive' else (-f_res['score'] if f_res['label'] == 'negative' else 0)
            
            # Note: Your code is doing 0.6 FinBERT and 0.4 VADER. 
            # This is actually better than your prompt (which said the reverse) 
            # because FinBERT has better financial context.
            h_score = (0.6 * f_score) + (0.4 * v_score)
            
            # Penalty for model disagreement
            if (v_score > 0.1 and f_score < -0.1) or (v_score < -0.1 and f_score > 0.1):
                penalties += 1
            total_score += h_score
            
        # Final calculations
        final_sent = total_score / len(headlines)
        conf = max(10.0, 100.0 - ((penalties / len(headlines)) * 100))

        # Signal mapping
        if final_sent > 0.25: action = "🟢 EXTREME HYPE"
        elif final_sent > 0.05: action = "🟢 OPTIMISTIC"
        elif final_sent < -0.25: action = "🔴 PANIC SELLING"
        elif final_sent < -0.05: action = "🔴 FEARFUL"
        else: action = "⚪ NEUTRAL"
        
        # Consistent Success Return
        # Consistent Success Return
        return {
            "ticker": ticker.upper(),
            "ensemble_score": round(final_sent, 4),
            "confidence": round(conf, 2),
            "trading_signal": action,
            "headlines_analyzed": len(headlines),
            "top_headlines": headlines[:5]  # <-- ADDED THIS LINE
        }
        
    except Exception as e:
        # Catch errors gracefully without crashing the server
        return {
            "ticker": ticker.upper(),
            "ensemble_score": 0.0,
            "confidence": 0.0,
            "trading_signal": f"⚪ OFFLINE/ERROR",
            "error_details": str(e),
            "headlines_analyzed": 0,
            "top_headlines": []
        }

# ==========================================
# 3. THE QUANTITATIVE FORGE (LSTM)
# ==========================================
def forge_universal_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches and engineers the 12 features required for the LSTM."""
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
        
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['Returns'] = df['Close'].pct_change()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df.dropna(inplace=True)
    return df

def generate_lstm_prediction(ticker: str) -> dict:
    """Runs the data through the LSTM and returns the forecast."""
    end_d = datetime.date.today().strftime('%Y-%m-%d')
    start_d = (datetime.date.today() - datetime.timedelta(days=150)).strftime('%Y-%m-%d')
    df = forge_universal_data(ticker, start_d, end_d)
    
    if df is None or df.empty or len(df) < 60:
        return {"error": "Insufficient market data or API block."}
        
    core_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'SMA_20', 'EMA_50', 'Volatility', 'RSI_14', 'MACD', 'Signal_Line']
    last_60 = df[core_features].tail(60).astype(float)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(last_60)
    
    X_latest = np.array([scaled_data])
    scaled_pred = lstm_model.predict(X_latest, verbose=0)
    
    dummy = np.zeros((1, len(core_features)))
    close_idx = core_features.index('Close')
    dummy[0, close_idx] = scaled_pred[0][0]
    
    predicted_price = float(scaler.inverse_transform(dummy)[0, close_idx])
    current_price = float(last_60['Close'].iloc[-1])

    # Aggressive memory wipe to protect your server
    del last_60, scaled_data, X_latest, dummy 
    gc.collect()
    
    return {
        "ticker": ticker.upper(),
        "current_price": round(current_price, 2),
        "predicted_price": round(predicted_price, 2),
        "expected_return_pct": round(((predicted_price - current_price) / current_price) * 100, 2)
    }
