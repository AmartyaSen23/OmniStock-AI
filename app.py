import streamlit as st
import yfinance as yf
yf.set_tz_cache_location("omnistock_cache") # Forces timezone data into a safe, writable folder!
import pandas as pd
import numpy as np
import urllib.request
import xml.etree.ElementTree as ET
import datetime

# --- NEW IMPORTS ---
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & UI SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="OmniStock AI", page_icon="⚡", layout="wide")
st.title("⚡ OmniStock AI: Universal Market Intelligence")
st.markdown("---")

# ---------------------------------------------------------
# 2. CACHED AI MODELS (The Weight Transfer Protocol)
# ---------------------------------------------------------
@st.cache_resource
def load_heavy_artillery():
    # 1. Load NLP
    nltk.download('vader_lexicon', quiet=True)
    vader = SentimentIntensityAnalyzer()
    finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    
    # 2. Rebuild the LSTM Blueprint manually to bypass Keras 3 bugs
    lstm_model = Sequential()
    lstm_model.add(Input(shape=(60, 12))) # 60 Time Steps, 12 Features
    lstm_model.add(LSTM(units=64, return_sequences=True))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(units=64, return_sequences=False))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(units=1))
    
    # 3. Load ONLY the trained math (weights), ignoring the broken config!
    lstm_model.load_weights('OmniStock_LSTM_Best.h5')
    
    return vader, finbert, lstm_model

with st.spinner("Booting Neural Cores..."):
    vader_analyzer, finbert_analyzer, lstm_model = load_heavy_artillery()
    
# ---------------------------------------------------------
# 3. THE ENGINE FUNCTIONS
# ---------------------------------------------------------
def forge_universal_data(ticker, start_date, end_date):
    # Rip out the manual requests.Session() code!
    # The upgraded yfinance engine handles stealth mode automatically now.
    
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
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

def get_live_sentiment(ticker):
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        root = ET.fromstring(response.read())
        headlines = [item.find('title').text for item in root.findall('./channel/item') if item.find('title').text]
        
        if not headlines:
            return 0, "⚪ NEUTRAL", 100.0, []
            
        total_score, penalties = 0, 0
        for text in headlines:
            v_score = vader_analyzer.polarity_scores(text)['compound']
            f_res = finbert_analyzer(text)[0]
            f_score = f_res['score'] if f_res['label'] == 'positive' else (-f_res['score'] if f_res['label'] == 'negative' else 0)
            
            h_score = (0.6 * f_score) + (0.4 * v_score)
            if (v_score > 0.1 and f_score < -0.1) or (v_score < -0.1 and f_score > 0.1):
                penalties += 1
            total_score += h_score
            
        final_sent = total_score / len(headlines)
        conf = max(10.0, 100.0 - ((penalties / len(headlines)) * 100))
        
        if final_sent > 0.25: em = "🟢 EXTREME HYPE"
        elif final_sent > 0.05: em = "🟢 OPTIMISTIC"
        elif final_sent < -0.25: em = "🔴 PANIC SELLING"
        elif final_sent < -0.05: em = "🔴 FEARFUL"
        else: em = "⚪ NEUTRAL"
            
        return final_sent, em, conf, headlines[:5]
    except:
        return 0, "⚪ OFFLINE", 0.0, []

def get_lstm_prediction(ticker):
    end_d = datetime.date.today().strftime('%Y-%m-%d')
    start_d = (datetime.date.today() - datetime.timedelta(days=150)).strftime('%Y-%m-%d')
    df = forge_universal_data(ticker, start_d, end_d)
    
    if df is None or df.empty or len(df) < 60:
        st.error("🚨 Market API Intercepted: Yahoo Finance temporarily blocked the cloud server's IP. Please wait 60 seconds and click Engage again.")
        st.stop()
        
    # --- THE TITANIUM SHIELD ---
    # Explicitly lock in our 12 exact math columns. 
    # This completely blocks any rogue strings ('Date', 'Ticker') from crashing the scaler!
    core_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'SMA_20', 'EMA_50', 'Volatility', 'RSI_14', 'MACD', 'Signal_Line']
    
    # Filter the dataframe to ONLY those 12 columns, then cast to float
    last_60 = df[core_features].tail(60).astype(float)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(last_60)
    
    X_latest = np.array([scaled_data])
    scaled_pred = lstm_model.predict(X_latest, verbose=0)
    
    # De-compress using the exact length of our core_features (12)
    dummy = np.zeros((1, len(core_features)))
    close_idx = core_features.index('Close')
    dummy[0, close_idx] = scaled_pred[0][0]
    
    predicted_price = scaler.inverse_transform(dummy)[0, close_idx]
    current_price = last_60['Close'].iloc[-1]
    
    return current_price, predicted_price

# ---------------------------------------------------------
# 4. FRONTEND DASHBOARD
# ---------------------------------------------------------
st.sidebar.header("⚙️ Target Acquisition")
target_ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, NVDA, TSLA):", "NVDA").upper()
run_button = st.sidebar.button("ENGAGE OMNISTOCK")

if run_button:
    st.sidebar.success(f"Uplink established for {target_ticker}")
    
    with st.spinner("Forging Data & Running Neural Pipelines..."):
        # 1. Fetch Prices
        curr_price, pred_price = get_lstm_prediction(target_ticker)
        exp_return = ((pred_price - curr_price) / curr_price) * 100
        
        # 2. Fetch Sentiment
        sent_val, emotion, conf, top_news = get_live_sentiment(target_ticker)
        
        # 3. King Engine Logic
        if exp_return > 1.0 and sent_val > 0.05 and conf >= 50: signal, strat, color = "🟢 STRONG BUY", "Math + Emotion Aligned. Execute.", "green"
        elif exp_return < -1.0 and sent_val < -0.05: signal, strat, color = "🔴 STRONG SELL", "Accelerated Decay Detected. Liquidate.", "red"
        elif exp_return > 1.0 and sent_val < -0.20: signal, strat, color = "🟡 HOLD (BEAR TRAP)", "Math upside, but violent negative news.", "orange"
        elif exp_return < -1.0 and sent_val > 0.20: signal, strat, color = "🟡 HOLD (BULL TRAP)", "Fake Pump Detected. Do not buy.", "orange"
        else: signal, strat, color = "⚪ NEUTRAL / HOLD", "No dominant confluence. Protect Capital.", "gray"

    # --- DISPLAY METRICS ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${curr_price:.2f}")
    col2.metric("LSTM Predicted (T+1)", f"${pred_price:.2f}", f"{exp_return:+.2f}%")
    col3.metric("NLP Market Emotion", emotion, f"Conf: {conf:.1f}%")
    
    st.markdown("---")
    
    # --- FINAL DECISION BANNER ---
    st.markdown(f"""
    <div style="background-color:{color};padding:20px;border-radius:10px;text-align:center;">
        <h2 style="color:white;margin:0;">KING ENGINE SIGNAL: {signal}</h2>
        <p style="color:white;font-size:18px;margin-top:10px;">{strat}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📰 Real-Time Intercepted Intelligence")
    for n in top_news:
        st.write(f"- {n}")