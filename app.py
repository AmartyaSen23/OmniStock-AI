import streamlit as st
import yfinance as yf
yf.set_tz_cache_location("omnistock_cache") # Forces timezone data into a safe, writable folder!
import pandas as pd
import numpy as np
import urllib.request
import xml.etree.ElementTree as ET
import datetime
import urllib.parse
import json

# --- NEW IMPORTS ---
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import plotly.graph_objects as go
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
@st.cache_data(ttl=3600)
def resolve_ticker(query):
    if not query:
        return "NVDA", None, []
        
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(query)}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        data = json.loads(response.read())
        quotes = data.get('quotes', [])
        
        if not quotes:
            return query.upper(), None, []
            
        # The top match
        best_match = quotes[0]['symbol']
        best_name = quotes[0].get('shortname', quotes[0].get('longname', best_match))
        
        # The alternative matches (in case the AI guessed wrong)
        alts = []
        for q in quotes[1:4]:
            if 'symbol' in q:
                name = q.get('shortname', q.get('longname', q['symbol']))
                alts.append(f"{name} ({q['symbol']})")
                
        return best_match, best_name, alts
    except Exception:
        return query.upper(), None, []

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
    url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
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

def get_fundamental_intel(ticker):
    try:
        stock = yf.Ticker(ticker)
        
        # Bypassing the blocked .info endpoint by using .fast_info
        fast = stock.fast_info
        mcap = fast['market_cap']
        high_52 = fast['year_high']
        low_52 = fast['year_low']

        # P/E Ratio is the only metric strictly locked inside .info
        # We will try to fetch it, but gracefully catch it if Yahoo blocks us
        try:
            pe = stock.info.get('trailingPE', 'N/A')
        except:
            pe = 'Hidden'

        # Cleanly format the Market Cap (Trillions, Billions, Millions)
        if mcap >= 1e12: mcap_str = f"${mcap/1e12:.3f}T"
        elif mcap >= 1e9: mcap_str = f"${mcap/1e9:.3f}B"
        elif mcap >= 1e6: mcap_str = f"${mcap/1e6:.3f}M"
        else: mcap_str = f"${mcap}"

        pe_str = f"{pe:.2f}" if isinstance(pe, (int, float)) else str(pe)
        high_str = f"${high_52:.2f}" if isinstance(high_52, (int, float)) else str(high_52)
        low_str = f"${low_52:.2f}" if isinstance(low_52, (int, float)) else str(low_52)

        return mcap_str, pe_str, high_str, low_str
        
    except Exception as e:
        print(f"Intel Error: {e}")
        return "N/A", "N/A", "N/A", "N/A"

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
    
    return current_price, predicted_price, df

# ---------------------------------------------------------
# 4. FRONTEND DASHBOARD
# ---------------------------------------------------------
st.sidebar.header("⚙️ Target Acquisition")
raw_input = st.sidebar.text_input("Enter Company Name or Ticker (e.g., Tata, Meta, AAPL):", "NVDA").strip()
target_ticker, resolved_name, alt_matches = resolve_ticker(raw_input)
if resolved_name and raw_input.upper() != target_ticker.upper():
    st.sidebar.success(f"💡 Auto-Targeting: **{resolved_name}** (`{target_ticker}`)")
    
    if alt_matches:
        with st.sidebar.expander("Not what you meant?"):
            st.write("Other possibilities:")
            for alt in alt_matches:
                st.markdown(f"- {alt}")

run_button = st.sidebar.button("ENGAGE OMNISTOCK")

# --- THE MEMORY BANK (SESSION STATE) ---
# This stops the app from wiping the screen when the user clicks the simulator
if 'engine_engaged' not in st.session_state:
    st.session_state.engine_engaged = False

# 1. FETCH AND STORE (Only runs when the button is clicked)
if run_button:
    st.session_state.engine_engaged = True
    st.session_state.active_ticker = target_ticker
    st.sidebar.success(f"Uplink established for {target_ticker}")
    
    with st.spinner("Forging Data & Running Neural Pipelines..."):
        # We save the results directly into Streamlit's memory
        st.session_state.curr_price, st.session_state.pred_price, st.session_state.raw_data = get_lstm_prediction(target_ticker)
        st.session_state.sent_val, st.session_state.emotion, st.session_state.conf, st.session_state.top_news = get_live_sentiment(target_ticker)
        st.session_state.mcap, st.session_state.pe, st.session_state.high52, st.session_state.low52 = get_fundamental_intel(target_ticker)

# 2. RENDER DASHBOARD (Reads from memory so it never resets!)
if st.session_state.engine_engaged:
    # Pull data from memory
    curr_price = st.session_state.curr_price
    pred_price = st.session_state.pred_price
    raw_data = st.session_state.raw_data
    sent_val = st.session_state.sent_val
    emotion = st.session_state.emotion
    conf = st.session_state.conf
    top_news = st.session_state.top_news
    active_ticker = st.session_state.active_ticker
    
    exp_return = ((pred_price - curr_price) / curr_price) * 100
    
    # 3. King Engine Logic
    if exp_return > 1.0 and sent_val > 0.05 and conf >= 50: signal, strat, color = "🟢 STRONG BUY", "Math + Emotion Aligned. Execute.", "green"
    elif exp_return < -1.0 and sent_val < -0.05: signal, strat, color = "🔴 STRONG SELL", "Accelerated Decay Detected. Liquidate.", "red"
    elif exp_return > 1.0 and sent_val < -0.20: signal, strat, color = "🟡 HOLD (BEAR TRAP)", "Math upside, but violent negative news.", "orange"
    elif exp_return < -1.0 and sent_val > 0.20: signal, strat, color = "🟡 HOLD (BULL TRAP)", "Fake Pump Detected. Do not buy.", "orange"
    else: signal, strat, color = "⚪ NEUTRAL / HOLD", "No dominant confluence. Protect Capital.", "gray"

    # --- DISPLAY METRICS ---
    st.subheader("📡 Live Neural Telemetry")
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${curr_price:.2f}")
    col2.metric("LSTM Predicted (T+1)", f"${pred_price:.2f}", f"{exp_return:+.2f}%")
    col3.metric("NLP Market Emotion", emotion, f"Conf: {conf:.1f}%")

    # --- NEW: FUNDAMENTAL METRICS ---
    st.markdown("---")
    st.subheader("🏢 Fundamental Corporate Intel")
    col4, col5, col6, col7 = st.columns(4)
    col4.metric("Market Capitalization", st.session_state.mcap)
    col5.metric("P/E Ratio (Valuation)", st.session_state.pe)
    col6.metric("52-Week High", st.session_state.high52)
    col7.metric("52-Week Low", st.session_state.low52)
    
    st.markdown("---")
    
    # --- FINAL DECISION BANNER ---
    st.markdown(f"""
    <div style="background-color:{color};padding:20px;border-radius:10px;text-align:center;">
        <h2 style="color:white;margin:0;">KING ENGINE SIGNAL: {signal}</h2>
        <p style="color:white;font-size:18px;margin-top:10px;">{strat}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # --- NEW: T+1 CAPITAL SIMULATOR ---
    st.subheader("💰 T+1 Capital Simulator")
    st.write("Simulate your projected returns based on the King Engine's LSTM forecast.")
    
    sim_col1, sim_col2 = st.columns([1, 2])
    
    with sim_col1:
        # User inputs their cash stack
        capital = st.number_input("Investment Capital ($):", min_value=10.0, value=1000.0, step=100.0)
        
    with sim_col2:
        # The Math
        shares_bought = capital / curr_price
        projected_value = shares_bought * pred_price
        projected_profit = projected_value - capital
        
        profit_color = "normal" if projected_profit == 0 else ("green" if projected_profit > 0 else "red")
        trend_icon = "📈" if projected_profit > 0 else "📉"
        
        st.info(f"**Simulation Results:**\n"
                f"- **Purchasing Power:** `{shares_bought:.4f} Shares`\n"
                f"- **Projected T+1 Value:** `${projected_value:.2f}`\n"
                f"- **Estimated P/L:** :{profit_color}[{trend_icon} **${projected_profit:+.2f}**]")
                
    st.markdown("---")

    # --- PLOTLY CANDLESTICK MATRIX ---
    st.subheader(f"📊 Market Matrix: {active_ticker}")
    
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(x=raw_data.index,
                open=raw_data['Open'], high=raw_data['High'],
                low=raw_data['Low'], close=raw_data['Close'],
                name='Market Price'))
    
    fig.add_trace(go.Scatter(x=raw_data.index, y=raw_data['SMA_20'], 
                             line=dict(color='#00ff00', width=1.5), name='SMA 20'))
    fig.add_trace(go.Scatter(x=raw_data.index, y=raw_data['EMA_50'], 
                             line=dict(color='#ff00ff', width=1.5), name='EMA 50'))
    
    fig.update_layout(
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --- TECHNICAL SUB-PLOTS (RSI & MACD) ---
    st.markdown("---")
    st.subheader("🎛️ Technical Momentum Indicators")
    
    col_rsi, col_macd = st.columns(2)
    
    with col_rsi:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=raw_data.index, y=raw_data['RSI_14'], line=dict(color='#ff9900', width=2), name='RSI 14'))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        fig_rsi.update_layout(template='plotly_dark', height=300, margin=dict(l=0, r=0, t=30, b=0), title="Relative Strength Index (RSI)")
        st.plotly_chart(fig_rsi, use_container_width=True)
        
    with col_macd:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=raw_data.index, y=raw_data['MACD'], line=dict(color='#00bfff', width=2), name='MACD'))
        fig_macd.add_trace(go.Scatter(x=raw_data.index, y=raw_data['Signal_Line'], line=dict(color='#ff00ff', width=2), name='Signal'))
        fig_macd.update_layout(template='plotly_dark', height=300, margin=dict(l=0, r=0, t=30, b=0), title="MACD & Signal Line")
        st.plotly_chart(fig_macd, use_container_width=True)

    # --- THE UNDER-THE-HOOD EXPANDER ---
    st.markdown("---")
    with st.expander("🔬 Under the Hood: Raw Engineering Data"):
        st.write("The exact multi-dimensional mathematical matrix feeding the LSTM Neural Network.")
        st.dataframe(raw_data.iloc[::-1].head(15), use_container_width=True)

    # --- NEWS SECTION ---
    st.markdown("---")
    st.subheader("📰 Real-Time Intercepted Intelligence")
    for n in top_news:
        st.write(f"- {n}")