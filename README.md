# ⚡ OmniStock AI: Universal Market Intelligence

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00.svg?logo=tensorflow)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C.svg?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg?logo=streamlit)

OmniStock AI is an institutional-grade, full-stack quantitative financial dashboard. It fuses deep learning (LSTM) for price forecasting with natural language processing (FinBERT) for real-time market sentiment analysis. 

Built to simulate a true quantitative analyst's workflow, OmniStock bypasses traditional retail noise to deliver mathematically grounded T+1 predictive signals.

---

## 🚀 The "King Engine" Architecture

OmniStock operates on a dual-stage neural architecture:

1. **The Math (LSTM Predictor):** A Long Short-Term Memory neural network trained on a heavily engineered 12-dimensional matrix (including SMA, EMA, MACD, RSI, and Volatility). It analyzes the last 60 trading days to forecast the T+1 closing price.
2. **The Crowd (NLP Sentiment):** A real-time web scraper that intercepts financial news and processes it through a hybrid NLP pipeline: **FinBERT** (for deep financial context) and **VADER** (for raw emotional polarity). 

The Engine cross-references the expected LSTM return with the NLP sentiment to generate a final weighted market signal (Strong Buy, Lean Buy, Hold, Lean Sell, Strong Sell), complete with trap-avoidance logic (Bull/Bear traps).

---

## 📊 Core Features

* **Real-Time Global Search:** Dynamically resolves company names to their correct financial tickers using API interception.
* **Interactive Candlestick Matrix:** Dark-mode native Plotly integration featuring raw price action overlaid with 20-day and 50-day moving averages.
* **T+1 Capital Simulator:** Allows users to input hypothetical investment capital and calculates exact fractional share purchasing power and projected T+1 P/L.
* **30-Day Proxy Backtest Engine:** A lightweight historical simulator that applies the AI's core momentum signals (MACD vs Signal Line + RSI constraints) against the last 30 days to generate a localized Win Rate against the standard Buy-and-Hold market baseline.
* **Under the Hood (Data Export):** Real-time access to the engineered 12-dimensional Pandas DataFrame feeding the LSTM, complete with CSV export capabilities for external quantitative analysis.

---

## 🛠️ Tech Stack

* **Machine Learning:** TensorFlow / Keras (LSTM)
* **Natural Language Processing:** HuggingFace Transformers (ProsusAI/finbert), NLTK (VADER)
* **Data Engineering:** Pandas, NumPy, Scikit-learn (MinMaxScaler)
* **Data Pipelines:** yfinance, urllib, XML parsing
* **Frontend / Visualization:** Streamlit, Plotly Graph Objects

---

## ⚙️ Local Installation & Execution

To run OmniStock AI locally, ensure you have Python 3.10+ installed. 

```bash
# 1. Clone the repository
git clone [https://github.com/AmartyaSen23/OmniStock-AI.git](https://github.com/YourUsername/OmniStock-AI.git)
cd OmniStock-AI

# 2. Install the required dependencies
pip install -r requirements.txt

# 3. Ignite the terminal
streamlit run app.py