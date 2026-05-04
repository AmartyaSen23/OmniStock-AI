# main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from engine import analyze_sentiment, generate_lstm_prediction
from supabase import create_client, Client
import time
import os
from dotenv import load_dotenv

# ==========================================
# 0. DATABASE CONNECTION
# ==========================================
load_dotenv() 

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("🚨 Missing Supabase environment variables! Check your .env file.")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize the FastAPI application
app = FastAPI(
    title="OmniStock Core API",
    description="High-performance AI engine with background telemetry.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    ticker: str

# ==========================================
# 1. TELEMETRY WORKER (Runs in the background)
# ==========================================
def log_to_supabase(ticker: str, endpoint: str, latency: float, output: str):
    """Silently pushes execution data to PostgreSQL without slowing down the API."""
    try:
        data = {
            "ticker": ticker,
            "endpoint": endpoint,
            "latency_ms": latency,
            "engine_output": str(output)
        }
        supabase.table("api_logs").insert(data).execute()
        print(f"📡 Telemetry Logged: {ticker} | {latency}ms")
    except Exception as e:
        print(f"⚠️ DB Logging Error: {e}")

# ==========================================
# 2. API ROUTES
# ==========================================
@app.get("/", tags=["Health Check"])
def system_status():
    return {"status": "online", "message": "OmniStock Engine & Supabase Linked."}

@app.post("/api/v1/analyze", tags=["Trading Engine"])
def trigger_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    start_time = time.time()
    ticker = request.ticker.strip()
    
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol cannot be empty.")
        
    try:
        result = analyze_sentiment(ticker)
        
        # Calculate how many milliseconds the RTX 4060 and CPU took to process
        latency = round((time.time() - start_time) * 1000, 2)
        
        # Dispatch the save command to the background
        background_tasks.add_task(log_to_supabase, ticker, "/analyze", latency, result.get("trading_signal", "Error"))
        
        return {"success": True, "data": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine runtime error: {str(e)}")

@app.post("/api/v1/predict", tags=["Trading Engine"])
def trigger_prediction(request: AnalysisRequest, background_tasks: BackgroundTasks):
    start_time = time.time()
    ticker = request.ticker.strip()
    
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol cannot be empty.")
        
    try:
        result = generate_lstm_prediction(ticker)
        if "error" in result:
            raise HTTPException(status_code=503, detail=result["error"])
            
        latency = round((time.time() - start_time) * 1000, 2)
        
        # Dispatch to background
        background_tasks.add_task(log_to_supabase, ticker, "/predict", latency, f"Yield: {result.get('expected_return_pct')}%")
            
        return {"success": True, "data": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LSTM runtime error: {str(e)}")